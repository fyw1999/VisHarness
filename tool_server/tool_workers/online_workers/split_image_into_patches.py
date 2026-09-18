"""Concurrent CPU worker for splitting images into overlapping patches.

The image-processing algorithm and response schema intentionally match the
serial implementation in ``split_image_into_patches_serial.py``.  Independent
requests execute in a bounded thread pool instead of blocking FastAPI's event
loop.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import argparse
from PIL import Image, ImageDraw, ImageFont
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None


def get_direct_super_resolution_factor(image_name):
    """Return the scale encoded by trailing ``_4x`` operations only."""
    base_name = image_name.rsplit(".", 1)[0] if "." in image_name else image_name
    factor = 1
    while base_name.endswith("_4x"):
        factor *= 4
        base_name = base_name[:-3]
    return factor


def downsample_direct_super_resolution_visual(image, image_name):
    """Restore a direct SuperResolution result to its parent display size."""
    factor = get_direct_super_resolution_factor(image_name)
    if factor == 1:
        return image
    width, height = image.size
    return image.resize(
        (max(1, width // factor), max(1, height // factor)),
        resample=Image.Resampling.LANCZOS,
    )

class SplitImageIntoPatches(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_name = "",
                 worker_addr = "auto",
                 no_register = False,
                 tool_name = "SplitImageIntoPatches",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 cpu_worker_threads = None,
                 ):
        requested_workers = (
            limit_model_concurrency
            if cpu_worker_threads is None
            else cpu_worker_threads
        )
        self.cpu_worker_threads = min(
            max(1, int(requested_workers)),
            max(1, int(limit_model_concurrency)),
        )
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self.cpu_worker_threads,
            thread_name_prefix="split-image-patches",
        )
        super().__init__(
            controller_addr,
            worker_name,
            worker_addr,
            no_register,
            tool_name,
            limit_model_concurrency,
            host,
            port
            )

    def init_model(self):
        logger.info(f"No need to initialize model {self.tool_name}.")
        logger.info(
            "SplitImageIntoPatches CPU worker threads: "
            f"{self.cpu_worker_threads} (request concurrency limit: "
            f"{self.limit_model_concurrency})"
        )
        self.model = None

    async def generate_gate_async(self, params):
        """Run the unchanged synchronous implementation outside the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_executor,
            self.generate_gate,
            params,
        )

    def get_segments(self, total_length, step_size, min_usage_ratio=0.3):
        segments = []
        start = 0
        while start < total_length:
            end = start + step_size
            if end >= total_length:
                segments.append((start, total_length))
                break
            remaining = total_length - end
            if remaining < (step_size * min_usage_ratio):
                segments.append((start, total_length))
                break
            else:
                segments.append((start, end))
                start = end
        return segments

    def generate(self, params):
        image_dict = params.get("image_dict", None)
        if image_dict is None:
            logger.error("Missing required inputs: image.")
            ret = {"message": "No image_dict provided.", "status": "error"}
            return ret
        ret = {"message": "", "status": ""}
        results = {}
        try:
            overlap_ratio = 0.2
            for image_name, image_data in image_dict.items():
                image_pil = bytes_to_pil(image_data["image_bytes"])
                patch_size = image_data["patch_size"]
                # Extract coordinates and point indices
                img_w, img_h = image_pil.size
                results[image_name] = {
                    "patches": {},
                    "overview": None
                }
                y_segments = self.get_segments(img_h, patch_size, min_usage_ratio=0.3)
                x_segments = self.get_segments(img_w, patch_size, min_usage_ratio=0.3)
                visual_image = image_pil.copy()
                draw = ImageDraw.Draw(visual_image)
                font_size = int(img_w / 35)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default(size=font_size)

                for r_idx, (y_start, y_end) in enumerate(y_segments):
                    for c_idx, (x_start, x_end) in enumerate(x_segments):
                        seg_h = y_end - y_start
                        seg_w = x_end - x_start

                        pad_h = int(seg_h * overlap_ratio)
                        pad_w = int(seg_w * overlap_ratio)

                        crop_y1 = max(0, y_start - pad_h)
                        crop_y2 = min(img_h, y_end + pad_h)
                        crop_x1 = max(0, x_start - pad_w)
                        crop_x2 = min(img_w, x_end + pad_w)

                        patch = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        patch_name = f"{image_name}_r{r_idx + 1}_c{c_idx + 1}"
                        results[image_name]["patches"][patch_name] = {
                            "image_bytes": pil_to_bytes(patch),
                            "offset_x": crop_x1,
                            "offset_y": crop_y1,
                            "width": crop_x2 - crop_x1,
                            "height": crop_y2 - crop_y1
                        }

                        line_width = max(2, int(img_w / 200))
                        draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=line_width)

                        label_text = patch_name
                        text_bbox = draw.textbbox((x_start, y_start), label_text, font=font)
                        draw.rectangle(text_bbox, fill="yellow")
                        draw.text((x_start, y_start), label_text, fill="black", font=font)

                # The overview is model-facing context only. Restore it to the
                # parent display size when the split input is itself a direct
                # SuperResolution result (name ends in ``_4x``). The cropped
                # patches above deliberately remain at their native, enhanced
                # resolution for subsequent visual-tool calls.
                overview = downsample_direct_super_resolution_visual(
                    visual_image,
                    image_name,
                )
                results[image_name]["overview"] = pil_to_bytes(overview)

            ret["results"] = results
            ret["status"] = "success"
            ret["message"] = f"Tool SplitImageIntoPatches executed successfully."
        except ValueError as e:
            logger.error(f"Error processing line parameters: {e}")
            raise
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="SplitImageIntoPatches")
    parser.add_argument("--worker_name", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20013)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--cpu-worker-threads",
        type=int,
        default=8,
        help=(
            "Maximum number of SplitImageIntoPatches requests that execute "
            "CPU work simultaneously. It is capped by "
            "--limit-model-concurrency."
        ),
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SplitImageIntoPatches(
        controller_addr=args.controller_address,
        worker_name = args.worker_name,
        worker_addr=args.worker_address,
        tool_name = args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        cpu_worker_threads = args.cpu_worker_threads,
    )
    worker.run()
