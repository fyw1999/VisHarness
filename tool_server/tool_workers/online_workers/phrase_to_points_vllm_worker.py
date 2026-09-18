"""
A model worker executes the model.
"""
import asyncio
import os
os.environ.pop("LD_LIBRARY_PATH", None)
import uuid
import re
import argparse
import torch
import numpy as np
from PIL import ImageDraw
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import (
    BaseToolWorker,
    build_worker_error_response,
)
import traceback
from scipy.spatial import cKDTree
from transformers import AutoProcessor, AutoModelForImageTextToText
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

def get_downsample_factor(img_name):
    if '.' in img_name:
        base_name = img_name.rsplit('.', 1)[0]
    else:
        base_name = img_name

    super_res_count = 0
    while base_name.endswith('_4x'):
        super_res_count += 1
        base_name = base_name[:-3]

    downsample_factor = 4 ** super_res_count

    return downsample_factor

def parse_points(points_str: str, img):
    match = re.search(r'coords="([\d\s]+)', points_str)
    if not match:
        print("not found coords")
        return np.empty((0, 2))

    coords_str = match.group(1)

    raw_nums = []
    for n in coords_str.split():
        if n.isdigit():
            raw_nums.append(int(n))

    if len(raw_nums) < 4:
        print("no complete points found in data")
        return np.empty((0, 2))

    point_data_all = raw_nums[1:]

    num_complete_points = len(point_data_all) // 3
    if num_complete_points == 0:
        return np.empty((0, 2))

    point_data = point_data_all[:num_complete_points * 3]

    width, height = img.size

    raw_array = np.array(point_data)

    points_norm = raw_array.reshape(-1, 3)[:, 1:3]

    scale = 1000.0
    points_abs = points_norm / scale * [width, height]

    return points_abs

def show_points(points, img, image_name):
    down_factor = get_downsample_factor(image_name)
    if down_factor != 1:
        current_width, current_height = img.size
        new_width = current_width // down_factor
        new_height = current_height // down_factor
        img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        points = points / down_factor

    draw = ImageDraw.Draw(img)
    num_points = len(points)
    MIN_RADIUS = 1.0
    MAX_RADIUS = 5.0
    BETA = 0.3
    radii = []

    if num_points <= 1:
        radii = [MAX_RADIUS] * num_points
    else:
        tree = cKDTree(points)

        dists, _ = tree.query(points, k=2)

        nearest_dists = dists[:, 1]

        calculated_radii = nearest_dists * BETA

        radii = np.clip(calculated_radii, MIN_RADIUS, MAX_RADIUS)

    for (x, y), r in zip(points, radii):
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill='DeepPink',
            outline='white',
            width=1
        )

    return img

def format_np_one_decimal(t):
    t = t.tolist()  # 转 Python list
    return [[float(f"{v:.1f}".rstrip('0').rstrip('.')) for v in row] for row in t]

class PhraseToPointWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_name = "",
                 worker_addr = "auto",
                 no_register = False,
                 model_path = "",
                 tool_name = "Point",
                 load_8bit = False,
                 load_4bit = False,
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 max_tokens = 2048,
                 max_batch_size = 12,
                 gpu_memory_utilization = 0.5,
                 max_num_batched_tokens = 40000
                 ):
        self.max_tokens = max_tokens
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = int(max_num_batched_tokens)
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
        logger.info(f"Initializing model {self.tool_name}...")
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_tokens,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        logger.info(f"load model {self.tool_name} success")

    def _prepare_image_request(self, image_name, image_data, phrase):
        image_pil = bytes_to_pil(image_data)
        text_prompt = f"Point to {phrase} in the image."

        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="image", image=image_pil),
                    dict(type="text", text=text_prompt),
                ],
            }
        ]

        prompt_str = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_input = {
            "prompt": prompt_str,
            "multi_modal_data": {"image": image_pil},
        }
        return image_name, image_pil, prompt_input

    def _build_image_result(self, image_name, image, phrase, generated_text):
        points = parse_points(generated_text, image)
        visual_img = show_points(points, image, image_name)

        return image_name, {
            "text_response": (
                f"For the phrase '{phrase}', there are {len(points)} "
                f"corresponding objects in the image {image_name}. "
                "The centers of the corresponding objects are marked with pink dots."
            ),
            "visual_image": pil_to_bytes(visual_img),
            "points": points.tolist(),
        }

    async def _generate_one_image(self, image_name, image_data, phrase):
        image_name, image, prompt_input = self._prepare_image_request(
            image_name,
            image_data,
            phrase,
        )
        request_id = f"{worker_id}-{uuid.uuid4().hex}"
        final_output = None

        logger.info(
            f"Submit vLLM request {request_id} for image {image_name}"
        )
        async for output in self.engine.generate(
            prompt=prompt_input,
            sampling_params=self.sampling_params,
            request_id=request_id,
        ):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise RuntimeError(
                f"vLLM returned no output for request {request_id}"
            )

        generated_text = final_output.outputs[0].text.strip()
        return self._build_image_result(
            image_name,
            image,
            phrase,
            generated_text,
        )

    async def async_generate(self, params):
        phrase = params.get("phrase", None)
        image_dict = params.get("image_dict", None)  # dict of {image_name: base64_image_data}

        if image_dict is None or phrase is None:
            logger.error("Missing required inputs: image or prompts.")
            return {"message": "Missing required inputs: image or prompts", "status": "error"}

        phrase = phrase.strip()
        ret = {"message": "", "status": ""}
        logger.info("read arguments success")

        try:
            results = {}
            image_items = list(image_dict.items())
            total_images = len(image_items)
            logger.info(
                f"Total images to process: {total_images}, "
                f"max concurrent images per request: {self.max_batch_size}"
            )

            for i in range(0, total_images, self.max_batch_size):
                batch_items = image_items[i : i + self.max_batch_size]

                logger.info(
                    f"Begin to submit PhraseToPoint requests for chunk "
                    f"{i // self.max_batch_size + 1}"
                )
                tasks = [
                    asyncio.create_task(
                        self._generate_one_image(
                            image_name,
                            image_data,
                            phrase,
                        )
                    )
                    for image_name, image_data in batch_items
                ]

                try:
                    batch_outputs = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

                for image_name, image_result in batch_outputs:
                    results[image_name] = image_result

            ret["message"] = "Tool PhraseToPoint executed successfully."
            ret["results"] = results
            ret["status"] = "success"

        except Exception as e:
            remote_traceback = traceback.format_exc()
            logger.error(f"Error when using PhraseToPoint: {e}")
            logger.error(remote_traceback)
            ret = build_worker_error_response(
                self.tool_name,
                e,
                remote_traceback=remote_traceback,
            )

        return ret

    async def generate_gate_async(self, params):
        return await self.async_generate(params)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="PhraseToPoint")
    parser.add_argument("--worker_name", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, required=True)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--max_batch_size", type=int, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, required=True)
    parser.add_argument("--max_num_batched_tokens", type=int, required=True)
    parser.add_argument("--load-8bit", type=str2bool, default=False, help="Use 8-bit quantization")
    parser.add_argument("--load-4bit",  type=str2bool, default=False, help="Use 4-bit quantization (nf4)")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = PhraseToPointWorker(
        controller_addr=args.controller_address,
        worker_name = args.worker_name,
        worker_addr=args.worker_address,
        tool_name = args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
        load_8bit = args.load_8bit,
        load_4bit = args.load_4bit,
        max_tokens = args.max_tokens,
        max_batch_size = args.max_batch_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_num_batched_tokens = args.max_num_batched_tokens

    )
    worker.run()
