"""Native SAM3 PointToBoxMask worker with request-level dynamic batching.

Concurrent HTTP requests are coalesced and their images are fed through the
existing native SAM3 image batch path.  A single executor owns GPU execution;
this improves throughput without racing the stateful interactive predictor or
overlapping independent CUDA memory peaks. Image backbones remain batched,
while each image's independent point prompts are decoded in bounded chunks.
"""
import asyncio
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from PIL import ImageDraw
import torch
import uuid
import gc
import argparse
from pycocotools import mask as mask_utils
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import (
    BaseToolWorker,
    build_worker_error_response,
    is_cuda_oom_error,
)
from tool_server.tool_workers.oom_utils import response_indicates_cuda_oom
import numpy as np

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

@dataclass
class _PendingRequest:
    params: dict
    future: asyncio.Future


def release_torch_memory(device=None):
    gc.collect()

    if not torch.cuda.is_available():
        return

    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception as ipc_error:
                    logger.warning(f"CUDA ipc_collect failed during cleanup: {ipc_error}")
        else:
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as ipc_error:
                logger.warning(f"CUDA ipc_collect failed during cleanup: {ipc_error}")
    except Exception as cleanup_error:
        logger.warning(f"CUDA memory cleanup failed: {cleanup_error}")


def reset_sam3_predictor_state(model):
    """Release predictor-owned feature references left behind by failed inference."""
    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        return

    predictor._features = None
    predictor._is_image_set = False
    predictor._is_batch = False


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

def show_bboxes(image, box_coords, image_name, width_ratio=0.003):
    down_factor = get_downsample_factor(image_name)
    if down_factor != 1:
        current_width, current_height = image.size
        new_width = current_width // down_factor
        new_height = current_height // down_factor
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        box_coords = box_coords / down_factor

    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    adaptive_width = max(3, int((img_width + img_height) / 2 * width_ratio))

    for box in box_coords:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        draw.rectangle(
            [x0, y0, x1, y1],
            outline='green',
            width=adaptive_width
        )
    return image

def filter_valid_box_masks(bboxes, masks_nhw):
    valid_bboxes = []
    valid_masks = []

    bboxes = np.asarray(bboxes, dtype=np.float32)
    masks_nhw = np.asarray(masks_nhw)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4 or masks_nhw.ndim != 3:
        mask_shape = masks_nhw.shape[1:] if masks_nhw.ndim == 3 else (0, 0)
        return np.empty((0, 4), dtype=np.float32), np.empty((0, *mask_shape), dtype=bool)

    for bbox, mask in zip(bboxes, masks_nhw):
        if not np.all(np.isfinite(bbox)):
            continue
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue
        if not np.any(mask):
            continue
        valid_bboxes.append(bbox)
        valid_masks.append(mask.astype(bool))

    if not valid_bboxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, *masks_nhw.shape[1:]), dtype=bool)

    return np.asarray(valid_bboxes, dtype=np.float32), np.asarray(valid_masks, dtype=bool)

class PointToBoxMaskWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_name = "",
                 worker_addr = "auto",
                 no_register = False,
                 tool_name = "",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_path = "",
                 max_batch_size = 4,
                 batch_wait_ms = 10.0,
                 max_batch_requests = 0,
                 max_points_per_forward_confidence = 96,
                 max_points_per_forward_area = 32,
                 point_mask_postprocess_budget_mb = 2048,
                 ):
        self.model_path = model_path
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_ms = max(0.0, float(batch_wait_ms))
        self.max_batch_requests = (
            max(1, int(max_batch_requests))
            if int(max_batch_requests) > 0
            else max(1, int(limit_model_concurrency))
        )
        self.max_points_per_forward = {
            "confidence": max(1, int(max_points_per_forward_confidence)),
            "area": max(1, int(max_points_per_forward_area)),
        }
        self.point_mask_postprocess_budget_bytes = (
            max(1, int(point_mask_postprocess_budget_mb)) * 1024 * 1024
        )
        self.use_bfloat16 = False
        self._request_queue = None
        self._batch_worker_task = None
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="point-to-box-mask-gpu",
        )
        self._inference_lock = threading.Lock()
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_bfloat16 = torch.cuda.is_bf16_supported()
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_bfloat16 = False
        else:
            self.device = torch.device("cpu")
            self.use_bfloat16 = False

        print(f"using device: {self.device}")
        print(f"using bfloat16: {self.use_bfloat16}")
        logger.info(f"using device: {self.device}, bfloat16 supported/enabled: {self.use_bfloat16}")

        if self.device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        sam3_root = os.path.join(os.path.dirname(sam3.__file__))
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        self.sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=self.model_path,
            enable_inst_interactivity=True
        ).to(self.device)
        self.sam3_model.eval()
        self.processor = Sam3Processor(self.sam3_model)
        logger.info(f"load model {self.tool_name} success")

    def _autocast_context(self):
        if self.use_bfloat16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    async def _ensure_batch_worker(self):
        if self._request_queue is None:
            self._request_queue = asyncio.Queue()

        if self._batch_worker_task is None or self._batch_worker_task.done():
            if (
                self._batch_worker_task is not None
                and not self._batch_worker_task.cancelled()
            ):
                previous_error = self._batch_worker_task.exception()
                if previous_error is not None:
                    logger.error(
                        "Restarting stopped PointToBoxMask batch worker: "
                        f"{previous_error}"
                    )
            self._batch_worker_task = asyncio.create_task(
                self._batch_worker_loop(),
                name="point-to-box-mask-dynamic-batcher",
            )

    async def generate_gate_async(self, params):
        """Queue one HTTP request for cross-request native SAM3 batching."""
        await self._ensure_batch_worker()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._request_queue.put(_PendingRequest(params=params, future=future))
        return await future

    async def _batch_worker_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            first_request = await self._request_queue.get()
            pending_requests = [first_request]
            deadline = loop.time() + self.batch_wait_ms / 1000.0

            try:
                while len(pending_requests) < self.max_batch_requests:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        pending_requests.append(
                            await asyncio.wait_for(
                                self._request_queue.get(),
                                timeout=remaining,
                            )
                        )
                    except asyncio.TimeoutError:
                        break

                params_batch = [pending.params for pending in pending_requests]
                try:
                    responses = await loop.run_in_executor(
                        self._inference_executor,
                        self._process_request_batch,
                        params_batch,
                    )
                    if len(responses) != len(pending_requests):
                        raise RuntimeError(
                            "PointToBoxMask batcher returned a different number "
                            f"of responses ({len(responses)}) than requests "
                            f"({len(pending_requests)})"
                        )
                except Exception as error:
                    logger.error(f"PointToBoxMask batch execution failed: {error}")
                    logger.error(traceback.format_exc())
                    responses = [
                        self._unexpected_error_response(error)
                        for _ in pending_requests
                    ]

                for pending, response in zip(pending_requests, responses):
                    if not pending.future.done():
                        pending.future.set_result(response)
            finally:
                for _ in pending_requests:
                    self._request_queue.task_done()

    @staticmethod
    def _success_response(results):
        return {
            "message": "Tool PointToBoxMaskWorker executed successfully.",
            "status": "success",
            "results": results,
        }

    @staticmethod
    def _unexpected_error_response(error):
        return build_worker_error_response(
            "PointToBoxMask",
            error,
            remote_traceback=traceback.format_exc(),
        )

    def generate(self, params):
        """Keep the historical synchronous API while sharing GPU ownership."""
        with self._inference_lock:
            return self._generate_serial(params)

    def _process_request_batch(self, params_batch):
        # Sam3Processor and inst_interactive_predictor retain mutable image
        # state. The lock protects them from direct ``generate`` calls racing
        # with the dynamic batcher's GPU executor.
        with self._inference_lock:
            return self._process_request_batch_locked(params_batch)

    def _process_request_batch_locked(self, params_batch):
        responses = [None] * len(params_batch)
        request_results = [{} for _ in params_batch]
        requests_by_mode = {"confidence": [], "area": []}

        for request_index, params in enumerate(params_batch):
            try:
                image_dict = params.get("image_dict", None)
                mode = str(params.get("mode", "confidence")).lower()
            except Exception as error:
                responses[request_index] = self._unexpected_error_response(error)
                continue

            if image_dict is None:
                logger.error("Missing required inputs: images.")
                responses[request_index] = {
                    "message": "Missing required inputs: images",
                    "status": "error",
                }
                continue
            if mode not in {"confidence", "area"}:
                responses[request_index] = {
                    "message": (
                        f"Unsupported mode: {mode}. Supported modes are "
                        "'confidence' and 'area'."
                    ),
                    "status": "error",
                }
                continue

            try:
                image_items = list(image_dict.items())
            except Exception:
                # Let the historical implementation construct the exact error
                # response for malformed image_dict values.
                responses[request_index] = self._generate_serial(params)
                continue

            if not image_items:
                responses[request_index] = self._success_response({})
                continue

            # Non-string names already fail in the historical visualization
            # path. Keep such malformed requests isolated rather than changing
            # their behavior by converting the names to strings.
            if any(not isinstance(image_name, str) for image_name, _ in image_items):
                responses[request_index] = self._generate_serial(params)
                continue

            requests_by_mode[mode].append(
                (request_index, params, image_items)
            )

        for mode, request_entries in requests_by_mode.items():
            if not request_entries:
                continue

            if len(request_entries) == 1:
                request_index, params, _ = request_entries[0]
                responses[request_index] = self._generate_serial(params)
                continue

            combined_image_dict = {}
            name_mapping = {}
            for request_index, _, image_items in request_entries:
                for image_index, (image_name, image_data) in enumerate(image_items):
                    synthetic_name = (
                        f"__point_batch_r{request_index}_i{image_index}__{image_name}"
                    )
                    combined_image_dict[synthetic_name] = image_data
                    name_mapping[synthetic_name] = (request_index, image_name)

            logger.info(
                "PointToBoxMask dynamic batch: "
                f"mode={mode}, requests={len(request_entries)}, "
                f"images={len(combined_image_dict)}, "
                f"max_image_batch_size={self.max_batch_size}"
            )

            combined_response = self._generate_serial(
                {"image_dict": combined_image_dict, "mode": mode}
            )
            combined_results = combined_response.get("results", {})
            combined_ok = (
                combined_response.get("status") == "success"
                and set(combined_results) == set(name_mapping)
            )

            if combined_ok:
                for synthetic_name, image_result in combined_results.items():
                    request_index, image_name = name_mapping[synthetic_name]
                    # Only the no-detection text mentions the image name. The
                    # replacement restores the exact public response while the
                    # synthetic prefix remains invisible to callers.
                    text_response = image_result.get("text_response")
                    if isinstance(text_response, str):
                        image_result["text_response"] = text_response.replace(
                            synthetic_name,
                            image_name,
                        )
                    request_results[request_index][image_name] = image_result
                continue

            # Splitting a coalesced request is an OOM recovery only. Retrying a
            # non-OOM failure independently would hide a batching/model bug.
            if not response_indicates_cuda_oom(combined_response):
                raise RuntimeError(
                    "Combined PointToBoxMask batch failed or returned incomplete "
                    f"results: {combined_response!r}"
                )
            logger.warning(
                "Combined PointToBoxMask batch OOM; retrying requests independently. "
                f"mode={mode}, status={combined_response.get('status')}"
            )
            for request_index, params, _ in request_entries:
                responses[request_index] = self._generate_serial(params)

        for request_index, response in enumerate(responses):
            if response is None:
                responses[request_index] = self._success_response(
                    request_results[request_index]
                )
        return responses

    def _set_image_batch(self, images):
        with torch.inference_mode(), self._autocast_context():
            return self.processor.set_image_batch(images)

    def _activate_predictor_from_state(self, inference_state):
        """Install one encoded image batch in SAM3's interactive predictor.

        This is the feature-setup half of SAM3 ``predict_inst_batch``.  Keeping
        it separate lets the worker reuse one batched backbone encoding while
        bounding the number of independent point prompts sent to each mask
        decoder forward.
        """
        with torch.inference_mode(), self._autocast_context():
            predictor = self.sam3_model.inst_interactive_predictor
            if predictor is None:
                raise RuntimeError(
                    "PointToBoxMask requires SAM3 instance interactivity."
                )

            try:
                backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
                original_heights = inference_state["original_heights"]
                original_widths = inference_state["original_widths"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "Invalid SAM3 batched inference state for PointToBoxMask."
                ) from error

            (
                _,
                vision_feats,
                _,
                _,
            ) = predictor.model._prepare_backbone_features(backbone_out)
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

            batch_size = vision_feats[-1].shape[1]
            if not (
                batch_size == len(original_heights) == len(original_widths)
            ):
                raise ValueError(
                    "PointToBoxMask SAM3 batch size mismatch: "
                    f"features={batch_size}, heights={len(original_heights)}, "
                    f"widths={len(original_widths)}."
                )

            features = [
                feature.permute(1, 2, 0).view(batch_size, -1, *feature_size)
                for feature, feature_size in zip(
                    vision_feats[::-1], predictor._bb_feat_sizes[::-1]
                )
            ][::-1]
            predictor._features = {
                "image_embed": features[-1],
                "high_res_feats": features[:-1],
            }
            predictor._is_image_set = True
            predictor._is_batch = True
            predictor._orig_hw = list(zip(original_heights, original_widths))
            return predictor

    def _predict_image_point_chunk(
        self,
        image_index,
        chunk_points,
        chunk_labels,
        mode,
    ):
        """Predict one bounded point chunk for one image in the active batch."""
        with torch.inference_mode(), self._autocast_context():
            predictor = self.sam3_model.inst_interactive_predictor
            if (
                predictor is None
                or not predictor._is_image_set
                or not predictor._is_batch
                or predictor._features is None
            ):
                raise RuntimeError(
                    "PointToBoxMask SAM3 batch predictor is not active."
                )

            mask_input, coords, labels, box = predictor._prep_prompts(
                chunk_points,
                chunk_labels,
                None,
                None,
                True,
                img_idx=image_index,
            )
            masks, scores, low_res_masks = predictor._predict(
                coords,
                labels,
                box,
                mask_input,
                multimask_output=(mode == "area"),
                return_logits=False,
                img_idx=image_index,
            )

            # Match SAM3 predict_batch's mask conversion exactly. Scores and
            # low-resolution logits were never part of this tool's response.
            # ``return_logits=False`` makes masks boolean. Move them off the
            # GPU before the historical float32 conversion so a full-size
            # float copy is never allocated on CUDA.
            masks_numpy = masks.squeeze(0).detach().cpu().float().numpy()
            scores = None
            low_res_masks = None
            return masks_numpy

    @staticmethod
    def _call_with_oom_capture(operation):
        """Return ``(result, oom_message)`` without retaining an OOM traceback."""
        try:
            return operation(), None
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise
            oom_message = str(error)
            error.__traceback__ = None
            return None, oom_message

    @staticmethod
    def _prepare_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            points = np.empty((0, 1, 2), dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 2:
            points = points[:, None, :]
        if points.ndim != 3 or points.shape[1:] != (1, 2):
            raise ValueError(
                "PointToBoxMask points must have shape (N, 2) or (N, 1, 2), "
                f"but received {points.shape}."
            )
        labels = np.ones((points.shape[0], 1), dtype=np.int32)
        return points, labels

    @staticmethod
    def _select_valid_masks_and_boxes(result_masks, mode):
        masks = result_masks
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().float().numpy()
        else:
            masks = np.asarray(masks)

        if masks.ndim == 3:
            masks = masks[None, ...]

        if masks.ndim != 4:
            raise ValueError(f"Unexpected PointToBoxMask mask shape: {masks.shape}")

        if mode == "area":
            areas = masks.sum(axis=(2, 3))
            max_indices = areas.argmax(axis=1)
            masks = masks[np.arange(masks.shape[0]), max_indices]
        else:
            if masks.shape[1] != 1:
                raise ValueError(
                    "Confidence mode expected one mask per point, "
                    f"but received mask shape {masks.shape}."
                )
            masks = np.squeeze(masks, axis=1)

        masks_bool = masks > 0
        bboxes = []
        for mask in masks_bool:
            if np.any(mask):
                y_indices, x_indices = np.where(mask)
                bboxes.append(
                    [
                        int(x_indices.min()),
                        int(y_indices.min()),
                        int(x_indices.max()),
                        int(y_indices.max()),
                    ]
                )
            else:
                bboxes.append([0, 0, 0, 0])

        return filter_valid_box_masks(bboxes, masks_bool)

    @staticmethod
    def _encode_masks(masks_nhw):
        if len(masks_nhw) == 0:
            return []

        masks_hwn = np.transpose(masks_nhw, (1, 2, 0))
        encoded_masks = mask_utils.encode(
            np.asfortranarray(masks_hwn.astype(np.uint8))
        )
        if isinstance(encoded_masks, dict):
            return [encoded_masks]
        return list(encoded_masks)

    @staticmethod
    def _build_image_result(image_name, image, bboxes, rle_masks):
        bboxes_array = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        # Drawing must not mutate the decoded image reused by another result
        # consumer or a subsequent post-processing path.
        visual_image = show_bboxes(image.copy(), bboxes_array, image_name)

        if len(bboxes_array) > 0:
            text_response = (
                "The bounding boxes and masks for all points have been obtained. For clearer visualization, "
                "only the bounding boxes are displayed on the original image, while the masks are not visualized."
            )
        else:
            text_response = (
                f"For the points in image {image_name}, no valid bounding boxes or masks were obtained."
            )

        return {
            "visual_image": pil_to_bytes(visual_image),
            "text_response": text_response,
            "bboxes": bboxes_array.tolist(),
            "masks": rle_masks,
        }

    def _choose_point_chunk_size(self, num_points, height, width, mode):
        """Bound a decoder forward by both point count and output pixels."""
        if num_points <= 0:
            return 1, 1

        candidate_masks_per_point = 3 if mode == "area" else 1
        # Full-resolution interpolation, thresholding, and the binary result
        # briefly overlap. Eight bytes per output pixel is a conservative
        # mixed BF16/FP32 estimate, matching PhraseToBoxMask budgeting.
        estimated_bytes_per_point = max(
            1,
            int(height)
            * int(width)
            * candidate_masks_per_point
            * 8,
        )
        budget_limited_size = max(
            1,
            self.point_mask_postprocess_budget_bytes
            // estimated_bytes_per_point,
        )
        selected_size = max(
            1,
            min(
                int(num_points),
                self.max_points_per_forward[mode],
                budget_limited_size,
            ),
        )
        return selected_size, budget_limited_size

    def _decode_image_in_point_chunks(
        self,
        inference_state,
        image_index,
        image_name,
        image,
        points,
        labels,
        mode,
    ):
        """Decode every point for one encoded image with a bounded fanout."""
        if len(points) == 0:
            return self._build_image_result(image_name, image, [], [])

        configured_chunk_size = self.max_points_per_forward[mode]
        chunk_size, budget_limited_size = self._choose_point_chunk_size(
            len(points),
            image.height,
            image.width,
            mode,
        )
        if len(points) > chunk_size:
            logger.info(
                "PointToBoxMask proactive point chunking: "
                f"image={image_name}, num_points={len(points)}, mode={mode}, "
                f"size={image.width}x{image.height}, "
                f"max_points_per_forward={configured_chunk_size}, "
                f"pixel_budget_limit={budget_limited_size}, "
                f"selected_chunk_size={chunk_size}"
            )

        all_bboxes = []
        all_rle_masks = []
        point_start = 0
        while point_start < len(points):
            current_size = min(chunk_size, len(points) - point_start)
            chunk_points = points[point_start : point_start + current_size]
            chunk_labels = labels[point_start : point_start + current_size]

            result_masks, chunk_oom_message = self._call_with_oom_capture(
                lambda: self._predict_image_point_chunk(
                    image_index,
                    chunk_points,
                    chunk_labels,
                    mode,
                )
            )

            if chunk_oom_message is not None:
                result_masks = None
                reset_sam3_predictor_state(self.sam3_model)
                release_torch_memory(self.device)

                if current_size == 1:
                    raise torch.cuda.OutOfMemoryError(
                        "PointToBoxMask still OOM with one point on image "
                        f"{image_name}: {chunk_oom_message}"
                    )

                chunk_size = max(1, current_size // 2)
                logger.warning(
                    "PointToBoxMask point chunk OOM: "
                    f"image={image_name}, failed_chunk_size={current_size}, "
                    f"retry_chunk_size={chunk_size}"
                )
                # The failed decoder call may leave SAM3's mutable predictor
                # state incomplete. Rebuild lightweight predictor views from
                # the already-computed backbone state; do not re-encode images.
                self._activate_predictor_from_state(inference_state)
                continue

            chunk_bboxes, chunk_masks = self._select_valid_masks_and_boxes(
                result_masks,
                mode,
            )
            chunk_rle_masks = self._encode_masks(chunk_masks)
            all_bboxes.extend(chunk_bboxes.tolist())
            all_rle_masks.extend(chunk_rle_masks)
            point_start += current_size

            result_masks = None
            chunk_masks = None
            chunk_rle_masks = None

        return self._build_image_result(
            image_name,
            image,
            all_bboxes,
            all_rle_masks,
        )

    def _process_encoded_image_batch_once(self, prepared_items, mode):
        """Encode one image batch once, then decode each image's point chunks."""
        inference_state = None
        try:
            inference_state = self._set_image_batch(
                [item[1] for item in prepared_items]
            )
            self._activate_predictor_from_state(inference_state)

            batch_results = {}
            for image_index, (image_name, image, points, labels) in enumerate(
                prepared_items
            ):
                batch_results[image_name] = self._decode_image_in_point_chunks(
                    inference_state,
                    image_index,
                    image_name,
                    image,
                    points,
                    labels,
                    mode,
                )
            return batch_results
        finally:
            reset_sam3_predictor_state(self.sam3_model)
            inference_state = None

    def _process_prepared_image_batch(self, prepared_items, mode):
        """Run a prepared image batch, recursively shrinking it only on OOM."""
        batch_results, oom_message = self._call_with_oom_capture(
            lambda: self._process_encoded_image_batch_once(prepared_items, mode)
        )
        if oom_message is None:
            return batch_results

        batch_results = None
        reset_sam3_predictor_state(self.sam3_model)
        release_torch_memory(self.device)

        if len(prepared_items) > 1:
            split_index = len(prepared_items) // 2
            logger.warning(
                "PointToBoxMask image batch OOM during encoding or chunked decoding; "
                f"splitting image batch from {len(prepared_items)} into "
                f"{split_index}+{len(prepared_items) - split_index}. "
                f"error={oom_message}"
            )
            split_results = {}
            split_results.update(
                self._process_prepared_image_batch(
                    prepared_items[:split_index],
                    mode,
                )
            )
            split_results.update(
                self._process_prepared_image_batch(
                    prepared_items[split_index:],
                    mode,
                )
            )
            return split_results

        image_name = prepared_items[0][0]
        raise torch.cuda.OutOfMemoryError(
            f"PointToBoxMask OOM while processing image {image_name}: {oom_message}"
        )

    def _generate_serial(self, params):
        image_dict = params.get("image_dict", None)
        mode = str(params.get("mode", "confidence")).lower()
        if image_dict is None:
            logger.error("Missing required inputs: images.")
            return {"message": "Missing required inputs: images", "status": "error"}
        if mode not in {"confidence", "area"}:
            return {
                "message": f"Unsupported mode: {mode}. Supported modes are 'confidence' and 'area'.",
                "status": "error",
            }

        ret = {"message": "", "status": ""}
        needs_cuda_cleanup = False
        results = None
        image_items = None
        batch_image_items = None
        prepared_items = None
        try:
            max_batch_size = self.max_batch_size
            results = {}
            image_items = list(image_dict.items())
            total_images = len(image_items)
            logger.info(f"Total images: {total_images}. Using max_batch_size: {max_batch_size}")

            for i in range(0, total_images, max_batch_size):
                batch_start = i
                batch_end = min(i + max_batch_size, total_images)
                batch_image_items = image_items[batch_start:batch_end]
                logger.info(f"Processing batch {batch_start} to {batch_end}...")

                prepared_items = []
                for image_name, data in batch_image_items:
                    img = bytes_to_pil(data['image_bytes']).convert("RGB")
                    pts, lbls = self._prepare_points(data['points'])
                    prepared_items.append((image_name, img, pts, lbls))

                    logger.info(
                        f"Prepared image={image_name}, size={img.width}x{img.height}, "
                        f"num_points={len(pts)}, mode={mode}"
                    )

                results.update(
                    self._process_prepared_image_batch(prepared_items, mode)
                )
                prepared_items = None

            ret['message'] = "Tool PointToBoxMaskWorker executed successfully."
            ret['status'] = "success"
            ret['results'] = results
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error when using PointToBoxMaskWorker: {e}")
            logger.error(f"Traceback: {tb}")
            ret = build_worker_error_response(
                self.tool_name,
                e,
                remote_traceback=tb,
            )
            if is_cuda_oom_error(e):
                needs_cuda_cleanup = True
                logger.error("CUDA OOM detected in PointToBoxMaskWorker. Cleaning request tensors and CUDA cache.")
        finally:
            image_items = None
            batch_image_items = None
            prepared_items = None
            if ret.get("status") != "success":
                results = None
            if needs_cuda_cleanup:
                reset_sam3_predictor_state(self.sam3_model)
                release_torch_memory(self.device)
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="PointToBoxMask")
    parser.add_argument("--worker_name", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20036)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--batch_wait_ms", type=float, default=30.0)
    parser.add_argument("--max_batch_requests", type=int, default=0)
    parser.add_argument(
        "--max_points_per_forward_confidence",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--max_points_per_forward_area",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--point_mask_postprocess_budget_mb",
        type=int,
        default=2048,
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = PointToBoxMaskWorker(
        controller_addr=args.controller_address,
        worker_name = args.worker_name,
        worker_addr=args.worker_address,
        tool_name = args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path=args.model_path,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_requests=args.max_batch_requests,
        max_points_per_forward_confidence=(
            args.max_points_per_forward_confidence
        ),
        max_points_per_forward_area=args.max_points_per_forward_area,
        point_mask_postprocess_budget_mb=(
            args.point_mask_postprocess_budget_mb
        ),
    )
    worker.run()
