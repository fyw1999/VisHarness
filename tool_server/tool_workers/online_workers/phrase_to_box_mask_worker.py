"""Native SAM3 PhraseToBoxMask worker with request-level dynamic batching."""

import argparse
import asyncio
import gc
import hashlib
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from tool_server.tool_workers.online_workers.base_tool_worker import (
    BaseToolWorker,
    build_worker_error_response,
    is_cuda_oom_error,
)
from tool_server.utils.server_utils import *
from tool_server.utils.utils import *


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

@dataclass
class _PendingRequest:
    params: dict
    future: asyncio.Future


@dataclass
class _ResultConsumer:
    request_index: int
    image_name: str
    display_image: object


@dataclass
class _QuerySpec:
    query_id: int
    phrase: str
    consumers: list[_ResultConsumer] = field(default_factory=list)


@dataclass
class _ImageGroup:
    image_key: bytes
    model_image: object
    queries: list[_QuerySpec] = field(default_factory=list)
    queries_by_phrase: dict[str, _QuerySpec] = field(default_factory=dict)


def release_torch_memory(device=None):
    """Release cached CUDA blocks after an OOM; never called on the normal path."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception as cleanup_error:
        logger.warning(f"CUDA memory cleanup failed: {cleanup_error}")


def get_downsample_factor(img_name):
    if "." in img_name:
        base_name = img_name.rsplit(".", 1)[0]
    else:
        base_name = img_name

    super_res_count = 0
    while base_name.endswith("_4x"):
        super_res_count += 1
        base_name = base_name[:-3]

    return 4**super_res_count


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
        draw.rectangle([x0, y0, x1, y1], outline="green", width=adaptive_width)
    return image


class PhraseToBoxMaskWorker(BaseToolWorker):
    """Serve native SAM3 text-prompt segmentation with continuous micro-batching.

    HTTP requests are accepted concurrently and coalesced for a short window. A
    single executor owns GPU execution, so concurrency raises throughput through
    batching instead of overlapping independent CUDA forwards and their memory
    peaks.
    """

    DETECTION_THRESHOLD = 0.4
    MASK_THRESHOLD = 0.5

    def __init__(
        self,
        controller_addr,
        worker_name="",
        worker_addr="auto",
        no_register=False,
        tool_name="",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        model_path="",
        max_batch_size=4,
        batch_wait_ms=10.0,
        max_batch_requests=0,
        max_queries_per_batch=8,
        mask_chunk_size=0,
        mask_postprocess_budget_mb=512,
    ):
        self.model_path = model_path
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_ms = max(0.0, float(batch_wait_ms))
        self.max_batch_requests = (
            max(1, int(max_batch_requests))
            if int(max_batch_requests) > 0
            else max(1, int(limit_model_concurrency))
        )
        self.max_queries_per_batch = max(1, int(max_queries_per_batch))
        # Zero means no fixed cap: use the largest chunk that fits the memory
        # budget. This keeps ordinary requests on the official one-shot path.
        self.mask_chunk_size = int(mask_chunk_size)
        self.mask_postprocess_budget_bytes = (
            max(1, int(mask_postprocess_budget_mb)) * 1024 * 1024
        )
        self.use_bfloat16 = False

        self._request_queue = None
        self._batch_worker_task = None
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="phrase-to-box-mask-gpu",
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
            port,
        )

    @staticmethod
    def _resolve_checkpoint_path(model_path):
        if not model_path:
            return None
        if os.path.isdir(model_path):
            checkpoint_path = os.path.join(model_path, "sam3.pt")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Native SAM3 expects sam3.pt inside model directory {model_path}"
                )
            return checkpoint_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {model_path}")
        return model_path

    def init_model(self):
        logger.info(f"Initializing native SAM3 model {self.tool_name}...")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The official native SAM3 image implementation requires a CUDA GPU."
            )
        self.device = torch.device("cuda")
        self.use_bfloat16 = self._require_bfloat16_gemm()

        if (
            self.device.type == "cuda"
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam3_root = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        checkpoint_path = self._resolve_checkpoint_path(self.model_path)

        logger.info(
            f"using device={self.device}, bfloat16={self.use_bfloat16}, "
            f"checkpoint={checkpoint_path or 'official Hugging Face download'}"
        )
        print(f"using device: {self.device}")
        print(f"using bfloat16: {self.use_bfloat16}")

        self.sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path
        ).to(self.device)
        self.sam3_model.eval()

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008,
                    max_size=1008,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        logger.info(f"load native SAM3 model {self.tool_name} success")

    @staticmethod
    def _require_bfloat16_gemm():
        """Validate the operation required by native SAM3 inference.

        Some CUDA/PyTorch combinations report BF16 hardware support while
        ``cublasGemmEx`` itself is unusable. Native SAM3 also contains fused
        MLP kernels that explicitly produce BF16 tensors, so silently falling
        back to FP32 is not valid; fail at startup with a useful error instead.
        """
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The official native SAM3 implementation requires BF16 CUDA support."
            )
        left = None
        right = None
        try:
            left = torch.ones((1, 16), device="cuda", dtype=torch.bfloat16)
            right = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
            _ = left @ right
            torch.cuda.synchronize()
            return True
        except Exception as error:
            raise RuntimeError(
                "CUDA reports BF16 support, but BF16 GEMM failed. The official native "
                "SAM3 fused kernels cannot run in this CUDA/PyTorch environment. "
                f"Original error: {error}"
            ) from error
        finally:
            left = None
            right = None

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
                        f"Restarting stopped PhraseToBoxMask batch worker: {previous_error}"
                    )
            self._batch_worker_task = asyncio.create_task(
                self._batch_worker_loop(),
                name="phrase-to-box-mask-dynamic-batcher",
            )

    async def generate_gate_async(self, params):
        """Queue a request and await a result produced by the shared GPU batcher."""
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
                                self._request_queue.get(), timeout=remaining
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
                            "PhraseToBoxMask batcher returned a different number of responses "
                            f"({len(responses)}) than requests ({len(pending_requests)})"
                        )
                except Exception as error:
                    logger.error(f"PhraseToBoxMask batch execution failed: {error}")
                    logger.error(traceback.format_exc())
                    responses = [self._error_response(error) for _ in pending_requests]

                for pending, response in zip(pending_requests, responses):
                    if not pending.future.done():
                        pending.future.set_result(response)
            finally:
                for _ in pending_requests:
                    self._request_queue.task_done()

    @staticmethod
    def _error_response(error, remote_traceback=None):
        if remote_traceback is None:
            active_traceback = traceback.format_exc()
            if active_traceback.strip() != "NoneType: None":
                remote_traceback = active_traceback
        return build_worker_error_response(
            "PhraseToBoxMask",
            error,
            remote_traceback=remote_traceback,
        )

    @staticmethod
    def _success_response(results):
        return {
            "message": "Tool PhraseToBoxMask executed successfully.",
            "status": "success",
            "results": results,
        }

    def generate(self, params):
        """Keep the historical synchronous Python API for non-HTTP callers."""
        return self._process_request_batch([params])[0]

    def _process_request_batch(self, params_batch):
        # This lock also protects against a direct synchronous ``generate`` call
        # racing with the HTTP batch executor.
        with self._inference_lock:
            return self._process_request_batch_locked(params_batch)

    def _process_request_batch_locked(self, params_batch):
        responses = [None] * len(params_batch)
        request_results = [{} for _ in params_batch]
        image_groups_by_key = {}
        image_groups = []
        next_query_id = 1

        for request_index, params in enumerate(params_batch):
            try:
                image_dict = params.get("image_dict", None)
                phrase = params.get("phrase", None)
                phrase = phrase.strip() if phrase is not None else None

                if image_dict is None or phrase is None:
                    logger.error("Missing required inputs: image or prompts.")
                    responses[request_index] = {
                        "message": "Missing required inputs: image or prompts",
                        "status": "error",
                    }
                    continue

                image_items = list(image_dict.items())
                prepared_items = []
                for image_name, image_bytes in image_items:
                    raw_image_bytes = bytes(image_bytes)
                    decoded_image = bytes_to_pil(raw_image_bytes)
                    prepared_items.append(
                        (
                            image_name,
                            raw_image_bytes,
                            decoded_image.convert("RGB"),
                            decoded_image,
                        )
                    )

                for (
                    image_name,
                    raw_image_bytes,
                    model_image,
                    display_image,
                ) in prepared_items:
                    image_key = hashlib.sha256(raw_image_bytes).digest()
                    image_group = image_groups_by_key.get(image_key)
                    if image_group is None:
                        image_group = _ImageGroup(
                            image_key=image_key,
                            model_image=model_image,
                        )
                        image_groups_by_key[image_key] = image_group
                        image_groups.append(image_group)

                    query = image_group.queries_by_phrase.get(phrase)
                    if query is None:
                        query = _QuerySpec(query_id=next_query_id, phrase=phrase)
                        next_query_id += 1
                        image_group.queries_by_phrase[phrase] = query
                        image_group.queries.append(query)

                    query.consumers.append(
                        _ResultConsumer(
                            request_index=request_index,
                            image_name=image_name,
                            display_image=display_image,
                        )
                    )
            except Exception as error:
                logger.error(f"Error preparing PhraseToBoxMask request: {error}")
                responses[request_index] = self._error_response(error)

        valid_groups = []
        for image_group in image_groups:
            valid_queries = [
                query
                for query in image_group.queries
                if any(responses[c.request_index] is None for c in query.consumers)
            ]
            if valid_queries:
                valid_groups.append(
                    _ImageGroup(
                        image_key=image_group.image_key,
                        model_image=image_group.model_image,
                        queries=valid_queries,
                        queries_by_phrase={
                            query.phrase: query for query in valid_queries
                        },
                    )
                )

        logger.info(
            "PhraseToBoxMask dynamic batch: "
            f"requests={len(params_batch)}, unique_images={len(valid_groups)}, "
            f"unique_queries={sum(len(group.queries) for group in valid_groups)}"
        )

        for microbatch in self._iter_microbatches(valid_groups):
            try:
                query_results, query_oom_errors = (
                    self._infer_groups_with_oom_fallback(microbatch)
                )
                for image_group in microbatch:
                    for query in image_group.queries:
                        query_oom_error = query_oom_errors.get(query.query_id)
                        if query_oom_error is not None:
                            for consumer in query.consumers:
                                if responses[consumer.request_index] is None:
                                    responses[consumer.request_index] = (
                                        self._error_response(query_oom_error)
                                    )
                            continue

                        if query.query_id not in query_results:
                            raise RuntimeError(
                                f"Missing native SAM3 result for query_id={query.query_id}"
                            )
                        boxes, rle_masks = query_results[query.query_id]
                        for consumer in query.consumers:
                            if responses[consumer.request_index] is not None:
                                continue
                            request_results[consumer.request_index][
                                consumer.image_name
                            ] = self._build_image_result(
                                consumer.image_name,
                                consumer.display_image,
                                query.phrase,
                                boxes,
                                rle_masks,
                            )
            except Exception as error:
                remote_traceback = traceback.format_exc()
                logger.error(f"Error in PhraseToBoxMask native microbatch: {error}")
                logger.error(remote_traceback)
                for image_group in microbatch:
                    for query in image_group.queries:
                        for consumer in query.consumers:
                            if responses[consumer.request_index] is None:
                                responses[consumer.request_index] = (
                                    self._error_response(error, remote_traceback)
                                )

        for request_index, response in enumerate(responses):
            if response is None:
                responses[request_index] = self._success_response(
                    request_results[request_index]
                )
        return responses

    def _iter_microbatches(self, image_groups):
        current_groups = []
        current_query_count = 0

        for image_group in image_groups:
            for query_start in range(
                0, len(image_group.queries), self.max_queries_per_batch
            ):
                query_slice = image_group.queries[
                    query_start : query_start + self.max_queries_per_batch
                ]
                split_group = _ImageGroup(
                    image_key=image_group.image_key,
                    model_image=image_group.model_image,
                    queries=query_slice,
                    queries_by_phrase={query.phrase: query for query in query_slice},
                )

                would_exceed_images = len(current_groups) >= self.max_batch_size
                would_exceed_queries = (
                    current_query_count + len(query_slice) > self.max_queries_per_batch
                )
                if current_groups and (would_exceed_images or would_exceed_queries):
                    yield current_groups
                    current_groups = []
                    current_query_count = 0

                current_groups.append(split_group)
                current_query_count += len(query_slice)

        if current_groups:
            yield current_groups

    def _make_datapoint(self, image_group):
        image = image_group.model_image.copy()
        width, height = image.size
        datapoint = Datapoint(
            find_queries=[],
            images=[SAMImage(data=image, objects=[], size=(height, width))],
        )

        for query in image_group.queries:
            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=query.phrase,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query.query_id,
                        original_image_id=query.query_id,
                        original_category_id=1,
                        original_size=(height, width),
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )
        return self.transform(datapoint)

    def _infer_groups_with_oom_fallback(self, image_groups):
        """Infer a microbatch while isolating irreducible OOMs by query.

        Returns ``(successful_results, query_oom_errors)``.  A batch-only OOM
        is retried in smaller pieces.  If one image/phrase query still OOMs at
        the minimum unit, only that query is recorded as failed; successful
        sibling queries are retained instead of being discarded with the
        original microbatch.
        """
        try:
            return self._infer_groups(image_groups), {}
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise

            error_message = str(error)
            error.__traceback__ = None
            release_torch_memory(self.device)
            total_queries = sum(len(group.queries) for group in image_groups)
            if total_queries <= 1:
                if not image_groups or not image_groups[0].queries:
                    raise RuntimeError(
                        "Native SAM3 OOM fallback received an empty query batch"
                    )
                query_id = image_groups[0].queries[0].query_id
                query_error = torch.cuda.OutOfMemoryError(
                    "Native SAM3 OOM for one image/query "
                    f"(query_id={query_id}): {error_message}"
                )
                return {}, {query_id: query_error}

            left_groups, right_groups = self._split_groups_in_half(image_groups)
            logger.warning(
                "Native SAM3 microbatch OOM; retrying two smaller batches: "
                f"images={len(image_groups)}, queries={total_queries}, "
                f"left_queries={sum(len(g.queries) for g in left_groups)}, "
                f"right_queries={sum(len(g.queries) for g in right_groups)}"
            )
            left_results, left_errors = self._infer_groups_with_oom_fallback(
                left_groups
            )
            right_results, right_errors = self._infer_groups_with_oom_fallback(
                right_groups
            )
            overlap = set(left_results).intersection(right_results)
            if overlap:
                raise RuntimeError(
                    f"Duplicate native SAM3 query results after OOM split: {sorted(overlap)}"
                )
            error_overlap = set(left_errors).intersection(right_errors)
            if error_overlap:
                raise RuntimeError(
                    f"Duplicate native SAM3 query errors after OOM split: {sorted(error_overlap)}"
                )
            left_results.update(right_results)
            left_errors.update(right_errors)
            return left_results, left_errors

    @staticmethod
    def _split_groups_in_half(image_groups):
        flattened = [
            (image_group, query)
            for image_group in image_groups
            for query in image_group.queries
        ]
        midpoint = max(1, len(flattened) // 2)

        def rebuild(items):
            rebuilt = []
            current_source = None
            current_group = None
            for source_group, query in items:
                if source_group is not current_source:
                    current_source = source_group
                    current_group = _ImageGroup(
                        image_key=source_group.image_key,
                        model_image=source_group.model_image,
                    )
                    rebuilt.append(current_group)
                current_group.queries.append(query)
                current_group.queries_by_phrase[query.phrase] = query
            return rebuilt

        return rebuild(flattened[:midpoint]), rebuild(flattened[midpoint:])

    def _infer_groups(self, image_groups):
        datapoints = None
        batch = None
        output = None
        try:
            datapoints = [
                self._make_datapoint(image_group) for image_group in image_groups
            ]
            batch = collate_fn_api(datapoints, dict_key="phrase_to_box_mask")[
                "phrase_to_box_mask"
            ]
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            with torch.inference_mode(), self._autocast_context():
                output = self.sam3_model(batch)

            return self._postprocess_native_output(output, batch.find_metadatas)
        finally:
            output = None
            batch = None
            datapoints = None

    def _postprocess_native_output(self, find_stages, find_metadatas):
        if find_stages.loss_stages is not None:
            find_metadatas = [
                find_metadatas[index] for index in find_stages.loss_stages
            ]

        results = {}
        for outputs, metadata in zip(find_stages, find_metadatas):
            stage_results = self._postprocess_stage(outputs, metadata)
            overlap = set(results).intersection(stage_results)
            if overlap:
                raise RuntimeError(
                    f"Duplicate native SAM3 query ids in output: {sorted(overlap)}"
                )
            results.update(stage_results)
        return results

    def _postprocess_stage(self, outputs, metadata):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_masks = outputs["pred_masks"]

        probabilities = pred_logits.sigmoid()
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            probabilities = probabilities * presence_score
        scores, _ = probabilities.max(-1)

        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_sizes = metadata.original_size.to(device=boxes_xyxy.device)
        image_height, image_width = target_sizes.unbind(1)
        scale = torch.stack(
            [image_width, image_height, image_width, image_height], dim=1
        )
        boxes_xyxy = boxes_xyxy * scale[:, None, :]

        query_ids = metadata.coco_image_id
        stage_results = {}
        for query_index, query_id_tensor in enumerate(query_ids):
            keep = scores[query_index] > self.DETECTION_THRESHOLD
            query_boxes = boxes_xyxy[query_index][keep].detach().cpu().float().numpy()
            query_masks = pred_masks[query_index][keep]
            target_height, target_width = (
                int(target_sizes[query_index, 0].item()),
                int(target_sizes[query_index, 1].item()),
            )
            valid_boxes, rle_masks = self._resize_filter_and_encode_masks(
                query_masks,
                query_boxes,
                target_height,
                target_width,
            )
            stage_results[int(query_id_tensor.item())] = (valid_boxes, rle_masks)
        return stage_results

    def _choose_mask_chunk_size(self, num_masks, height, width):
        if num_masks <= 0:
            return 1
        # Interpolation, sigmoid and thresholding briefly coexist. Eight bytes
        # per output pixel is a conservative BF16/FP32 mixed estimate.
        estimated_bytes_per_mask = max(1, int(height) * int(width) * 8)
        budget_limited_size = max(
            1,
            self.mask_postprocess_budget_bytes // estimated_bytes_per_mask,
        )
        configured_limit = (
            self.mask_chunk_size if self.mask_chunk_size > 0 else num_masks
        )
        return max(1, min(num_masks, configured_limit, budget_limited_size))

    def _resize_filter_and_encode_masks(self, mask_logits, boxes, height, width):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), []

        finite = np.isfinite(boxes).all(axis=1)
        positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid_box_indices = np.flatnonzero(finite & positive_area)
        if len(valid_box_indices) == 0:
            return np.empty((0, 4), dtype=np.float32), []

        index_tensor = torch.as_tensor(
            valid_box_indices,
            dtype=torch.long,
            device=mask_logits.device,
        )
        mask_logits = mask_logits.index_select(0, index_tensor)
        boxes = boxes[valid_box_indices]

        valid_boxes = []
        rle_masks = []
        mask_start = 0
        chunk_size = self._choose_mask_chunk_size(len(mask_logits), height, width)

        while mask_start < len(mask_logits):
            current_size = min(chunk_size, len(mask_logits) - mask_start)
            current_logits = mask_logits[mask_start : mask_start + current_size]
            try:
                resized_logits = F.interpolate(
                    current_logits.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                binary_masks = (resized_logits.sigmoid() > self.MASK_THRESHOLD).squeeze(
                    1
                )
            except Exception as error:
                if not is_cuda_oom_error(error):
                    raise
                error.__traceback__ = None
                resized_logits = None
                binary_masks = None
                release_torch_memory(self.device)
                if current_size > 1:
                    chunk_size = max(1, current_size // 2)
                    logger.warning(
                        "PhraseToBoxMask mask resize OOM; reducing chunk size: "
                        f"failed={current_size}, retry={chunk_size}, target={width}x{height}"
                    )
                    continue

                logger.warning(
                    "PhraseToBoxMask mask resize OOM for one mask; using CPU fallback: "
                    f"target={width}x{height}"
                )
                # Move the low-resolution logits off CUDA before converting
                # to FP32, avoiding another GPU allocation during OOM recovery.
                cpu_logits = current_logits.detach().cpu().float()
                resized_logits = F.interpolate(
                    cpu_logits.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                binary_masks = (resized_logits.sigmoid() > self.MASK_THRESHOLD).squeeze(
                    1
                )

            masks_numpy = binary_masks.detach().cpu().numpy().astype(bool, copy=False)
            nonempty = masks_numpy.reshape(len(masks_numpy), -1).any(axis=1)
            if np.any(nonempty):
                nonempty_masks = masks_numpy[nonempty]
                mask_hwn = np.transpose(nonempty_masks, (1, 2, 0))
                encoded = mask_utils.encode(
                    np.asfortranarray(mask_hwn.astype(np.uint8))
                )
                if isinstance(encoded, dict):
                    encoded = [encoded]
                rle_masks.extend(encoded)
                chunk_boxes = boxes[mask_start : mask_start + current_size]
                valid_boxes.extend(chunk_boxes[nonempty].tolist())

            mask_start += current_size
            resized_logits = None
            binary_masks = None
            masks_numpy = None

        if not valid_boxes:
            return np.empty((0, 4), dtype=np.float32), []
        return np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 4), rle_masks

    @staticmethod
    def _build_image_result(image_name, image, phrase, boxes, rle_masks):
        boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        visual_image = show_bboxes(image.copy(), boxes_array, image_name)
        return {
            "visual_image": pil_to_bytes(visual_image),
            "text_response": (
                f"For the phrase '{phrase}', there are {len(rle_masks)} corresponding objects "
                f"in the image {image_name}. For clearer visualization, only the bounding boxes "
                "are displayed on the original image, while the masks are not visualized."
            ),
            "bboxes": boxes_array.tolist(),
            "masks": rle_masks,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="PhraseToBoxMask")
    parser.add_argument("--worker_name", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20036)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--batch_wait_ms", type=float, default=30.0)
    parser.add_argument("--max_batch_requests", type=int, default=0)
    parser.add_argument("--max_queries_per_batch", type=int, default=8)
    parser.add_argument("--mask_chunk_size", type=int, default=0)
    parser.add_argument("--mask_postprocess_budget_mb", type=int, default=512)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = PhraseToBoxMaskWorker(
        controller_addr=args.controller_address,
        worker_name=args.worker_name,
        worker_addr=args.worker_address,
        tool_name=args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
        model_path=args.model_path,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_requests=args.max_batch_requests,
        max_queries_per_batch=args.max_queries_per_batch,
        mask_chunk_size=args.mask_chunk_size,
        mask_postprocess_budget_mb=args.mask_postprocess_budget_mb,
    )
    worker.run()
