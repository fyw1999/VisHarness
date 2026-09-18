"""Concurrent Real-ESRGAN/GFPGAN tool worker.

Requests are accepted concurrently and coalesced for a short scheduling window.
The worker keeps one GPU execution thread: true throughput gains come from
batching GFPGAN face crops and same-sized Real-ESRGAN inputs, not from racing
stateful GFPGANer/RealESRGANer helper objects on multiple CUDA threads.
"""

import argparse
import asyncio
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torchvision.transforms.functional import normalize

from tool_server.tool_workers.online_workers.base_tool_worker import (
    BaseToolWorker,
    build_worker_error_response,
    is_cuda_oom_error,
)
from tool_server.utils.server_utils import *
from tool_server.utils.utils import *


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")


@dataclass
class _PendingRequest:
    params: dict
    future: asyncio.Future


@dataclass
class _ImageTask:
    request_index: int
    image_name: str
    input_bgr: np.ndarray
    cropped_faces: list[np.ndarray] = field(default_factory=list)
    affine_matrices: list[np.ndarray] = field(default_factory=list)
    restored_faces: list[np.ndarray] = field(default_factory=list)
    parse_masks: list[np.ndarray] = field(default_factory=list)
    background_output: np.ndarray | None = None
    final_output: np.ndarray | None = None


class SuperResolutionWorker(BaseToolWorker):
    """Serve semantically compatible SuperResolution calls with dynamic batching."""

    def __init__(
        self,
        controller_addr,
        worker_name,
        worker_addr="auto",
        no_register=False,
        tool_name="",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        model_path="",
        outscale=4,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        denoise_strength=0.5,
        face_enhance=True,
        fp32=False,
        batch_wait_ms=30.0,
        max_batch_requests=0,
        max_batch_size=2,
        max_batch_input_pixels=524288,
        max_face_detection_batch_size=2,
        max_face_batch_size=8,
        max_face_parse_batch_size=8,
    ):
        self.model_path = model_path
        numeric_outscale = float(outscale)
        self.outscale = (
            int(numeric_outscale)
            if numeric_outscale.is_integer()
            else numeric_outscale
        )
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        self.pre_pad = int(pre_pad)
        self.denoise_strength = float(denoise_strength)
        self.face_enhance = bool(face_enhance)
        self.fp32 = bool(fp32)

        self.batch_wait_ms = max(0.0, float(batch_wait_ms))
        self.max_batch_requests = (
            max(1, int(max_batch_requests))
            if int(max_batch_requests) > 0
            else max(1, int(limit_model_concurrency))
        )
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_batch_input_pixels = max(0, int(max_batch_input_pixels))
        self.max_face_detection_batch_size = max(
            1, int(max_face_detection_batch_size)
        )
        self.max_face_batch_size = max(1, int(max_face_batch_size))
        self.max_face_parse_batch_size = max(
            1, int(max_face_parse_batch_size)
        )

        self._request_queue = None
        self._batch_worker_task = None
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="super-resolution-gpu",
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

    def init_model(self):
        logger.info(f"Initializing model {self.tool_name}...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")

        if "RealESRGAN_x4plus" in self.model_path:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
        elif "RealESRNet_x4plus" in self.model_path:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
        elif "RealESRGAN_x4plus_anime_6B" in self.model_path:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
        elif "RealESRGAN_x2plus" in self.model_path:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            netscale = 2
        elif "realesr-animevideov3" in self.model_path:
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type="prelu",
            )
            netscale = 4
        elif "realesr-general-x4v3" in self.model_path:
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            netscale = 4
        else:
            raise ValueError(f"Unsupported Real-ESRGAN checkpoint: {self.model_path}")

        resolved_model_path = self.model_path
        dni_weight = None
        if (
            "realesr-general-x4v3" in self.model_path
            and self.denoise_strength != 1
        ):
            wdn_model_path = self.model_path.replace(
                "realesr-general-x4v3", "realesr-general-wdn-x4v3"
            )
            resolved_model_path = [self.model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=resolved_model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            device=self.device,
        )

        self.face_enhancer = None
        if self.face_enhance:
            from gfpgan import GFPGANer

            self.face_enhancer = GFPGANer(
                model_path=self.model_path.replace(
                    "realesr-general-x4v3", "GFPGANv1.3"
                ),
                upscale=self.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
                device=self.device,
            )

        logger.info(
            f"load model {self.tool_name} success: device={self.device}, "
            f"face_enhance={self.face_enhance}, max_batch_size={self.max_batch_size}, "
            f"max_face_batch_size={self.max_face_batch_size}"
        )

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
                        f"Restarting stopped SuperResolution batch worker: {previous_error}"
                    )
            self._batch_worker_task = asyncio.create_task(
                self._batch_worker_loop(),
                name="super-resolution-dynamic-batcher",
            )

    async def generate_gate_async(self, params):
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
                            "SuperResolution batcher returned a different number of "
                            f"responses ({len(responses)}) than requests "
                            f"({len(pending_requests)})"
                        )
                except Exception as error:
                    logger.error(f"SuperResolution batch execution failed: {error}")
                    remote_traceback = traceback.format_exc()
                    logger.error(remote_traceback)
                    responses = [
                        self._error_response(error, remote_traceback)
                        for _ in pending_requests
                    ]

                for pending, response in zip(pending_requests, responses):
                    if not pending.future.done():
                        pending.future.set_result(response)
            finally:
                for _ in pending_requests:
                    self._request_queue.task_done()

    @staticmethod
    def _missing_input_response():
        return build_worker_error_response(
            "SuperResolution",
            ValueError("Missing required inputs: image"),
        )

    @staticmethod
    def _error_response(error, remote_traceback=None):
        return build_worker_error_response(
            "SuperResolution",
            error,
            remote_traceback=remote_traceback,
        )

    @staticmethod
    def _success_response(results):
        return {
            "message": "Tool SuperResolution executed successfully.",
            "status": "success",
            "results": results,
        }

    def generate(self, params):
        """Keep the historical synchronous Python API for non-HTTP callers."""
        return self._process_request_batch([params])[0]

    def _process_request_batch(self, params_batch):
        # GFPGANer, FaceRestoreHelper and RealESRGANer contain mutable request
        # state. This lock also protects a direct generate() call from racing
        # the asynchronous GPU executor.
        with self._inference_lock:
            return self._process_request_batch_locked(params_batch)

    def _process_request_batch_locked(
        self,
        params_batch,
        *,
        isolate_request_oom=True,
    ):
        started_at = time.perf_counter()
        responses = [None] * len(params_batch)
        request_tasks = [[] for _ in params_batch]
        all_tasks = []

        for request_index, params in enumerate(params_batch):
            image_dict = params.get("image_dict", None)
            if image_dict is None:
                logger.error("Missing required inputs: image.")
                responses[request_index] = self._missing_input_response()
                continue

            try:
                for image_name, image_data in image_dict.items():
                    image_pil = bytes_to_pil(image_data).convert("RGB")
                    input_rgb = np.asarray(image_pil)
                    input_bgr = np.ascontiguousarray(
                        cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
                    )
                    task = _ImageTask(
                        request_index=request_index,
                        image_name=image_name,
                        input_bgr=input_bgr,
                    )
                    request_tasks[request_index].append(task)
                    all_tasks.append(task)
            except Exception as error:
                logger.error(f"Error preparing SuperResolution request: {error}")
                responses[request_index] = self._error_response(
                    error,
                    traceback.format_exc(),
                )

        valid_tasks = [
            task for task in all_tasks if responses[task.request_index] is None
        ]
        logger.info(
            "SuperResolution dynamic batch: "
            f"requests={len(params_batch)}, images={len(valid_tasks)}, "
            f"face_enhance={self.face_enhance}"
        )

        if valid_tasks:
            try:
                self._run_task_pipeline(valid_tasks)
            except Exception as error:
                if not is_cuda_oom_error(error):
                    raise

                remote_traceback = traceback.format_exc()
                error_message = str(error)
                error.__traceback__ = None
                self._release_cuda_cache()
                if not isolate_request_oom:
                    raise torch.cuda.OutOfMemoryError(error_message) from None

                valid_request_indices = [
                    request_index
                    for request_index, response in enumerate(responses)
                    if response is None
                ]
                logger.warning(
                    "SuperResolution combined request batch reached an irreducible "
                    "OOM; retrying each original HTTP request independently: "
                    f"requests={len(valid_request_indices)}, images={len(valid_tasks)}, "
                    f"error={error_message}"
                )
                if len(valid_request_indices) == 1:
                    request_index = valid_request_indices[0]
                    responses[request_index] = self._error_response(
                        torch.cuda.OutOfMemoryError(error_message),
                        remote_traceback,
                    )
                else:
                    for request_index in valid_request_indices:
                        self._release_cuda_cache()
                        try:
                            # Rebuild every _ImageTask from the immutable HTTP
                            # params. The failed combined pipeline may already
                            # have populated mutable face/background fields.
                            responses[request_index] = (
                                self._process_request_batch_locked(
                                    [params_batch[request_index]],
                                    isolate_request_oom=False,
                                )[0]
                            )
                        except Exception as request_error:
                            if not is_cuda_oom_error(request_error):
                                raise
                            request_traceback = traceback.format_exc()
                            request_error.__traceback__ = None
                            self._release_cuda_cache()
                            responses[request_index] = self._error_response(
                                request_error,
                                request_traceback,
                            )

                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                logger.info(
                    "SuperResolution request-level OOM isolation completed: "
                    f"requests={len(params_batch)}, images={len(valid_tasks)}, "
                    f"elapsed_ms={elapsed_ms:.1f}"
                )
                return responses

        for request_index, response in enumerate(responses):
            if response is not None:
                continue

            try:
                results = OrderedDict()
                for task in request_tasks[request_index]:
                    if task.final_output is None:
                        raise RuntimeError(
                            f"Missing output for image {task.image_name}"
                        )
                    new_name = f"{task.image_name}_{self.outscale}x"
                    output_rgb = cv2.cvtColor(
                        task.final_output, cv2.COLOR_BGR2RGB
                    )
                    results[new_name] = pil_to_bytes(Image.fromarray(output_rgb))
                responses[request_index] = self._success_response(results)
            except Exception as error:
                logger.error(f"Error encoding SuperResolution result: {error}")
                responses[request_index] = self._error_response(
                    error,
                    traceback.format_exc(),
                )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "SuperResolution dynamic batch completed: "
            f"requests={len(params_batch)}, images={len(valid_tasks)}, "
            f"elapsed_ms={elapsed_ms:.1f}"
        )
        return responses

    def _run_task_pipeline(self, tasks):
        detection_ms = 0.0
        restoration_ms = 0.0
        background_ms = 0.0
        parsing_ms = 0.0
        paste_ms = 0.0
        total_faces = 0

        if self.face_enhance:
            stage_started = time.perf_counter()
            self._prepare_face_batches(tasks)
            detection_ms = (time.perf_counter() - stage_started) * 1000.0

            face_items = []
            for task in tasks:
                task.restored_faces = [None] * len(task.cropped_faces)
                for face_index, cropped_face in enumerate(task.cropped_faces):
                    face_items.append((task, face_index, cropped_face))

            total_faces = len(face_items)
            stage_started = time.perf_counter()
            for start in range(0, len(face_items), self.max_face_batch_size):
                self._restore_faces_with_fallback(
                    face_items[start : start + self.max_face_batch_size]
                )
            restoration_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        self._run_background_batches(tasks)
        background_ms = (time.perf_counter() - stage_started) * 1000.0

        if self.face_enhance:
            parse_items = []
            for task in tasks:
                task.parse_masks = [None] * len(task.restored_faces)
                for face_index, restored_face in enumerate(task.restored_faces):
                    parse_items.append((task, face_index, restored_face))
            stage_started = time.perf_counter()
            for start in range(0, len(parse_items), self.max_face_parse_batch_size):
                self._parse_faces_with_fallback(
                    parse_items[start : start + self.max_face_parse_batch_size]
                )
            parsing_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        for task in tasks:
            if not self.face_enhance:
                task.final_output = task.background_output
                continue
            self._paste_faces(task)
        paste_ms = (time.perf_counter() - stage_started) * 1000.0

        logger.info(
            "SuperResolution pipeline stages: "
            f"images={len(tasks)}, faces={total_faces}, "
            f"detection_ms={detection_ms:.1f}, "
            f"restoration_ms={restoration_ms:.1f}, "
            f"background_ms={background_ms:.1f}, "
            f"parsing_ms={parsing_ms:.1f}, paste_ms={paste_ms:.1f}"
        )

    def _prepare_face_batches(self, tasks):
        detector = self.face_enhancer.face_helper.face_det
        if not hasattr(detector, "batched_detect_faces"):
            for task in tasks:
                self._prepare_faces_single(task)
            return

        buckets = OrderedDict()
        for task in tasks:
            height, width = task.input_bgr.shape[:2]
            buckets.setdefault((height, width), []).append(task)

        for shape, shape_tasks in buckets.items():
            current_batch = []
            current_pixels = 0
            image_pixels = shape[0] * shape[1]
            for task in shape_tasks:
                exceeds_count = (
                    len(current_batch) >= self.max_face_detection_batch_size
                )
                exceeds_pixels = (
                    self.max_batch_input_pixels > 0
                    and current_batch
                    and current_pixels + image_pixels > self.max_batch_input_pixels
                )
                if exceeds_count or exceeds_pixels:
                    self._prepare_face_batch_with_fallback(current_batch)
                    current_batch = []
                    current_pixels = 0
                current_batch.append(task)
                current_pixels += image_pixels

            if current_batch:
                self._prepare_face_batch_with_fallback(current_batch)

    def _prepare_face_batch_with_fallback(self, tasks):
        if not tasks:
            return
        if len(tasks) == 1:
            # Keep the exact official single-image detector path when there is
            # no batching opportunity.
            self._prepare_faces_single(tasks[0])
            return

        try:
            self._prepare_faces_batch(tasks)
            return
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise
            error_message = str(error)
            error.__traceback__ = None
            self._release_cuda_cache()

        midpoint = max(1, len(tasks) // 2)
        logger.warning(
            "RetinaFace detection batch failed; retrying smaller batches: "
            f"images={len(tasks)}, error={error_message}"
        )
        self._prepare_face_batch_with_fallback(tasks[:midpoint])
        self._prepare_face_batch_with_fallback(tasks[midpoint:])

    def _prepare_faces_batch(self, tasks):
        detector = self.face_enhancer.face_helper.face_det
        frames = torch.from_numpy(
            np.stack(
                [task.input_bgr.astype(np.float32) for task in tasks],
                axis=0,
            )
        )
        with torch.no_grad():
            boxes_batch, landmarks_batch = detector.batched_detect_faces(
                frames,
                conf_threshold=0.97,
                nms_threshold=0.4,
                use_origin_size=True,
            )

        if len(boxes_batch) != len(tasks) or len(landmarks_batch) != len(tasks):
            raise RuntimeError(
                "RetinaFace returned a batch length different from its input"
            )

        for task, boxes, landmarks in zip(tasks, boxes_batch, landmarks_batch):
            self._build_face_context_from_detections(task, boxes, landmarks)

    def _build_face_context_from_detections(self, task, boxes, landmarks):
        helper = self.face_enhancer.face_helper
        helper.clean_all()
        try:
            helper.input_img = task.input_bgr
            for box, landmark_flat in zip(boxes, landmarks):
                eye_distance = np.linalg.norm(
                    [
                        landmark_flat[0] - landmark_flat[2],
                        landmark_flat[1] - landmark_flat[3],
                    ]
                )
                if eye_distance < 5:
                    continue
                landmark = np.asarray(landmark_flat, dtype=np.float32).reshape(5, 2)
                helper.all_landmarks_5.append(landmark)
                helper.det_faces.append(np.asarray(box, dtype=np.float32))

            helper.align_warp_face()
            task.cropped_faces = [face.copy() for face in helper.cropped_faces]
            task.affine_matrices = [
                matrix.copy() for matrix in helper.affine_matrices
            ]
        finally:
            helper.clean_all()
            helper.input_img = None

    def _prepare_faces_single(self, task):
        helper = self.face_enhancer.face_helper
        helper.clean_all()
        try:
            helper.read_image(task.input_bgr)
            helper.get_face_landmarks_5(
                only_center_face=False,
                eye_dist_threshold=5,
            )
            helper.align_warp_face()
            task.cropped_faces = [face.copy() for face in helper.cropped_faces]
            task.affine_matrices = [
                matrix.copy() for matrix in helper.affine_matrices
            ]
        finally:
            helper.clean_all()
            helper.input_img = None

    def _restore_faces_with_fallback(self, face_items):
        if not face_items:
            return
        try:
            restored_faces = self._restore_face_batch(face_items)
            for (task, face_index, _), restored_face in zip(
                face_items, restored_faces
            ):
                task.restored_faces[face_index] = restored_face
            return
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise
            error_message = str(error)
            error.__traceback__ = None
            self._release_cuda_cache()

        if len(face_items) > 1:
            midpoint = max(1, len(face_items) // 2)
            logger.warning(
                "GFPGAN face batch failed; retrying smaller batches: "
                f"faces={len(face_items)}, error={error_message}"
            )
            self._restore_faces_with_fallback(face_items[:midpoint])
            self._restore_faces_with_fallback(face_items[midpoint:])
            return

        raise torch.cuda.OutOfMemoryError(
            f"GFPGAN restoration OOM for one face: {error_message}"
        )

    def _restore_face_batch(self, face_items):
        face_tensors = []
        for _, _, cropped_face in face_items:
            face_tensor = img2tensor(
                cropped_face / 255.0,
                bgr2rgb=True,
                float32=True,
            )
            normalize(
                face_tensor,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                inplace=True,
            )
            face_tensors.append(face_tensor)

        input_batch = torch.stack(face_tensors, dim=0).to(self.device)
        with torch.no_grad():
            output_batch = self.face_enhancer.gfpgan(
                input_batch,
                return_rgb=False,
                weight=0.5,
            )[0]

        restored_faces = []
        for output in output_batch:
            restored_face = tensor2img(
                output,
                rgb2bgr=True,
                min_max=(-1, 1),
            ).astype(np.uint8)
            restored_faces.append(restored_face)
        return restored_faces

    def _run_background_batches(self, tasks):
        if not tasks:
            return

        if self.upsampler.tile_size > 0:
            # The official tiled helper mutates tile assembly state. Keep it
            # serial and byte-compatible when a non-default tile mode is used.
            for task in tasks:
                task.background_output = self.upsampler.enhance(
                    task.input_bgr,
                    outscale=self.outscale,
                )[0]
            return

        buckets = OrderedDict()
        for task in tasks:
            height, width = task.input_bgr.shape[:2]
            buckets.setdefault((height, width), []).append(task)

        for shape, shape_tasks in buckets.items():
            current_batch = []
            current_pixels = 0
            image_pixels = shape[0] * shape[1]
            for task in shape_tasks:
                exceeds_count = len(current_batch) >= self.max_batch_size
                exceeds_pixels = (
                    self.max_batch_input_pixels > 0
                    and current_batch
                    and current_pixels + image_pixels > self.max_batch_input_pixels
                )
                if exceeds_count or exceeds_pixels:
                    self._upsample_background_with_fallback(current_batch)
                    current_batch = []
                    current_pixels = 0
                current_batch.append(task)
                current_pixels += image_pixels

            if current_batch:
                self._upsample_background_with_fallback(current_batch)

    def _upsample_background_with_fallback(self, tasks):
        if not tasks:
            return
        try:
            outputs = self._upsample_background_batch(tasks)
            for task, output in zip(tasks, outputs):
                task.background_output = output
            return
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise
            error_message = str(error)
            error.__traceback__ = None
            self._release_cuda_cache()

        if len(tasks) > 1:
            midpoint = max(1, len(tasks) // 2)
            logger.warning(
                "Real-ESRGAN background batch failed; retrying smaller batches: "
                f"images={len(tasks)}, error={error_message}"
            )
            self._upsample_background_with_fallback(tasks[:midpoint])
            self._upsample_background_with_fallback(tasks[midpoint:])
            return

        raise torch.cuda.OutOfMemoryError(
            f"Real-ESRGAN background OOM for one image: {error_message}"
        )

    def _upsample_background_batch(self, tasks):
        # The public worker always converts incoming PIL images to RGB and then
        # BGR uint8 before inference. This mirrors the corresponding RGB branch
        # of RealESRGANer.enhance without touching its mutable self.img/output.
        input_arrays = []
        for task in tasks:
            image = task.input_bgr.astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_arrays.append(np.transpose(image, (2, 0, 1)))

        input_batch = torch.from_numpy(np.stack(input_arrays, axis=0)).float()
        input_batch = input_batch.to(self.device)
        if self.upsampler.half:
            input_batch = input_batch.half()

        if self.upsampler.pre_pad != 0:
            input_batch = F.pad(
                input_batch,
                (0, self.upsampler.pre_pad, 0, self.upsampler.pre_pad),
                "reflect",
            )

        mod_scale = None
        if self.upsampler.scale == 2:
            mod_scale = 2
        elif self.upsampler.scale == 1:
            mod_scale = 4

        mod_pad_h = 0
        mod_pad_w = 0
        if mod_scale is not None:
            height, width = input_batch.shape[-2:]
            mod_pad_h = (mod_scale - height % mod_scale) % mod_scale
            mod_pad_w = (mod_scale - width % mod_scale) % mod_scale
            if mod_pad_h or mod_pad_w:
                input_batch = F.pad(
                    input_batch,
                    (0, mod_pad_w, 0, mod_pad_h),
                    "reflect",
                )

        with torch.no_grad():
            output_batch = self.upsampler.model(input_batch)

        if mod_scale is not None:
            output_height, output_width = output_batch.shape[-2:]
            output_batch = output_batch[
                :,
                :,
                : output_height - mod_pad_h * self.upsampler.scale,
                : output_width - mod_pad_w * self.upsampler.scale,
            ]
        if self.upsampler.pre_pad != 0:
            output_height, output_width = output_batch.shape[-2:]
            output_batch = output_batch[
                :,
                :,
                : output_height - self.upsampler.pre_pad * self.upsampler.scale,
                : output_width - self.upsampler.pre_pad * self.upsampler.scale,
            ]

        output_batch = output_batch.float().cpu().clamp_(0, 1).numpy()
        outputs = []
        for task, output_tensor in zip(tasks, output_batch):
            output_image = np.transpose(
                output_tensor[[2, 1, 0], :, :],
                (1, 2, 0),
            )
            output_image = (output_image * 255.0).round().astype(np.uint8)
            if self.outscale != float(self.upsampler.scale):
                input_height, input_width = task.input_bgr.shape[:2]
                output_image = cv2.resize(
                    output_image,
                    (
                        int(input_width * self.outscale),
                        int(input_height * self.outscale),
                    ),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            outputs.append(output_image)
        return outputs

    def _parse_faces_with_fallback(self, face_items):
        if not face_items:
            return
        try:
            parse_masks = self._parse_face_batch(face_items)
            for (task, face_index, _), parse_mask in zip(
                face_items, parse_masks
            ):
                task.parse_masks[face_index] = parse_mask
            return
        except Exception as error:
            if not is_cuda_oom_error(error):
                raise
            error_message = str(error)
            error.__traceback__ = None
            self._release_cuda_cache()

        if len(face_items) > 1:
            midpoint = max(1, len(face_items) // 2)
            logger.warning(
                "Face parsing batch failed; retrying smaller batches: "
                f"faces={len(face_items)}, error={error_message}"
            )
            self._parse_faces_with_fallback(face_items[:midpoint])
            self._parse_faces_with_fallback(face_items[midpoint:])
            return

        raise torch.cuda.OutOfMemoryError(
            f"Face parsing OOM for one face: {error_message}"
        )

    def _parse_face_batch(self, face_items):
        face_tensors = []
        for _, _, restored_face in face_items:
            face_input = cv2.resize(
                restored_face,
                (512, 512),
                interpolation=cv2.INTER_LINEAR,
            )
            face_tensor = img2tensor(
                face_input.astype(np.float32) / 255.0,
                bgr2rgb=True,
                float32=True,
            )
            normalize(
                face_tensor,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                inplace=True,
            )
            face_tensors.append(face_tensor)

        input_batch = torch.stack(face_tensors, dim=0).to(self.device)
        with torch.no_grad():
            parse_logits = self.face_enhancer.face_helper.face_parse(input_batch)[0]
        parse_classes = parse_logits.argmax(dim=1).cpu().numpy()

        mask_colormap = [
            0,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            0,
            255,
            0,
            0,
            0,
        ]
        parse_masks = []
        for (_, _, restored_face), parse_class in zip(
            face_items, parse_classes
        ):
            mask = np.zeros(parse_class.shape)
            for class_index, color in enumerate(mask_colormap):
                mask[parse_class == class_index] = color
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            threshold = 10
            mask[:threshold, :] = 0
            mask[-threshold:, :] = 0
            mask[:, :threshold] = 0
            mask[:, -threshold:] = 0
            mask = mask / 255.0
            mask = cv2.resize(mask, restored_face.shape[:2])
            parse_masks.append(mask)
        return parse_masks

    def _paste_faces(self, task):
        try:
            if any(face is None for face in task.restored_faces):
                raise RuntimeError(
                    f"Missing restored face for image {task.image_name}"
                )
            if any(mask is None for mask in task.parse_masks):
                raise RuntimeError(
                    f"Missing face parse mask for image {task.image_name}"
                )

            input_height, input_width = task.input_bgr.shape[:2]
            output_height = int(input_height * self.outscale)
            output_width = int(input_width * self.outscale)
            output = cv2.resize(
                task.background_output,
                (output_width, output_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

            for restored_face, affine_matrix, parse_mask in zip(
                task.restored_faces,
                task.affine_matrices,
                task.parse_masks,
            ):
                inverse_affine = cv2.invertAffineTransform(affine_matrix)
                inverse_affine *= self.outscale
                extra_offset = 0.5 * self.outscale if self.outscale > 1 else 0
                inverse_affine[:, 2] += extra_offset

                pasted_face = cv2.warpAffine(
                    restored_face,
                    inverse_affine,
                    (output_width, output_height),
                )
                soft_mask = cv2.warpAffine(
                    parse_mask,
                    inverse_affine,
                    (output_width, output_height),
                    flags=3,
                )[:, :, None]
                output = soft_mask * pasted_face + (1 - soft_mask) * output

            task.final_output = (
                output.astype(np.uint16)
                if np.max(output) > 256
                else output.astype(np.uint8)
            )
        except Exception as error:
            raise RuntimeError(
                f"face paste-back failed for image {task.image_name}: {error}"
            ) from error

    def _release_cuda_cache(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except Exception as error:
            logger.warning(f"Failed to release CUDA cache after OOM: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="SuperResolution")
    parser.add_argument("--worker_name", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20038)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--outscale", type=float, default=4)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=0)
    parser.add_argument("--denoise-strength", type=float, default=0.5)
    parser.add_argument(
        "--face-enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--batch_wait_ms", type=float, default=30.0)
    parser.add_argument("--max_batch_requests", type=int, default=0)
    parser.add_argument("--max_batch_size", type=int, default=2)
    parser.add_argument("--max_batch_input_pixels", type=int, default=524288)
    parser.add_argument("--max_face_detection_batch_size", type=int, default=2)
    parser.add_argument("--max_face_batch_size", type=int, default=8)
    parser.add_argument("--max_face_parse_batch_size", type=int, default=8)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SuperResolutionWorker(
        controller_addr=args.controller_address,
        worker_name=args.worker_name,
        worker_addr=args.worker_address,
        tool_name=args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
        model_path=args.model_path,
        outscale=args.outscale,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        denoise_strength=args.denoise_strength,
        face_enhance=args.face_enhance,
        fp32=args.fp32,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_requests=args.max_batch_requests,
        max_batch_size=args.max_batch_size,
        max_batch_input_pixels=args.max_batch_input_pixels,
        max_face_detection_batch_size=args.max_face_detection_batch_size,
        max_face_batch_size=args.max_face_batch_size,
        max_face_parse_batch_size=args.max_face_parse_batch_size,
    )
    worker.run()
