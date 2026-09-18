from io import BytesIO

import pytest
import torch
from PIL import Image


pytest.importorskip("basicsr")
pytest.importorskip("realesrgan")
pytest.importorskip("gfpgan")

from tool_server.tool_workers.online_workers.super_resolution_worker import (
    SuperResolutionWorker,
)


def _png_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_worker():
    worker = SuperResolutionWorker.__new__(SuperResolutionWorker)
    worker.face_enhance = False
    worker.outscale = 4
    worker.device = torch.device("cpu")
    worker._release_cuda_cache = lambda: None
    return worker


def test_super_resolution_combined_oom_retries_original_requests_independently():
    worker = _make_worker()
    task_object_ids = []

    def run_pipeline(tasks):
        task_object_ids.append([id(task) for task in tasks])
        if len({task.request_index for task in tasks}) > 1:
            tasks[0].cropped_faces.append("partial-state")
            raise torch.cuda.OutOfMemoryError("combined CUDA out of memory")
        if any(task.image_name == "bad.png" for task in tasks):
            raise torch.cuda.OutOfMemoryError("single CUDA out of memory")
        for task in tasks:
            task.final_output = task.input_bgr.copy()

    worker._run_task_pipeline = run_pipeline
    responses = worker._process_request_batch_locked(
        [
            {"image_dict": {"good-a.png": _png_bytes("red")}},
            {"image_dict": {"bad.png": _png_bytes("green")}},
            {"image_dict": {"good-b.png": _png_bytes("blue")}},
        ]
    )

    assert [response["status"] for response in responses] == [
        "success",
        "error",
        "success",
    ]
    assert responses[1]["error_type"] == "cuda_oom"
    assert set(responses[0]["results"]) == {"good-a.png_4x"}
    assert set(responses[2]["results"]) == {"good-b.png_4x"}
    parent_task_ids = set(task_object_ids[0])
    assert all(
        parent_task_ids.isdisjoint(retry_ids) for retry_ids in task_object_ids[1:]
    )


def test_super_resolution_multi_image_request_is_atomic_on_leaf_oom():
    worker = _make_worker()

    def run_pipeline(tasks):
        if any(task.image_name == "bad.png" for task in tasks):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        for task in tasks:
            task.final_output = task.input_bgr.copy()

    worker._run_task_pipeline = run_pipeline
    response = worker._process_request_batch_locked(
        [
            {
                "image_dict": {
                    "good.png": _png_bytes("red"),
                    "bad.png": _png_bytes("green"),
                }
            }
        ]
    )[0]

    assert response["status"] == "error"
    assert response["error_type"] == "cuda_oom"


def test_super_resolution_non_oom_batch_failure_is_not_hidden():
    worker = _make_worker()
    worker._run_task_pipeline = lambda _tasks: (_ for _ in ()).throw(
        ValueError("broken model output")
    )

    with pytest.raises(ValueError, match="broken model output"):
        worker._process_request_batch_locked(
            [{"image_dict": {"image.png": _png_bytes("red")}}]
        )
