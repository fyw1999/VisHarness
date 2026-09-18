import asyncio
import threading
from types import MethodType

import httpx
import msgpack
import pytest
import torch

from tool_server.tool_workers.online_workers.base_tool_worker import (
    BaseToolWorker,
    build_worker_error_response,
)


def _make_worker(error):
    worker = BaseToolWorker.__new__(BaseToolWorker)
    worker.tool_name = "TestTool"
    worker.model_semaphore = asyncio.BoundedSemaphore(1)
    worker.active_requests = 0
    worker.global_counter = 0
    worker.queue_lock = threading.Lock()

    async def generate_gate_async(self, params):
        raise error

    worker.generate_gate_async = MethodType(generate_gate_async, worker)
    from fastapi import FastAPI

    worker.app = FastAPI()
    worker.setup_routes()
    return worker


async def _post(worker, payload):
    transport = httpx.ASGITransport(app=worker.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://worker",
        timeout=None,
    ) as client:
        return await client.post("/worker_generate", content=payload)


def test_worker_serializes_non_oom_with_exception_type_and_traceback():
    response = asyncio.run(
        _post(
            _make_worker(KeyError("missing-output")),
            msgpack.packb({"image_dict": {}}, use_bin_type=True),
        )
    )
    body = msgpack.unpackb(response.content, raw=False)

    assert body["status"] == "error"
    assert body["error_type"] == "tool_execution_error"
    assert body["error_code"] == 50001
    assert body["exception_type"] == "KeyError"
    assert "missing-output" in body["message"]
    assert "KeyError" in body["remote_traceback"]


def test_worker_serializes_cuda_oom_as_the_only_retryable_error():
    response = asyncio.run(
        _post(
            _make_worker(torch.cuda.OutOfMemoryError("CUDA out of memory")),
            msgpack.packb({"image_dict": {}}, use_bin_type=True),
        )
    )
    body = msgpack.unpackb(response.content, raw=False)

    assert body["status"] == "error"
    assert body["error_type"] == "cuda_oom"
    assert body["error_code"] == 50002
    assert body["exception_type"] == "OutOfMemoryError"


@pytest.mark.parametrize(
    "message",
    [
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUDNN_STATUS_ALLOC_FAILED",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CUDA malloc failed",
    ],
)
def test_worker_uses_shared_cuda_oom_patterns_for_runtime_errors(message):
    body = build_worker_error_response("TestTool", RuntimeError(message))

    assert body["status"] == "error"
    assert body["error_type"] == "cuda_oom"
    assert body["error_code"] == 50002


def test_worker_serializes_malformed_request_instead_of_returning_raw_http_500():
    response = asyncio.run(_post(_make_worker(AssertionError("unused")), b"bad-msgpack"))
    body = msgpack.unpackb(response.content, raw=False)

    assert body["status"] == "error"
    assert body["error_type"] == "tool_execution_error"
    assert body["error_code"] == 50001
    assert body["exception_type"] in {"ExtraData", "ValueError"}
    assert "remote_traceback" in body
