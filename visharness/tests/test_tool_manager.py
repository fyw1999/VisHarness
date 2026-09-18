import asyncio

import msgpack
import pytest

from tool_server.tool_workers.tool_manager import base_manager


class _FakeAsyncResponse:
    def __init__(self, *, json_data=None, packed_data=None, status=200):
        self._json_data = json_data
        self._packed_data = packed_data
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return self._json_data

    async def read(self):
        return self._packed_data


class _FakeAsyncSession:
    def __init__(self, worker_responses):
        self.worker_responses = list(worker_responses)
        self.controller_calls = 0
        self.worker_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        if url.endswith("/get_worker_address"):
            self.controller_calls += 1
            return _FakeAsyncResponse(
                json_data={
                    "worker_name": "PointToBoxMask_h20_2",
                    "worker_addr": "http://worker:8103",
                }
            )

        assert url == "http://worker:8103/worker_generate"
        response = self.worker_responses[self.worker_calls]
        self.worker_calls += 1
        if isinstance(response, BaseException):
            raise response
        packed_data = (
            response
            if isinstance(response, bytes)
            else msgpack.packb(response, use_bin_type=True)
        )
        return _FakeAsyncResponse(packed_data=packed_data)


def _make_manager(max_consecutive_oom=5):
    manager = base_manager.ToolManager.__new__(base_manager.ToolManager)
    manager.controller_addr = "http://controller:21001"
    manager.headers = {"VisionAgent": "Client"}
    manager.max_consecutive_oom = max_consecutive_oom
    return manager


@pytest.mark.parametrize(
    "response",
    [
        {"status": "error", "error_code": 50002, "message": "allocation failed"},
        {"status": "error", "error_type": "cuda_oom", "message": "allocation failed"},
        {"status": "error", "message": "torch.cuda.OutOfMemoryError"},
        {"status": "error", "message": "CUBLAS_STATUS_ALLOC_FAILED"},
        {"status": "error", "message": "CUDNN_STATUS_ALLOC_FAILED"},
        {"status": "error", "message": "CUDA_ERROR_OUT_OF_MEMORY"},
    ],
)
def test_tool_manager_recognizes_all_shared_cuda_oom_forms(response):
    assert base_manager._is_oom_response(response)


def test_tool_manager_does_not_retry_unrelated_runtime_error_as_oom():
    response = {
        "status": "error",
        "error_type": "tool_execution_error",
        "error_code": 50001,
        "message": "tensor shape mismatch",
    }
    assert not base_manager._is_oom_response(response)


def test_async_tool_manager_stops_immediately_after_fifth_oom(monkeypatch):
    oom_response = {
        "status": "error",
        "error_code": 50002,
        "message": "CUDA out of memory",
    }
    session = _FakeAsyncSession([oom_response] * 6)
    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", lambda **kwargs: session)

    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(base_manager.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        _make_manager().async_dynamic_call_tool("PointToBoxMask", b"packed-parameters")
    )

    assert session.worker_calls == 5
    assert session.controller_calls == 5
    assert sleep_calls == [20, 20, 20, 20]
    assert result["status"] == "error"
    assert result["error_type"] == "tool_oom_retry_exhausted"
    assert result["oom_attempts"] == 5
    assert len(result["oom_worker_history"]) == 5


def test_async_tool_manager_returns_success_before_oom_limit(monkeypatch):
    oom_response = {
        "status": "error",
        "error_code": 50002,
        "message": "OutOfMemoryError",
    }
    success_response = {"status": "success", "results": {"boxes": []}}
    session = _FakeAsyncSession([oom_response] * 4 + [success_response])
    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", lambda **kwargs: session)

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(base_manager.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        _make_manager().async_dynamic_call_tool("PointToBoxMask", b"packed-parameters")
    )

    assert session.worker_calls == 5
    assert result == success_response


def test_async_tool_manager_raises_non_oom_with_remote_traceback_without_retry(monkeypatch):
    failure = {
        "status": "error",
        "error_type": "tool_execution_error",
        "error_code": 50001,
        "exception_type": "KeyError",
        "message": "missing model output",
        "remote_traceback": "Traceback (most recent call last):\n  KeyError: missing model output",
    }
    session = _FakeAsyncSession([failure])
    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", lambda **kwargs: session)

    with pytest.raises(base_manager.ToolWorkerExecutionError) as exc_info:
        asyncio.run(
            _make_manager().async_dynamic_call_tool(
                "PointToBoxMask", b"packed-parameters"
            )
        )

    assert session.controller_calls == 1
    assert session.worker_calls == 1
    assert "KeyError: missing model output" in str(exc_info.value)
    assert "Remote worker traceback" in str(exc_info.value)


def test_async_tool_manager_raises_invalid_msgpack_without_retry(monkeypatch):
    session = _FakeAsyncSession([b"not-msgpack"])
    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", lambda **kwargs: session)

    with pytest.raises(base_manager.ToolWorkerProtocolError):
        asyncio.run(
            _make_manager().async_dynamic_call_tool(
                "PointToBoxMask", b"packed-parameters"
            )
        )

    assert session.controller_calls == 1
    assert session.worker_calls == 1


def test_async_tool_manager_raises_worker_transport_error_without_retry(monkeypatch):
    session = _FakeAsyncSession([base_manager.aiohttp.ClientConnectionError("down")])
    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", lambda **kwargs: session)

    with pytest.raises(base_manager.ToolTransportError):
        asyncio.run(
            _make_manager().async_dynamic_call_tool(
                "PointToBoxMask", b"packed-parameters"
            )
        )

    assert session.controller_calls == 1
    assert session.worker_calls == 1


def test_async_tool_manager_disables_client_timeout(monkeypatch):
    success_response = {"status": "success", "results": {}}
    session = _FakeAsyncSession([success_response])
    client_session_kwargs = {}

    def make_session(**kwargs):
        client_session_kwargs.update(kwargs)
        return session

    monkeypatch.setattr(base_manager.aiohttp, "ClientSession", make_session)

    asyncio.run(
        _make_manager().async_dynamic_call_tool(
            "PointToBoxMask", b"packed-parameters"
        )
    )

    timeout = client_session_kwargs["timeout"]
    assert timeout.total is None
    assert timeout.connect is None
    assert timeout.sock_connect is None
    assert timeout.sock_read is None
