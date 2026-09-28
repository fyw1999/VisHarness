import requests

from tool_server.tool_workers.online_workers import base_tool_worker
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker


class _TestWorker(BaseToolWorker):
    def __init__(self, events, *, no_register=False):
        self.events = events
        super().__init__(
            controller_addr="http://controller:20001",
            worker_name="test-worker",
            worker_addr="http://localhost:29999",
            no_register=no_register,
            tool_name="TestTool",
            port=29999,
        )

    def init_model(self):
        self.events.append("model_initialized")


class _Response:
    def __init__(self, payload=None):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_constructor_initializes_model_without_registering(monkeypatch):
    def unexpected_post(*args, **kwargs):
        raise AssertionError("The worker registered during construction")

    monkeypatch.setattr(base_tool_worker.requests, "post", unexpected_post)
    events = []
    worker = _TestWorker(events)

    assert events == ["model_initialized"]
    assert worker.registration_thread is None
    assert worker.heart_beat_thread is None


def test_worker_registers_and_starts_heartbeat_only_after_http_is_ready(monkeypatch):
    events = []
    worker = _TestWorker(events)
    readiness_attempts = 0

    def fake_post(url, **kwargs):
        nonlocal readiness_attempts
        if url.endswith("/worker_get_status"):
            readiness_attempts += 1
            events.append("readiness_probe")
            if readiness_attempts == 1:
                raise requests.ConnectionError("server is not listening")
            return _Response(
                {
                    "model_names": [worker.tool_name],
                    "speed": 1,
                    "queue_length": 0,
                    "worker_addr": worker.worker_addr,
                }
            )
        if url.endswith("/register_worker"):
            events.append("registered")
            return _Response()
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(base_tool_worker.requests, "post", fake_post)
    monkeypatch.setattr(base_tool_worker, "WORKER_READY_CHECK_INTERVAL", 0)
    monkeypatch.setattr(
        worker,
        "start_heart_beat_thread",
        lambda: events.append("heart_beat_started"),
    )

    worker.wait_until_ready_and_register()

    assert events == [
        "model_initialized",
        "readiness_probe",
        "readiness_probe",
        "registered",
        "heart_beat_started",
    ]


def test_run_starts_readiness_thread_before_uvicorn(monkeypatch):
    events = []
    worker = _TestWorker(events)

    class _Thread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            events.append("readiness_thread_started")

    monkeypatch.setattr(base_tool_worker.threading, "Thread", _Thread)
    monkeypatch.setattr(worker, "release_port", lambda port: events.append("port_released"))
    monkeypatch.setattr(
        base_tool_worker.uvicorn,
        "run",
        lambda *args, **kwargs: events.append("uvicorn_started"),
    )

    worker.run()

    assert events == [
        "model_initialized",
        "port_released",
        "readiness_thread_started",
        "uvicorn_started",
    ]
    assert worker.stop_event.is_set()
