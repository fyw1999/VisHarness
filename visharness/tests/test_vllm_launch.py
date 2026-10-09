"""Automatic KV capacity registration, without starting a real vLLM server."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from visharness.trajectory_runner import vllm_launch


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(vllm_launch, "_registry_path", lambda port: tmp_path / f"{port}.json")
    return tmp_path


def _recorder(tmp_path, dp=1):
    return vllm_launch.StartupRecorder([
        "serve", str(tmp_path / "model"), "--served-model-name", "VisHarness",
        "--tensor-parallel-size", "1", "--data-parallel-size", str(dp),
    ], os.getpid(), "startup.log")


def _engine_line(engine, tokens=350464):
    return (
        f"(EngineCore_DP{engine} pid={os.getpid()}) INFO "
        f"GPU KV cache size: {tokens:,} tokens\n"
    )


def _load(tmp_path, **kwargs):
    options = dict(
        metrics_url="http://localhost:8000/metrics", model_name="VisHarness",
        model_config_path=tmp_path / "model",
    )
    options.update(kwargs)
    return vllm_launch.load_startup_record(**options)


def test_four_replicas_are_registered_without_yaml_settings(registry):
    recorder = _recorder(registry, dp=4)
    recorder.observe("Initializing engine dtype=torch.bfloat16, kv_cache_dtype=auto")
    for engine in range(3):
        recorder.observe(_engine_line(engine))
    assert _load(registry) is None
    recorder.observe(_engine_line(3))
    record = _load(registry)
    assert recorder.complete
    assert set(record["engines"]) == {"0", "1", "2", "3"}
    assert record["model_dtype"] == "bfloat16"
    assert record["engines"]["3"]["capacity_tokens"] == 350464
    assert record["tensor_parallel_size"] == 1


@pytest.mark.parametrize("kwargs", [
    {"model_name": "different-model"},
    {"model_config_path": "/different/checkpoint"},
    {"metrics_url": "http://localhost:8001/metrics"},
    {"metrics_url": "http://remote-server:8000/metrics"},
])
def test_mismatched_service_is_never_used(registry, kwargs):
    recorder = _recorder(registry)
    recorder.observe("dtype=torch.bfloat16")
    recorder.observe(_engine_line(0))
    assert _load(registry, **kwargs) is None


def test_reused_pid_with_different_start_time_is_rejected(registry):
    recorder = _recorder(registry)
    recorder.observe("dtype=torch.bfloat16")
    recorder.observe(_engine_line(0))
    record = json.loads(recorder.path.read_text())
    record["server_process"]["start_ticks"] = "wrong-previous-process"
    recorder.path.write_text(json.dumps(record))
    assert _load(registry) is None


def test_dead_engine_is_rejected(registry):
    recorder = _recorder(registry)
    recorder.observe("dtype=torch.bfloat16")
    recorder.observe(_engine_line(0))
    record = json.loads(recorder.path.read_text())
    record["engines"]["0"]["process"]["pid"] = 999999999
    recorder.path.write_text(json.dumps(record))
    assert _load(registry) is None


def test_stale_engine_scrollback_is_ignored(registry, monkeypatch):
    recorder = _recorder(registry)
    recorder.observe("dtype=torch.bfloat16")
    monkeypatch.setattr(vllm_launch, "_is_descendant", lambda *args: False)
    recorder.observe(_engine_line(0))
    assert recorder.record["engines"] == {}


def test_corrupt_record_is_unavailable_not_fatal(registry):
    (registry / "8000.json").write_text("{invalid")
    assert _load(registry) is None


def test_dtype_does_not_accidentally_match_cache_dtype(registry):
    recorder = _recorder(registry)
    recorder.observe("dtype=torch.bfloat16, kv_cache_dtype=auto")
    recorder.observe("kv_cache_dtype=fp8")
    assert recorder.record["model_dtype"] == "bfloat16"


@pytest.mark.parametrize("fail_log", [False, True])
def test_launcher_preserves_arguments_output_and_exit_code(registry, monkeypatch, capsys, fail_log):
    class FakeProcess:
        pid = os.getpid()
        stdout = ["dtype=torch.bfloat16\n", _engine_line(0), "request log\n"]

        def poll(self):
            return 7

        def wait(self):
            return 7

    class FakeStdout(list):
        def close(self):
            pass

    process = FakeProcess()
    process.stdout = FakeStdout(process.stdout)
    seen = []

    def popen(command, **kwargs):
        seen.append((command, kwargs))
        return process

    monkeypatch.setattr(vllm_launch.shutil, "which", lambda _: "/fake/vllm")
    monkeypatch.setattr(vllm_launch.subprocess, "Popen", popen)
    # Put startup logs in the test's temporary directory, not the checkout.
    monkeypatch.setattr(vllm_launch, "__file__", str(registry / "a" / "b" / "vllm_launch.py"))
    if fail_log:
        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if path.name == "startup.log":
                raise OSError("disk full")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_path)
    arguments = ["serve", str(registry / "model"), "--served-model-name", "VisHarness", "--async-scheduling"]
    assert vllm_launch.main(arguments) == 7
    assert seen[0][0] == ["/fake/vllm", *arguments]
    assert seen[0][1]["start_new_session"] is True
    record = _load(registry)
    assert record is not None
    assert "request log" in capsys.readouterr().out
    if not fail_log:
        log = Path(record["startup_log"]).read_text()
        assert "GPU KV cache size" in log
        assert "request log" not in log  # Do not grow an unbounded HTTP log.


def test_launcher_forwards_termination_to_child_without_orphans(tmp_path):
    launcher = tmp_path / "a" / "b" / "vllm_launch.py"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(vllm_launch.__file__, launcher)
    executable = tmp_path / "vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "print('dtype=torch.bfloat16', flush=True)\n"
        "print(f'(EngineCore_DP0 pid={os.getpid()}) GPU KV cache size: 16 tokens', flush=True)\n"
        "print('READY', flush=True)\n"
        "while True: time.sleep(1)\n"
    )
    executable.chmod(0o755)
    with socket.socket() as lease:
        lease.bind(("127.0.0.1", 0))
        port = lease.getsockname()[1]
        process = subprocess.Popen([
            sys.executable, "-S", str(launcher), "serve", str(tmp_path / "model"),
            "--port", str(port),
        ], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        path = vllm_launch._registry_path(port)
        try:
            deadline = time.monotonic() + 5
            ready = False
            while time.monotonic() < deadline:
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["engines"]:
                        ready = True
                        break
                time.sleep(0.02)
            assert ready
            child_pid = record["server_process"]["pid"]
            process.terminate()
            output, errors = process.communicate(timeout=5)
            assert process.returncode == 143, errors
            assert "READY" in output
            assert vllm_launch._process_identity(child_pid) is None
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
            path.unlink(missing_ok=True)


def test_standalone_launcher_imports_no_training_dependencies():
    script = Path(vllm_launch.__file__)
    result = subprocess.run([
        "python", "-S", "-c",
        "import runpy; runpy.run_path(__import__('sys').argv[1], run_name='launcher_test')",
        str(script),
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
