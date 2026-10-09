"""Resource accounting tests without GPU access or serving requests."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import pytest

from visharness.trajectory_runner import benchmark
from visharness.trajectory_runner.benchmark import (
    InferenceResourceMonitor,
    _calculate_kv_cache_capacity,
    _parse_prometheus_samples,
)


def _monitor(monkeypatch, text, **config):
    current = {"text": text}
    monkeypatch.setattr(
        benchmark.urllib.request,
        "urlopen",
        lambda *args, **kwargs: BytesIO(current["text"].encode()),
    )
    monitor = InferenceResourceMonitor(
        {"enabled": True, "vllm_metrics_url": "http://unused/metrics", **config},
        trace_path=None,
        run_id="test",
    )
    return monitor, current


def _cache(engine, pool_bytes="None", blocks="None"):
    return (
        f'vllm:cache_config_info{{engine="{engine}",block_size="16",'
        f'num_gpu_blocks="{blocks}",kv_cache_memory_bytes="{pool_bytes}",'
        'cache_dtype="auto"} 1\n'
    )


@pytest.mark.parametrize("blocks", [None, "None", "null", ""])
def test_explicit_capacity_does_not_require_gpu_blocks(blocks):
    config = {"num_gpu_blocks": blocks, "kv_cache_memory_bytes": str(3 * 1024**3)}
    capacity = _calculate_kv_cache_capacity(
        config, model_config_path=None, tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    assert capacity["pool_gib_per_gpu"] == 3.0
    override = _calculate_kv_cache_capacity(
        config, model_config_path=None, tensor_parallel_size=1,
        pipeline_parallel_size=1, pool_gib_override=4.0,
    )
    assert override["pool_gib_per_gpu"] == 4.0


def test_prometheus_preserves_labels_spaces_and_ignores_nonfinite():
    samples = _parse_prometheus_samples(
        'vllm:kv_cache_usage_perc{engine="1",model_name="my model"} 0.5 1234\n'
        'vllm:kv_cache_usage_perc{engine="2"} NaN\n'
    )
    assert samples["vllm:kv_cache_usage_perc"] == [
        ({"engine": "1", "model_name": "my model"}, 0.5)
    ]


def test_different_dp_pools_are_matched_to_their_own_usage(monkeypatch):
    text = (
        _cache("0", 2 * 1024**3) + _cache("1", 8 * 1024**3)
        + 'vllm:kv_cache_usage_perc{engine="0"} 0.9\n'
        + 'vllm:kv_cache_usage_perc{engine="1"} 0.5\n'
        + 'vllm:num_requests_running{engine="0"} 2\n'
        + 'vllm:num_requests_running{engine="1"} 3\n'
    )
    monitor, current = _monitor(monkeypatch, text)
    monitor._sample()
    current["text"] = text.replace('engine="1"} 0.5', 'engine="1"} 0.1')
    summary = monitor.stop()
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] == 4.0
    assert summary["peak_active_kv_engine"] == "1"
    assert summary["peak_active_kv_cache_usage_percent"] == 50.0
    assert summary["peak_kv_cache_usage_percent"] == 90.0
    assert summary["kv_cache_pool_gib_per_gpu"] is None
    assert summary["mixed_kv_cache_pool_capacities"] is True
    assert summary["active_kv_metrics_complete"] is True
    assert summary["peak_requests_running"] == 5.0
    assert summary["peak_requests_waiting"] is None


@pytest.mark.parametrize("override", [10.0, {"0": 10.0}, {0: 10.0}])
def test_manual_pool_works_without_cache_info(monkeypatch, override):
    monitor, _ = _monitor(
        monkeypatch, 'vllm:kv_cache_usage_perc{engine="0"} 0.2\n',
        kv_cache_pool_gib_per_gpu=override,
    )
    summary = monitor.stop()
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] == 2.0
    assert summary["active_kv_metrics_complete"] is True


def test_per_engine_manual_pool_overrides(monkeypatch):
    monitor, _ = _monitor(
        monkeypatch,
        _cache("0") + _cache("1")
        + 'vllm:kv_cache_usage_perc{engine="0"} 0.8\n'
        + 'vllm:kv_cache_usage_perc{engine="1"} 0.4\n',
        kv_cache_pool_gib_per_gpu={"0": 2.0, "1": 10.0},
    )
    assert monitor.stop()["peak_active_kv_cache_memory_per_gpu_gib"] == 4.0


def test_unknown_capacity_keeps_usage_but_not_fake_memory(monkeypatch):
    monitor, _ = _monitor(
        monkeypatch, _cache("0") + 'vllm:kv_cache_usage_perc{engine="0"} 0.2\n',
    )
    summary = monitor.stop()
    assert summary["peak_kv_cache_usage_percent"] == 20.0
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] is None
    assert summary["active_kv_metrics_complete"] is False
    assert "start_vLLM_VisHarness.bash" in summary["kv_cache_capacity_error"]


def test_capacity_can_be_resolved_after_startup(monkeypatch):
    monitor, current = _monitor(
        monkeypatch, _cache("0") + 'vllm:kv_cache_usage_perc{engine="0"} 0.2\n',
    )
    monitor._sample()
    assert monitor._kv_cache_capacity is None
    current["text"] = (
        _cache("0", 10 * 1024**3) + 'vllm:kv_cache_usage_perc{engine="0"} 0.1\n'
    )
    summary = monitor.stop()
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] == 2.0
    assert summary["kv_cache_capacity_error"] is None


def test_partial_replica_metrics_do_not_claim_complete_peak(monkeypatch):
    monitor, _ = _monitor(
        monkeypatch, _cache("0", 10 * 1024**3) + _cache("1", 10 * 1024**3)
        + 'vllm:kv_cache_usage_perc{engine="0"} 0.2\n',
    )
    summary = monitor.stop()
    assert summary["active_kv_metrics_complete"] is False
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] is None


def test_selected_model_filters_gauges_and_counters(monkeypatch):
    monitor, _ = _monitor(
        monkeypatch,
        'vllm:kv_cache_usage_perc{engine="0",model_name="VisHarness"} 0.2\n'
        'vllm:kv_cache_usage_perc{engine="0",model_name="other"} 0.9\n'
        'vllm:prompt_tokens_total{engine="0",model_name="other"} 100\n'
        'vllm:num_requests_running{engine="0",model_name="VisHarness"} 0\n',
        model_name="VisHarness", kv_cache_pool_gib_per_gpu=10,
    )
    summary = monitor.stop()
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] == 2.0
    assert summary["peak_requests_running"] == 0.0
    assert summary["peak_requests_waiting"] is None
    assert summary["vllm_counter_start"] == {}


def test_counters_that_appear_later_get_their_own_baseline(monkeypatch):
    monitor, current = _monitor(monkeypatch, "vllm:prompt_tokens_total 10\n")
    monitor._sample()
    current["text"] += "vllm:generation_tokens_total 100\n"
    monitor._sample()
    current["text"] = (
        "vllm:prompt_tokens_total 20\nvllm:generation_tokens_total 105\n"
    )
    summary = monitor.stop()
    assert summary["vllm_counter_delta"] == {
        "prompt_tokens_total": 10.0, "generation_tokens_total": 5.0,
    }


def test_tp_does_not_default_to_dp_gpu_count():
    monitor = InferenceResourceMonitor(
        {"gpu_indices": [0, 1, 2, 3]}, trace_path=None, run_id="test"
    )
    assert monitor.tensor_parallel_size == 1


def test_disabled_monitor_does_not_poll_or_initialize_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled resource monitor must not perform I/O")

    monitor = InferenceResourceMonitor({}, trace_path=None, run_id="test")
    monkeypatch.setattr(monitor, "_initialize_gpu_monitor", forbidden)
    monkeypatch.setattr(monitor, "_sample", forbidden)
    monitor.start()
    assert monitor.stop() == {"enabled": False, "sample_count": 0, "monitor_errors": []}


def test_hybrid_geometry_requires_an_explicit_pool(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"layer_types": ["linear_attention"]}))
    with pytest.raises(ValueError, match="hybrid/MLA"):
        _calculate_kv_cache_capacity(
            {"num_gpu_blocks": "100", "block_size": "16"},
            model_config_path=path, tensor_parallel_size=1, pipeline_parallel_size=1,
        )


def test_startup_tokens_resolve_four_dp_capacities_automatically(monkeypatch, tmp_path):
    # The model file can have FP32 weights even when the server runs BF16.
    model = tmp_path / "config.json"
    model.write_text(json.dumps({
        "num_hidden_layers": 36, "num_key_value_heads": 8,
        "num_attention_heads": 32, "head_dim": 128, "torch_dtype": "float32",
    }))
    startup = {
        "model_config_path": str(model), "model_dtype": "bfloat16",
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "server_process": {"pid": 123, "start_ticks": "test"},
        "startup_log": "startup.log",
        "engines": {str(i): {"capacity_tokens": 350464} for i in range(4)},
    }
    monkeypatch.setattr(benchmark, "load_startup_record", lambda *args, **kwargs: startup)
    text = "".join(
        _cache(str(i)) + f'vllm:kv_cache_usage_perc{{engine="{i}"}} {0.1 * (i + 1)}\n'
        for i in range(4)
    )
    monitor, _ = _monitor(monkeypatch, text, model_config_path=model)
    summary = monitor.stop()
    assert summary["kv_cache_pool_gib_per_gpu"] == 48.12890625
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] == pytest.approx(48.12890625 * 0.4)
    assert summary["active_kv_metrics_complete"] is True
    assert summary["kv_cache_capacity"]["calculation_method"] == "vllm_startup_tokens_and_model_kv_geometry"
    assert summary["kv_cache_capacity"]["resolved_cache_dtype"] == "bfloat16"


def test_startup_known_engines_require_complete_usage_samples(monkeypatch, tmp_path):
    startup = {
        "model_config_path": str(tmp_path / "unused"), "model_dtype": "bfloat16",
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "server_process": {"pid": 123}, "startup_log": "startup.log",
        "engines": {str(i): {"capacity_tokens": 16} for i in range(4)},
    }
    monkeypatch.setattr(benchmark, "load_startup_record", lambda *args, **kwargs: startup)
    monitor, _ = _monitor(
        monkeypatch, 'vllm:kv_cache_usage_perc{engine="0"} 0.5\n',
        kv_cache_pool_gib_per_gpu=10,
    )
    summary = monitor.stop()
    assert summary["active_kv_metrics_complete"] is False
    assert summary["peak_active_kv_cache_memory_per_gpu_gib"] is None


@pytest.fixture
def startup_log(monkeypatch, tmp_path):
    identity = benchmark._process_identity(os.getpid())
    path = tmp_path / "server.log"
    options = argparse.Namespace(
        model=str(tmp_path / "model"), host="0.0.0.0", port=8000,
        served_model_name=["VisHarness"], tensor_parallel_size=1,
        pipeline_parallel_size=1, data_parallel_size=4,
        decode_context_parallel_size=1, prefill_context_parallel_size=1,
    )
    header = (
        f'VISHARNESS_VLLM_STARTUP pid={identity["pid"]} '
        f'start_ticks={identity["start_ticks"]} boot_id={identity["boot_id"]}\n'
    )
    body = "Initializing a V1 LLM engine with dtype=torch.bfloat16, kv_cache_dtype=auto\n"
    body += "".join(
        f'(EngineCore_DP{i} pid={identity["pid"]}) GPU KV cache size: 350,464 tokens\n'
        for i in range(4)
    )
    path.write_text(header + body)
    monkeypatch.setattr(benchmark, "_startup_log_paths", lambda _: [path])
    monkeypatch.setattr(benchmark, "_serving_options", lambda _: options)
    return path, options


def _read_startup(options, **overrides):
    kwargs = {
        "metrics_url": "http://localhost:8000/metrics", "model_name": "VisHarness",
        "model_config_path": options.model,
    }
    kwargs.update(overrides)
    return benchmark.load_startup_record(**kwargs)


def test_direct_cli_log_resolves_all_four_replicas(startup_log):
    path, options = startup_log
    record = _read_startup(options)
    assert record["startup_log"] == str(path)
    assert record["model_dtype"] == "bfloat16"
    assert set(record["engines"]) == {"0", "1", "2", "3"}
    assert record["engines"]["3"]["capacity_tokens"] == 350464
    assert record["tensor_parallel_size"] == 1
    assert _read_startup(options, model_config_path=Path(options.model) / "config.json")


@pytest.mark.parametrize("kwargs", [
    {"model_name": "another-model"},
    {"model_config_path": "/another/checkpoint"},
    {"metrics_url": "http://localhost:8001/metrics"},
    {"metrics_url": "http://remote-host:8000/metrics"},
])
def test_direct_cli_log_requires_matching_service(startup_log, kwargs):
    _, options = startup_log
    assert _read_startup(options, **kwargs) is None


@pytest.mark.parametrize("corruption", ["start_ticks", "boot_id", "header", "partial", "dead_engine"])
def test_old_or_incomplete_startup_log_is_not_used(startup_log, corruption):
    path, options = startup_log
    text = path.read_text()
    if corruption == "start_ticks":
        text = text.replace('start_ticks=', 'start_ticks=99')
    elif corruption == "boot_id":
        text = text.replace('boot_id=', 'boot_id=old-')
    elif corruption == "header":
        text = text.split("\n", 1)[1]
    elif corruption == "partial":
        text = "\n".join(text.splitlines()[:-1])
    else:
        text = text.replace(f"DP3 pid={os.getpid()}", "DP3 pid=999999999")
    path.write_text(text)
    assert _read_startup(options) is None


def test_startup_log_rejects_unrelated_engine_processes(startup_log, monkeypatch):
    _, options = startup_log
    monkeypatch.setattr(benchmark, "_is_descendant", lambda *args: False)
    assert _read_startup(options) is None


def test_startup_dtype_ignores_unrelated_log_messages(startup_log):
    path, options = startup_log
    with path.open("a") as file:
        file.write("request mentions dtype=float32, kv_cache_dtype=fp8\n")
    assert _read_startup(options)["model_dtype"] == "bfloat16"


@pytest.mark.parametrize("host", ["127.0.0.2", "192.0.2.1"])
def test_startup_log_rejects_different_bind_address(startup_log, host):
    _, options = startup_log
    options.host = host
    assert _read_startup(options) is None


def test_single_engine_cli_log_is_supported(startup_log):
    path, options = startup_log
    options.data_parallel_size = 1
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:3]).replace("EngineCore_DP0", "EngineCore"))
    assert set(_read_startup(options)["engines"]) == {"0"}


def test_monitor_caches_log_but_rejects_reused_server_pid(startup_log, monkeypatch):
    _, options = startup_log
    loads = []
    original = benchmark.load_startup_record

    def read(*args, **kwargs):
        loads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "load_startup_record", read)
    monitor, _ = _monitor(
        monkeypatch, _cache("0") + 'vllm:kv_cache_usage_perc{engine="0"} 0.2\n',
        model_config_path=options.model,
    )
    monitor.vllm_metrics_url = "http://localhost:8000/metrics"
    monitor._sample()
    assert monitor._startup_record is not None
    monitor._sample()
    assert len(loads) == 1
    monkeypatch.setattr(benchmark, "_process_identity", lambda _: None)
    assert monitor.stop()["peak_active_kv_cache_memory_per_gpu_gib"] is None
    assert monitor._startup_record is None


def test_startup_log_search_includes_serving_checkout(monkeypatch, tmp_path):
    serving = tmp_path / "serving"
    inference = tmp_path / "inference"
    monkeypatch.setattr(benchmark, "__file__", str(inference / "visharness/trajectory_runner/benchmark.py"))
    paths = benchmark._startup_log_paths(serving / "checkpoints/model")
    assert paths[0] == inference / "outputs/vllm/server.log"
    assert serving / "outputs/vllm/server.log" in paths


def test_bash_launcher_executes_cli_directly_and_preserves_exit_status(tmp_path):
    root = tmp_path / "checkout"
    script = root / "recipe/visharness/scripts/trajectory_runner/start_vLLM_VisHarness.bash"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / script.relative_to(root)
    shutil.copyfile(source, script)
    executable = tmp_path / "vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print('FAKE_CLI=' + json.dumps({'pid': os.getpid(), 'args': sys.argv[1:], "
        "'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}), flush=True)\n"
        "sys.exit(7)\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "MODEL": "checkpoints/model", "CONDA_PREFIX": str(tmp_path)},
    )
    assert result.returncode == 7, result.stderr
    fake = json.loads(next(
        line.removeprefix("FAKE_CLI=")
        for line in result.stdout.splitlines() if line.startswith("FAKE_CLI=")
    ))
    log = (root / "outputs/vllm/server.log").read_text()
    assert f'pid={fake["pid"]} ' in log.splitlines()[0]
    assert fake["args"][:2] == ["serve", str(root / "checkpoints/model")]
    assert fake["gpus"] == "0,1,2,3"
    assert fake["args"][fake["args"].index("--data-parallel-size") + 1] == "4"
    assert "FAKE_CLI=" in log


def test_direct_cli_log_reader_works_with_real_process_without_gpu(tmp_path):
    # A fake executable exercises the real Bash/PID/log-reader path, not vLLM.
    root = tmp_path / "checkout"
    script = root / "recipe/visharness/scripts/trajectory_runner/start_vLLM_VisHarness.bash"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / script.relative_to(root)
    shutil.copyfile(source, script)
    executable = tmp_path / "vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "print('Initializing a V1 LLM engine dtype=torch.bfloat16, kv_cache_dtype=auto', flush=True)\n"
        "for i in range(4):\n"
        " print(f'(EngineCore_DP{i} pid={os.getpid()}) GPU KV cache size: 350,464 tokens', flush=True)\n"
        "print('READY', flush=True)\n"
        "while True: time.sleep(1)\n"
    )
    executable.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "MODEL": str(root / "checkpoints/model"), "CONDA_PREFIX": str(tmp_path)},
    )
    try:
        deadline = time.monotonic() + 5
        record = None
        while time.monotonic() < deadline:
            record = benchmark.load_startup_record(
                "http://localhost:8000/metrics", model_name="VisHarness",
                model_config_path=root / "checkpoints/model",
            )
            if record is not None:
                break
            time.sleep(0.02)
        assert record is not None
        assert record["server_process"]["pid"] == process.pid  # Bash exec, no wrapper.
        assert set(record["engines"]) == {"0", "1", "2", "3"}
        process.terminate()
        output, errors = process.communicate(timeout=5)
        assert process.returncode == -15, errors
        assert "READY" in output
        assert benchmark._process_identity(process.pid) is None
        assert benchmark.load_startup_record(
            "http://localhost:8000/metrics", model_name="VisHarness",
            model_config_path=root / "checkpoints/model",
        ) is None
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
