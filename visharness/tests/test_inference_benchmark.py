"""Resource accounting tests without GPU access or serving requests."""

import json
from io import BytesIO

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
    assert "benchmark.kv_cache_pool_gib_per_gpu" in summary["kv_cache_capacity_error"]


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
