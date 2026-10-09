"""Runtime resource monitoring for trajectory-runner efficiency benchmarks."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VLLM_METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
}
_VLLM_CACHE_CONFIG_METRIC = "vllm:cache_config_info"
_PROMETHEUS_LABEL_PATTERN = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"'
)
_DTYPE_BYTES = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_inc": 1,
    "fp8_ds_mla": 1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_gpu_indices(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    indices: list[int] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError("benchmark.gpu_indices must contain GPU integers")
        index = int(item)
        if index < 0:
            raise ValueError("benchmark.gpu_indices cannot contain negatives")
        if index not in indices:
            indices.append(index)
    return indices


def _parse_prometheus_samples(
    text: str,
) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Keep engine labels so DP replicas are not mistaken for one KV pool."""

    values: dict[str, list[tuple[dict[str, str], float]]] = {
        name: [] for name in _VLLM_METRICS | {_VLLM_CACHE_CONFIG_METRIC}
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r'([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?', line
        )
        if match is None:
            continue
        metric_name, raw_labels, raw_value = match.groups()
        if metric_name not in values:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels: dict[str, str] = {}
        for label_match in _PROMETHEUS_LABEL_PATTERN.finditer(raw_labels or ""):
            key, raw_label_value = label_match.groups()
            try:
                label_value = json.loads(f'"{raw_label_value}"')
            except json.JSONDecodeError:
                label_value = raw_label_value
            labels[key] = str(label_value)
        values[metric_name].append((labels, value))
    return values


def _parse_prometheus_metrics(text: str) -> dict[str, list[float]]:
    samples = _parse_prometheus_samples(text)
    return {name: [value for _, value in samples[name]] for name in _VLLM_METRICS}


def _parse_prometheus_metric_labels(
    text: str,
    metric_name: str,
) -> dict[str, str]:
    """Return labels from the first sample of a Prometheus info metric."""

    samples = _parse_prometheus_samples(text).get(metric_name, [])
    return samples[0][0] if samples else {}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _model_config_file(path_value: Any) -> Path:
    if not path_value:
        raise ValueError(
            "benchmark.model_config_path is required to calculate automatic "
            "KV-cache memory"
        )
    path = Path(str(path_value)).expanduser()
    config_path = path if path.is_file() else path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return config_path


def _resolve_cache_dtype(
    cache_dtype: str,
    model_config: dict[str, Any],
    text_config: dict[str, Any],
) -> tuple[str, int]:
    dtype = str(cache_dtype or "auto").lower()
    if dtype == "auto":
        dtype = str(
            text_config.get("dtype")
            or text_config.get("torch_dtype")
            or model_config.get("dtype")
            or model_config.get("torch_dtype")
            or ""
        ).lower()
    dtype = dtype.removeprefix("torch.")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported resolved KV-cache dtype: {dtype!r}")
    return dtype, _DTYPE_BYTES[dtype]


def _calculate_kv_cache_capacity(
    cache_config: dict[str, str],
    *,
    model_config_path: Any,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    pool_gib_override: Any = None,
) -> dict[str, Any]:
    """Calculate the physical KV-cache pool capacity allocated per GPU."""

    # Newer vLLM versions expose num_gpu_blocks="None". An explicit
    # capacity must still work without these optional block metadata.
    if pool_gib_override is not None:
        pool_gib = _positive_float(
            pool_gib_override,
            "benchmark.kv_cache_pool_gib_per_gpu",
        )
        pool_bytes = int(round(pool_gib * 1024**3))
        return {
            "available": True,
            "calculation_method": "configured_pool_gib_override",
            "pool_bytes_per_gpu": pool_bytes,
            "pool_gib_per_gpu": pool_bytes / 1024**3,
        }

    raw_explicit_bytes = cache_config.get("kv_cache_memory_bytes")
    if raw_explicit_bytes not in {None, "", "None", "null"}:
        pool_bytes = _positive_int(
            raw_explicit_bytes,
            "vLLM kv_cache_memory_bytes",
        )
        return {
            "available": True,
            "calculation_method": "vllm_kv_cache_memory_bytes",
            "pool_bytes_per_gpu": pool_bytes,
            "pool_gib_per_gpu": pool_bytes / 1024**3,
        }

    try:
        num_gpu_blocks = _positive_int(
            cache_config.get("num_gpu_blocks"), "vLLM num_gpu_blocks"
        )
        block_size = _positive_int(
            cache_config.get("block_size"), "vLLM block_size"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "vLLM does not expose usable KV-cache capacity; set "
            "benchmark.kv_cache_pool_gib_per_gpu to the actual KV pool "
            "capacity per GPU (not total GPU memory). Usage remains available."
        ) from exc
    capacity_tokens = num_gpu_blocks * block_size

    config_path = _model_config_file(model_config_path)
    with config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)
    if not isinstance(model_config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    raw_text_config = model_config.get("text_config")
    text_config = (
        raw_text_config
        if isinstance(raw_text_config, dict)
        else model_config
    )
    if (
        text_config.get("kv_lora_rank") is not None
        or text_config.get("use_mla")
        or text_config.get("use_sliding_window")
        or any(
            layer_type != "full_attention"
            for layer_type in (text_config.get("layer_types") or [])
        )
    ):
        raise ValueError(
            "Automatic KV capacity requires full-attention KV geometry; "
            "set benchmark.kv_cache_pool_gib_per_gpu for hybrid/MLA models"
        )

    total_layers = _positive_int(
        text_config.get("num_hidden_layers"),
        "model num_hidden_layers",
    )
    total_kv_heads = _positive_int(
        text_config.get("num_key_value_heads")
        or text_config.get("num_attention_heads"),
        "model num_key_value_heads",
    )
    attention_heads = _positive_int(
        text_config.get("num_attention_heads"),
        "model num_attention_heads",
    )
    head_size_value = text_config.get("head_dim")
    if head_size_value is None:
        hidden_size = _positive_int(
            text_config.get("hidden_size"),
            "model hidden_size",
        )
        if hidden_size % attention_heads != 0:
            raise ValueError(
                "model hidden_size must be divisible by num_attention_heads"
            )
        head_size_value = hidden_size // attention_heads
    head_size = _positive_int(head_size_value, "model head_dim")

    tensor_parallel_size = _positive_int(
        tensor_parallel_size,
        "benchmark.tensor_parallel_size",
    )
    pipeline_parallel_size = _positive_int(
        pipeline_parallel_size,
        "benchmark.pipeline_parallel_size",
    )
    if (
        total_kv_heads % tensor_parallel_size
        and tensor_parallel_size % total_kv_heads
    ):
        raise ValueError("KV heads and tensor parallel size are incompatible")
    kv_heads_per_gpu = max(1, total_kv_heads // tensor_parallel_size)
    layers_per_gpu = math.ceil(total_layers / pipeline_parallel_size)
    resolved_dtype, dtype_bytes = _resolve_cache_dtype(
        cache_config.get("cache_dtype", "auto"),
        model_config,
        text_config,
    )

    # This matches vLLM FullAttentionSpec.real_page_size_bytes:
    # K and V * tokens * local KV heads * head size * dtype bytes.
    bytes_per_token_per_layer_per_gpu = (
        2 * kv_heads_per_gpu * head_size * dtype_bytes
    )
    bytes_per_token_per_gpu = (
        layers_per_gpu * bytes_per_token_per_layer_per_gpu
    )
    pool_bytes = capacity_tokens * bytes_per_token_per_gpu
    return {
        "available": True,
        "calculation_method": "vllm_blocks_and_model_kv_geometry",
        "model_config_path": str(config_path),
        "num_gpu_blocks": num_gpu_blocks,
        "block_size_tokens": block_size,
        "capacity_tokens": capacity_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "total_hidden_layers": total_layers,
        "layers_per_gpu": layers_per_gpu,
        "total_kv_heads": total_kv_heads,
        "kv_heads_per_gpu": kv_heads_per_gpu,
        "head_size": head_size,
        "resolved_cache_dtype": resolved_dtype,
        "cache_dtype_bytes": dtype_bytes,
        "bytes_per_token_per_gpu": bytes_per_token_per_gpu,
        "pool_bytes_per_gpu": pool_bytes,
        "pool_gib_per_gpu": pool_bytes / 1024**3,
    }


class InferenceResourceMonitor:
    """Sample serving-GPU memory and vLLM metrics in a background thread."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        trace_path: Path | None,
        run_id: str,
    ):
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.sample_interval_seconds = max(
            float(config.get("sample_interval_seconds", 0.2)),
            0.05,
        )
        self.gpu_indices = _normalize_gpu_indices(config.get("gpu_indices"))
        metrics_url = config.get("vllm_metrics_url")
        self.vllm_metrics_url = (
            str(metrics_url).strip() if metrics_url else None
        )
        self.metrics_timeout_seconds = max(
            float(config.get("metrics_timeout_seconds", 1.0)),
            0.05,
        )
        self.trace_enabled = bool(config.get("save_resource_trace", True))
        self.trace_path = trace_path if self.trace_enabled else None
        self.run_id = run_id
        self.model_name = config.get("model_name")
        self.model_config_path = config.get("model_config_path")
        configured_tp = config.get("tensor_parallel_size")
        self.tensor_parallel_size = _positive_int(
            (
                configured_tp
                if configured_tp is not None
                else 1
            ),
            "benchmark.tensor_parallel_size",
        )
        self.pipeline_parallel_size = _positive_int(
            config.get("pipeline_parallel_size", 1),
            "benchmark.pipeline_parallel_size",
        )
        self.kv_cache_pool_gib_per_gpu_override = config.get(
            "kv_cache_pool_gib_per_gpu"
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._trace_handle = None
        self._pynvml: Any = None
        self._gpu_handles: dict[int, Any] = {}
        self._gpu_metadata: dict[str, dict[str, Any]] = {}
        self._sample_count = 0
        self._started_perf_counter: float | None = None
        self._baseline_gpu_memory_mib: dict[str, float] = {}
        self._peak_gpu_memory_mib: dict[str, float] = {}
        self._peak_kv_cache_usage: float | None = None
        self._vllm_cache_config: dict[str, str] = {}
        self._cache_config_by_engine: dict[str, dict[str, str]] = {}
        self._capacity_by_engine: dict[str, dict[str, Any]] = {}
        self._capacity_signatures: dict[str, str] = {}
        self._capacity_errors: dict[str, str] = {}
        self._peak_kv_usage_by_engine: dict[str, float] = {}
        self._kv_cache_capacity: dict[str, Any] | None = None
        self._kv_cache_capacity_error: str | None = None
        self._peak_requests_running: float | None = None
        self._peak_requests_waiting: float | None = None
        self._counter_start: dict[str, float] = {}
        self._counter_end: dict[str, float] = {}
        self._errors: list[str] = []
        self._error_set: set[str] = set()

    def _record_error(self, source: str, exc: BaseException) -> None:
        message = f"{source}: {type(exc).__name__}: {exc}"
        if message in self._error_set:
            return
        self._error_set.add(message)
        self._errors.append(message)
        logger.warning("Resource monitor %s", message)

    def _initialize_gpu_monitor(self) -> None:
        if not self.gpu_indices:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except Exception as exc:
            self._record_error("NVML initialization failed", exc)
            return

        for index in self.gpu_indices:
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(index)
                raw_name = self._pynvml.nvmlDeviceGetName(handle)
                name = (
                    raw_name.decode("utf-8", errors="replace")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._gpu_handles[index] = handle
                self._gpu_metadata[str(index)] = {
                    "name": name,
                    "total_memory_mib": float(memory.total / (1024**2)),
                }
            except Exception as exc:
                self._record_error(f"GPU {index} initialization failed", exc)

    def _sample_gpu_memory(self) -> dict[str, float]:
        result: dict[str, float] = {}
        if self._pynvml is None:
            return result
        for index, handle in self._gpu_handles.items():
            try:
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                result[str(index)] = float(memory.used / (1024**2))
            except Exception as exc:
                self._record_error(f"GPU {index} sampling failed", exc)
        return result

    def _sample_vllm_metrics(self) -> dict[str, Any]:
        if not self.vllm_metrics_url:
            return {}
        try:
            with urllib.request.urlopen(
                self.vllm_metrics_url,
                timeout=self.metrics_timeout_seconds,
            ) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            self._record_error("vLLM metrics sampling failed", exc)
            return {}

        parsed = {
            name: [
                (labels, value)
                for labels, value in samples
                if (
                    not self.model_name
                    or "model_name" not in labels
                    or labels["model_name"] == self.model_name
                )
            ]
            for name, samples in _parse_prometheus_samples(text).items()
        }

        def engine_id(labels: dict[str, str]) -> str:
            return labels.get("engine", labels.get("engine_id", "0"))

        for labels, _ in parsed[_VLLM_CACHE_CONFIG_METRIC]:
            self._cache_config_by_engine[engine_id(labels)] = labels
        usage_by_engine: dict[str, float] = {}
        for labels, value in parsed["vllm:kv_cache_usage_perc"]:
            engine = engine_id(labels)
            usage_by_engine[engine] = max(usage_by_engine.get(engine, 0.0), value)

        engines = set(self._cache_config_by_engine) | set(usage_by_engine)
        for engine in engines:
            cache_config = self._cache_config_by_engine.get(engine, {})
            override = self.kv_cache_pool_gib_per_gpu_override
            if isinstance(override, dict):
                override = override.get(
                    engine,
                    override.get(int(engine)) if engine.isdigit() else None,
                )
            signature = json.dumps([cache_config, override], sort_keys=True)
            if self._capacity_signatures.get(engine) == signature:
                continue
            self._capacity_signatures[engine] = signature
            try:
                self._capacity_by_engine[engine] = _calculate_kv_cache_capacity(
                    cache_config,
                    model_config_path=self.model_config_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    pool_gib_override=override,
                )
                self._capacity_errors.pop(engine, None)
            except Exception as exc:
                self._capacity_by_engine.pop(engine, None)
                self._capacity_errors[engine] = f"{type(exc).__name__}: {exc}"
                self._record_error(
                    f"KV-cache capacity calculation failed for engine {engine}",
                    exc,
                )

        self._vllm_cache_config = next(
            iter(self._cache_config_by_engine.values()), {}
        )
        capacities = list(self._capacity_by_engine.values())
        self._kv_cache_capacity = (
            capacities[0]
            if (
                capacities and not self._capacity_errors and all(
                    capacity["pool_bytes_per_gpu"] == capacities[0]["pool_bytes_per_gpu"]
                    for capacity in capacities
                )
            )
            else None
        )
        self._kv_cache_capacity_error = (
            "; ".join(
                f"engine {key}: {value}"
                for key, value in sorted(self._capacity_errors.items())
            )
            or None
        )

        result: dict[str, Any] = {}
        for metric_name in _VLLM_METRICS:
            metric_values = [value for _, value in parsed[metric_name]]
            if not metric_values:
                continue
            short_name = metric_name.removeprefix("vllm:")
            if metric_name == "vllm:kv_cache_usage_perc":
                result[short_name] = max(metric_values)
            else:
                result[short_name] = sum(metric_values)
        if usage_by_engine:
            result["kv_cache_usage_perc_by_engine"] = usage_by_engine
        return result

    def _write_trace(self, sample: dict[str, Any]) -> None:
        if self._trace_handle is None:
            return
        try:
            self._trace_handle.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )
            self._trace_handle.flush()
        except Exception as exc:
            self._record_error("resource trace write failed", exc)
            try:
                self._trace_handle.close()
            except Exception:
                pass
            self._trace_handle = None

    def _sample(self) -> dict[str, Any]:
        elapsed_seconds = (
            time.perf_counter() - self._started_perf_counter
            if self._started_perf_counter is not None
            else 0.0
        )
        gpu_memory_mib = self._sample_gpu_memory()
        vllm_metrics = self._sample_vllm_metrics()
        sample = {
            "run_id": self.run_id,
            "timestamp": _utc_now(),
            "elapsed_seconds": float(elapsed_seconds),
            "gpu_memory_mib": gpu_memory_mib,
            "vllm": vllm_metrics,
        }
        self._sample_count += 1

        if not self._baseline_gpu_memory_mib and gpu_memory_mib:
            self._baseline_gpu_memory_mib = dict(gpu_memory_mib)
        for index, value in gpu_memory_mib.items():
            self._baseline_gpu_memory_mib.setdefault(index, value)
            self._peak_gpu_memory_mib[index] = max(
                value,
                self._peak_gpu_memory_mib.get(index, value),
            )

        current_kv_usage = vllm_metrics.get("kv_cache_usage_perc")
        if current_kv_usage is not None:
            self._peak_kv_cache_usage = max(
                self._peak_kv_cache_usage or 0.0,
                float(current_kv_usage),
            )
        for engine, usage in vllm_metrics.get(
            "kv_cache_usage_perc_by_engine", {}
        ).items():
            self._peak_kv_usage_by_engine[engine] = max(
                self._peak_kv_usage_by_engine.get(engine, 0.0), usage
            )
        for name, attribute in (
            ("num_requests_running", "_peak_requests_running"),
            ("num_requests_waiting", "_peak_requests_waiting"),
        ):
            if name in vllm_metrics:
                previous = getattr(self, attribute)
                setattr(self, attribute, max(previous or 0.0, vllm_metrics[name]))
        counter_values = {
            name: float(vllm_metrics[name])
            for name in (
                "prompt_tokens_total",
                "generation_tokens_total",
                "request_success_total",
            )
            if name in vllm_metrics
        }
        for name, value in counter_values.items():
            self._counter_start.setdefault(name, value)
            self._counter_end[name] = value

        self._write_trace(sample)
        return sample

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            try:
                self._sample()
            except Exception as exc:
                self._record_error("background sampling failed", exc)

    def start(self) -> None:
        if not self.enabled:
            return
        self._initialize_gpu_monitor()
        if self.trace_path is not None:
            try:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._trace_handle = self.trace_path.open(
                    "a",
                    encoding="utf-8",
                )
            except Exception as exc:
                self._record_error("resource trace open failed", exc)
                self._trace_handle = None
        self._started_perf_counter = time.perf_counter()
        try:
            self._sample()
        except Exception as exc:
            self._record_error("initial resource sampling failed", exc)
        self._thread = threading.Thread(
            target=self._run,
            name="trajectory-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "sample_count": 0,
                "monitor_errors": [],
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(
                timeout=(
                    self.sample_interval_seconds
                    + self.metrics_timeout_seconds
                    + 2.0
                )
            )
        try:
            self._sample()
        except Exception as exc:
            self._record_error("final resource sampling failed", exc)
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as exc:
                self._record_error("NVML shutdown failed", exc)

        incremental_peak_mib = {
            index: max(
                self._peak_gpu_memory_mib[index]
                - self._baseline_gpu_memory_mib.get(index, 0.0),
                0.0,
            )
            for index in self._peak_gpu_memory_mib
        }
        counter_delta = {
            name: max(
                self._counter_end.get(name, start_value) - start_value,
                0.0,
            )
            for name, start_value in self._counter_start.items()
        }
        kv_cache_pool_gib = (
            self._kv_cache_capacity.get("pool_gib_per_gpu")
            if self._kv_cache_capacity is not None
            else None
        )
        known_engines = (
            set(self._cache_config_by_engine) | set(self._peak_kv_usage_by_engine)
        )
        active_kv_complete = bool(known_engines) and all(
            engine in self._capacity_by_engine
            and engine in self._peak_kv_usage_by_engine
            for engine in known_engines
        )
        active_kv_by_engine = {
            engine: capacity["pool_gib_per_gpu"] * self._peak_kv_usage_by_engine[engine]
            for engine, capacity in self._capacity_by_engine.items()
            if engine in self._peak_kv_usage_by_engine
        }
        peak_active_engine = (
            max(active_kv_by_engine, key=active_kv_by_engine.get)
            if active_kv_complete else None
        )
        peak_active_kv_cache_gib = (
            active_kv_by_engine[peak_active_engine]
            if peak_active_engine is not None else None
        )
        return {
            "enabled": True,
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": self._sample_count,
            "gpu_indices": list(self.gpu_indices),
            "gpu_metadata": self._gpu_metadata,
            "gpu_idle_memory_mib": self._baseline_gpu_memory_mib,
            "gpu_peak_memory_mib": self._peak_gpu_memory_mib,
            "gpu_incremental_peak_memory_mib": incremental_peak_mib,
            "idle_gpu_memory_per_gpu_gib": (
                max(self._baseline_gpu_memory_mib.values()) / 1024.0
                if self._baseline_gpu_memory_mib
                else None
            ),
            "peak_gpu_memory_per_gpu_mib": (
                max(self._peak_gpu_memory_mib.values())
                if self._peak_gpu_memory_mib
                else None
            ),
            "peak_gpu_memory_per_gpu_gib": (
                max(self._peak_gpu_memory_mib.values()) / 1024.0
                if self._peak_gpu_memory_mib
                else None
            ),
            "incremental_peak_gpu_memory_per_gpu_gib": (
                max(incremental_peak_mib.values()) / 1024.0
                if incremental_peak_mib
                else None
            ),
            "vllm_metrics_url": self.vllm_metrics_url,
            "vllm_cache_config": self._vllm_cache_config,
            "vllm_cache_config_by_engine": self._cache_config_by_engine,
            "kv_cache_capacity": self._kv_cache_capacity,
            "kv_cache_capacity_by_engine": self._capacity_by_engine,
            "mixed_kv_cache_pool_capacities": (
                len({
                    capacity["pool_bytes_per_gpu"]
                    for capacity in self._capacity_by_engine.values()
                }) > 1
            ),
            "kv_cache_capacity_error": self._kv_cache_capacity_error,
            "kv_cache_pool_gib_per_gpu": kv_cache_pool_gib,
            "peak_kv_cache_usage": self._peak_kv_cache_usage,
            "peak_kv_cache_usage_percent": (
                self._peak_kv_cache_usage * 100.0
                if self._peak_kv_cache_usage is not None
                else None
            ),
            "peak_active_kv_cache_memory_per_gpu_gib": (
                peak_active_kv_cache_gib
            ),
            "active_kv_metrics_complete": active_kv_complete,
            "peak_active_kv_engine": peak_active_engine,
            "peak_active_kv_cache_usage_percent": (
                self._peak_kv_usage_by_engine[peak_active_engine] * 100.0
                if peak_active_engine is not None else None
            ),
            "peak_kv_cache_usage_by_engine": self._peak_kv_usage_by_engine,
            "peak_requests_running": self._peak_requests_running,
            "peak_requests_waiting": self._peak_requests_waiting,
            "vllm_counter_start": self._counter_start,
            "vllm_counter_end": self._counter_end,
            "vllm_counter_delta": counter_delta,
            "resource_trace_path": (
                str(self.trace_path) if self.trace_path is not None else None
            ),
            "monitor_errors": list(self._errors),
        }
