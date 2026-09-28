"""Validation aggregation for checkpoint selection.

Mask/count metrics use sufficient statistics emitted by ``vision_reward`` and
therefore match the standalone evaluators instead of averaging per-batch
approximations.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TASK_NAMES = {
    "gres_rle_mask": "GRES",
    "reasonseg_rle_mask": "ReasonSeg",
    "point": "REC8K",
}


def _at(values: dict[str, Sequence[Any]], key: str, index: int, default: Any = None) -> Any:
    sequence = values.get(key, ())
    return sequence[index] if index < len(sequence) else default


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [_float(record.get(key)) for record in records]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def _add_distribution(
    metrics: dict[str, float],
    prefix: str,
    records: list[dict[str, Any]],
    key: str,
) -> None:
    values = _finite_array(records, key)
    if not values.size:
        return
    metrics[f"{prefix}/mean"] = float(values.mean())
    metrics[f"{prefix}/p50"] = float(np.quantile(values, 0.5))
    metrics[f"{prefix}/p90"] = float(np.quantile(values, 0.9))
    metrics[f"{prefix}/max"] = float(values.max())


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = _finite_array(records, key)
    return float(values.mean()) if values.size else 0.0


def _safe_reason(value: Any) -> str:
    if value is None or str(value).strip() in {"", "None"}:
        return "none"
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in str(value))


def build_validation_records(
    data_sources: Sequence[Any],
    sample_uids: Sequence[Any],
    reward_extra_infos: dict[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    sample_count = max(
        len(sample_uids),
        len(data_sources),
        max((len(values) for values in reward_extra_infos.values()), default=0),
    )
    records: list[dict[str, Any]] = []
    for index in range(sample_count):
        task_type = str(_at(reward_extra_infos, "task_type", index, "unknown"))
        fallback_reward = _float(_at(reward_extra_infos, "reward", index), 0.0)
        task_reward = _float(_at(reward_extra_infos, "task_reward", index))
        if not math.isfinite(task_reward):
            task_reward = fallback_reward
        trajectory_reward = _float(_at(reward_extra_infos, "trajectory_reward", index))
        if not math.isfinite(trajectory_reward):
            trajectory_reward = fallback_reward
        assistant_turns = _float(
            _at(reward_extra_infos, "num_assistant_turns", index)
        )
        truncated_turn_count = _float(
            _at(reward_extra_infos, "truncated_turn_count", index)
        )
        stored_output_format_error_count = _float(
            _at(reward_extra_infos, "output_format_error_count", index)
        )
        stored_tool_args_error_count = _float(
            _at(reward_extra_infos, "tool_args_error_count", index)
        )
        raw_output_format_failure_count = _float(
            _at(reward_extra_infos, "raw_output_format_failure_count", index)
        )
        raw_tool_args_failure_count = _float(
            _at(reward_extra_infos, "raw_tool_args_failure_count", index)
        )
        has_exclusive_error_schema = math.isfinite(
            raw_output_format_failure_count
        ) or math.isfinite(raw_tool_args_failure_count)
        if has_exclusive_error_schema:
            output_format_error_count = stored_output_format_error_count
            tool_args_error_count = stored_tool_args_error_count
        else:
            # Before raw failure fields were introduced, the format-error count
            # also included truncations. Keep it only as a raw diagnostic; do
            # not silently relabel historical data as an exclusive error count.
            output_format_error_count = float("nan")
            tool_args_error_count = float("nan")
        if not math.isfinite(raw_output_format_failure_count):
            raw_output_format_failure_count = stored_output_format_error_count
        if not math.isfinite(raw_tool_args_failure_count):
            raw_tool_args_failure_count = stored_tool_args_error_count

        output_format_success_rate = _float(
            _at(reward_extra_infos, "output_format_success_rate", index)
        )
        tool_args_success_rate = _float(
            _at(reward_extra_infos, "tool_args_success_rate", index)
        )
        output_format_error_rate = _float(
            _at(reward_extra_infos, "output_format_error_rate", index)
        )
        tool_args_error_rate = _float(
            _at(reward_extra_infos, "tool_args_error_rate", index)
        )
        raw_output_format_failure_rate = _float(
            _at(reward_extra_infos, "raw_output_format_failure_rate", index)
        )
        raw_tool_args_failure_rate = _float(
            _at(reward_extra_infos, "raw_tool_args_failure_rate", index)
        )
        truncated_turn_rate = _float(
            _at(reward_extra_infos, "truncated_turn_rate", index)
        )
        if math.isfinite(assistant_turns) and assistant_turns > 0:
            if has_exclusive_error_schema and not math.isfinite(output_format_error_rate):
                output_format_error_rate = output_format_error_count / assistant_turns
            if has_exclusive_error_schema and not math.isfinite(tool_args_error_rate):
                tool_args_error_rate = tool_args_error_count / assistant_turns
            if not math.isfinite(raw_output_format_failure_rate):
                raw_output_format_failure_rate = (
                    raw_output_format_failure_count / assistant_turns
                )
            if not math.isfinite(raw_tool_args_failure_rate):
                raw_tool_args_failure_rate = raw_tool_args_failure_count / assistant_turns
            if not math.isfinite(truncated_turn_rate):
                truncated_turn_rate = truncated_turn_count / assistant_turns

        exclusive_hard_error_count = float("nan")
        if all(
            math.isfinite(value)
            for value in (
                output_format_error_count,
                tool_args_error_count,
                truncated_turn_count,
            )
        ):
            exclusive_hard_error_count = (
                output_format_error_count
                + tool_args_error_count
                + truncated_turn_count
            )
        record = {
            "uid": str(sample_uids[index]) if index < len(sample_uids) else str(index),
            "data_source": str(data_sources[index]) if index < len(data_sources) else "unknown",
            "task": TASK_NAMES.get(task_type, task_type),
            "task_type": task_type,
            "task_reward": task_reward,
            "trajectory_reward": trajectory_reward,
            "trajectory_step_cost": _float(_at(reward_extra_infos, "trajectory_step_cost", index)),
            "eval_iou": _float(_at(reward_extra_infos, "eval_iou", index)),
            "eval_reward": _float(_at(reward_extra_infos, "eval_reward", index)),
            "eval_intersection": _float(_at(reward_extra_infos, "eval_intersection", index)),
            "eval_union": _float(_at(reward_extra_infos, "eval_union", index)),
            "eval_ground_truth_empty": _bool(
                _at(reward_extra_infos, "eval_ground_truth_empty", index, False)
            ),
            "eval_empty_correct": _bool(_at(reward_extra_infos, "eval_empty_correct", index, False)),
            "eval_gt_count": _float(_at(reward_extra_infos, "eval_gt_count", index)),
            "eval_pred_count": _float(_at(reward_extra_infos, "eval_pred_count", index)),
            "eval_absolute_error": _float(_at(reward_extra_infos, "eval_absolute_error", index)),
            "eval_squared_error": _float(_at(reward_extra_infos, "eval_squared_error", index)),
            "eval_exact_count": _float(_at(reward_extra_infos, "eval_exact_count", index)),
            "eval_tp": _float(_at(reward_extra_infos, "eval_tp", index)),
            "eval_fp": _float(_at(reward_extra_infos, "eval_fp", index)),
            "eval_fn": _float(_at(reward_extra_infos, "eval_fn", index)),
            "assistant_output_tokens": _float(
                _at(reward_extra_infos, "assistant_response_token_count", index)
            ),
            "observation_tokens": _float(_at(reward_extra_infos, "observation_token_count", index)),
            "trajectory_response_tokens": _float(
                _at(reward_extra_infos, "trajectory_response_token_count", index)
            ),
            "assistant_turns": assistant_turns,
            "tool_calls": _float(_at(reward_extra_infos, "tool_call_count", index)),
            "generate_seconds": _float(_at(reward_extra_infos, "generate_time_seconds", index)),
            "tool_seconds": _float(_at(reward_extra_infos, "tool_time_seconds", index)),
            "trajectory_seconds": _float(
                _at(reward_extra_infos, "trajectory_elapsed_seconds", index)
            ),
            "trajectory_finished": _bool(
                _at(reward_extra_infos, "trajectory_finished", index, False)
            ),
            "trajectory_invalid": _bool(
                _at(reward_extra_infos, "trajectory_invalid", index, False)
            ),
            "trajectory_aborted": _bool(
                _at(reward_extra_infos, "trajectory_aborted", index, False)
            ),
            "max_agent_turns_reached": _bool(
                _at(reward_extra_infos, "max_agent_turns_reached", index, False)
            ),
            "invalid_reason": _at(reward_extra_infos, "invalid_reason", index),
            "output_format_error_count": output_format_error_count,
            "raw_output_format_failure_count": raw_output_format_failure_count,
            "tool_args_error_count": tool_args_error_count,
            "raw_tool_args_failure_count": raw_tool_args_failure_count,
            "truncated_turn_count": truncated_turn_count,
            "exclusive_hard_error_count": exclusive_hard_error_count,
            "output_format_success_rate": output_format_success_rate,
            "tool_args_success_rate": tool_args_success_rate,
            "output_format_error_rate": output_format_error_rate,
            "tool_args_error_rate": tool_args_error_rate,
            "raw_output_format_failure_rate": raw_output_format_failure_rate,
            "raw_tool_args_failure_rate": raw_tool_args_failure_rate,
            "truncated_turn_rate": truncated_turn_rate,
        }
        records.append(record)
    return records


def _add_common_metrics(
    metrics: dict[str, float],
    records: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    core_prefix = f"val-core/{label}"
    aux_prefix = f"val-aux/{label}"

    def add_trajectory_incidence(metric_name: str, count_key: str) -> None:
        counts = _finite_array(records, count_key)
        if counts.size:
            metrics[f"{aux_prefix}/{metric_name}"] = float(np.mean(counts > 0))

    def add_micro_turn_rate(metric_name: str, count_key: str) -> None:
        numerator = 0.0
        denominator = 0.0
        for record in records:
            count = _float(record.get(count_key))
            turn_count = _float(record.get("assistant_turns"))
            if not math.isfinite(count) or not math.isfinite(turn_count) or turn_count <= 0:
                continue
            numerator += count
            denominator += turn_count
        if denominator > 0:
            metrics[f"{aux_prefix}/{metric_name}"] = numerator / denominator

    metrics[f"{core_prefix}/samples"] = float(len(records))
    metrics[f"{core_prefix}/task_reward_mean"] = _mean(records, "task_reward")
    metrics[f"{core_prefix}/trajectory_reward_mean"] = _mean(records, "trajectory_reward")
    eval_rewards = _finite_array(records, "eval_reward")
    if eval_rewards.size:
        metrics[f"{core_prefix}/evaluation_reward_mean"] = float(eval_rewards.mean())
    metrics[f"{aux_prefix}/finished_rate"] = float(
        np.mean([record["trajectory_finished"] for record in records])
    )
    metrics[f"{aux_prefix}/invalid_rate"] = float(
        np.mean([record["trajectory_invalid"] for record in records])
    )
    metrics[f"{aux_prefix}/aborted_rate"] = float(
        np.mean([record["trajectory_aborted"] for record in records])
    )
    metrics[f"{aux_prefix}/max_turns_reached_rate"] = float(
        np.mean([record["max_agent_turns_reached"] for record in records])
    )
    for metric_name, count_key in (
        ("trajectories_with_format_error_rate", "output_format_error_count"),
        ("trajectories_with_raw_format_failure_rate", "raw_output_format_failure_count"),
        ("trajectories_with_args_error_rate", "tool_args_error_count"),
        ("trajectories_with_raw_args_failure_rate", "raw_tool_args_failure_count"),
        ("trajectories_with_truncation_rate", "truncated_turn_count"),
    ):
        add_trajectory_incidence(metric_name, count_key)
    metrics[f"{aux_prefix}/output_format_success_rate"] = _mean(
        records, "output_format_success_rate"
    )
    metrics[f"{aux_prefix}/tool_args_success_rate"] = _mean(records, "tool_args_success_rate")
    for metric_name, count_key in (
        ("turn_output_format_error_rate", "output_format_error_count"),
        ("turn_tool_args_error_rate", "tool_args_error_count"),
        ("raw_output_format_failure_rate", "raw_output_format_failure_count"),
        ("raw_tool_args_failure_rate", "raw_tool_args_failure_count"),
        ("turn_truncation_rate", "truncated_turn_count"),
        ("turn_hard_error_rate", "exclusive_hard_error_count"),
    ):
        add_micro_turn_rate(metric_name, count_key)

    distributions = {
        "assistant_output_tokens": "assistant_output_tokens",
        "trajectory_response_tokens": "trajectory_response_tokens",
        "assistant_turns": "assistant_turns",
        "tool_calls": "tool_calls",
        "generate_seconds": "generate_seconds",
        "tool_seconds": "tool_seconds",
        "trajectory_seconds": "trajectory_seconds",
    }
    for metric_name, record_key in distributions.items():
        _add_distribution(metrics, f"{aux_prefix}/{metric_name}", records, record_key)

    reason_counts = Counter(_safe_reason(record["invalid_reason"]) for record in records)
    for reason, count in reason_counts.items():
        if reason != "none":
            metrics[f"{aux_prefix}/invalid_reason/{reason}_rate"] = count / len(records)


def summarize_validation(
    data_sources: Sequence[Any],
    sample_uids: Sequence[Any],
    reward_extra_infos: dict[str, Sequence[Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records = build_validation_records(data_sources, sample_uids, reward_extra_infos)
    if not records:
        return {"val-core/samples": 0.0}, []

    metrics: dict[str, float] = {
        "val-core/samples": float(len(records)),
        "val-core/task_reward_mean": _mean(records, "task_reward"),
        "val-core/trajectory_reward_mean": _mean(records, "trajectory_reward"),
    }
    eval_rewards = _finite_array(records, "eval_reward")
    if eval_rewards.size:
        metrics["val-core/evaluation_reward_mean"] = float(eval_rewards.mean())
    _add_common_metrics(metrics, records, label="overall")

    by_task: dict[str, list[dict[str, Any]]] = {}
    for task in TASK_NAMES.values():
        task_records = [record for record in records if record["task"] == task]
        if task_records:
            by_task[task] = task_records
            _add_common_metrics(metrics, task_records, label=task)

    for task in ("GRES", "ReasonSeg"):
        task_records = by_task.get(task, [])
        if not task_records:
            continue
        ious = _finite_array(task_records, "eval_iou")
        intersections = _finite_array(task_records, "eval_intersection")
        unions = _finite_array(task_records, "eval_union")
        total_intersection = float(intersections.sum())
        total_union = float(unions.sum())
        metrics[f"val-core/{task}/gIoU"] = float(ious.mean()) if ious.size else 0.0
        metrics[f"val-core/{task}/cIoU"] = (
            total_intersection / total_union if total_union else 0.0
        )
        metrics[f"val-aux/{task}/total_intersection"] = total_intersection
        metrics[f"val-aux/{task}/total_union"] = total_union
        empty_records = [record for record in task_records if record["eval_ground_truth_empty"]]
        if task == "GRES" or empty_records:
            metrics[f"val-core/{task}/N_acc"] = (
                float(np.mean([record["eval_empty_correct"] for record in empty_records]))
                if empty_records
                else 0.0
            )
            metrics[f"val-aux/{task}/empty_ground_truth_samples"] = float(len(empty_records))

    rec_records = by_task.get("REC8K", [])
    if rec_records:
        squared_errors = _finite_array(rec_records, "eval_squared_error")
        tp = float(_finite_array(rec_records, "eval_tp").sum())
        fp = float(_finite_array(rec_records, "eval_fp").sum())
        fn = float(_finite_array(rec_records, "eval_fn").sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics["val-core/REC8K/MAE"] = _mean(rec_records, "eval_absolute_error")
        metrics["val-core/REC8K/RMSE"] = (
            float(np.sqrt(squared_errors.mean())) if squared_errors.size else 0.0
        )
        metrics["val-core/REC8K/precision"] = precision
        metrics["val-core/REC8K/recall"] = recall
        metrics["val-core/REC8K/F1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        metrics["val-core/REC8K/exact_count_accuracy"] = _mean(
            rec_records, "eval_exact_count"
        )
        metrics["val-aux/REC8K/TP"] = tp
        metrics["val-aux/REC8K/FP"] = fp
        metrics["val-aux/REC8K/FN"] = fn

    return metrics, records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_validation_artifacts(
    output_root: str | Path,
    *,
    global_step: int,
    metrics: dict[str, float],
    records: list[dict[str, Any]],
    generation_args: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_root = Path(output_root)
    step_dir = output_root / f"global_step_{int(global_step)}"
    summary = {
        "schema_version": 1,
        "global_step": int(global_step),
        "sample_count": len(records),
        "generation_args": generation_args,
        "metadata": metadata or {},
        "metrics": metrics,
    }
    summary_text = json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n"
    samples_text = "".join(
        json.dumps(
            _json_safe({"global_step": int(global_step), **record}),
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )
    _atomic_write(step_dir / "metrics.json", summary_text)
    _atomic_write(step_dir / "samples.jsonl", samples_text)
    _atomic_write(output_root / "latest_metrics.json", summary_text)
    return step_dir
