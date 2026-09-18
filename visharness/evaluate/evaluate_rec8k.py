"""Evaluate REC-8K counting and point-localization predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from visharness.evaluate.common import (
    CompactPrediction,
    load_compact_predictions,
    load_exif_transposed_image_size,
    load_manifest,
    prediction_status_summary,
    resolve_image_path,
    restore_boxes_to_original,
    summarize_efficiency_metrics,
    validate_final_result_consistency,
    write_json,
    write_jsonl,
)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rec8k_item_id(image_name: str, phrase: str) -> str:
    return f"{Path(image_name).stem}-{'_'.join(phrase.split())}"


def _build_split_lookup(
    split_items: Any,
) -> dict[str, tuple[str, str]]:
    if not isinstance(split_items, list):
        raise ValueError("REC-8K split must be a list")
    lookup: dict[str, tuple[str, str]] = {}
    for entry in split_items:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"Invalid REC-8K split entry: {entry!r}")
        image_name, phrase = str(entry[0]), str(entry[1])
        item_id = _rec8k_item_id(image_name, phrase)
        if item_id in lookup:
            raise ValueError(f"Duplicate REC-8K split id: {item_id}")
        lookup[item_id] = (image_name, phrase)
    return lookup


def _restore_prediction_boxes(
    prediction: CompactPrediction | None,
    *,
    original_width: int,
    original_height: int,
    aspect_ratio_tolerance: float,
) -> tuple[list[list[float]], str]:
    if prediction is None:
        return [], "missing"
    if not prediction.has_visual_result:
        return (
            [],
            (
                "explicit_empty"
                if prediction.explicit_empty_prediction
                else prediction.termination_reason
            ),
        )
    validate_final_result_consistency(prediction)
    boxes = restore_boxes_to_original(
        prediction.final_bboxes,
        prediction.final_masks,
        original_width=original_width,
        original_height=original_height,
        aspect_ratio_tolerance=aspect_ratio_tolerance,
    )
    return boxes, "visual_result"


def _localization_matches(
    pred_boxes: Sequence[Sequence[float]],
    gt_points: Sequence[Sequence[float]],
) -> int:
    """Preserve the legacy REC-8K Hungarian/sigma matching rule."""

    if not pred_boxes or not gt_points:
        return 0

    pred_points: list[list[float]] = []
    widths: list[float] = []
    heights: list[float] = []
    for box in pred_boxes:
        x1, y1, x2, y2 = map(float, box)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            raise ValueError(f"Prediction box has non-positive area: {box}")
        pred_points.append([x1 + width / 2.0, y1 + height / 2.0])
        widths.append(width)
        heights.append(height)

    gt_array = np.asarray(gt_points, dtype=np.float64)
    if gt_array.ndim != 2 or gt_array.shape[1] != 2 or not np.all(np.isfinite(gt_array)):
        raise ValueError(f"REC-8K GT points must have shape [N, 2], got {gt_array.shape}")

    areas = np.asarray(widths) * np.asarray(heights)
    sorted_indices = np.argsort(areas)
    count = len(pred_boxes)
    if count % 2 == 1:
        median_index = sorted_indices[count // 2]
        sigma = np.hypot(widths[median_index], heights[median_index]) / 2.0
    else:
        first = sorted_indices[count // 2 - 1]
        second = sorted_indices[count // 2]
        sigma = (
            np.hypot(widths[first], heights[first])
            + np.hypot(widths[second], heights[second])
        ) / 4.0

    distance_matrix = cdist(
        np.asarray(pred_points, dtype=np.float64),
        gt_array,
        metric="euclidean",
    )
    row_indices, column_indices = linear_sum_assignment(distance_matrix)
    return sum(
        distance_matrix[row_index, column_index] <= sigma
        for row_index, column_index in zip(row_indices, column_indices)
    )


def evaluate_rec8k(
    prediction_paths: str | Path | Sequence[str | Path],
    manifest_path: str | Path,
    dataset_root: str | Path,
    split: str,
    *,
    metrics_output: str | Path | None = None,
    per_sample_output: str | Path | None = None,
    aspect_ratio_tolerance: float = 0.01,
    strict_jsonl: bool = True,
    kv_cache_pool_gib_per_gpu_override: float | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    annotations = _load_json_object(dataset_root / "annotations.json")
    splits = _load_json_object(dataset_root / "splits.json")
    if split not in splits:
        raise ValueError(f"Unknown REC-8K split {split!r}; available={sorted(splits)}")
    split_lookup = _build_split_lookup(splits[split])

    manifest = load_manifest(manifest_path)
    predictions, load_report = load_compact_predictions(
        prediction_paths,
        strict=strict_jsonl,
    )
    expected_ids = [str(item["id"]) for item in manifest]

    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    global_tp = 0
    global_fp = 0
    global_fn = 0
    per_sample: list[dict[str, Any]] = []

    for manifest_item in manifest:
        item_id = str(manifest_item["id"])
        lookup_id = item_id.removeprefix("REC8K-")
        if lookup_id not in split_lookup:
            raise ValueError(
                f"Manifest item {item_id} does not belong to REC-8K split {split!r}"
            )
        image_name, phrase = split_lookup[lookup_id]
        if image_name not in annotations or phrase not in annotations[image_name]:
            raise KeyError(f"Missing REC-8K annotation for {image_name!r}, {phrase!r}")
        gt_points = annotations[image_name][phrase].get("points", [])
        if not isinstance(gt_points, list):
            raise ValueError(f"REC-8K points are not a list for {item_id}")

        image_path = resolve_image_path(
            manifest_item,
            dataset_root=dataset_root,
            manifest_path=manifest_path,
        )
        original_width, original_height = load_exif_transposed_image_size(image_path)
        prediction = predictions.get(item_id)
        pred_boxes, prediction_kind = _restore_prediction_boxes(
            prediction,
            original_width=original_width,
            original_height=original_height,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        )

        gt_count = len(gt_points)
        pred_count = len(pred_boxes)
        absolute_error = abs(pred_count - gt_count)
        squared_error = (pred_count - gt_count) ** 2
        true_positives = int(_localization_matches(pred_boxes, gt_points))
        false_positives = pred_count - true_positives
        false_negatives = gt_count - true_positives

        absolute_errors.append(float(absolute_error))
        squared_errors.append(float(squared_error))
        global_tp += true_positives
        global_fp += false_positives
        global_fn += false_negatives
        per_sample.append(
            {
                "id": item_id,
                "termination_reason": (
                    prediction.termination_reason if prediction is not None else "missing"
                ),
                "prediction_kind": prediction_kind,
                "has_visual_result": bool(prediction and prediction.has_visual_result),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
            }
        )

    sample_count = len(manifest)
    mae = float(np.mean(absolute_errors)) if absolute_errors else 0.0
    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
    precision = global_tp / (global_tp + global_fp) if global_tp + global_fp else 0.0
    recall = global_tp / (global_tp + global_fn) if global_tp + global_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "task": "REC8K",
        "split": split,
        "sample_count": sample_count,
        "MAE": mae,
        "RMSE": rmse,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "TP": global_tp,
        "FP": global_fp,
        "FN": global_fn,
        "status": prediction_status_summary(expected_ids, predictions, load_report),
        "efficiency": summarize_efficiency_metrics(
            expected_ids,
            predictions,
            prediction_paths,
            kv_cache_pool_gib_per_gpu_override=(
                kv_cache_pool_gib_per_gpu_override
            ),
        ),
    }
    write_json(metrics_output, metrics)
    write_jsonl(per_sample_output, per_sample)
    return metrics


def evaluate_rec8k_from_gt(
    jsonl_path: str | Path,
    dataset_root: str | Path,
    split: str = "val",
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if manifest_path is None:
        manifest_path = Path(dataset_root) / f"REC8K_QA_{split}.json"
    return evaluate_rec8k(jsonl_path, manifest_path, dataset_root, split)


def _format_table_metric(
    value: Any,
    *,
    precision: int = 2,
    unit: str = "",
) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(numeric_value):
        return "N/A"
    formatted = f"{numeric_value:.{precision}f}"
    return f"{formatted} {unit}".rstrip()


def print_rec8k_summary_table(metrics: dict[str, Any]) -> None:
    """Print the compact task/efficiency table used for benchmark comparison."""

    efficiency = metrics.get("efficiency")
    if not isinstance(efficiency, dict):
        efficiency = {}
    per_trajectory = efficiency.get("per_trajectory")
    if not isinstance(per_trajectory, dict):
        per_trajectory = {}
    run_level = efficiency.get("run_level")
    if not isinstance(run_level, dict):
        run_level = {}

    peak_active_kv_memory = _format_table_metric(
        run_level.get(
            "peak_active_kv_cache_memory_per_gpu_gib"
        ),
        precision=2,
        unit="GiB/GPU",
    )
    peak_kv_usage_percent = run_level.get(
        "peak_kv_cache_usage_percent"
    )
    if peak_kv_usage_percent is None:
        peak_kv_usage = run_level.get("peak_kv_cache_usage")
        if peak_kv_usage is not None:
            try:
                peak_kv_usage_percent = float(peak_kv_usage) * 100.0
            except (TypeError, ValueError):
                peak_kv_usage_percent = None
    formatted_peak_kv_usage = _format_table_metric(
        peak_kv_usage_percent,
        precision=2,
    )
    if formatted_peak_kv_usage != "N/A":
        formatted_peak_kv_usage = f"{formatted_peak_kv_usage}%"
    if (
        peak_active_kv_memory != "N/A"
        and formatted_peak_kv_usage != "N/A"
    ):
        peak_active_kv_memory = (
            f"{peak_active_kv_memory} ({formatted_peak_kv_usage})"
        )

    headers = (
        "MAE",
        "RMSE",
        "Avg. Visual Tokens/Trajectory",
        "Avg. Latency",
        "Peak active KV memory",
        "rollout throughput",
    )
    values = (
        _format_table_metric(metrics.get("MAE"), precision=4),
        _format_table_metric(metrics.get("RMSE"), precision=4),
        _format_table_metric(
            per_trajectory.get("average_cumulative_visual_tokens"),
            precision=2,
        ),
        _format_table_metric(
            per_trajectory.get("average_trajectory_elapsed_seconds"),
            precision=2,
            unit="s/trajectory",
        ),
        peak_active_kv_memory,
        _format_table_metric(
            run_level.get(
                "rollout_throughput_trajectories_per_minute"
            ),
            precision=2,
            unit="trajectories/min",
        ),
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    print("| " + " | ".join(values) + " |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--aspect-ratio-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--kv-cache-pool-gib-per-gpu",
        type=float,
        default=None,
        help=(
            "KV-cache pool capacity per GPU for legacy benchmark summaries "
            "that predate automatic capacity recording."
        ),
    )
    parser.add_argument("--allow-malformed-lines", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_rec8k(
        args.predictions,
        args.manifest,
        args.dataset_root,
        args.split,
        metrics_output=args.metrics_output,
        per_sample_output=args.per_sample_output,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        strict_jsonl=not args.allow_malformed_lines,
        kv_cache_pool_gib_per_gpu_override=(
            args.kv_cache_pool_gib_per_gpu
        ),
    )
    print_rec8k_summary_table(metrics)


if __name__ == "__main__":
    main()
