"""Evaluate Dense200 boxes using Rex-Omni's published matching/averaging rules.

Match in GT order, average precision/recall across QA samples, then compute
F1 at each IoU threshold. The aggregate is the mean of these F1 scores, not
the harmonic mean of threshold-averaged precision/recall.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from visharness.evaluate.common import (
    CompactPrediction,
    load_compact_predictions,
    load_exif_transposed_image_size,
    load_manifest,
    load_records,
    prediction_status_summary,
    resolve_image_path,
    restore_boxes_to_original,
    summarize_efficiency_metrics,
    validate_final_result_consistency,
    write_json,
    write_jsonl,
)


def _validate_boxes(boxes: Any, *, context: str) -> list[list[float]]:
    if not isinstance(boxes, list):
        raise ValueError(f"{context} boxes must be a list")
    validated: list[list[float]] = []
    for index, box in enumerate(boxes):
        array = np.asarray(box, dtype=np.float64)
        if array.shape != (4,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{context} box {index} must contain four finite coordinates")
        x1, y1, x2, y2 = array.tolist()
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"{context} box {index} has non-positive area: {box}")
        validated.append([x1, y1, x2, y2])
    return validated


def _build_ground_truth(
    annotation_path: str | Path,
) -> dict[str, dict[str, Any]]:
    ground_truth: dict[str, dict[str, Any]] = {}
    for item in load_records(annotation_path):
        image_path = str(item.get("image_path", ""))
        image_name = Path(image_path).stem
        categories = item.get("gt")
        if not isinstance(categories, dict):
            raise ValueError(f"Dense200 annotation {image_path} has no gt mapping")
        for category, boxes in categories.items():
            category_name = str(category)
            item_id = f"Dense200-{image_name}-{category_name.replace(' ', '_')}"
            if item_id in ground_truth:
                raise ValueError(f"Duplicate Dense200 annotation id: {item_id}")
            ground_truth[item_id] = {
                "category": category_name,
                "boxes": _validate_boxes(boxes, context=f"Dense200 GT {item_id}"),
            }
    return ground_truth


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


def _harmonic_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _gt_order_iou_matches(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> int:
    """Match each GT to its best unused prediction; ties keep prediction order.

    This is Rex-Omni's calculate_detection_metrics box-matching rule. Vectorize
    each GT's IoUs without changing greedy GT order or adding an IoU epsilon.
    The shared legacy matcher remains unchanged for other evaluation tasks.
    """
    if not pred_boxes or not gt_boxes:
        return 0
    predictions = np.asarray(pred_boxes, dtype=np.float64)
    pred_areas = (predictions[:, 2] - predictions[:, 0]) * (
        predictions[:, 3] - predictions[:, 1]
    )
    used = np.zeros(len(predictions), dtype=bool)
    matches = 0
    for target in gt_boxes:
        top_left = np.maximum(predictions[:, :2], target[:2])
        bottom_right = np.minimum(predictions[:, 2:], target[2:])
        overlap = np.maximum(bottom_right - top_left, 0.0)
        intersection = overlap[:, 0] * overlap[:, 1]
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        union = pred_areas + target_area - intersection
        ious = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        ious[used] = -1.0
        best_index = int(np.argmax(ious))
        if ious[best_index] > 0 and ious[best_index] >= iou_threshold:
            used[best_index] = True
            matches += 1
    return matches


def _compute_sample_metrics(
    samples: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> tuple[dict[float, dict[str, float]], dict[str, float]]:
    metrics: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        precisions: list[float] = []
        recalls: list[float] = []
        for sample in samples:
            tp = sample["tp_by_iou"][f"{threshold:.2f}"]
            pred_count, gt_count = sample["pred_count"], sample["gt_count"]
            precisions.append(tp / pred_count if pred_count else 0.0)
            recalls.append(tp / gt_count if gt_count else 0.0)
        precision = mean(precisions) if precisions else 0.0
        recall = mean(recalls) if recalls else 0.0
        metrics[threshold] = {
            "precision": precision,
            "recall": recall,
            "F1": _harmonic_f1(precision, recall),
        }

    aggregate = {
        key: mean(metrics[t][key] for t in thresholds) if thresholds else 0.0
        for key in ("precision", "recall", "F1")
    }
    return metrics, aggregate


def evaluate_dense200(
    prediction_paths: str | Path | Sequence[str | Path],
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    annotation_path: str | Path | None = None,
    metrics_output: str | Path | None = None,
    per_sample_output: str | Path | None = None,
    aspect_ratio_tolerance: float = 0.01,
    strict_jsonl: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    annotation_path = annotation_path or dataset_root / "Dense200.jsonl"
    ground_truth = _build_ground_truth(annotation_path)
    manifest = load_manifest(manifest_path)
    predictions, load_report = load_compact_predictions(
        prediction_paths,
        strict=strict_jsonl,
    )
    expected_ids = [str(item["id"]) for item in manifest]

    thresholds = [round(float(value), 2) for value in np.arange(0.5, 1.0, 0.05)]
    categories: set[str] = set()
    total_predictions = 0
    total_ground_truth = 0
    per_sample: list[dict[str, Any]] = []

    for manifest_item in manifest:
        item_id = str(manifest_item["id"])
        if item_id not in ground_truth:
            raise KeyError(f"Dense200 manifest id has no annotation: {item_id}")
        gt_data = ground_truth[item_id]
        gt_boxes = gt_data["boxes"]
        category = gt_data["category"]
        categories.add(category)

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

        per_threshold_tp: dict[str, int] = {}
        for threshold in thresholds:
            true_positives = _gt_order_iou_matches(pred_boxes, gt_boxes, threshold)
            per_threshold_tp[f"{threshold:.2f}"] = true_positives

        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)
        per_sample.append(
            {
                "id": item_id,
                "category": category,
                "termination_reason": (
                    prediction.termination_reason if prediction is not None else "missing"
                ),
                "prediction_kind": prediction_kind,
                "has_visual_result": bool(prediction and prediction.has_visual_result),
                "pred_count": len(pred_boxes),
                "gt_count": len(gt_boxes),
                "tp_by_iou": per_threshold_tp,
            }
        )

    threshold_metrics, mean_metrics = _compute_sample_metrics(per_sample, thresholds)
    metrics = {
        "task": "Dense200",
        "sample_count": len(manifest),
        "category_count": len(categories),
        "total_predictions": total_predictions,
        "total_ground_truth": total_ground_truth,
        "F1@0.50": threshold_metrics[0.5]["F1"],
        "F1@0.95": threshold_metrics[0.95]["F1"],
        "F1@0.50:0.95": mean_metrics["F1"],
        "precision@0.50": threshold_metrics[0.5]["precision"],
        "recall@0.50": threshold_metrics[0.5]["recall"],
        "precision@0.95": threshold_metrics[0.95]["precision"],
        "recall@0.95": threshold_metrics[0.95]["recall"],
        "precision@0.50:0.95": mean_metrics["precision"],
        "recall@0.50:0.95": mean_metrics["recall"],
        "per_threshold": {
            f"{threshold:.2f}": threshold_metrics[threshold] for threshold in thresholds
        },
        "protocol": {
            "name": "rex_omni_dense200",
            "source_url": (
                "https://github.com/IDEA-Research/Rex-Omni/blob/"
                "181bf00c92a5db28d4838fd5126bf0f40b598ade/"
                "evaluation/metrics/other_metric.py"
            ),
            "matching": "GT-order greedy matching; prediction-order IoU ties",
            "averaging": (
                "Mean precision and recall across QA samples, then harmonic F1 "
                "at each threshold; mean F1 across IoU 0.50:0.05:0.95"
            ),
            "units": "fractions in [0, 1]; multiply by 100 for paper tables",
        },
        "status": prediction_status_summary(expected_ids, predictions, load_report),
        "efficiency": summarize_efficiency_metrics(
            expected_ids,
            predictions,
            prediction_paths,
        ),
    }
    write_json(metrics_output, metrics)
    write_jsonl(per_sample_output, per_sample)
    return metrics


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


def print_dense200_summary_table(metrics: dict[str, Any]) -> None:
    """Print compact Dense200 task and efficiency metrics as a Markdown table."""

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
        run_level.get("peak_active_kv_cache_memory_per_gpu_gib"),
        precision=2,
        unit="GiB/GPU",
    )
    peak_kv_usage_percent = run_level.get(
        "peak_active_kv_cache_usage_percent",
        run_level.get("peak_kv_cache_usage_percent"),
    )
    if peak_kv_usage_percent is None and "peak_active_kv_cache_usage_percent" not in run_level:
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
    if peak_active_kv_memory != "N/A" and formatted_peak_kv_usage != "N/A":
        peak_active_kv_memory = (
            f"{peak_active_kv_memory} ({formatted_peak_kv_usage})"
        )

    headers = (
        "F1@0.50",
        "F1@0.95",
        "Mean F1@IoU 0.50:0.95",
        "Avg. Visual Tokens/Trajectory",
        "Avg. Latency",
        "Peak active KV memory",
        "rollout throughput",
    )
    values = (
        _format_table_metric(metrics.get("F1@0.50"), precision=4),
        _format_table_metric(metrics.get("F1@0.95"), precision=4),
        _format_table_metric(metrics.get("F1@0.50:0.95"), precision=4),
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
            run_level.get("rollout_throughput_trajectories_per_minute"),
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
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--aspect-ratio-tolerance", type=float, default=0.01)
    parser.add_argument("--allow-malformed-lines", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_dense200(
        args.predictions,
        args.manifest,
        args.dataset_root,
        annotation_path=args.annotations,
        metrics_output=args.metrics_output,
        per_sample_output=args.per_sample_output,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        strict_jsonl=not args.allow_malformed_lines,
    )
    print_dense200_summary_table(metrics)


if __name__ == "__main__":
    main()
