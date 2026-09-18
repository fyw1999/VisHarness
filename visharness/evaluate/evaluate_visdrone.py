"""Evaluate VisDrone box predictions with global/micro F1 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from visharness.evaluate.common import (
    CompactPrediction,
    greedy_iou_matches,
    load_compact_predictions,
    load_exif_transposed_image_size,
    load_manifest,
    load_records,
    prediction_status_summary,
    print_metrics,
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
            raise ValueError(f"VisDrone annotation {image_path} has no gt mapping")
        for category, boxes in categories.items():
            category_name = str(category)
            item_id = f"VisDrone-{image_name}-{category_name.replace(' ', '_')}"
            if item_id in ground_truth:
                raise ValueError(f"Duplicate VisDrone annotation id: {item_id}")
            ground_truth[item_id] = {
                "category": category_name,
                "boxes": _validate_boxes(boxes, context=f"VisDrone GT {item_id}"),
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
    return (
        restore_boxes_to_original(
            prediction.final_bboxes,
            prediction.final_masks,
            original_width=original_width,
            original_height=original_height,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        ),
        "visual_result",
    )


def _f1(true_positives: int, prediction_count: int, ground_truth_count: int) -> float:
    if true_positives == 0 or prediction_count == 0 or ground_truth_count == 0:
        return 0.0
    precision = true_positives / prediction_count
    recall = true_positives / ground_truth_count
    return 2 * precision * recall / (precision + recall)


def evaluate_visdrone(
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
    annotation_path = annotation_path or dataset_root / "VisDrone.jsonl"
    ground_truth = _build_ground_truth(annotation_path)
    manifest = load_manifest(manifest_path)
    predictions, load_report = load_compact_predictions(
        prediction_paths,
        strict=strict_jsonl,
    )
    expected_ids = [str(item["id"]) for item in manifest]

    thresholds = [round(float(value), 2) for value in np.arange(0.5, 1.0, 0.05)]
    true_positives = {threshold: 0 for threshold in thresholds}
    total_predictions = 0
    total_ground_truth = 0
    per_sample: list[dict[str, Any]] = []

    for manifest_item in manifest:
        item_id = str(manifest_item["id"])
        if item_id not in ground_truth:
            raise KeyError(f"VisDrone manifest id has no annotation: {item_id}")
        gt_data = ground_truth[item_id]
        gt_boxes = gt_data["boxes"]

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
            matched = greedy_iou_matches(pred_boxes, gt_boxes, threshold)
            true_positives[threshold] += matched
            per_threshold_tp[f"{threshold:.2f}"] = matched

        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)
        per_sample.append(
            {
                "id": item_id,
                "category": gt_data["category"],
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

    f1_by_threshold = {
        threshold: _f1(
            true_positives[threshold],
            total_predictions,
            total_ground_truth,
        )
        for threshold in thresholds
    }
    precision_by_threshold = {
        threshold: (
            true_positives[threshold] / total_predictions if total_predictions else 0.0
        )
        for threshold in thresholds
    }
    recall_by_threshold = {
        threshold: (
            true_positives[threshold] / total_ground_truth if total_ground_truth else 0.0
        )
        for threshold in thresholds
    }

    metrics = {
        "task": "VisDrone",
        "sample_count": len(manifest),
        "total_predictions": total_predictions,
        "total_ground_truth": total_ground_truth,
        "F1@0.50": f1_by_threshold[0.5],
        "F1@0.95": f1_by_threshold[0.95],
        "F1@0.50:0.95": float(np.mean(list(f1_by_threshold.values()))),
        "per_threshold": {
            f"{threshold:.2f}": {
                "TP": true_positives[threshold],
                "precision": precision_by_threshold[threshold],
                "recall": recall_by_threshold[threshold],
                "F1": f1_by_threshold[threshold],
            }
            for threshold in thresholds
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
    metrics = evaluate_visdrone(
        args.predictions,
        args.manifest,
        args.dataset_root,
        annotation_path=args.annotations,
        metrics_output=args.metrics_output,
        per_sample_output=args.per_sample_output,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        strict_jsonl=not args.allow_malformed_lines,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
