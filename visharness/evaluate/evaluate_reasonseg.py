"""Evaluate ReasonSeg predictions produced by the VisHarness trajectory runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from visharness.evaluate.common import (
    CompactPrediction,
    load_compact_predictions,
    load_exif_transposed_image_size,
    load_manifest,
    prediction_status_summary,
    resolve_image_path,
    restore_mask_to_original,
    summarize_efficiency_metrics,
    validate_final_result_consistency,
    write_json,
    write_jsonl,
)


def _read_reasonseg_annotation(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except UnicodeDecodeError:
        with path.open("r", encoding="cp1252") as file:
            payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("shapes"), list):
        raise ValueError(f"Invalid ReasonSeg annotation: {path}")
    return payload


def get_mask_from_json(json_path: str | Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Render ReasonSeg target/ignore polygons on the original image canvas."""

    annotation = _read_reasonseg_annotation(Path(json_path))
    height, width = image_shape

    polygons: list[tuple[int, dict[str, Any]]] = []
    for shape in annotation["shapes"]:
        if not isinstance(shape, dict):
            raise ValueError(f"Invalid polygon entry in {json_path}: {shape!r}")
        label = str(shape.get("label", ""))
        if label.lower() == "flag":
            continue
        points = np.asarray(shape.get("points"), dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"Invalid polygon points in {json_path}: {shape.get('points')!r}")

        area_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(area_mask, [points], 1)
        polygons.append((int(area_mask.sum()), shape))

    # Match the legacy ReasonSeg rendering order: larger polygons first, then
    # smaller polygons overwrite them.
    polygons.sort(key=lambda item: item[0], reverse=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, shape in polygons:
        label = str(shape["label"])
        points = np.asarray(shape["points"], dtype=np.int32)
        value = 255 if "ignore" in label.lower() else 1
        cv2.polylines(mask, [points], True, value, 1)
        cv2.fillPoly(mask, [points], value)
    return mask


def _prediction_mask(
    prediction: CompactPrediction | None,
    *,
    original_width: int,
    original_height: int,
    aspect_ratio_tolerance: float,
) -> tuple[np.ndarray, str, bool]:
    if prediction is None:
        return (
            np.zeros((original_height, original_width), dtype=bool),
            "missing",
            False,
        )
    if not prediction.has_visual_result:
        explicit_empty = prediction.explicit_empty_prediction
        return (
            np.zeros((original_height, original_width), dtype=bool),
            "explicit_empty" if explicit_empty else prediction.termination_reason,
            explicit_empty,
        )

    validate_final_result_consistency(prediction)
    mask = restore_mask_to_original(
        prediction.final_masks,
        original_width=original_width,
        original_height=original_height,
        aspect_ratio_tolerance=aspect_ratio_tolerance,
    )
    return mask, "visual_result", False


def evaluate_reasonseg(
    prediction_paths: str | Path | Sequence[str | Path],
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    metrics_output: str | Path | None = None,
    per_sample_output: str | Path | None = None,
    aspect_ratio_tolerance: float = 0.01,
    strict_jsonl: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    predictions, load_report = load_compact_predictions(
        prediction_paths,
        strict=strict_jsonl,
    )
    expected_ids = [str(item["id"]) for item in manifest]

    total_intersection = 0
    total_union = 0
    iou_sum = 0.0
    empty_gt_count = 0
    empty_gt_correct = 0
    per_sample: list[dict[str, Any]] = []

    for manifest_item in manifest:
        item_id = str(manifest_item["id"])
        if not item_id.startswith("ReasonSeg-"):
            raise ValueError(f"Unexpected ReasonSeg item id: {item_id}")

        image_path = resolve_image_path(
            manifest_item,
            dataset_root=dataset_root,
            manifest_path=manifest_path,
        )
        original_width, original_height = load_exif_transposed_image_size(image_path)
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.is_file():
            fallback = Path(dataset_root) / f"{item_id.removeprefix('ReasonSeg-')}.json"
            if fallback.is_file():
                annotation_path = fallback
            else:
                raise FileNotFoundError(annotation_path)

        gt_mask = get_mask_from_json(
            annotation_path,
            (original_height, original_width),
        )
        prediction = predictions.get(item_id)
        pred_target, prediction_kind, explicit_empty = _prediction_mask(
            prediction,
            original_width=original_width,
            original_height=original_height,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        )

        gt_target = gt_mask == 1
        valid_area = gt_mask != 255
        valid_prediction = pred_target & valid_area
        valid_target = gt_target & valid_area
        is_gt_empty = not bool(np.any(valid_target))

        intersection = int(np.logical_and(valid_prediction, valid_target).sum())
        union = int(np.logical_or(valid_prediction, valid_target).sum())
        if is_gt_empty:
            empty_gt_count += 1
            # Missing, failed, and max-round trajectories are not intentional
            # empty predictions. Match the GRES null-target semantics.
            if explicit_empty:
                iou = 1.0
                empty_gt_correct += 1
            else:
                iou = 0.0
        else:
            iou = intersection / union if union else 0.0

        total_intersection += intersection
        total_union += union
        iou_sum += iou
        per_sample.append(
            {
                "id": item_id,
                "termination_reason": (
                    prediction.termination_reason if prediction is not None else "missing"
                ),
                "prediction_kind": prediction_kind,
                "has_visual_result": bool(prediction and prediction.has_visual_result),
                "ground_truth_empty": is_gt_empty,
                "intersection": intersection,
                "union": union,
                "iou": iou,
                "null_correct": bool(is_gt_empty and explicit_empty),
            }
        )

    sample_count = len(manifest)
    metrics = {
        "task": "ReasonSeg",
        "sample_count": sample_count,
        "gIoU": iou_sum / sample_count if sample_count else 0.0,
        "cIoU": total_intersection / total_union if total_union else 0.0,
        "N_acc": empty_gt_correct / empty_gt_count if empty_gt_count else 0.0,
        "empty_ground_truth_samples": empty_gt_count,
        "empty_ground_truth_correct": empty_gt_correct,
        "total_intersection": total_intersection,
        "total_union": total_union,
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


# Backward-compatible function name used by the old script.
def evaluate_lisa_results(
    jsonl_path: str | Path,
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if manifest_path is None:
        root = Path(dataset_root)
        split_name = root.name if root.name in {"train", "val", "test"} else "test"
        manifest_path = root.parent / f"ReasonSeg_QA_{split_name}.json"
        dataset_root = root.parent
    return evaluate_reasonseg(jsonl_path, manifest_path, dataset_root)


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


def print_reasonseg_summary_table(metrics: dict[str, Any]) -> None:
    """Print compact ReasonSeg task and efficiency metrics as a Markdown table."""

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
        "gIoU",
        "cIoU",
        "Avg. Visual Tokens/Trajectory",
        "Avg. Latency",
        "Peak active KV memory",
        "rollout throughput",
    )
    values = (
        _format_table_metric(metrics.get("gIoU"), precision=4),
        _format_table_metric(metrics.get("cIoU"), precision=4),
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
    parser.add_argument("--predictions", nargs="+", required=True, help="Checkpoint JSONL file(s)")
    parser.add_argument(
        "--manifest",
        required=True,
        help="ReasonSeg QA manifest used for inference",
    )
    parser.add_argument("--dataset-root", required=True, help="ReasonSeg dataset root")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--aspect-ratio-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--allow-malformed-lines",
        action="store_true",
        help="Skip malformed JSONL lines instead of failing fast",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_reasonseg(
        args.predictions,
        args.manifest,
        args.dataset_root,
        metrics_output=args.metrics_output,
        per_sample_output=args.per_sample_output,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        strict_jsonl=not args.allow_malformed_lines,
    )
    print_reasonseg_summary_table(metrics)


if __name__ == "__main__":
    main()
