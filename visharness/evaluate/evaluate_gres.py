"""Evaluate full or manifest-defined GRES splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from visharness.evaluate.common import (
    CompactPrediction,
    load_compact_predictions,
    load_manifest,
    prediction_status_summary,
    restore_mask_to_original,
    summarize_efficiency_metrics,
    validate_final_result_consistency,
    write_json,
    write_jsonl,
)
from visharness.evaluate.grefer import G_REFER


def _parse_ref_id(item_id: str) -> int:
    for prefix in ("GRES-", "GERS-"):
        if item_id.startswith(prefix):
            item_id = item_id[len(prefix) :]
            break
    try:
        return int(item_id)
    except ValueError as exc:
        raise ValueError(f"Invalid GRES item id: {item_id!r}") from exc


def _load_ref(gref: G_REFER, ref_id: int) -> dict[str, Any]:
    try:
        refs = gref.loadRefs([ref_id])
    except Exception:
        refs = []
    if refs:
        return refs[0]
    ref = gref.Refs.get(ref_id)
    if not ref:
        raise KeyError(f"Unable to load GRES ref_id={ref_id}")
    return ref


def _prediction_mask(
    prediction: CompactPrediction | None,
    *,
    original_width: int,
    original_height: int,
    aspect_ratio_tolerance: float,
) -> tuple[np.ndarray, str, bool]:
    """Return mask, prediction kind, and whether empty was explicitly answered."""

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
    return (
        restore_mask_to_original(
            prediction.final_masks,
            original_width=original_width,
            original_height=original_height,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        ),
        "visual_result",
        False,
    )


def evaluate_gres(
    prediction_paths: str | Path | Sequence[str | Path],
    manifest_path: str | Path,
    dataset_root: str | Path,
    split: str,
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

    gref = G_REFER(str(dataset_root), dataset="grefcoco", splitBy="unc")
    valid_ref_ids = set(gref.getRefIds(split=split))

    total_intersection = 0
    total_union = 0
    iou_sum = 0.0
    empty_gt_count = 0
    empty_gt_correct = 0
    per_sample: list[dict[str, Any]] = []

    for manifest_item in manifest:
        item_id = str(manifest_item["id"])
        ref_id = _parse_ref_id(item_id)
        if ref_id not in valid_ref_ids:
            raise ValueError(
                f"Manifest item {item_id} does not belong to GRES split {split!r}"
            )

        ref = _load_ref(gref, ref_id)
        mask_info = gref.getMaskByRef(ref=ref, merge=True)
        is_gt_empty = bool(mask_info.get("empty", False))
        image_info = gref.Imgs[ref["image_id"]]
        original_width = int(image_info["width"])
        original_height = int(image_info["height"])
        if is_gt_empty:
            gt_mask = np.zeros((original_height, original_width), dtype=bool)
        else:
            raw_gt_mask = np.asarray(mask_info["mask"])
            if raw_gt_mask.shape != (original_height, original_width):
                raise ValueError(
                    f"GRES GT mask shape mismatch for {item_id}: "
                    f"{raw_gt_mask.shape} != {(original_height, original_width)}"
                )
            gt_mask = raw_gt_mask > 0

        prediction = predictions.get(item_id)
        pred_mask, prediction_kind, explicit_empty = _prediction_mask(
            prediction,
            original_width=original_width,
            original_height=original_height,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        )

        intersection = int(np.logical_and(pred_mask, gt_mask).sum())
        union = int(np.logical_or(pred_mask, gt_mask).sum())

        if is_gt_empty:
            empty_gt_count += 1
            # A missing/failed trajectory is not a correct null prediction.
            # Only an explicit final answer with no visual result receives the
            # generalized IoU credit for a correctly rejected expression.
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
                "ref_id": ref_id,
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
        "task": "GRES",
        "split": split,
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


def evaluate_gres_rigorous(
    jsonl_path: str | Path,
    dataset_root: str | Path,
    split: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible entry point with manifest-driven completeness."""

    if manifest_path is None:
        manifest_path = Path(dataset_root) / f"GRES_QA_{split}.json"
    return evaluate_gres(jsonl_path, manifest_path, dataset_root, split)


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


def print_gres_summary_table(metrics: dict[str, Any]) -> None:
    """Print compact GRES task and efficiency metrics as a Markdown table."""

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


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--predictions", nargs="+", required=True, help="Checkpoint JSONL file(s)")
    parser.add_argument("--manifest", required=True, help="GRES QA manifest used for inference")
    parser.add_argument("--dataset-root", required=True, help="GRES dataset root")
    parser.add_argument("--split", required=True, choices=["train", "val", "testA", "testB"])
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--aspect-ratio-tolerance", type=float, default=0.01)
    parser.add_argument("--allow-malformed-lines", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_gres(
        args.predictions,
        args.manifest,
        args.dataset_root,
        args.split,
        metrics_output=args.metrics_output,
        per_sample_output=args.per_sample_output,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        strict_jsonl=not args.allow_malformed_lines,
    )


def main() -> None:
    metrics = run_from_args(build_parser().parse_args())
    print_gres_summary_table(metrics)


if __name__ == "__main__":
    main()
