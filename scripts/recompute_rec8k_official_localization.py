"""Re-score saved REC-8K boxes with the CVPR 2024 authors' localization rule.

This does not change the main evaluator or overwrite evaluation_metrics.json.
Run with ``python scripts/recompute_rec8k_official_localization.py --dataset-root ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visharness.evaluate.common import (
    load_compact_predictions,
    load_exif_transposed_image_size,
    load_manifest,
    resolve_image_path,
    write_json,
    write_jsonl,
)
from visharness.evaluate.evaluate_rec8k import (
    _build_split_lookup,
    _localization_matches,
    _restore_prediction_boxes,
)


OFFICIAL_COMMIT = "e8f5a54f295fd7443665b5c70499e5a85bccc0e4"
OFFICIAL_URL = (
    "https://github.com/sydai/referring-expression-counting/blob/"
    f"{OFFICIAL_COMMIT}/train_test.py"
)


def _legacy_localization_matches(
    pixel_boxes: list[list[float]], pixel_gt_points: list[list[float]]
) -> int:
    """Audit pre-official results only; never use this as the default evaluator."""
    if not pixel_boxes or not pixel_gt_points:
        return 0
    widths = [box[2] - box[0] for box in pixel_boxes]
    heights = [box[3] - box[1] for box in pixel_boxes]
    indices = np.argsort(np.asarray(widths) * np.asarray(heights))
    count = len(pixel_boxes)
    if count % 2:
        middle = indices[count // 2]
        threshold = np.hypot(widths[middle], heights[middle]) / 2.0
    else:
        first, second = indices[count // 2 - 1 : count // 2 + 1]
        threshold = (
            np.hypot(widths[first], heights[first])
            + np.hypot(widths[second], heights[second])
        ) / 4.0
    pred_points = np.asarray([
        [box[0] + width / 2.0, box[1] + height / 2.0]
        for box, width, height in zip(pixel_boxes, widths, heights)
    ], dtype=np.float64)
    distances = cdist(pred_points, np.asarray(pixel_gt_points, dtype=np.float64), metric="euclidean")
    pred_indices, gt_indices = linear_sum_assignment(distances)
    return int(np.count_nonzero(distances[pred_indices, gt_indices] <= threshold))


def aggregate_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}


def evaluate_saved_run(run_directory: str, dataset_directory: str, split: str) -> dict[str, Any]:
    folder, dataset_root = Path(run_directory), Path(dataset_directory)
    manifest_path = dataset_root / f"REC8K_QA_{split}.json"
    prediction_path = folder / f"{folder.name}_ckpt.jsonl"
    with (folder / "evaluation_metrics.json").open(encoding="utf-8") as file:
        previous_metrics = json.load(file)
    with (dataset_root / "annotations.json").open(encoding="utf-8") as file:
        annotations = json.load(file)
    with (dataset_root / "splits.json").open(encoding="utf-8") as file:
        split_lookup = _build_split_lookup(json.load(file)[split])
    manifest = load_manifest(manifest_path)
    print(f"Loading {folder.name}: {prediction_path.stat().st_size / 2**30:.2f} GiB", flush=True)
    predictions, load_report = load_compact_predictions(prediction_path, strict=True)
    expected_ids = {str(item["id"]) for item in manifest}
    if set(predictions) != expected_ids or load_report.duplicate_records:
        raise ValueError(f"{folder.name}: missing/unexpected/duplicate prediction IDs")
    if len(manifest) != len(split_lookup):
        raise ValueError(f"{folder.name}: manifest does not cover the full official split")

    rows: list[dict[str, Any]] = []
    old_tp = new_tp = pred_total = gt_total = 0
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    image_sizes: dict[Path, tuple[int, int]] = {}
    for index, item in enumerate(manifest, start=1):
        item_id = str(item["id"])
        image_name, phrase = split_lookup[item_id.removeprefix("REC8K-")]
        gt_points = annotations[image_name][phrase]["points"]
        # Official prepare_targets substitutes a dummy point for empty GT.
        # Neither official val nor test has empty GT; fail rather than alter it.
        if not gt_points:
            raise ValueError(f"{item_id}: empty GT requires a separately specified protocol")
        image_path = resolve_image_path(item, dataset_root=dataset_root, manifest_path=manifest_path)
        if image_path not in image_sizes:
            image_sizes[image_path] = load_exif_transposed_image_size(image_path)
        width, height = image_sizes[image_path]
        boxes, prediction_kind = _restore_prediction_boxes(
            predictions[item_id],
            original_width=width,
            original_height=height,
            aspect_ratio_tolerance=0.01,
        )
        tp = _localization_matches(
            boxes, gt_points, original_width=width, original_height=height
        )
        legacy_tp = _legacy_localization_matches(boxes, gt_points)
        pred_count, gt_count = len(boxes), len(gt_points)
        old_tp += legacy_tp
        new_tp += tp
        pred_total += pred_count
        gt_total += gt_count
        absolute_errors.append(float(abs(pred_count - gt_count)))
        squared_errors.append(float((pred_count - gt_count) ** 2))
        rows.append(
            {
                "id": item_id,
                "image_name": image_name,
                "original_width": width,
                "original_height": height,
                "prediction_kind": prediction_kind,
                "termination_reason": predictions[item_id].termination_reason,
                "pred_count": pred_count,
                "gt_count": gt_count,
                "TP": tp,
                "FP": pred_count - tp,
                "FN": gt_count - tp,
                "legacy_TP": legacy_tp,
                "TP_change": tp - legacy_tp,
            }
        )
        if index % 1000 == 0:
            print(f"{folder.name}: scored {index}/{len(manifest)}", flush=True)

    legacy_metrics = aggregate_metrics(old_tp, pred_total - old_tp, gt_total - old_tp)
    official_metrics = aggregate_metrics(new_tp, pred_total - new_tp, gt_total - new_tp)
    previous_is_official = (
        previous_metrics.get("localization_protocol", {}).get("name")
        == "CVPR2024_REC_official_localization"
    )
    expected_metrics = official_metrics if previous_is_official else legacy_metrics
    mae, rmse = float(np.mean(absolute_errors)), float(np.sqrt(np.mean(squared_errors)))
    for key, value in {**expected_metrics, "sample_count": len(manifest), "MAE": mae, "RMSE": rmse}.items():
        if not math.isclose(value, previous_metrics[key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{folder.name}: original {key} not reproduced: {value} != {previous_metrics[key]}")

    metrics = {
        "task": "REC8K",
        "split": split,
        "run": folder.name,
        "sample_count": len(manifest),
        **official_metrics,
        "MAE": mae,
        "RMSE": rmse,
        "protocol": {
            "name": "CVPR2024_REC_official_localization",
            "official_source_url": OFFICIAL_URL,
            "coordinate_system": "x/width, y/height; normalized cxcywh boxes",
            "gt_dtype": "float32, matching official prepare_targets",
            "matching": "Hungarian minimum total Euclidean distance",
            "threshold": "half diagonal of area-sorted box at index N//2 (upper median)",
            "acceptance": "distance < threshold",
            "aggregation": "dataset-wide TP/FP/FN (micro)",
            "prediction_handling": "identical to existing evaluator; no sample excluded",
        },
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
        "previous_metrics_reproduced": True,
        "previous_metrics_protocol": "official" if previous_is_official else "legacy",
        "legacy_metrics_reproduced": not previous_is_official,
        "legacy_localization": legacy_metrics,
        "samples_with_changed_tp": sum(row["TP_change"] != 0 for row in rows),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(folder / "evaluation_localization_official.json", metrics)
    write_jsonl(folder / "localization_official_per_sample.jsonl", rows)
    print(
        f"{folder.name}: P={metrics['precision']:.8f} R={metrics['recall']:.8f} "
        f"F1={metrics['F1']:.8f}; old F1={legacy_metrics['F1']:.8f}; saved metrics reproduced",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = [
        (str(args.results_root / f"VisHarness_{model}_REC8K_{split}"), str(args.dataset_root), split)
        for model in ("SFT", "GRPO_550")
        for split in ("val", "test")
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_saved_run, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    ordering = {Path(job[0]).name: index for index, job in enumerate(jobs)}
    results.sort(key=lambda result: ordering[result["run"]])
    write_json(args.results_root / "REC8K_official_localization_comparison.json", results)
    lines = [
        "# REC-8K localization using the CVPR 2024 official matching protocol",
        "",
        "Precision, Recall and F1 are percentages. Original evaluation files are unchanged.",
        "",
        f"Reference: {OFFICIAL_URL}",
        "",
        "| Model | Split | Precision (%) | Recall (%) | F1 (%) | Previous F1 (%) | F1 change (pp) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for metrics in results:
        model = "SFT" if "_SFT_" in metrics["run"] else "RL550"
        old_f1 = metrics["legacy_localization"]["F1"]
        lines.append(
            f"| {model} | {metrics['split']} | {100 * metrics['precision']:.2f} | "
            f"{100 * metrics['recall']:.2f} | {100 * metrics['F1']:.2f} | "
            f"{100 * old_f1:.2f} | {100 * (metrics['F1'] - old_f1):+.2f} |"
        )
    comparison_path = args.results_root / "REC8K_official_localization_comparison.md"
    comparison_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
