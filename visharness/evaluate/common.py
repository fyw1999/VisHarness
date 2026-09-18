"""Shared utilities for evaluating VisHarness trajectory-runner outputs.

The trajectory checkpoint is intentionally rich: every JSONL record may contain
the full conversation, archived images, raw tool responses, and a merged
visualization.  Evaluation only needs a small subset of those fields.  This
module streams the checkpoint and keeps compact per-item records so evaluating
large result files does not require memory proportional to the file size.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils


TERMINATION_ANSWER = "answer"
TERMINATION_FAILED = "failed"
TERMINATION_MAX_ROUNDS = "max_rounds_reached"
TERMINATION_INCOMPLETE = "incomplete"


@dataclass(slots=True)
class CompactPrediction:
    """Only the checkpoint fields needed by the evaluators."""

    item_id: str
    final_bboxes: list[Any] = field(default_factory=list)
    final_masks: list[dict[str, Any]] = field(default_factory=list)
    count: int | None = None
    status: str | None = None
    termination_reason: str = TERMINATION_INCOMPLETE
    trajectory_finished: bool | None = None
    max_rounds_reached: bool | None = None
    current_round: int | None = None
    max_rounds: int | None = None
    error: str | None = None
    efficiency_metrics: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    line_number: int = 0

    @property
    def has_visual_result(self) -> bool:
        return bool(
            self.final_bboxes
            or self.final_masks
            or (self.count is not None and self.count > 0)
        )

    @property
    def process_success(self) -> bool:
        return self.termination_reason == TERMINATION_ANSWER

    @property
    def explicit_empty_prediction(self) -> bool:
        return self.process_success and not self.has_visual_result


@dataclass(slots=True)
class PredictionLoadReport:
    source_paths: list[str] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    malformed_lines: int = 0
    invalid_id_lines: int = 0
    duplicate_records: int = 0
    duplicate_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duplicate_ids"] = sorted(self.duplicate_ids)
        result["unique_duplicate_ids"] = len(self.duplicate_ids)
        return result


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_termination_reason(item: dict[str, Any]) -> str:
    """Infer new and legacy trajectory termination semantics.

    New checkpoints explicitly store ``termination_reason``.  Old checkpoints
    only stored current/max rounds; an old trajectory that stopped before the
    round limit is treated as an explicit answer, matching the legacy runner.
    """

    if item.get("status") == "failed" or item.get("trajectory_invalid") is True:
        return TERMINATION_FAILED

    explicit_reason = item.get("termination_reason")
    if explicit_reason in {
        TERMINATION_ANSWER,
        TERMINATION_FAILED,
        TERMINATION_MAX_ROUNDS,
    }:
        return str(explicit_reason)

    if item.get("max_rounds_reached") is True:
        return TERMINATION_MAX_ROUNDS
    if item.get("trajectory_finished") is True:
        return TERMINATION_ANSWER
    if item.get("answer") is not None:
        return TERMINATION_ANSWER

    current_round = _optional_int(item.get("current_round"))
    max_rounds = _optional_int(item.get("max_rounds"))
    if current_round is not None and max_rounds is not None:
        if current_round >= max_rounds:
            return TERMINATION_MAX_ROUNDS
        # Legacy records either omitted status or used status="finished"
        # without trajectory_finished/termination_reason. Finishing before the
        # maximum number of rounds meant the model emitted its final answer.
        if item.get("status") in {None, "finished"}:
            return TERMINATION_ANSWER

    if item.get("status") == "finished":
        return TERMINATION_ANSWER

    return TERMINATION_INCOMPLETE


def _extract_item_id(item: dict[str, Any]) -> str | None:
    item_id = item.get("id")
    if item_id is None and isinstance(item.get("meta_data"), dict):
        item_id = item["meta_data"].get("id")
    if item_id is None:
        return None
    return str(item_id)


def compact_prediction(
    item: dict[str, Any],
    *,
    source_path: str,
    line_number: int,
) -> CompactPrediction:
    item_id = _extract_item_id(item)
    if item_id is None:
        raise ValueError("prediction record does not contain id or meta_data.id")

    final_results = item.get("final_results")
    if not isinstance(final_results, dict):
        final_results = {}

    final_bboxes = final_results.get("final_bboxes")
    final_masks = final_results.get("final_masks")
    count = _optional_int(final_results.get("count"))
    efficiency_metrics = item.get("efficiency_metrics")
    if not isinstance(efficiency_metrics, dict):
        efficiency_metrics = {}

    return CompactPrediction(
        item_id=item_id,
        final_bboxes=list(final_bboxes) if isinstance(final_bboxes, list) else [],
        final_masks=list(final_masks) if isinstance(final_masks, list) else [],
        count=count,
        status=str(item["status"]) if item.get("status") is not None else None,
        termination_reason=infer_termination_reason(item),
        trajectory_finished=item.get("trajectory_finished")
        if isinstance(item.get("trajectory_finished"), bool)
        else None,
        max_rounds_reached=item.get("max_rounds_reached")
        if isinstance(item.get("max_rounds_reached"), bool)
        else None,
        current_round=_optional_int(item.get("current_round")),
        max_rounds=_optional_int(item.get("max_rounds")),
        error=str(item["error"]) if item.get("error") is not None else None,
        efficiency_metrics=copy.deepcopy(efficiency_metrics),
        source_path=source_path,
        line_number=line_number,
    )


def load_compact_predictions(
    paths: str | Path | Sequence[str | Path],
    *,
    strict: bool = True,
) -> tuple[dict[str, CompactPrediction], PredictionLoadReport]:
    """Stream one or more checkpoint JSONL files.

    Later records replace earlier records for the same item id.  This mirrors
    append/resume semantics and also permits resumed fragments to be supplied
    in chronological order.
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("at least one prediction JSONL path is required")

    predictions: dict[str, CompactPrediction] = {}
    report = PredictionLoadReport(source_paths=[str(Path(path)) for path in paths])

    for path_value in paths:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                report.total_lines += 1
                if not line.strip():
                    report.blank_lines += 1
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.malformed_lines += 1
                    if strict:
                        raise ValueError(f"Malformed JSON in {path}:{line_number}: {exc}") from exc
                    continue
                if not isinstance(item, dict):
                    report.malformed_lines += 1
                    if strict:
                        raise ValueError(
                            f"Prediction record in {path}:{line_number} must be a JSON object"
                        )
                    continue
                try:
                    prediction = compact_prediction(
                        item,
                        source_path=str(path),
                        line_number=line_number,
                    )
                except ValueError:
                    report.invalid_id_lines += 1
                    if strict:
                        raise
                    continue

                if prediction.item_id in predictions:
                    report.duplicate_records += 1
                    report.duplicate_ids.add(prediction.item_id)
                predictions[prediction.item_id] = prediction

    return predictions, report


def load_records(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(item)
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError(f"Manifest {path} must contain a JSON list")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Manifest {path} item {index} is not a JSON object")
        return payload

    raise ValueError(f"Unsupported record file: {path}")


def load_manifest(path_value: str | Path) -> list[dict[str, Any]]:
    records = load_records(path_value)
    seen: set[str] = set()
    for index, item in enumerate(records):
        if item.get("id") is None:
            raise ValueError(f"Manifest item {index} does not contain an id")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Manifest contains duplicate id: {item_id}")
        seen.add(item_id)
    return records


def resolve_image_path(
    manifest_item: dict[str, Any],
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
) -> Path:
    image_path_value = manifest_item.get("image_path") or manifest_item.get("image")
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(f"Manifest item {manifest_item.get('id')} has no image_path")

    image_path = Path(image_path_value)
    if image_path.is_absolute():
        return image_path

    dataset_candidate = Path(dataset_root) / image_path
    if dataset_candidate.exists():
        return dataset_candidate

    manifest_candidate = Path(manifest_path).parent / image_path
    if manifest_candidate.exists():
        return manifest_candidate

    # Return the canonical dataset-root location so the eventual error clearly
    # reports the path expected by this evaluation.
    return dataset_candidate


def load_exif_transposed_image_size(path_value: str | Path) -> tuple[int, int]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        return int(transposed.width), int(transposed.height)


def normalize_rles(rles: Any) -> list[dict[str, Any]]:
    if isinstance(rles, dict):
        rles = [rles]
    if not isinstance(rles, (list, tuple)) or not rles:
        raise ValueError("final_masks must be a non-empty COCO RLE list")

    normalized = copy.deepcopy(list(rles))
    for index, rle in enumerate(normalized):
        if not isinstance(rle, dict):
            raise ValueError(f"final_masks[{index}] must be a COCO RLE object")
        size = rle.get("size")
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or isinstance(size[0], bool)
            or isinstance(size[1], bool)
        ):
            raise ValueError(f"final_masks[{index}].size must be [height, width]")
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"final_masks[{index}] has invalid size {size}")
        rle["size"] = [height, width]
        if "counts" not in rle:
            raise ValueError(f"final_masks[{index}] has no counts")
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
    return normalized


def rle_canvas_size(rles: Any) -> tuple[int, int]:
    normalized = normalize_rles(rles)
    first_size = tuple(normalized[0]["size"])
    for index, rle in enumerate(normalized[1:], start=1):
        if tuple(rle["size"]) != first_size:
            raise ValueError(
                "All final masks must share one canvas size: "
                f"mask 0={first_size}, mask {index}={tuple(rle['size'])}"
            )
    return int(first_size[0]), int(first_size[1])


def decode_merged_mask(rles: Any) -> np.ndarray:
    normalized = normalize_rles(rles)
    canvas_size = tuple(normalized[0]["size"])
    for index, rle in enumerate(normalized[1:], start=1):
        if tuple(rle["size"]) != canvas_size:
            raise ValueError(
                "All final masks must share one canvas size: "
                f"mask 0={canvas_size}, mask {index}={tuple(rle['size'])}"
            )
    decoded = mask_utils.decode(normalized)
    if decoded.ndim == 2:
        decoded = decoded[:, :, np.newaxis]
    if decoded.ndim != 3:
        raise ValueError(f"Decoded final masks must be 3D, got shape={decoded.shape}")
    return np.any(decoded > 0, axis=2)


def _validate_aspect_ratio(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    tolerance: float,
) -> None:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "Image dimensions must be positive: "
            f"source={source_width}x{source_height}, target={target_width}x{target_height}"
        )
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    relative_error = abs(source_ratio / target_ratio - 1.0)
    if relative_error > tolerance:
        raise ValueError(
            "Prediction and ground-truth canvases have incompatible aspect ratios: "
            f"prediction={source_width}x{source_height}, "
            f"target={target_width}x{target_height}, "
            f"relative_error={relative_error:.6f}, tolerance={tolerance:.6f}"
        )


def validate_final_result_consistency(prediction: CompactPrediction) -> None:
    """Validate the shared MergeBoxMask result contract."""

    if not prediction.has_visual_result:
        return
    if not prediction.final_bboxes or not prediction.final_masks:
        raise ValueError(
            f"{prediction.item_id}: a visual result must contain both final_bboxes and final_masks"
        )
    if len(prediction.final_bboxes) != len(prediction.final_masks):
        raise ValueError(
            f"{prediction.item_id}: bbox/mask count mismatch: "
            f"{len(prediction.final_bboxes)} != {len(prediction.final_masks)}"
        )
    if prediction.count is not None and prediction.count != len(prediction.final_bboxes):
        raise ValueError(
            f"{prediction.item_id}: count={prediction.count} but "
            f"len(final_bboxes)={len(prediction.final_bboxes)}"
        )
    rle_canvas_size(prediction.final_masks)


def restore_mask_to_original(
    final_masks: Any,
    *,
    original_width: int,
    original_height: int,
    aspect_ratio_tolerance: float = 0.01,
) -> np.ndarray:
    """Decode and restore a merged mask from processed ``img_0`` to GT size."""

    mask = decode_merged_mask(final_masks)
    processed_height, processed_width = mask.shape
    _validate_aspect_ratio(
        processed_width,
        processed_height,
        original_width,
        original_height,
        tolerance=aspect_ratio_tolerance,
    )
    if (processed_width, processed_height) == (original_width, original_height):
        return mask.astype(bool)
    restored = cv2.resize(
        mask.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return restored.astype(bool)


def restore_boxes_to_original(
    final_bboxes: Any,
    final_masks: Any,
    *,
    original_width: int,
    original_height: int,
    aspect_ratio_tolerance: float = 0.01,
) -> list[list[float]]:
    """Restore merged boxes using the authoritative RLE canvas dimensions."""

    if not isinstance(final_bboxes, (list, tuple)):
        raise ValueError("final_bboxes must be a list")
    processed_height, processed_width = rle_canvas_size(final_masks)
    _validate_aspect_ratio(
        processed_width,
        processed_height,
        original_width,
        original_height,
        tolerance=aspect_ratio_tolerance,
    )

    scale_x = original_width / processed_width
    scale_y = original_height / processed_height
    restored: list[list[float]] = []
    boundary_tolerance = 1e-4

    for index, box in enumerate(final_bboxes):
        box_array = np.asarray(box, dtype=np.float64)
        if box_array.shape != (4,) or not np.all(np.isfinite(box_array)):
            raise ValueError(f"final_bboxes[{index}] must contain four finite coordinates")
        x1, y1, x2, y2 = box_array.tolist()
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"final_bboxes[{index}] has non-positive area: {box}")
        if (
            x1 < -boundary_tolerance
            or y1 < -boundary_tolerance
            or x2 > processed_width + boundary_tolerance
            or y2 > processed_height + boundary_tolerance
        ):
            raise ValueError(
                f"final_bboxes[{index}]={box} lies outside prediction canvas "
                f"{processed_width}x{processed_height}"
            )

        x1 = float(np.clip(x1, 0.0, processed_width) * scale_x)
        x2 = float(np.clip(x2, 0.0, processed_width) * scale_x)
        y1 = float(np.clip(y1, 0.0, processed_height) * scale_y)
        y2 = float(np.clip(y2, 0.0, processed_height) * scale_y)
        restored.append([x1, y1, x2, y2])

    return restored


def greedy_iou_matches(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> int:
    """Legacy confidence-free greedy IoU matching used by Dense200/VisDrone."""

    if not pred_boxes or not gt_boxes:
        return 0

    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                pairs.append((iou, pred_index, gt_index))
    # Keep the legacy evaluator's stable tie behavior: pairs with equal IoU
    # remain in prediction/ground-truth traversal order.
    pairs.sort(key=lambda item: item[0], reverse=True)

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches = 0
    for _, pred_index, gt_index in pairs:
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches += 1
    return matches


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("IoU boxes must contain four coordinates")
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )
    union = area1 + area2 - intersection
    # Preserve the legacy evaluators' numerical convention.  Boxes are
    # validated before reaching this function, so the epsilon is only relevant
    # for threshold comparisons at the floating-point boundary.
    return intersection / (union + 1e-6) if union > 0 else 0.0


def prediction_status_summary(
    expected_ids: Iterable[str],
    predictions: dict[str, CompactPrediction],
    load_report: PredictionLoadReport,
) -> dict[str, Any]:
    expected_list = [str(item_id) for item_id in expected_ids]
    expected_set = set(expected_list)
    matched = [predictions[item_id] for item_id in expected_list if item_id in predictions]
    outcome_counts = Counter(prediction.termination_reason for prediction in matched)
    actual_rounds = [
        prediction.current_round
        for prediction in matched
        if prediction.current_round is not None
    ]

    return {
        "expected_samples": len(expected_list),
        "unique_prediction_records": len(predictions),
        "matched_predictions": len(matched),
        "missing_predictions": len(expected_set - predictions.keys()),
        "unexpected_predictions": len(predictions.keys() - expected_set),
        "duplicate_prediction_records": load_report.duplicate_records,
        "unique_duplicate_ids": len(load_report.duplicate_ids),
        "explicit_answers": outcome_counts[TERMINATION_ANSWER],
        "failed_trajectories": outcome_counts[TERMINATION_FAILED],
        "max_rounds_reached": outcome_counts[TERMINATION_MAX_ROUNDS],
        "incomplete_trajectories": outcome_counts[TERMINATION_INCOMPLETE],
        "valid_visual_results": sum(prediction.has_visual_result for prediction in matched),
        "explicit_empty_predictions": sum(
            prediction.explicit_empty_prediction for prediction in matched
        ),
        "agent_success_rate": (
            outcome_counts[TERMINATION_ANSWER] / len(expected_list) if expected_list else 0.0
        ),
        "average_actual_rounds": (
            sum(actual_rounds) / len(actual_rounds) if actual_rounds else 0.0
        ),
        "actual_rounds_sample_count": len(actual_rounds),
        "actual_rounds_missing_count": len(expected_list) - len(actual_rounds),
        "load_report": load_report.to_dict(),
    }


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _distribution(
    values: list[float],
    *,
    expected_count: int,
) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "missing_count": max(expected_count - len(values), 0),
        "mean": float(np.mean(values)) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "sum": float(sum(values)) if values else None,
    }


def _normalize_prediction_paths(
    paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path) for path in paths]


def _load_benchmark_run_summaries(
    prediction_paths: str | Path | Sequence[str | Path],
) -> list[tuple[Path, dict[str, Any]]]:
    summaries: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    for prediction_path in _normalize_prediction_paths(prediction_paths):
        summary_path = prediction_path.parent / "benchmark_run.json"
        try:
            resolved_path = summary_path.resolve()
        except OSError:
            resolved_path = summary_path
        if resolved_path in seen_paths or not summary_path.is_file():
            continue
        seen_paths.add(resolved_path)
        with summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Benchmark summary {summary_path} must contain a JSON object"
            )
        checkpoint_path = payload.get("prediction_checkpoint_path")
        if checkpoint_path:
            try:
                recorded_checkpoint = Path(checkpoint_path).resolve()
                current_checkpoint = prediction_path.resolve()
            except OSError:
                recorded_checkpoint = Path(checkpoint_path)
                current_checkpoint = prediction_path
            if recorded_checkpoint != current_checkpoint:
                continue
        summaries.append((summary_path, payload))
    return summaries


def summarize_efficiency_metrics(
    expected_ids: Iterable[str],
    predictions: dict[str, CompactPrediction],
    prediction_paths: str | Path | Sequence[str | Path],
    *,
    kv_cache_pool_gib_per_gpu_override: float | None = None,
) -> dict[str, Any]:
    """Aggregate online-recorded trajectory metrics and run-level resources."""

    pool_override = _finite_float(kv_cache_pool_gib_per_gpu_override)
    if (
        kv_cache_pool_gib_per_gpu_override is not None
        and (pool_override is None or pool_override <= 0)
    ):
        raise ValueError(
            "kv_cache_pool_gib_per_gpu_override must be positive"
        )

    expected_list = [str(item_id) for item_id in expected_ids]
    matched = [
        predictions[item_id]
        for item_id in expected_list
        if item_id in predictions
    ]
    with_metrics = [
        prediction
        for prediction in matched
        if prediction.efficiency_metrics
    ]
    archive_modes = {
        bool(prediction.efficiency_metrics["archive_previous_images"])
        for prediction in with_metrics
        if isinstance(
            prediction.efficiency_metrics.get(
                "archive_previous_images"
            ),
            bool,
        )
    }
    metric_names = (
        "trajectory_elapsed_seconds",
        "llm_generate_seconds",
        "tool_seconds",
        "cumulative_prompt_tokens",
        "cumulative_visual_tokens",
        "cumulative_completion_tokens",
        "max_visual_tokens_per_call",
        "model_call_count",
        "tool_call_count",
    )
    distributions: dict[str, dict[str, Any]] = {}
    for metric_name in metric_names:
        values = [
            value
            for prediction in matched
            if (
                value := _finite_float(
                    prediction.efficiency_metrics.get(metric_name)
                )
            )
            is not None
        ]
        distributions[metric_name] = _distribution(
            values,
            expected_count=len(expected_list),
        )

    benchmark_summaries = _load_benchmark_run_summaries(prediction_paths)
    run_records: list[dict[str, Any]] = []
    total_processed = 0.0
    total_rollout_wall = 0.0
    idle_gpu_memory_gib: list[float] = []
    peak_gpu_memory_gib: list[float] = []
    incremental_peak_gpu_memory_gib: list[float] = []
    peak_kv_cache_usage: list[float] = []
    kv_cache_pool_gib_per_gpu: list[float] = []
    peak_active_kv_cache_memory_gib: list[float] = []
    for summary_path, summary in benchmark_summaries:
        processed = _finite_float(summary.get("processed_trajectories"))
        rollout_wall = _finite_float(summary.get("rollout_wall_seconds"))
        if processed is not None:
            total_processed += processed
        if rollout_wall is not None:
            total_rollout_wall += rollout_wall
        resource_metrics = summary.get("resource_metrics")
        if not isinstance(resource_metrics, dict):
            resource_metrics = {}
        peak_gpu = _finite_float(
            resource_metrics.get("peak_gpu_memory_per_gpu_gib")
        )
        idle_gpu = _finite_float(
            resource_metrics.get("idle_gpu_memory_per_gpu_gib")
        )
        incremental_peak_gpu = _finite_float(
            resource_metrics.get(
                "incremental_peak_gpu_memory_per_gpu_gib"
            )
        )
        peak_kv = _finite_float(
            resource_metrics.get("peak_kv_cache_usage")
        )
        kv_pool = _finite_float(
            resource_metrics.get("kv_cache_pool_gib_per_gpu")
        )
        if kv_pool is None:
            capacity = resource_metrics.get("kv_cache_capacity")
            if isinstance(capacity, dict):
                kv_pool = _finite_float(
                    capacity.get("pool_gib_per_gpu")
                )
        if kv_pool is None:
            kv_pool = pool_override
        peak_active_kv = _finite_float(
            resource_metrics.get(
                "peak_active_kv_cache_memory_per_gpu_gib"
            )
        )
        if (
            peak_active_kv is None
            and peak_kv is not None
            and kv_pool is not None
        ):
            peak_active_kv = peak_kv * kv_pool
        if idle_gpu is not None:
            idle_gpu_memory_gib.append(idle_gpu)
        if peak_gpu is not None:
            peak_gpu_memory_gib.append(peak_gpu)
        if incremental_peak_gpu is not None:
            incremental_peak_gpu_memory_gib.append(
                incremental_peak_gpu
            )
        if peak_kv is not None:
            peak_kv_cache_usage.append(peak_kv)
        if kv_pool is not None:
            kv_cache_pool_gib_per_gpu.append(kv_pool)
        if peak_active_kv is not None:
            peak_active_kv_cache_memory_gib.append(peak_active_kv)
        run_records.append(
            {
                "path": str(summary_path),
                "run_id": summary.get("run_id"),
                "mode": summary.get("mode"),
                "archive_previous_images": summary.get(
                    "archive_previous_images"
                ),
                "batch_size": summary.get("batch_size"),
                "max_rounds": summary.get("max_rounds"),
                "attempted_trajectories": summary.get(
                    "attempted_trajectories"
                ),
                "processed_trajectories": summary.get(
                    "processed_trajectories"
                ),
                "rollout_wall_seconds": summary.get(
                    "rollout_wall_seconds"
                ),
                "rollout_throughput_trajectories_per_minute": (
                    summary.get(
                        "rollout_throughput_trajectories_per_minute"
                    )
                ),
                "idle_gpu_memory_per_gpu_gib": idle_gpu,
                "peak_gpu_memory_per_gpu_gib": peak_gpu,
                "incremental_peak_gpu_memory_per_gpu_gib": (
                    incremental_peak_gpu
                ),
                "peak_kv_cache_usage": peak_kv,
                "peak_kv_cache_usage_percent": (
                    peak_kv * 100.0 if peak_kv is not None else None
                ),
                "kv_cache_pool_gib_per_gpu": kv_pool,
                "peak_active_kv_cache_memory_per_gpu_gib": (
                    peak_active_kv
                ),
            }
        )

    return {
        "enabled": bool(with_metrics),
        "definition": (
            "Per-trajectory efficiency includes every matched attempted "
            "trajectory, including failed and max-round trajectories. "
            "Rollout throughput excludes checkpoint serialization time."
        ),
        "expected_trajectories": len(expected_list),
        "matched_prediction_records": len(matched),
        "records_with_efficiency_metrics": len(with_metrics),
        "records_without_efficiency_metrics": (
            len(matched) - len(with_metrics)
        ),
        "archive_previous_images": (
            next(iter(archive_modes)) if len(archive_modes) == 1 else None
        ),
        "mixed_archive_modes": len(archive_modes) > 1,
        "per_trajectory": {
            "average_trajectory_elapsed_seconds": distributions[
                "trajectory_elapsed_seconds"
            ]["mean"],
            "p50_trajectory_elapsed_seconds": distributions[
                "trajectory_elapsed_seconds"
            ]["p50"],
            "p95_trajectory_elapsed_seconds": distributions[
                "trajectory_elapsed_seconds"
            ]["p95"],
            "average_llm_generate_seconds": distributions[
                "llm_generate_seconds"
            ]["mean"],
            "average_tool_seconds": distributions["tool_seconds"]["mean"],
            "average_cumulative_prompt_tokens": distributions[
                "cumulative_prompt_tokens"
            ]["mean"],
            "average_cumulative_visual_tokens": distributions[
                "cumulative_visual_tokens"
            ]["mean"],
            "average_cumulative_completion_tokens": distributions[
                "cumulative_completion_tokens"
            ]["mean"],
            "average_max_visual_tokens_per_call": distributions[
                "max_visual_tokens_per_call"
            ]["mean"],
            "average_model_calls": distributions["model_call_count"]["mean"],
            "average_tool_calls": distributions["tool_call_count"]["mean"],
            "distributions": distributions,
        },
        "run_level": {
            "available": bool(benchmark_summaries),
            "run_count": len(benchmark_summaries),
            "processed_trajectories": total_processed,
            "run_scope_matches_evaluation_records": (
                bool(benchmark_summaries)
                and int(total_processed) == len(matched)
            ),
            "rollout_wall_seconds": total_rollout_wall,
            "rollout_throughput_trajectories_per_second": (
                total_processed / total_rollout_wall
                if total_rollout_wall > 0
                else None
            ),
            "rollout_throughput_trajectories_per_minute": (
                total_processed * 60.0 / total_rollout_wall
                if total_rollout_wall > 0
                else None
            ),
            "peak_gpu_memory_per_gpu_gib": (
                max(peak_gpu_memory_gib)
                if peak_gpu_memory_gib
                else None
            ),
            "idle_gpu_memory_per_gpu_gib": (
                max(idle_gpu_memory_gib)
                if idle_gpu_memory_gib
                else None
            ),
            "incremental_peak_gpu_memory_per_gpu_gib": (
                max(incremental_peak_gpu_memory_gib)
                if incremental_peak_gpu_memory_gib
                else None
            ),
            "peak_kv_cache_usage": (
                max(peak_kv_cache_usage)
                if peak_kv_cache_usage
                else None
            ),
            "peak_kv_cache_usage_percent": (
                max(peak_kv_cache_usage) * 100.0
                if peak_kv_cache_usage
                else None
            ),
            "kv_cache_pool_gib_per_gpu": (
                kv_cache_pool_gib_per_gpu[0]
                if (
                    kv_cache_pool_gib_per_gpu
                    and all(
                        math.isclose(
                            value,
                            kv_cache_pool_gib_per_gpu[0],
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                        for value in kv_cache_pool_gib_per_gpu[1:]
                    )
                )
                else None
            ),
            "mixed_kv_cache_pool_capacities": (
                len(
                    {
                        round(value, 9)
                        for value in kv_cache_pool_gib_per_gpu
                    }
                )
                > 1
            ),
            "peak_active_kv_cache_memory_per_gpu_gib": (
                max(peak_active_kv_cache_memory_gib)
                if peak_active_kv_cache_memory_gib
                else None
            ),
            "runs": run_records,
        },
    }


def write_json(path_value: str | Path | None, payload: Any) -> None:
    if path_value is None:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path_value: str | Path | None, records: Iterable[dict[str, Any]]) -> None:
    if path_value is None:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_metrics(metrics: dict[str, Any]) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def validate_finite_metric(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {name} is not finite: {value}")
    return value
