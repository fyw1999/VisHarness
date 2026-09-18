"""Per-trajectory acceptance scoring for generated SFT data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visharness.evaluate.common import (
    CompactPrediction,
    TERMINATION_ANSWER,
    load_exif_transposed_image_size,
    restore_mask_to_original,
    validate_final_result_consistency,
)
from visharness.evaluate.evaluate_gres import _load_ref, _parse_ref_id
from visharness.evaluate.evaluate_reasonseg import get_mask_from_json
from visharness.evaluate.grefer import G_REFER

from .schema import SFTFilterDecision


@dataclass(frozen=True, slots=True)
class FilterThresholds:
    gres_iou: float = 0.7
    reasonseg_iou: float = 0.7
    rec8k_relative_count_error: float = 0.3
    aspect_ratio_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for name in ("gres_iou", "reasonseg_iou"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        for name in ("rec8k_relative_count_error", "aspect_ratio_tolerance"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _prediction_kind(prediction: CompactPrediction) -> str:
    if prediction.has_visual_result:
        return "visual_result"
    if prediction.explicit_empty_prediction:
        return "explicit_empty"
    return prediction.termination_reason


def _rejected_lifecycle(prediction: CompactPrediction) -> SFTFilterDecision | None:
    # A task metric is only meaningful for a completed agent trajectory.  A
    # visual result left behind by MergeBoxMask is not sufficient when the
    # model subsequently exhausts its turn budget without submitting the
    # final answer.  Conversely, an answer submitted on the last available
    # turn still has TERMINATION_ANSWER and is eligible for metric scoring.
    if prediction.termination_reason == TERMINATION_ANSWER:
        return None
    return SFTFilterDecision(
        trajectory_id=prediction.item_id,
        accepted=False,
        task=_task_name(prediction.item_id),
        reason=f"trajectory_{prediction.termination_reason}",
        termination_reason=prediction.termination_reason,
        prediction_kind=_prediction_kind(prediction),
    )


def _task_name(item_id: str) -> str:
    if item_id.startswith("REC8K-"):
        return "REC8K"
    if item_id.startswith(("GRES-", "GERS-")):
        return "GRES"
    if item_id.startswith("ReasonSeg-"):
        return "ReasonSeg"
    raise ValueError(f"Unsupported trajectory id prefix: {item_id!r}")


def _mask_iou(prediction_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction_mask, target_mask).sum())
    union = int(np.logical_or(prediction_mask, target_mask).sum())
    return intersection / union if union else 0.0


class TrajectoryAcceptanceScorer:
    """Lazily load task annotations and score one compact prediction."""

    def __init__(
        self,
        *,
        rec8k_annotations_path: str | Path | None = None,
        gres_dataset_root: str | Path | None = None,
        reasonseg_dataset_root: str | Path | None = None,
        thresholds: FilterThresholds | None = None,
    ) -> None:
        self.rec8k_annotations_path = (
            Path(rec8k_annotations_path) if rec8k_annotations_path else None
        )
        self.gres_dataset_root = Path(gres_dataset_root) if gres_dataset_root else None
        self.reasonseg_dataset_root = (
            Path(reasonseg_dataset_root) if reasonseg_dataset_root else None
        )
        self.thresholds = thresholds or FilterThresholds()
        self._rec8k_annotations: dict[str, Any] | None = None
        self._gref: G_REFER | None = None

    def score(self, prediction: CompactPrediction) -> SFTFilterDecision:
        lifecycle_rejection = _rejected_lifecycle(prediction)
        if lifecycle_rejection is not None:
            return lifecycle_rejection
        task = _task_name(prediction.item_id)
        if task == "REC8K":
            return self._score_rec8k(prediction)
        if task == "GRES":
            return self._score_gres(prediction)
        return self._score_reasonseg(prediction)

    def _rec8k_annotation(self, item_id: str) -> dict[str, Any]:
        if self.rec8k_annotations_path is None:
            raise ValueError("REC8K filtering requires rec8k_annotations_path")
        if self._rec8k_annotations is None:
            self._rec8k_annotations = _load_json_object(self.rec8k_annotations_path)

        rest = item_id.removeprefix("REC8K-")
        candidates: list[tuple[str, str]] = []
        if "-" in rest:
            image_name, phrase = rest.rsplit("-", 1)
            candidates.extend(
                (f"{image_name}{suffix}", " ".join(phrase.split("_")))
                for suffix in (".jpg", ".png")
            )
        parts = rest.split("-")
        for split_index in range(len(parts) - 2, 0, -1):
            image_name = "-".join(parts[:split_index])
            phrase = " ".join("-".join(parts[split_index:]).split("_"))
            candidates.extend(
                (f"{image_name}{suffix}", phrase)
                for suffix in (".jpg", ".png")
            )
        for image_name, phrase in candidates:
            image_annotations = self._rec8k_annotations.get(image_name)
            if isinstance(image_annotations, dict) and phrase in image_annotations:
                annotation = image_annotations[phrase]
                if not isinstance(annotation, dict):
                    raise ValueError(
                        f"Invalid REC8K annotation for {image_name!r}, {phrase!r}"
                    )
                return annotation
        raise ValueError(f"Unable to match REC8K annotation for {item_id}")

    def _score_rec8k(self, prediction: CompactPrediction) -> SFTFilterDecision:
        annotation = self._rec8k_annotation(prediction.item_id)
        points = annotation.get("points", [])
        if not isinstance(points, list):
            raise ValueError(f"REC8K points must be a list for {prediction.item_id}")
        gt_count = len(points)
        pred_count = len(prediction.final_bboxes) if prediction.has_visual_result else 0
        kind = _prediction_kind(prediction)
        if gt_count == 0:
            accepted = prediction.explicit_empty_prediction
            return SFTFilterDecision(
                trajectory_id=prediction.item_id,
                accepted=accepted,
                task="REC8K",
                reason="passed" if accepted else "empty_target_not_explicitly_answered",
                metric="empty_target_accuracy",
                value=1.0 if accepted else 0.0,
                threshold=1.0,
                termination_reason=prediction.termination_reason,
                prediction_kind=kind,
            )
        if prediction.has_visual_result:
            validate_final_result_consistency(prediction)
        relative_error = abs(pred_count - gt_count) / gt_count
        accepted = prediction.has_visual_result and (
            relative_error <= self.thresholds.rec8k_relative_count_error
        )
        return SFTFilterDecision(
            trajectory_id=prediction.item_id,
            accepted=accepted,
            task="REC8K",
            reason="passed" if accepted else "relative_count_error_above_threshold",
            metric="relative_count_error",
            value=float(relative_error),
            threshold=self.thresholds.rec8k_relative_count_error,
            termination_reason=prediction.termination_reason,
            prediction_kind=kind,
        )

    def _get_gref(self) -> G_REFER:
        if self.gres_dataset_root is None:
            raise ValueError("GRES filtering requires gres_dataset_root")
        if self._gref is None:
            self._gref = G_REFER(
                str(self.gres_dataset_root),
                dataset="grefcoco",
                splitBy="unc",
            )
        return self._gref

    def _score_gres(self, prediction: CompactPrediction) -> SFTFilterDecision:
        gref = self._get_gref()
        ref = _load_ref(gref, _parse_ref_id(prediction.item_id))
        mask_info = gref.getMaskByRef(ref=ref, merge=True)
        image_info = gref.Imgs[ref["image_id"]]
        width = int(image_info["width"])
        height = int(image_info["height"])
        is_empty = bool(mask_info.get("empty", False))
        kind = _prediction_kind(prediction)
        if is_empty:
            accepted = prediction.explicit_empty_prediction
            return SFTFilterDecision(
                trajectory_id=prediction.item_id,
                accepted=accepted,
                task="GRES",
                reason="passed" if accepted else "empty_target_not_explicitly_answered",
                metric="iou",
                value=1.0 if accepted else 0.0,
                threshold=self.thresholds.gres_iou,
                termination_reason=prediction.termination_reason,
                prediction_kind=kind,
            )
        if not prediction.has_visual_result:
            iou = 0.0
        else:
            validate_final_result_consistency(prediction)
            prediction_mask = restore_mask_to_original(
                prediction.final_masks,
                original_width=width,
                original_height=height,
                aspect_ratio_tolerance=self.thresholds.aspect_ratio_tolerance,
            )
            target_mask = np.asarray(mask_info["mask"]) > 0
            iou = _mask_iou(prediction_mask, target_mask)
        accepted = iou >= self.thresholds.gres_iou
        return SFTFilterDecision(
            trajectory_id=prediction.item_id,
            accepted=accepted,
            task="GRES",
            reason="passed" if accepted else "iou_below_threshold",
            metric="iou",
            value=float(iou),
            threshold=self.thresholds.gres_iou,
            termination_reason=prediction.termination_reason,
            prediction_kind=kind,
        )

    def _score_reasonseg(self, prediction: CompactPrediction) -> SFTFilterDecision:
        if self.reasonseg_dataset_root is None:
            raise ValueError("ReasonSeg filtering requires reasonseg_dataset_root")
        base_name = prediction.item_id.removeprefix("ReasonSeg-")
        annotation_path = self.reasonseg_dataset_root / f"{base_name}.json"
        image_path = next(
            (
                path
                for suffix in (".jpg", ".png", ".jpeg")
                if (path := self.reasonseg_dataset_root / f"{base_name}{suffix}").is_file()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(
                f"Unable to find ReasonSeg image for {prediction.item_id} in {self.reasonseg_dataset_root}"
            )
        width, height = load_exif_transposed_image_size(image_path)
        gt_mask = get_mask_from_json(annotation_path, (height, width))
        valid_area = gt_mask != 255
        target_mask = (gt_mask == 1) & valid_area
        is_empty = not bool(np.any(target_mask))
        kind = _prediction_kind(prediction)
        if is_empty:
            accepted = prediction.explicit_empty_prediction
            return SFTFilterDecision(
                trajectory_id=prediction.item_id,
                accepted=accepted,
                task="ReasonSeg",
                reason="passed" if accepted else "empty_target_not_explicitly_answered",
                metric="iou",
                value=1.0 if accepted else 0.0,
                threshold=self.thresholds.reasonseg_iou,
                termination_reason=prediction.termination_reason,
                prediction_kind=kind,
            )
        if not prediction.has_visual_result:
            iou = 0.0
        else:
            validate_final_result_consistency(prediction)
            prediction_mask = restore_mask_to_original(
                prediction.final_masks,
                original_width=width,
                original_height=height,
                aspect_ratio_tolerance=self.thresholds.aspect_ratio_tolerance,
            )
            iou = _mask_iou(prediction_mask & valid_area, target_mask)
        accepted = iou >= self.thresholds.reasonseg_iou
        return SFTFilterDecision(
            trajectory_id=prediction.item_id,
            accepted=accepted,
            task="ReasonSeg",
            reason="passed" if accepted else "iou_below_threshold",
            metric="iou",
            value=float(iou),
            threshold=self.thresholds.reasonseg_iou,
            termination_reason=prediction.termination_reason,
            prediction_kind=kind,
        )
