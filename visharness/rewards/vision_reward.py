"""Trajectory-level rule reward for VisHarness."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

TRAJECTORY_STEP_COST = 0.2
TRAJECTORY_STEP_COST_ENABLED = True
EMPTY_CORRECT_REWARD = 5.0
MISSING_OR_FALSE_RESULT_PENALTY = -3.0

_REWARD_DEBUGGER_ATTACHED = False


def _maybe_wait_for_reward_debugger() -> None:
    """Optionally pause the reward worker so VSCode can attach to reward code."""
    global _REWARD_DEBUGGER_ATTACHED

    if os.getenv("VISHARNESS_DEBUG_REWARD", "0") != "1":
        return
    if _REWARD_DEBUGGER_ATTACHED and os.getenv("VISHARNESS_DEBUG_REWARD_ONCE", "1") != "0":
        return

    host = os.getenv("VISHARNESS_DEBUG_REWARD_HOST", "0.0.0.0")
    port = int(os.getenv("VISHARNESS_DEBUG_REWARD_PORT", "5683"))
    wait = os.getenv("VISHARNESS_DEBUG_REWARD_WAIT", "1") != "0"

    try:
        import debugpy
    except ImportError:
        print("[debug] debugpy is not installed in the reward worker; continuing without debugger.", flush=True)
        return

    try:
        debugpy.listen((host, port))
        print(f"[debug] VisHarness reward worker waiting for VSCode attach on {host}:{port}, pid={os.getpid()}", flush=True)
    except RuntimeError as exc:
        print(f"[debug] debugpy.listen({host}:{port}) in reward worker failed: {exc}", flush=True)
        if not debugpy.is_client_connected():
            return

    if wait and not debugpy.is_client_connected():
        debugpy.wait_for_client()
    _REWARD_DEBUGGER_ATTACHED = True
    debugpy.breakpoint()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def calculate_iou(mask1: np.ndarray | None, mask2: np.ndarray | None) -> float:
    """Calculate binary mask IoU, treating two empty masks as a perfect match."""
    if mask1 is None or mask2 is None or mask1.shape != mask2.shape:
        return 0.0

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        parsed = json.loads(ground_truth)
    elif isinstance(ground_truth, dict):
        parsed = ground_truth
    else:
        raise TypeError(f"ground_truth must be a JSON string or dict, got {type(ground_truth).__name__}")

    if not isinstance(parsed, dict) or "type" not in parsed or "data" not in parsed:
        raise ValueError("ground_truth must contain both 'type' and 'data'")
    return parsed


def _normalize_rles(rles: Any) -> list[dict[str, Any]]:
    """Normalize JSON-safe COCO RLEs before passing them to pycocotools."""
    if isinstance(rles, dict):
        rles = [rles]
    if not isinstance(rles, (list, tuple)):
        raise TypeError(f"Expected a COCO RLE or list of RLEs, got {type(rles).__name__}")

    normalized = copy.deepcopy(list(rles))
    for rle in normalized:
        if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
            raise ValueError("Each COCO RLE must contain 'size' and 'counts'")
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
    return normalized


def _decode_single_mask(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_utils.decode(_normalize_rles(rle))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def _decode_merged_prediction(final_results: dict[str, Any]) -> np.ndarray:
    rles = _normalize_rles(final_results["final_masks"])
    masks = mask_utils.decode(rles)
    if masks.ndim == 2:
        masks = masks[:, :, np.newaxis]
    return np.any(masks.astype(bool), axis=2)


def _resize_prediction_to_ground_truth(pred_mask: np.ndarray, gt_shape: tuple[int, int]) -> np.ndarray:
    if pred_mask.shape == gt_shape:
        return pred_mask

    pred_height, pred_width = pred_mask.shape
    gt_height, gt_width = gt_shape
    pred_ratio = pred_height / pred_width
    gt_ratio = gt_height / gt_width
    if abs(pred_ratio - gt_ratio) > 0.05:
        raise ValueError(
            "Prediction and ground-truth masks have incompatible aspect ratios: "
            f"prediction={pred_mask.shape}, ground_truth={gt_shape}"
        )

    resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (gt_width, gt_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _score_gres(final_results: dict[str, Any], ground_truth_data: dict[str, Any]) -> tuple[float, float]:
    gt_mask = _decode_single_mask(ground_truth_data)
    pred_mask = _resize_prediction_to_ground_truth(
        _decode_merged_prediction(final_results),
        gt_mask.shape,
    )
    iou = calculate_iou(pred_mask, gt_mask)
    return 8.0 * iou - 3.0, iou


def _score_reasonseg(final_results: dict[str, Any], ground_truth_data: dict[str, Any]) -> tuple[float, float]:
    gt_target = _decode_single_mask(ground_truth_data["target_rle"])
    gt_ignore = _decode_single_mask(ground_truth_data["ignore_rle"])
    pred_target = _resize_prediction_to_ground_truth(
        _decode_merged_prediction(final_results),
        gt_target.shape,
    )

    valid_area = ~gt_ignore
    intersection = np.logical_and(pred_target & valid_area, gt_target & valid_area).sum()
    union = np.logical_or(pred_target & valid_area, gt_target & valid_area).sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    return 8.0 * iou - 3.0, iou


def _reasonseg_ground_truth_is_empty(ground_truth_data: dict[str, Any]) -> bool:
    """Return whether ReasonSeg has no target pixels in the valid region."""

    gt_target = _decode_single_mask(ground_truth_data["target_rle"])
    gt_ignore = _decode_single_mask(ground_truth_data["ignore_rle"])
    if gt_target.shape != gt_ignore.shape:
        raise ValueError(
            "ReasonSeg target and ignore masks must have the same shape: "
            f"target={gt_target.shape}, ignore={gt_ignore.shape}"
        )
    return not bool(np.any(gt_target & ~gt_ignore))


def _ground_truth_is_empty(ground_truth: dict[str, Any]) -> bool:
    data_type = str(ground_truth["type"])
    if data_type == "gres_rle_mask":
        return ground_truth["data"] is None
    if data_type == "reasonseg_rle_mask":
        data = ground_truth["data"]
        if not isinstance(data, dict):
            raise TypeError("ReasonSeg ground-truth data must be a dictionary")
        return _reasonseg_ground_truth_is_empty(data)
    return False


def _score_point(final_results: dict[str, Any], ground_truth_data: list[Any]) -> tuple[float, int, int]:
    gt_count = len(ground_truth_data)
    pred_count = len(final_results["final_bboxes"])
    if gt_count == 0:
        reward = max(-3.0, 5.0 - pred_count * 2.0)
    else:
        relative_error = abs(gt_count - pred_count) / gt_count
        reward = max(-3.0, 5.0 - 8.0 * relative_error)
    return float(reward), gt_count, pred_count


def _has_visual_result(final_results: dict[str, Any] | None) -> bool:
    if not isinstance(final_results, dict):
        return False
    count = final_results.get("count")
    return bool(
        final_results.get("final_bboxes")
        or final_results.get("final_masks")
        or (isinstance(count, (int, float)) and count > 0)
    )


def _offline_aligned_eval_metrics(
    ground_truth: dict[str, Any],
    final_results: dict[str, Any] | None,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """Return sufficient statistics matching ``visharness.evaluate`` semantics."""

    evaluation_ground_truth = extra_info.get("evaluation_ground_truth")
    if evaluation_ground_truth is not None:
        evaluation_ground_truth = _parse_ground_truth(evaluation_ground_truth)
    else:
        evaluation_ground_truth = ground_truth

    data_type = str(evaluation_ground_truth["type"])
    original_width = int(
        extra_info.get("evaluation_original_width")
        or extra_info.get("image_width")
        or 0
    )
    original_height = int(
        extra_info.get("evaluation_original_height")
        or extra_info.get("image_height")
        or 0
    )
    has_visual_result = _has_visual_result(final_results)
    trajectory_finished = bool(extra_info.get("trajectory_finished", False))
    trajectory_invalid = bool(extra_info.get("trajectory_invalid", False))
    explicit_empty = trajectory_finished and not trajectory_invalid and not has_visual_result

    result: dict[str, Any] = {
        "eval_intersection": float("nan"),
        "eval_union": float("nan"),
        "eval_iou": float("nan"),
        "eval_reward": float("nan"),
        "eval_ground_truth_empty": False,
        "eval_empty_correct": False,
        "eval_gt_count": float("nan"),
        "eval_pred_count": float("nan"),
        "eval_absolute_error": float("nan"),
        "eval_squared_error": float("nan"),
        "eval_exact_count": float("nan"),
        "eval_tp": float("nan"),
        "eval_fp": float("nan"),
        "eval_fn": float("nan"),
        "eval_explicit_empty_prediction": explicit_empty,
        "eval_has_visual_result": has_visual_result,
    }
    if not bool(extra_info.get("validation_sample", False)):
        return result
    if has_visual_result:
        from visharness.evaluate.common import CompactPrediction, validate_final_result_consistency

        validate_final_result_consistency(
            CompactPrediction(
                item_id=str(extra_info.get("item_id") or "<validation-sample>"),
                final_bboxes=list(final_results.get("final_bboxes") or []),
                final_masks=list(final_results.get("final_masks") or []),
                count=(
                    int(final_results["count"])
                    if final_results.get("count") is not None
                    else None
                ),
            )
        )

    if data_type in {"gres_rle_mask", "reasonseg_rle_mask"}:
        from visharness.evaluate.common import restore_mask_to_original

        if data_type == "gres_rle_mask":
            is_empty = evaluation_ground_truth["data"] is None
            if is_empty:
                if original_width <= 0 or original_height <= 0:
                    raise ValueError("Original image size is required for empty GRES validation ground truth")
                gt_target = np.zeros((original_height, original_width), dtype=bool)
            else:
                gt_target = _decode_single_mask(evaluation_ground_truth["data"])
            gt_ignore = np.zeros_like(gt_target, dtype=bool)
        else:
            gt_target = _decode_single_mask(evaluation_ground_truth["data"]["target_rle"])
            gt_ignore = _decode_single_mask(evaluation_ground_truth["data"]["ignore_rle"])
            if gt_target.shape != gt_ignore.shape:
                raise ValueError(
                    "ReasonSeg target and ignore masks must have the same shape: "
                    f"target={gt_target.shape}, ignore={gt_ignore.shape}"
                )
            is_empty = not bool(np.any(gt_target & ~gt_ignore))

        if original_width <= 0 or original_height <= 0:
            original_height, original_width = gt_target.shape
        if gt_target.shape != (original_height, original_width):
            raise ValueError(
                "Evaluation ground-truth size does not match original image size: "
                f"ground_truth={gt_target.shape}, image={(original_height, original_width)}"
            )

        if has_visual_result:
            pred_target = restore_mask_to_original(
                final_results["final_masks"],
                original_width=original_width,
                original_height=original_height,
            )
        else:
            pred_target = np.zeros_like(gt_target, dtype=bool)

        valid_area = ~gt_ignore
        intersection = int(np.logical_and(pred_target & valid_area, gt_target & valid_area).sum())
        union = int(np.logical_or(pred_target & valid_area, gt_target & valid_area).sum())
        if is_empty:
            # A failed or max-turn trajectory is not a correct null prediction.
            iou = 1.0 if explicit_empty else 0.0
        else:
            iou = 1.0 if union == 0 else float(intersection / union)

        result.update(
            {
                "eval_intersection": intersection,
                "eval_union": union,
                "eval_iou": iou,
                "eval_reward": 8.0 * iou - 3.0,
                "eval_ground_truth_empty": is_empty,
                "eval_empty_correct": bool(is_empty and explicit_empty),
            }
        )
        return result

    if data_type == "point":
        from visharness.evaluate.common import restore_boxes_to_original
        from visharness.evaluate.evaluate_rec8k import _localization_matches

        gt_points = evaluation_ground_truth["data"]
        if not isinstance(gt_points, list):
            raise ValueError("Point evaluation ground truth must be a list")
        if has_visual_result:
            if original_width <= 0 or original_height <= 0:
                raise ValueError("Original image size is required for REC-8K validation")
            pred_boxes = restore_boxes_to_original(
                final_results["final_bboxes"],
                final_results["final_masks"],
                original_width=original_width,
                original_height=original_height,
            )
        else:
            pred_boxes = []

        gt_count = len(gt_points)
        pred_count = len(pred_boxes)
        absolute_error = abs(pred_count - gt_count)
        squared_error = (pred_count - gt_count) ** 2
        true_positives = int(_localization_matches(pred_boxes, gt_points))
        if gt_count == 0:
            eval_reward = max(-3.0, 5.0 - pred_count * 2.0)
        else:
            eval_reward = max(-3.0, 5.0 - 8.0 * absolute_error / gt_count)
        result.update(
            {
                "eval_gt_count": gt_count,
                "eval_pred_count": pred_count,
                "eval_absolute_error": absolute_error,
                "eval_squared_error": squared_error,
                "eval_exact_count": float(pred_count == gt_count),
                "eval_reward": float(eval_reward),
                "eval_tp": true_positives,
                "eval_fp": pred_count - true_positives,
                "eval_fn": gt_count - true_positives,
            }
        )
        return result

    raise ValueError(f"Unsupported evaluation ground-truth type: {data_type}")


def _score_final_result(
    ground_truth: dict[str, Any],
    final_results: dict[str, Any] | None,
    *,
    trajectory_finished: bool,
    trajectory_invalid: bool,
) -> dict[str, Any]:
    if _ground_truth_is_empty(ground_truth):
        correct_empty_prediction = (
            final_results is None
            and trajectory_finished
            and not trajectory_invalid
        )
        return {
            "task_reward": (
                EMPTY_CORRECT_REWARD
                if correct_empty_prediction
                else MISSING_OR_FALSE_RESULT_PENALTY
            ),
            "task_metric": 1.0 if correct_empty_prediction else 0.0,
            "ground_truth_empty": True,
        }

    if final_results is None:
        return {
            "task_reward": MISSING_OR_FALSE_RESULT_PENALTY,
            "task_metric": 0.0,
            "ground_truth_empty": False,
        }

    data_type = ground_truth["type"]
    if data_type == "gres_rle_mask":
        task_reward, iou = _score_gres(final_results, ground_truth["data"])
        return {
            "task_reward": task_reward,
            "task_metric": iou,
            "iou": iou,
            "ground_truth_empty": False,
        }
    if data_type == "reasonseg_rle_mask":
        task_reward, iou = _score_reasonseg(final_results, ground_truth["data"])
        return {
            "task_reward": task_reward,
            "task_metric": iou,
            "iou": iou,
            "ground_truth_empty": False,
        }
    if data_type == "point":
        task_reward, gt_count, pred_count = _score_point(final_results, ground_truth["data"])
        return {
            "task_reward": task_reward,
            "task_metric": max(0.0, 1.0 - abs(gt_count - pred_count) / max(gt_count, 1)),
            "ground_truth_empty": False,
            "gt_count": gt_count,
            "pred_count": pred_count,
        }
    raise ValueError(f"Unsupported VisHarness ground-truth type: {data_type}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    trajectory_step_cost_enabled: bool | None = None,
    trajectory_step_cost: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Score one complete VisHarness trajectory.

    ``turn_records`` and ``final_results`` are emitted by
    :class:`VisHarnessAgentLoop` and forwarded through verl's naive reward
    manager as ``tool_extra_fields``.
    """
    # _maybe_wait_for_reward_debugger()

    del data_source
    extra_info = extra_info or {}
    turn_records = extra_info.get("turn_records") or []
    final_results = extra_info.get("final_results")
    parsed_ground_truth = _parse_ground_truth(ground_truth)

    step_cost_enabled = (
        bool(trajectory_step_cost_enabled)
        if trajectory_step_cost_enabled is not None
        else _env_bool(
            "VISHARNESS_TRAJECTORY_STEP_COST_ENABLE",
            TRAJECTORY_STEP_COST_ENABLED,
        )
    )
    configured_step_cost_coef = abs(
        float(trajectory_step_cost)
        if trajectory_step_cost is not None
        else _env_float("VISHARNESS_TRAJECTORY_STEP_COST", TRAJECTORY_STEP_COST)
    )
    trajectory_step_cost_coef = configured_step_cost_coef if step_cost_enabled else 0.0
    num_assistant_turns = len(turn_records)
    trajectory_step_cost = trajectory_step_cost_coef * num_assistant_turns
    raw_output_format_failures = sum(
        not bool(turn.get("output_format_success", False)) for turn in turn_records
    )
    raw_tool_args_failures = sum(
        bool(turn.get("output_format_success", False))
        and not bool(turn.get("tool_args_success", True))
        for turn in turn_records
    )
    truncated_turns = sum(bool(turn.get("turn_truncated_by_length", False)) for turn in turn_records)
    output_format_errors = sum(
        not bool(turn.get("turn_truncated_by_length", False))
        and not bool(turn.get("output_format_success", False))
        for turn in turn_records
    )
    tool_args_errors = sum(
        not bool(turn.get("turn_truncated_by_length", False))
        and bool(turn.get("output_format_success", False))
        and not bool(turn.get("tool_args_success", True))
        for turn in turn_records
    )

    del solution_str

    trajectory_finished = bool(extra_info.get("trajectory_finished", False))
    trajectory_invalid = bool(extra_info.get("trajectory_invalid", False))
    result_metrics = _score_final_result(
        parsed_ground_truth,
        final_results,
        trajectory_finished=trajectory_finished,
        trajectory_invalid=trajectory_invalid,
    )
    evaluation_metrics = _offline_aligned_eval_metrics(
        parsed_ground_truth,
        final_results,
        extra_info,
    )
    trajectory_reward = float(result_metrics["task_reward"]) - trajectory_step_cost
    reward_info = {
        "score": float(result_metrics["task_reward"]),
        "task_reward": float(result_metrics["task_reward"]),
        "trajectory_reward": float(trajectory_reward),
        "trajectory_step_cost": float(trajectory_step_cost),
        "trajectory_step_cost_enabled": bool(step_cost_enabled),
        "trajectory_step_cost_coef": float(trajectory_step_cost_coef),
        "output_format_error_count": int(output_format_errors),
        "raw_output_format_failure_count": int(raw_output_format_failures),
        "tool_args_error_count": int(tool_args_errors),
        "raw_tool_args_failure_count": int(raw_tool_args_failures),
        "truncated_turn_count": int(truncated_turns),
        "output_format_success_rate": (
            float((num_assistant_turns - raw_output_format_failures) / num_assistant_turns)
            if num_assistant_turns
            else 0.0
        ),
        "tool_args_success_rate": (
            float(
                (
                    num_assistant_turns
                    - raw_output_format_failures
                    - raw_tool_args_failures
                )
                / num_assistant_turns
            )
            if num_assistant_turns
            else 0.0
        ),
        "output_format_error_rate": (
            float(output_format_errors / num_assistant_turns) if num_assistant_turns else 0.0
        ),
        "tool_args_error_rate": (
            float(tool_args_errors / num_assistant_turns) if num_assistant_turns else 0.0
        ),
        "raw_output_format_failure_rate": (
            float(raw_output_format_failures / num_assistant_turns)
            if num_assistant_turns
            else 0.0
        ),
        "raw_tool_args_failure_rate": (
            float(raw_tool_args_failures / num_assistant_turns)
            if num_assistant_turns
            else 0.0
        ),
        "truncated_turn_rate": float(truncated_turns / num_assistant_turns) if num_assistant_turns else 0.0,
        "num_assistant_turns": num_assistant_turns,
        "task_type": parsed_ground_truth["type"],
        "validation_sample": bool(extra_info.get("validation_sample", False)),
        "trajectory_finished": trajectory_finished,
        "trajectory_invalid": trajectory_invalid,
        "trajectory_aborted": bool(extra_info.get("trajectory_aborted", False)),
        "max_agent_turns_reached": bool(extra_info.get("max_agent_turns_reached", False)),
        "invalid_reason": extra_info.get("invalid_reason"),
        "tool_call_count": int(extra_info.get("tool_call_count") or 0),
        "assistant_response_token_count": int(
            extra_info.get("assistant_response_token_count")
            or sum(int(turn.get("turn_response_length") or 0) for turn in turn_records)
        ),
        "observation_token_count": int(
            extra_info.get("observation_token_count")
            or sum(int(turn.get("observation_token_length") or 0) for turn in turn_records)
        ),
        "trajectory_response_token_count": int(
            extra_info.get("trajectory_response_length")
            or sum(
                int(turn.get("turn_response_length") or 0)
                + int(turn.get("observation_token_length") or 0)
                for turn in turn_records
            )
        ),
        "generate_time_seconds": float(extra_info.get("generate_time_seconds") or 0.0),
        "tool_time_seconds": float(extra_info.get("tool_time_seconds") or 0.0),
        "trajectory_elapsed_seconds": float(extra_info.get("trajectory_elapsed_seconds") or 0.0),
        **evaluation_metrics,
        **result_metrics,
    }
    for optional_metric in ("iou", "gt_count", "pred_count"):
        reward_info.setdefault(optional_metric, float("nan"))
    return reward_info
