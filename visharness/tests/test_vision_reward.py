import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from visharness.rewards.vision_reward import compute_score


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def turn(
    completion="<think>done</think><answer>done</answer>",
    output_ok=True,
    tool_ok=True,
    truncated=False,
):
    return {
        "completion": completion,
        "output_format_success": output_ok,
        "tool_args_success": tool_ok,
        "turn_truncated_by_length": truncated,
    }


def final_results(mask=None, count=1):
    mask = np.ones((4, 4), dtype=np.uint8) if mask is None else mask
    return {
        "final_bboxes": [[0, 0, 3, 3]] * count,
        "final_masks": [encode_mask(mask)] if count else [],
        "count": count,
    }


def test_empty_ground_truth_rewards_no_result_and_charges_step_cost():
    result = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth=json.dumps({"type": "gres_rle_mask", "data": None}),
        extra_info={
            "turn_records": [turn()],
            "final_results": None,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
    )

    assert result["task_reward"] == 5.0
    assert result["trajectory_step_cost"] == 0.2
    assert result["trajectory_reward"] == 4.8
    assert result["score"] == pytest.approx(5.0)
    assert "legacy_total_score" not in result


def test_local_errors_are_diagnostics_not_trajectory_reward_terms():
    result = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth={"type": "gres_rle_mask", "data": None},
        extra_info={
            "turn_records": [
                turn("<think>bad</think>", output_ok=False, tool_ok=False),
                turn("<think>still no answer</think>", output_ok=True, tool_ok=True),
            ],
            "final_results": None,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
    )

    assert result["trajectory_step_cost"] == pytest.approx(0.4)
    assert result["trajectory_reward"] == pytest.approx(4.6)
    assert result["output_format_error_count"] == 1
    assert result["tool_args_error_count"] == 0
    assert result["truncated_turn_count"] == 0
    assert "answer_format_penalty" not in result
    assert "answer_format_success" not in result
    assert result["score"] == pytest.approx(5.0)
    assert "legacy_total_score" not in result


def test_truncation_is_not_double_counted_as_exclusive_format_error():
    result = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth={"type": "gres_rle_mask", "data": None},
        extra_info={
            "turn_records": [turn(output_ok=False, tool_ok=False, truncated=True)],
            "final_results": None,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
    )

    assert result["truncated_turn_count"] == 1
    assert result["output_format_error_count"] == 0
    assert result["raw_output_format_failure_count"] == 1
    assert result["tool_args_error_count"] == 0
    assert result["output_format_error_rate"] == 0.0
    assert result["raw_output_format_failure_rate"] == 1.0
    # Parser success remains a raw diagnostic rather than the complement of
    # the mutually exclusive format-error category.
    assert result["output_format_success_rate"] == 0.0


def test_trajectory_step_cost_can_be_disabled_without_changing_its_configured_value():
    result = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth={"type": "gres_rle_mask", "data": None},
        extra_info={
            "turn_records": [turn(), turn()],
            "final_results": None,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
        trajectory_step_cost_enabled=False,
        trajectory_step_cost=0.2,
    )

    assert result["task_reward"] == 5.0
    assert result["trajectory_step_cost_enabled"] is False
    assert result["trajectory_step_cost_coef"] == 0.0
    assert result["trajectory_step_cost"] == 0.0
    assert result["trajectory_reward"] == 5.0


def test_gres_uses_soft_iou_reward():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :] = 1
    result = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth={"type": "gres_rle_mask", "data": encode_mask(mask)},
        extra_info={"turn_records": [turn()], "final_results": final_results(mask)},
    )

    assert result["iou"] == 1.0
    assert result["task_reward"] == 5.0
    assert result["score"] == pytest.approx(5.0)
    assert "legacy_total_score" not in result


def test_reasonseg_ignores_prediction_errors_inside_ignore_region():
    target = np.zeros((4, 4), dtype=np.uint8)
    target[:2, :2] = 1
    ignore = np.zeros((4, 4), dtype=np.uint8)
    ignore[2:, :] = 1
    prediction = target | ignore

    result = compute_score(
        data_source="visharness/reasonseg",
        solution_str="",
        ground_truth={
            "type": "reasonseg_rle_mask",
            "data": {
                "target_rle": encode_mask(target),
                "ignore_rle": encode_mask(ignore),
            },
        },
        extra_info={"turn_records": [turn()], "final_results": final_results(prediction)},
    )

    assert result["iou"] == 1.0
    assert result["task_reward"] == 5.0


def test_point_reward_uses_relative_count_error():
    result = compute_score(
        data_source="visharness/rec8k",
        solution_str="",
        ground_truth={"type": "point", "data": [[1, 1], [2, 2], [3, 3], [4, 4]]},
        extra_info={"turn_records": [turn()], "final_results": final_results(count=3)},
    )

    assert result["gt_count"] == 4
    assert result["pred_count"] == 3
    assert result["task_reward"] == pytest.approx(3.0)
    assert result["score"] == pytest.approx(3.0)
    assert "legacy_total_score" not in result


def test_non_empty_ground_truth_without_final_result_gets_false_negative_penalty():
    result = compute_score(
        data_source="visharness/rec8k",
        solution_str="<think>done</think><answer>none</answer>",
        ground_truth={"type": "point", "data": [[1, 1]]},
        extra_info={"turn_records": [], "final_results": None},
    )

    assert result["task_reward"] == -3.0
    assert result["score"] == -3.0


def test_gres_empty_reward_requires_successful_valid_final_answer():
    ground_truth = {"type": "gres_rle_mask", "data": None}
    common_extra = {
        "validation_sample": True,
        "evaluation_ground_truth": json.dumps(ground_truth),
        "evaluation_original_width": 4,
        "evaluation_original_height": 4,
        "turn_records": [turn()],
        "final_results": None,
    }

    unfinished = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": False,
            "trajectory_invalid": False,
            "max_agent_turns_reached": True,
        },
    )
    finished = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
    )
    invalid = compute_score(
        data_source="visharness/gres",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": False,
            "trajectory_invalid": True,
            "invalid_reason": "rollout_prompt_overlong",
        },
    )

    assert unfinished["task_reward"] == -3.0
    assert unfinished["task_metric"] == 0.0
    assert unfinished["eval_iou"] == 0.0
    assert unfinished["eval_reward"] == -3.0
    assert unfinished["eval_empty_correct"] is False
    assert unfinished["max_agent_turns_reached"] is True

    assert finished["task_reward"] == 5.0
    assert finished["task_metric"] == 1.0
    assert finished["eval_iou"] == 1.0
    assert finished["eval_reward"] == 5.0
    assert finished["eval_empty_correct"] is True

    assert invalid["task_reward"] == -3.0
    assert invalid["task_metric"] == 0.0
    assert invalid["eval_iou"] == 0.0
    assert invalid["eval_reward"] == -3.0
    assert invalid["eval_empty_correct"] is False


def test_reasonseg_empty_reward_requires_successful_valid_final_answer():
    empty_mask = np.zeros((4, 4), dtype=np.uint8)
    ground_truth = {
        "type": "reasonseg_rle_mask",
        "data": {
            "target_rle": encode_mask(empty_mask),
            "ignore_rle": encode_mask(empty_mask),
        },
    }
    common_extra = {
        "validation_sample": True,
        "evaluation_ground_truth": json.dumps(ground_truth),
        "evaluation_original_width": 4,
        "evaluation_original_height": 4,
        "turn_records": [turn()],
        "final_results": None,
    }

    unfinished = compute_score(
        data_source="visharness/reasonseg",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": False,
            "trajectory_invalid": False,
            "max_agent_turns_reached": True,
        },
    )
    finished = compute_score(
        data_source="visharness/reasonseg",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": True,
            "trajectory_invalid": False,
        },
    )
    invalid = compute_score(
        data_source="visharness/reasonseg",
        solution_str="",
        ground_truth=ground_truth,
        extra_info={
            **common_extra,
            "trajectory_finished": False,
            "trajectory_invalid": True,
            "invalid_reason": "rollout_prompt_overlong",
        },
    )

    assert unfinished["task_reward"] == -3.0
    assert unfinished["task_metric"] == 0.0
    assert unfinished["ground_truth_empty"] is True
    assert unfinished["eval_iou"] == 0.0
    assert unfinished["eval_reward"] == -3.0
    assert unfinished["eval_ground_truth_empty"] is True
    assert unfinished["eval_empty_correct"] is False

    assert finished["task_reward"] == 5.0
    assert finished["task_metric"] == 1.0
    assert finished["ground_truth_empty"] is True
    assert finished["eval_iou"] == 1.0
    assert finished["eval_reward"] == 5.0
    assert finished["eval_ground_truth_empty"] is True
    assert finished["eval_empty_correct"] is True

    assert invalid["task_reward"] == -3.0
    assert invalid["task_metric"] == 0.0
    assert invalid["eval_iou"] == 0.0
    assert invalid["eval_reward"] == -3.0
    assert invalid["eval_empty_correct"] is False


def test_rec8k_validation_restores_boxes_before_localization_matching():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    prediction = {
        "final_bboxes": [[0, 0, 2, 2]],
        "final_masks": [encode_mask(mask)],
        "count": 1,
    }
    original_ground_truth = {"type": "point", "data": [[2.0, 2.0]]}

    result = compute_score(
        data_source="visharness/rec8k",
        solution_str="",
        ground_truth={"type": "point", "data": [[1.0, 1.0]]},
        extra_info={
            "validation_sample": True,
            "evaluation_ground_truth": json.dumps(original_ground_truth),
            "evaluation_original_width": 8,
            "evaluation_original_height": 8,
            "trajectory_finished": True,
            "turn_records": [turn()],
            "final_results": prediction,
        },
    )

    assert result["eval_gt_count"] == 1
    assert result["eval_pred_count"] == 1
    assert result["eval_absolute_error"] == 0
    assert result["eval_tp"] == 1
    assert result["eval_fp"] == 0
    assert result["eval_fn"] == 0
