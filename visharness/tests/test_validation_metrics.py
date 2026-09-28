import json
import math

import pytest

from visharness.rl_trainer.validation_metrics import (
    summarize_validation,
    write_validation_artifacts,
)


def test_validation_metrics_use_dataset_level_sufficient_statistics():
    task_types = [
        "gres_rle_mask",
        "gres_rle_mask",
        "reasonseg_rle_mask",
        "point",
        "point",
    ]
    extras = {
        "task_type": task_types,
        "task_reward": [5.0, -3.0, 1.0, 5.0, 1.0],
        "trajectory_reward": [4.8, -3.2, 0.8, 4.8, 0.8],
        "trajectory_step_cost": [0.2] * 5,
        "eval_iou": [1.0, 0.25, 0.5, float("nan"), float("nan")],
        "eval_intersection": [0, 1, 3, float("nan"), float("nan")],
        "eval_union": [0, 4, 6, float("nan"), float("nan")],
        "eval_ground_truth_empty": [True, False, False, False, False],
        "eval_empty_correct": [True, False, False, False, False],
        "eval_gt_count": [float("nan")] * 3 + [2, 4],
        "eval_pred_count": [float("nan")] * 3 + [2, 6],
        "eval_absolute_error": [float("nan")] * 3 + [0, 2],
        "eval_squared_error": [float("nan")] * 3 + [0, 4],
        "eval_exact_count": [float("nan")] * 3 + [1, 0],
        "eval_tp": [float("nan")] * 3 + [2, 3],
        "eval_fp": [float("nan")] * 3 + [0, 3],
        "eval_fn": [float("nan")] * 3 + [0, 1],
        "assistant_response_token_count": [100, 200, 300, 400, 500],
        "observation_token_count": [10] * 5,
        "trajectory_response_token_count": [110, 210, 310, 410, 510],
        "num_assistant_turns": [1, 2, 3, 4, 5],
        "tool_call_count": [0, 1, 2, 3, 4],
        "generate_time_seconds": [1, 2, 3, 4, 5],
        "tool_time_seconds": [0, 1, 1, 2, 2],
        "trajectory_elapsed_seconds": [1, 3, 4, 6, 7],
        "trajectory_finished": [True] * 5,
        "trajectory_invalid": [False] * 5,
        "trajectory_aborted": [False] * 5,
        "max_agent_turns_reached": [False] * 5,
        "invalid_reason": [None] * 5,
        "output_format_error_count": [0] * 5,
        "raw_output_format_failure_count": [1, 0, 0, 0, 0],
        "tool_args_error_count": [0] * 5,
        "raw_tool_args_failure_count": [0] * 5,
        "truncated_turn_count": [1, 0, 0, 0, 0],
        "output_format_success_rate": [0.0, 1.0, 1.0, 1.0, 1.0],
        "tool_args_success_rate": [0.0, 1.0, 1.0, 1.0, 1.0],
        "output_format_error_rate": [0.0] * 5,
        "tool_args_error_rate": [0.0] * 5,
        "raw_output_format_failure_rate": [1.0, 0.0, 0.0, 0.0, 0.0],
        "raw_tool_args_failure_rate": [0.0] * 5,
        "truncated_turn_rate": [1.0, 0.0, 0.0, 0.0, 0.0],
    }

    metrics, records = summarize_validation(
        ["visharness/gres"] * 2
        + ["visharness/reasonseg"]
        + ["visharness/rec8k"] * 2,
        ["g0", "g1", "r0", "c0", "c1"],
        extras,
    )

    assert len(records) == 5
    assert metrics["val-core/GRES/gIoU"] == pytest.approx(0.625)
    assert metrics["val-core/GRES/cIoU"] == pytest.approx(0.25)
    assert metrics["val-core/GRES/N_acc"] == 1.0
    assert metrics["val-core/ReasonSeg/gIoU"] == 0.5
    assert metrics["val-core/ReasonSeg/cIoU"] == 0.5
    assert metrics["val-core/REC8K/MAE"] == 1.0
    assert metrics["val-core/REC8K/RMSE"] == pytest.approx(2**0.5)
    assert metrics["val-core/REC8K/precision"] == pytest.approx(5 / 8)
    assert metrics["val-core/REC8K/recall"] == pytest.approx(5 / 6)
    assert metrics["val-core/REC8K/F1"] == pytest.approx(5 / 7)
    assert metrics["val-core/REC8K/exact_count_accuracy"] == 0.5
    assert metrics["val-aux/overall/assistant_output_tokens/p90"] == pytest.approx(460.0)
    assert metrics["val-aux/overall/trajectories_with_format_error_rate"] == 0.0
    assert metrics["val-aux/overall/trajectories_with_raw_format_failure_rate"] == pytest.approx(0.2)
    assert metrics["val-aux/overall/trajectories_with_truncation_rate"] == pytest.approx(0.2)
    assert metrics["val-aux/overall/turn_output_format_error_rate"] == 0.0
    assert metrics["val-aux/overall/raw_output_format_failure_rate"] == pytest.approx(1 / 15)
    assert metrics["val-aux/overall/turn_truncation_rate"] == pytest.approx(1 / 15)
    assert metrics["val-aux/overall/turn_hard_error_rate"] == pytest.approx(1 / 15)


def test_legacy_validation_errors_are_not_mislabeled_as_exclusive_format_errors():
    metrics, records = summarize_validation(
        ["visharness/gres"],
        ["legacy"],
        {
            "task_type": ["gres_rle_mask"],
            "num_assistant_turns": [1],
            # This was the old raw parser-failure count and included truncation.
            "output_format_error_count": [1],
            "tool_args_error_count": [0],
            "truncated_turn_count": [1],
            "output_format_success_rate": [0.0],
            "tool_args_success_rate": [0.0],
            "truncated_turn_rate": [1.0],
        },
    )

    assert records[0]["raw_output_format_failure_count"] == 1.0
    assert math.isnan(records[0]["output_format_error_count"])
    assert "val-aux/overall/trajectories_with_format_error_rate" not in metrics
    assert "val-aux/overall/turn_output_format_error_rate" not in metrics
    assert metrics["val-aux/overall/trajectories_with_raw_format_failure_rate"] == 1.0
    assert metrics["val-aux/overall/raw_output_format_failure_rate"] == 1.0
    assert metrics["val-aux/overall/turn_truncation_rate"] == 1.0


def test_reasonseg_empty_accuracy_is_reported_when_empty_targets_exist():
    metrics, _ = summarize_validation(
        ["visharness/reasonseg", "visharness/reasonseg"],
        ["empty-correct", "empty-failed"],
        {
            "task_type": ["reasonseg_rle_mask", "reasonseg_rle_mask"],
            "eval_iou": [1.0, 0.0],
            "eval_intersection": [0.0, 0.0],
            "eval_union": [0.0, 0.0],
            "eval_ground_truth_empty": [True, True],
            "eval_empty_correct": [True, False],
        },
    )

    assert metrics["val-core/ReasonSeg/N_acc"] == 0.5
    assert metrics["val-aux/ReasonSeg/empty_ground_truth_samples"] == 2.0


def test_validation_artifacts_are_structured_and_replace_latest(tmp_path):
    records = [{"uid": "sample-1", "task": "GRES", "eval_iou": float("nan")}]
    metrics = {"val-core/GRES/gIoU": 0.5}
    step_dir = write_validation_artifacts(
        tmp_path,
        global_step=10,
        metrics=metrics,
        records=records,
        generation_args={"max_tokens": 2048},
    )

    summary = json.loads((step_dir / "metrics.json").read_text())
    sample = json.loads((step_dir / "samples.jsonl").read_text())
    latest = json.loads((tmp_path / "latest_metrics.json").read_text())
    assert summary["global_step"] == 10
    assert summary["generation_args"]["max_tokens"] == 2048
    assert sample["eval_iou"] is None
    assert latest == summary
