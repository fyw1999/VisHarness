from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from visharness.evaluate import common
from visharness.evaluate.common import (
    TERMINATION_ANSWER,
    TERMINATION_FAILED,
    TERMINATION_MAX_ROUNDS,
    infer_termination_reason,
    load_compact_predictions,
    prediction_status_summary,
    restore_boxes_to_original,
    restore_mask_to_original,
    summarize_efficiency_metrics,
)
from visharness.evaluate.evaluate_gres import print_gres_summary_table
from visharness.evaluate.evaluate_reasonseg import (
    evaluate_reasonseg,
    print_reasonseg_summary_table,
)
from visharness.evaluate.evaluate_rec8k import (
    evaluate_rec8k,
    print_rec8k_summary_table,
)
from visharness.evaluate.evaluate_dense200 import evaluate_dense200


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def _prediction(
    item_id: str,
    *,
    box: list[float] | None = None,
    mask: np.ndarray | None = None,
    termination_reason: str = "answer",
    current_round: int = 1,
    efficiency_metrics: dict | None = None,
) -> dict:
    final_results = None
    if box is not None and mask is not None:
        final_results = {
            "final_bboxes": [box],
            "final_masks": [_rle(mask)],
            "count": 1,
            "final_visual_image": "large-field-not-needed-by-evaluation",
        }
    return {
        "meta_data": {"id": item_id},
        "status": "failed" if termination_reason == "failed" else "finished",
        "termination_reason": termination_reason,
        "trajectory_finished": termination_reason == "answer",
        "max_rounds_reached": termination_reason == "max_rounds_reached",
        "current_round": current_round,
        "max_rounds": 3,
        "conversation": [{"role": "user", "content": "unused"}],
        "images": {"img_0": {"image": "unused"}},
        "final_results": final_results,
        "efficiency_metrics": efficiency_metrics or {},
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def test_new_and_legacy_termination_semantics() -> None:
    assert infer_termination_reason({"status": "failed"}) == TERMINATION_FAILED
    assert (
        infer_termination_reason(
            {"status": "finished", "trajectory_invalid": True}
        )
        == TERMINATION_FAILED
    )
    assert (
        infer_termination_reason(
            {
                "status": "finished",
                "termination_reason": "max_rounds_reached",
            }
        )
        == TERMINATION_MAX_ROUNDS
    )
    assert (
        infer_termination_reason({"current_round": 2, "max_rounds": 3})
        == TERMINATION_ANSWER
    )
    assert (
        infer_termination_reason(
            {"status": "finished", "current_round": 2, "max_rounds": 3}
        )
        == TERMINATION_ANSWER
    )
    assert (
        infer_termination_reason({"current_round": 3, "max_rounds": 3})
        == TERMINATION_MAX_ROUNDS
    )


def test_stream_loader_keeps_latest_compact_record(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    first = _prediction("sample", termination_reason="failed")
    second = _prediction("sample", termination_reason="answer")
    _write_jsonl(prediction_path, [first, second])

    predictions, report = load_compact_predictions(prediction_path)

    assert predictions["sample"].termination_reason == TERMINATION_ANSWER
    assert report.duplicate_records == 1
    assert report.duplicate_ids == {"sample"}
    assert not hasattr(predictions["sample"], "conversation")
    assert not hasattr(predictions["sample"], "images")


def test_status_summary_distinguishes_empty_failure_and_missing(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _prediction(
                "empty-answer",
                termination_reason="answer",
                current_round=2,
            ),
            _prediction(
                "failed",
                termination_reason="failed",
                current_round=4,
            ),
        ],
    )
    predictions, report = load_compact_predictions(prediction_path)
    summary = prediction_status_summary(
        ["empty-answer", "failed", "missing"],
        predictions,
        report,
    )

    assert summary["explicit_empty_predictions"] == 1
    assert summary["failed_trajectories"] == 1
    assert summary["missing_predictions"] == 1
    assert summary["agent_success_rate"] == pytest.approx(1 / 3)
    assert summary["average_actual_rounds"] == pytest.approx(3.0)
    assert summary["actual_rounds_sample_count"] == 2
    assert summary["actual_rounds_missing_count"] == 1


def test_coordinate_restore_uses_rle_canvas_dimensions() -> None:
    processed_mask = np.zeros((3, 4), dtype=np.uint8)
    processed_mask[1:3, 1:3] = 1
    rle = _rle(processed_mask)

    restored_mask = restore_mask_to_original(
        [rle],
        original_width=8,
        original_height=6,
    )
    restored_boxes = restore_boxes_to_original(
        [[1.0, 1.0, 3.0, 2.0]],
        [rle],
        original_width=8,
        original_height=6,
    )

    assert restored_mask.shape == (6, 8)
    assert restored_boxes == [[2.0, 2.0, 6.0, 4.0]]


def test_coordinate_restore_rejects_wrong_aspect_ratio() -> None:
    mask = np.ones((3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="incompatible aspect ratios"):
        restore_mask_to_original(
            [_rle(mask)],
            original_width=10,
            original_height=6,
        )


def test_greedy_matching_preserves_legacy_iou_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = {
        ("p0", "g0"): 0.5,
        ("p0", "g1"): 0.5,
        ("p1", "g0"): 0.5,
        ("p1", "g1"): 0.0,
    }
    monkeypatch.setattr(
        common,
        "box_iou",
        lambda prediction, target: scores[(prediction[0], target[0])],
    )

    matches = common.greedy_iou_matches(
        [["p0"], ["p1"]],
        [["g0"], ["g1"]],
        0.5,
    )

    assert matches == 1


def test_dense200_end_to_end_coordinate_restore(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dense200"
    image_dir = dataset_root / "dense200"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(image_dir / "sample.jpg")

    manifest_path = dataset_root / "Dense200_QA_test.json"
    _write_json(
        manifest_path,
        [
            {
                "id": "Dense200-sample-object",
                "image_path": "dense200/sample.jpg",
                "question": "Detect the object.",
            }
        ],
    )
    annotation_path = dataset_root / "Dense200.jsonl"
    _write_jsonl(
        annotation_path,
        [
            {
                "image_path": "dense200/sample.jpg",
                "gt": {"object": [[2.0, 2.0, 6.0, 4.0]]},
            }
        ],
    )

    processed_mask = np.zeros((3, 4), dtype=np.uint8)
    processed_mask[1:2, 1:3] = 1
    prediction_path = tmp_path / "dense_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _prediction(
                "Dense200-sample-object",
                box=[1.0, 1.0, 3.0, 2.0],
                mask=processed_mask,
                termination_reason="max_rounds_reached",
            )
        ],
    )

    metrics = evaluate_dense200(
        prediction_path,
        manifest_path,
        dataset_root,
    )
    assert metrics["F1@0.50"] == pytest.approx(1.0)
    assert metrics["F1@0.95"] == pytest.approx(1.0)
    assert metrics["status"]["max_rounds_reached"] == 1
    assert metrics["status"]["valid_visual_results"] == 1


def test_rec8k_end_to_end_coordinate_restore(tmp_path: Path) -> None:
    dataset_root = tmp_path / "REC-8K"
    image_dir = dataset_root / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(image_dir / "sample.jpg")

    manifest_path = dataset_root / "REC8K_QA_test.json"
    _write_json(
        manifest_path,
        [
            {
                "id": "sample-object",
                "image_path": "images/sample.jpg",
                "question": "Count the number of object in the image.",
            }
        ],
    )
    _write_json(
        dataset_root / "splits.json",
        {"test": [["sample.jpg", "object"]]},
    )
    _write_json(
        dataset_root / "annotations.json",
        {"sample.jpg": {"object": {"points": [[4.0, 3.0]]}}},
    )

    processed_mask = np.zeros((3, 4), dtype=np.uint8)
    processed_mask[1:2, 1:3] = 1
    prediction_path = tmp_path / "rec_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _prediction(
                "sample-object",
                box=[1.0, 1.0, 3.0, 2.0],
                mask=processed_mask,
            )
        ],
    )

    metrics = evaluate_rec8k(
        prediction_path,
        manifest_path,
        dataset_root,
        "test",
    )
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["F1"] == pytest.approx(1.0)
    assert metrics["efficiency"]["enabled"] is False
    assert metrics["efficiency"]["run_level"]["available"] is False


@pytest.mark.parametrize(
    ("termination_reason", "expected_iou", "expected_n_acc"),
    [
        ("answer", 1.0, 1.0),
        ("failed", 0.0, 0.0),
        ("max_rounds_reached", 0.0, 0.0),
    ],
)
def test_reasonseg_empty_target_requires_explicit_empty_prediction(
    tmp_path: Path,
    termination_reason: str,
    expected_iou: float,
    expected_n_acc: float,
) -> None:
    dataset_root = tmp_path / "ReasonSeg"
    image_dir = dataset_root / "val"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(image_dir / "sample.jpg")
    _write_json(image_dir / "sample.json", {"shapes": []})

    manifest_path = dataset_root / "ReasonSeg_QA_val.json"
    _write_json(
        manifest_path,
        [
            {
                "id": "ReasonSeg-sample",
                "image_path": "val/sample.jpg",
                "question": "Segment the matching object.",
            }
        ],
    )
    prediction_path = tmp_path / f"reasonseg_{termination_reason}.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _prediction(
                "ReasonSeg-sample",
                termination_reason=termination_reason,
            )
        ],
    )

    metrics = evaluate_reasonseg(
        prediction_path,
        manifest_path,
        dataset_root,
    )

    assert metrics["gIoU"] == expected_iou
    assert metrics["cIoU"] == 0.0
    assert metrics["N_acc"] == expected_n_acc
    assert metrics["empty_ground_truth_samples"] == 1
    assert metrics["empty_ground_truth_correct"] == int(expected_n_acc)


def test_efficiency_summary_combines_trajectory_and_run_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction_path = run_dir / "run_ckpt.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _prediction(
                "sample-a",
                efficiency_metrics={
                    "trajectory_elapsed_seconds": 10.0,
                    "llm_generate_seconds": 7.0,
                    "tool_seconds": 2.0,
                    "cumulative_prompt_tokens": 100,
                    "cumulative_visual_tokens": 40,
                    "cumulative_completion_tokens": 20,
                    "max_visual_tokens_per_call": 25,
                    "model_call_count": 2,
                    "tool_call_count": 1,
                },
            ),
            _prediction(
                "sample-b",
                termination_reason="failed",
                efficiency_metrics={
                    "trajectory_elapsed_seconds": 20.0,
                    "llm_generate_seconds": 14.0,
                    "tool_seconds": 4.0,
                    "cumulative_prompt_tokens": 200,
                    "cumulative_visual_tokens": 80,
                    "cumulative_completion_tokens": 30,
                    "max_visual_tokens_per_call": 45,
                    "model_call_count": 3,
                    "tool_call_count": 2,
                },
            ),
        ],
    )
    _write_json(
        run_dir / "benchmark_run.json",
        {
            "run_id": "run-1",
            "processed_trajectories": 2,
            "rollout_wall_seconds": 40.0,
            "rollout_throughput_trajectories_per_minute": 3.0,
            "resource_metrics": {
                "peak_gpu_memory_per_gpu_gib": 50.0,
                "peak_kv_cache_usage": 0.75,
                "kv_cache_pool_gib_per_gpu": 20.0,
            },
        },
    )
    predictions, _ = load_compact_predictions(prediction_path)

    summary = summarize_efficiency_metrics(
        ["sample-a", "sample-b"],
        predictions,
        prediction_path,
    )

    assert summary["enabled"] is True
    assert summary["records_with_efficiency_metrics"] == 2
    assert (
        summary["per_trajectory"][
            "average_trajectory_elapsed_seconds"
        ]
        == pytest.approx(15.0)
    )
    assert (
        summary["per_trajectory"][
            "average_cumulative_visual_tokens"
        ]
        == pytest.approx(60.0)
    )
    assert (
        summary["run_level"][
            "rollout_throughput_trajectories_per_minute"
        ]
        == pytest.approx(3.0)
    )
    assert summary["run_level"]["peak_gpu_memory_per_gpu_gib"] == 50.0
    assert summary["run_level"]["peak_kv_cache_usage"] == 0.75
    assert (
        summary["run_level"][
            "peak_active_kv_cache_memory_per_gpu_gib"
        ]
        == pytest.approx(15.0)
    )


def test_print_rec8k_summary_table(capsys: pytest.CaptureFixture[str]) -> None:
    print_rec8k_summary_table(
        {
            "MAE": 1.25,
            "RMSE": 2.5,
            "efficiency": {
                "per_trajectory": {
                    "average_cumulative_visual_tokens": 1234.5,
                    "average_trajectory_elapsed_seconds": 6.75,
                },
                "run_level": {
                    "peak_gpu_memory_per_gpu_gib": 42.25,
                    "rollout_throughput_trajectories_per_minute": 8.5,
                    "peak_active_kv_cache_memory_per_gpu_gib": 6.5,
                    "peak_kv_cache_usage_percent": 13.0,
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert (
        "| MAE | RMSE | Avg. Visual Tokens/Trajectory | Avg. Latency | "
        "Peak active KV memory | rollout throughput |"
    ) in output
    assert (
        "| 1.2500 | 2.5000 | 1234.50 | 6.75 s/trajectory | "
        "6.50 GiB/GPU (13.00%) | 8.50 trajectories/min |"
    ) in output


def test_print_rec8k_summary_table_handles_missing_benchmark_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_rec8k_summary_table({"MAE": 1.0, "RMSE": 2.0})

    output = capsys.readouterr().out
    assert (
        "| 1.0000 | 2.0000 | N/A | N/A | N/A | N/A |"
    ) in output


def test_print_gres_summary_table(capsys: pytest.CaptureFixture[str]) -> None:
    print_gres_summary_table(
        {
            "gIoU": 0.625,
            "cIoU": 0.75,
            "efficiency": {
                "per_trajectory": {
                    "average_cumulative_visual_tokens": 1088.716,
                    "average_trajectory_elapsed_seconds": 17.1519,
                },
                "run_level": {
                    "rollout_throughput_trajectories_per_minute": 25.7532,
                    "peak_active_kv_cache_memory_per_gpu_gib": 2.1094,
                    "peak_kv_cache_usage_percent": 3.8251,
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert (
        "| gIoU | cIoU | Avg. Visual Tokens/Trajectory | Avg. Latency | "
        "Peak active KV memory | rollout throughput |"
    ) in output
    assert (
        "| 0.6250 | 0.7500 | 1088.72 | 17.15 s/trajectory | "
        "2.11 GiB/GPU (3.83%) | 25.75 trajectories/min |"
    ) in output


def test_print_gres_summary_table_handles_missing_benchmark_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_gres_summary_table({"gIoU": 0.5, "cIoU": 0.25})

    output = capsys.readouterr().out
    assert (
        "| 0.5000 | 0.2500 | N/A | N/A | N/A | N/A |"
    ) in output


def test_print_reasonseg_summary_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_reasonseg_summary_table(
        {
            "gIoU": 0.625,
            "cIoU": 0.75,
            "efficiency": {
                "per_trajectory": {
                    "average_cumulative_visual_tokens": 1234.567,
                    "average_trajectory_elapsed_seconds": 8.765,
                },
                "run_level": {
                    "rollout_throughput_trajectories_per_minute": 20.125,
                    "peak_active_kv_cache_memory_per_gpu_gib": 6.975,
                    "peak_kv_cache_usage_percent": 12.649,
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert (
        "| gIoU | cIoU | Avg. Visual Tokens/Trajectory | Avg. Latency | "
        "Peak active KV memory | rollout throughput |"
    ) in output
    assert (
        "| 0.6250 | 0.7500 | 1234.57 | 8.77 s/trajectory | "
        "6.97 GiB/GPU (12.65%) | 20.12 trajectories/min |"
    ) in output


def test_print_reasonseg_summary_table_handles_missing_benchmark_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_reasonseg_summary_table({"gIoU": 0.5, "cIoU": 0.25})

    output = capsys.readouterr().out
    assert (
        "| 0.5000 | 0.2500 | N/A | N/A | N/A | N/A |"
    ) in output
