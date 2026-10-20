"""REC-8K official localization parity and default-evaluator regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from visharness.evaluate.evaluate_rec8k import _localization_matches, evaluate_rec8k
from visharness.rewards.vision_reward import compute_score


def _official_reference(boxes, points, width, height):
    # Independent transcription of prepare_targets/distance_threshold_func/
    # calc_loc_metric at official commit e8f5a54f295fd7443665b5c70499e5a85bccc0e4.
    if not boxes or not points:
        return 0
    normalized = [
        [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
         (x2 - x1) / width, (y2 - y1) / height]
        for x1, y1, x2, y2 in boxes
    ]
    gt = (np.array(points) / np.array([width, height])).astype(np.float32)
    areas = [box[2] * box[3] for box in normalized]
    median_box = normalized[np.argsort(areas)[len(areas) // 2]]
    threshold = np.sqrt(median_box[2] ** 2 + median_box[3] ** 2) / 2.0
    costs = cdist(np.array([box[:2] for box in normalized]), gt, metric="euclidean")
    rows, cols = linear_sum_assignment(costs)
    return sum(costs[row, col] < threshold for row, col in zip(rows, cols))


def test_normalization_changes_matching_on_non_square_images():
    assert _localization_matches(
        [[0, 0, 100, 10]], [[50, 20]], original_width=1000, original_height=100
    ) == 0


def test_even_predictions_use_upper_median_not_average_radius():
    assert _localization_matches(
        [[10, 10, 30, 30], [600, 600, 1000, 1000]], [[220, 10]],
        original_width=1000, original_height=1000,
    ) == 1


@pytest.mark.parametrize(("point", "expected"), [
    ([400.0, 100.0], 0),       # Exactly at the threshold: strict <, not <=.
    ([399.0, 100.0], 1),
    ([399.999999, 100.0], 0),  # Official float32 GT rounds to the boundary.
])
def test_strict_threshold_and_float32_ground_truth(point, expected):
    assert _localization_matches(
        [[0, 0, 300, 200]], [point], original_width=800, original_height=400
    ) == expected


@pytest.mark.parametrize(("boxes", "points"), [
    ([], [[5, 5]]), ([[0, 0, 10, 10]], []), ([], []),
])
def test_empty_predictions_and_annotations_are_not_invented(boxes, points):
    assert _localization_matches(
        boxes, points, original_width=100, original_height=50
    ) == 0


def test_nonempty_matching_requires_original_image_dimensions():
    with pytest.raises(TypeError):
        _localization_matches([[0, 0, 10, 10]], [[5, 5]])
    with pytest.raises(ValueError, match="dimensions"):
        _localization_matches(
            [[0, 0, 10, 10]], [[5, 5]], original_width=0, original_height=50
        )


def test_randomized_parity_with_official_reference():
    rng = np.random.default_rng(20261020)
    for _ in range(1000):
        width, height = map(int, rng.integers(32, 4001, size=2))
        count = int(rng.integers(0, 16))
        xy1 = rng.random((count, 2))
        xy2 = xy1 + (1 - xy1) * rng.uniform(0.01, 1.0, size=(count, 2))
        boxes = np.column_stack((xy1, xy2)) * [width, height, width, height]
        points = rng.random((int(rng.integers(1, 20)), 2)) * [width, height]
        boxes, points = boxes.tolist(), points.tolist()
        assert _localization_matches(
            boxes, points, original_width=width, original_height=height
        ) == _official_reference(boxes, points, width, height)


def _final_results():
    mask = np.zeros((10, 100), dtype=np.uint8)
    mask[:1, :10] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"final_bboxes": [[0, 0, 10, 1]], "final_masks": [rle], "count": 1}


@pytest.mark.parametrize(("gt_point", "expected_tp"), [([50.0, 5.0], 1), ([50.0, 20.0], 0)])
def test_default_evaluation_and_training_validation_use_official_matching(
    tmp_path: Path, gt_point, expected_tp
):
    root = tmp_path / "REC-8K"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (1000, 100)).save(root / "images/sample.jpg")
    manifest = root / "REC8K_QA_val.json"
    manifest.write_text(json.dumps([
        {"id": "sample-object", "image_path": "images/sample.jpg", "question": "Count objects."}
    ]))
    (root / "splits.json").write_text(json.dumps({"val": [["sample.jpg", "object"]]}))
    (root / "annotations.json").write_text(json.dumps({
        "sample.jpg": {"object": {"points": [gt_point]}}
    }))
    predictions = tmp_path / "predictions.jsonl"
    final = _final_results()
    predictions.write_text(json.dumps({
        "meta_data": {"id": "sample-object"},
        "status": "finished", "termination_reason": "answer", "trajectory_finished": True,
        "current_round": 1, "max_rounds": 3, "final_results": final,
    }) + "\n")
    output = tmp_path / "metrics.json"
    metrics = evaluate_rec8k(predictions, manifest, root, "val", metrics_output=output)
    assert (metrics["TP"], metrics["FP"], metrics["FN"]) == (
        expected_tp, 1 - expected_tp, 1 - expected_tp
    )
    assert metrics["MAE"] == metrics["RMSE"] == 0.0
    assert metrics["F1"] == expected_tp
    assert metrics["sample_count"] == 1
    saved = json.loads(output.read_text())
    assert saved["localization_protocol"]["name"] == "CVPR2024_REC_official_localization"
    assert "e8f5a54" in saved["localization_protocol"]["source_url"]

    validation = compute_score(
        data_source="visharness/rec8k", solution_str="",
        ground_truth={"type": "point", "data": [[5.0, gt_point[1] / 10]]},
        extra_info={
            "validation_sample": True,
            "evaluation_ground_truth": {"type": "point", "data": [gt_point]},
            "evaluation_original_width": 1000, "evaluation_original_height": 100,
            "trajectory_finished": True, "final_results": final,
            "turn_records": [{
                "completion": "<think>done</think><answer>done</answer>",
                "output_format_success": True, "tool_args_success": True,
                "turn_truncated_by_length": False,
            }],
        },
    )
    assert (validation["eval_tp"], validation["eval_fp"], validation["eval_fn"]) == (
        expected_tp, 1 - expected_tp, 1 - expected_tp
    )
    assert validation["eval_absolute_error"] == validation["eval_squared_error"] == 0
    assert validation["task_reward"] == validation["eval_reward"] == 5.0
