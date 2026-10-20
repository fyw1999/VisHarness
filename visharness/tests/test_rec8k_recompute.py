"""The optional audit script accepts both legacy and new official metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from visharness.evaluate.evaluate_rec8k import evaluate_rec8k


@pytest.mark.parametrize("previous_protocol", ["legacy", "official"])
def test_recompute_preserves_old_audit_and_accepts_new_default(tmp_path, previous_protocol):
    script = Path(__file__).resolve().parents[2] / "scripts/recompute_rec8k_official_localization.py"
    spec = importlib.util.spec_from_file_location("rec8k_recompute_audit", script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (1000, 100)).save(root / "images/sample.jpg")
    manifest = root / "REC8K_QA_val.json"
    manifest.write_text(json.dumps([
        {"id": "sample-object", "image_path": "images/sample.jpg", "question": "Count objects."}
    ]))
    (root / "splits.json").write_text(json.dumps({"val": [["sample.jpg", "object"]]}))
    (root / "annotations.json").write_text(json.dumps({
        "sample.jpg": {"object": {"points": [[50, 20]]}}
    }))
    run = tmp_path / "saved_run"
    run.mkdir()
    mask = np.zeros((10, 100), dtype=np.uint8)
    mask[:1, :10] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    predictions = run / "saved_run_ckpt.jsonl"
    predictions.write_text(json.dumps({
        "meta_data": {"id": "sample-object"}, "status": "finished",
        "trajectory_finished": True, "termination_reason": "answer",
        "current_round": 1, "max_rounds": 3,
        "final_results": {
            "final_bboxes": [[0, 0, 10, 1]], "final_masks": [rle], "count": 1,
        },
    }) + "\n")
    previous = evaluate_rec8k(predictions, manifest, root, "val")
    if previous_protocol == "legacy":
        previous.pop("localization_protocol")
        previous.update(TP=1, FP=0, FN=0, precision=1.0, recall=1.0, F1=1.0)
    previous_path = run / "evaluation_metrics.json"
    previous_path.write_text(json.dumps(previous))
    before = previous_path.read_bytes()
    metrics = audit.evaluate_saved_run(str(run), str(root), "val")
    assert metrics["previous_metrics_protocol"] == previous_protocol
    assert metrics["previous_metrics_reproduced"] is True
    assert metrics["legacy_localization"]["TP"] == 1
    assert metrics["TP"] == 0
    assert metrics["MAE"] == metrics["RMSE"] == 0
    assert previous_path.read_bytes() == before
