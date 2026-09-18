import pytest
import numpy as np
import torch
from PIL import Image


pytest.importorskip("sam3")

from tool_server.tool_workers.online_workers.point_to_box_mask_worker import (
    PointToBoxMaskWorker,
)


def _make_budget_only_worker(budget_mb=2048):
    worker = PointToBoxMaskWorker.__new__(PointToBoxMaskWorker)
    worker.max_points_per_forward = {"confidence": 96, "area": 32}
    worker.point_mask_postprocess_budget_bytes = budget_mb * 1024 * 1024
    return worker


def test_point_chunk_budget_keeps_hard_limits_for_ordinary_images():
    worker = _make_budget_only_worker()

    confidence_size, _ = worker._choose_point_chunk_size(
        200, 1024, 1024, "confidence"
    )
    area_size, _ = worker._choose_point_chunk_size(200, 1024, 1024, "area")

    assert confidence_size == 96
    assert area_size == 32


def test_point_chunk_budget_shrinks_with_resolution_and_candidate_count():
    worker = _make_budget_only_worker()

    confidence_size, _ = worker._choose_point_chunk_size(
        200, 2160, 3840, "confidence"
    )
    area_size, _ = worker._choose_point_chunk_size(200, 2160, 3840, "area")

    assert confidence_size == 32
    assert area_size == 10


def test_point_chunk_budget_never_returns_zero():
    worker = _make_budget_only_worker(budget_mb=1)

    size, budget_limit = worker._choose_point_chunk_size(
        4, 10000, 10000, "area"
    )

    assert size == 1
    assert budget_limit == 1


def test_point_decoder_uses_resolution_limited_chunks_without_reordering():
    worker = _make_budget_only_worker()
    observed_chunk_starts = []

    def predict(_image_index, chunk_points, _chunk_labels, _mode):
        observed_chunk_starts.append((int(chunk_points[0, 0, 0]), len(chunk_points)))
        return np.zeros((len(chunk_points), 1, 1, 1), dtype=np.float32)

    worker._predict_image_point_chunk = predict
    worker._select_valid_masks_and_boxes = lambda _masks, _mode: (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 1, 1), dtype=bool),
    )
    worker._encode_masks = lambda _masks: []
    worker._build_image_result = (
        lambda image_name, _image, bboxes, masks: {
            "image_name": image_name,
            "bboxes": bboxes,
            "masks": masks,
        }
    )

    points = np.arange(100, dtype=np.float32)[:, None, None]
    points = np.concatenate([points, points], axis=2)
    labels = np.ones((100, 1), dtype=np.int32)
    result = worker._decode_image_in_point_chunks(
        inference_state={},
        image_index=0,
        image_name="large.png",
        image=Image.new("RGB", (3840, 2160)),
        points=points,
        labels=labels,
        mode="confidence",
    )

    assert observed_chunk_starts == [(0, 32), (32, 32), (64, 32), (96, 4)]
    assert result["image_name"] == "large.png"


def test_point_decoder_halves_only_the_failed_chunk_after_oom(monkeypatch):
    worker = _make_budget_only_worker()
    observed_sizes = []
    predictor_reactivations = []

    def predict(_image_index, chunk_points, _chunk_labels, _mode):
        observed_sizes.append(len(chunk_points))
        if len(observed_sizes) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return np.zeros((len(chunk_points), 1, 1, 1), dtype=np.float32)

    worker._predict_image_point_chunk = predict
    worker._select_valid_masks_and_boxes = lambda _masks, _mode: (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 1, 1), dtype=bool),
    )
    worker._encode_masks = lambda _masks: []
    worker._build_image_result = lambda *_args: {"status": "ok"}
    worker._activate_predictor_from_state = (
        lambda inference_state: predictor_reactivations.append(inference_state)
    )
    worker.sam3_model = object()
    worker.device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    points = np.zeros((40, 1, 2), dtype=np.float32)
    labels = np.ones((40, 1), dtype=np.int32)
    worker._decode_image_in_point_chunks(
        inference_state={"encoded": True},
        image_index=0,
        image_name="large.png",
        image=Image.new("RGB", (3840, 2160)),
        points=points,
        labels=labels,
        mode="confidence",
    )

    assert observed_sizes == [32, 16, 16, 8]
    assert predictor_reactivations == [{"encoded": True}]
