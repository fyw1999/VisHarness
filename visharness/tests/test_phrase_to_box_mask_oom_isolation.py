from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image


pytest.importorskip("sam3")

from tool_server.tool_workers.online_workers.phrase_to_box_mask_worker import (
    PhraseToBoxMaskWorker,
    _ImageGroup,
    _QuerySpec,
)


def _png_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_worker():
    worker = PhraseToBoxMaskWorker.__new__(PhraseToBoxMaskWorker)
    worker.max_batch_size = 8
    worker.max_queries_per_batch = 8
    worker.device = torch.device("cpu")
    return worker


def test_phrase_oom_fallback_preserves_successful_sibling_queries(monkeypatch):
    worker = _make_worker()
    groups = [
        _ImageGroup(
            image_key=bytes([query_id]),
            model_image=Image.new("RGB", (2, 2)),
            queries=[_QuerySpec(query_id=query_id, phrase=f"q{query_id}")],
        )
        for query_id in (1, 2, 3)
    ]

    def infer(image_groups):
        queries = [
            query
            for image_group in image_groups
            for query in image_group.queries
        ]
        if len(queries) > 1 or queries[0].query_id == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        query_id = queries[0].query_id
        return {query_id: (np.empty((0, 4), dtype=np.float32), [])}

    worker._infer_groups = infer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    results, errors = worker._infer_groups_with_oom_fallback(groups)

    assert set(results) == {1, 3}
    assert set(errors) == {2}
    assert isinstance(errors[2], torch.cuda.OutOfMemoryError)


def test_phrase_leaf_oom_is_returned_only_to_its_http_request():
    worker = _make_worker()

    def infer_with_isolation(image_groups):
        results = {}
        errors = {}
        for image_group in image_groups:
            for query in image_group.queries:
                if query.phrase == "bad":
                    errors[query.query_id] = torch.cuda.OutOfMemoryError(
                        "CUDA out of memory"
                    )
                else:
                    results[query.query_id] = (
                        np.empty((0, 4), dtype=np.float32),
                        [],
                    )
        return results, errors

    worker._infer_groups_with_oom_fallback = infer_with_isolation
    worker._build_image_result = lambda image_name, *_args: {
        "image_name": image_name
    }

    responses = worker._process_request_batch_locked(
        [
            {"image_dict": {"good-a.png": _png_bytes("red")}, "phrase": "good"},
            {"image_dict": {"bad.png": _png_bytes("green")}, "phrase": "bad"},
            {"image_dict": {"good-b.png": _png_bytes("blue")}, "phrase": "good"},
        ]
    )

    assert [response["status"] for response in responses] == [
        "success",
        "error",
        "success",
    ]
    assert responses[1]["error_type"] == "cuda_oom"
    assert set(responses[0]["results"]) == {"good-a.png"}
    assert set(responses[2]["results"]) == {"good-b.png"}


def test_phrase_shared_leaf_query_oom_marks_all_consuming_requests():
    worker = _make_worker()
    shared_image = _png_bytes("red")

    def fail_shared_query(image_groups):
        query = image_groups[0].queries[0]
        return {}, {
            query.query_id: torch.cuda.OutOfMemoryError("CUDA out of memory")
        }

    worker._infer_groups_with_oom_fallback = fail_shared_query

    responses = worker._process_request_batch_locked(
        [
            {"image_dict": {"a.png": shared_image}, "phrase": "same"},
            {"image_dict": {"b.png": shared_image}, "phrase": "same"},
        ]
    )

    assert [response["error_type"] for response in responses] == [
        "cuda_oom",
        "cuda_oom",
    ]
