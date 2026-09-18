import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from tool_server.tool_workers.online_workers.merge_box_mask_worker import MergeBoxMask
from tool_server.utils.utils import pil_to_bytes


WORKER_CLASSES = [MergeBoxMask]


def _encode_masks(*masks: np.ndarray):
    masks_hwn = np.stack(masks, axis=2).astype(np.uint8)
    return mask_utils.encode(np.asfortranarray(masks_hwn))


def _worker(worker_class):
    worker = worker_class.__new__(worker_class)
    worker.iou_threshold = 0.8
    worker.fragment_overlap_ratio = 0.2
    worker.fragment_edge_margin_ratio = 0.03
    return worker


def _source_entry(
    masks: list[np.ndarray],
    bboxes: list[list[float]],
    *,
    offset_x: float,
    offset_y: float,
    has_split_ancestor: bool = True,
):
    height, width = masks[0].shape
    return {
        "image_bytes": pil_to_bytes(Image.new("RGB", (width, height), "black")),
        "bboxes": bboxes,
        "masks": _encode_masks(*masks),
        "transform_to_img0": [[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]],
        "depth": 1,
        "has_split_ancestor": has_split_ancestor,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def _run(worker_class, original_size, sources):
    image_dict = {
        "img_0": {
            "image_bytes": pil_to_bytes(Image.new("RGB", original_size, "black")),
            "transform_to_img0": np.eye(3).tolist(),
            "depth": 0,
            "has_split_ancestor": False,
        },
        **sources,
    }
    response = _worker(worker_class).generate({"image_dict": image_dict})
    assert response["status"] == "success"
    return response["results"]


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_fragment_components_keep_same_source_instances_separate(worker_class):
    patch_a_first = np.zeros((20, 25), dtype=np.uint8)
    patch_a_first[2:8, 20:25] = 1
    patch_a_second = np.zeros((20, 25), dtype=np.uint8)
    patch_a_second[12:18, 20:25] = 1
    patch_b_oversized = np.zeros((20, 25), dtype=np.uint8)
    patch_b_oversized[2:18, 5:15] = 1

    result = _run(
        worker_class,
        (40, 20),
        {
            "img_0_r1_c1": _source_entry(
                [patch_a_first, patch_a_second],
                [[20, 2, 25, 8], [20, 12, 25, 18]],
                offset_x=0,
                offset_y=0,
            ),
            "img_0_r1_c2": _source_entry(
                [patch_b_oversized],
                [[5, 2, 15, 18]],
                offset_x=15,
                offset_y=0,
            ),
        },
    )

    assert result["count"] == 2


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_relevant_patch_edges_allow_two_fragments_to_merge(worker_class):
    left = np.zeros((20, 25), dtype=np.uint8)
    left[3:13, 20:25] = 1
    right = np.zeros((20, 25), dtype=np.uint8)
    right[3:13, 0:10] = 1

    result = _run(
        worker_class,
        (40, 20),
        {
            "img_0_r1_c1": _source_entry([left], [[20, 3, 25, 13]], offset_x=0, offset_y=0),
            "img_0_r1_c2": _source_entry([right], [[0, 3, 10, 13]], offset_x=15, offset_y=0),
        },
    )

    assert result["count"] == 1


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_one_sided_patch_edge_evidence_is_sufficient(worker_class):
    clipped = np.zeros((20, 25), dtype=np.uint8)
    clipped[3:13, 20:25] = 1
    complete = np.zeros((20, 25), dtype=np.uint8)
    complete[3:13, 5:15] = 1

    result = _run(
        worker_class,
        (40, 20),
        {
            "img_0_r1_c1": _source_entry(
                [clipped], [[20, 3, 25, 13]], offset_x=0, offset_y=0
            ),
            # The complete result is interior in its own patch, but it spans
            # the right crop boundary of the clipped result.
            "img_0_r1_c2": _source_entry(
                [complete], [[5, 3, 15, 13]], offset_x=15, offset_y=0
            ),
        },
    )

    assert result["count"] == 1


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_interior_overlap_without_patch_boundary_evidence_does_not_merge(worker_class):
    first = np.zeros((20, 25), dtype=np.uint8)
    first[5:15, 8:16] = 1
    second = np.zeros((20, 25), dtype=np.uint8)
    second[5:15, 5:13] = 1

    result = _run(
        worker_class,
        (40, 20),
        {
            "img_0_r1_c1": _source_entry([first], [[8, 5, 16, 15]], offset_x=0, offset_y=0),
            # Global x=[10,18], overlapping the first mask at x=[10,16],
            # while both detections remain inside their local patch borders.
            "img_0_r1_c2": _source_entry([second], [[5, 5, 13, 15]], offset_x=5, offset_y=0),
        },
    )

    assert result["count"] == 2


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_four_patch_fragments_form_one_component(worker_class):
    top_left = np.zeros((25, 25), dtype=np.uint8)
    top_left[20:25, 20:25] = 1
    top_right = np.zeros((25, 25), dtype=np.uint8)
    top_right[20:25, 0:10] = 1
    bottom_left = np.zeros((25, 25), dtype=np.uint8)
    bottom_left[0:10, 20:25] = 1
    bottom_right = np.zeros((25, 25), dtype=np.uint8)
    bottom_right[0:10, 0:10] = 1

    result = _run(
        worker_class,
        (40, 40),
        {
            "img_0_r1_c1": _source_entry(
                [top_left], [[20, 20, 25, 25]], offset_x=0, offset_y=0
            ),
            "img_0_r1_c2": _source_entry(
                [top_right], [[0, 20, 10, 25]], offset_x=15, offset_y=0
            ),
            "img_0_r2_c1": _source_entry(
                [bottom_left], [[20, 0, 25, 10]], offset_x=0, offset_y=15
            ),
            "img_0_r2_c2": _source_entry(
                [bottom_right], [[0, 0, 10, 10]], offset_x=15, offset_y=15
            ),
        },
    )

    assert result["count"] == 1


@pytest.mark.parametrize("worker_class", WORKER_CLASSES)
def test_full_image_and_super_resolution_are_not_treated_as_patch_fragments(worker_class):
    first = np.zeros((20, 20), dtype=np.uint8)
    first[2:10, 2:10] = 1
    second = np.zeros((20, 20), dtype=np.uint8)
    second[2:14, 2:10] = 1

    result = _run(
        worker_class,
        (20, 20),
        {
            "img_0_full": _source_entry(
                [first], [[2, 2, 10, 10]], offset_x=0, offset_y=0, has_split_ancestor=False
            ),
            "img_0_4x": _source_entry(
                [second], [[2, 2, 10, 14]], offset_x=0, offset_y=0, has_split_ancestor=False
            ),
        },
    )

    # IoU is below 0.8; without the split-lineage requirement these two
    # differently shaped full-image results would be fragment-merged.
    assert result["count"] == 2
