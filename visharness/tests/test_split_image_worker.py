from io import BytesIO

import pytest
from PIL import Image

from tool_server.tool_workers.online_workers import split_image_into_patches


ConcurrentSplitImageIntoPatches = split_image_into_patches.SplitImageIntoPatches


def _image_bytes(size=(400, 200)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _decoded_size(payload: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(payload)) as image:
        return image.size


@pytest.mark.parametrize(
    "worker_module",
    [split_image_into_patches],
)
@pytest.mark.parametrize(
    ("image_name", "expected_factor"),
    [
        ("img_0", 1),
        ("img_0_4x", 4),
        ("img_0_4x_r1_c1", 1),
        ("img_0_r1_c1_4x", 4),
        ("img_0_r1_c1_4x_4x", 16),
        ("img_0_r1_c1_4x.jpg", 4),
    ],
)
def test_direct_super_resolution_factor_uses_trailing_operations_only(
    worker_module,
    image_name,
    expected_factor,
):
    assert worker_module.get_direct_super_resolution_factor(image_name) == expected_factor


@pytest.mark.parametrize(
    "worker_class",
    [ConcurrentSplitImageIntoPatches],
)
@pytest.mark.parametrize(
    ("image_name", "expected_overview_size"),
    [
        ("img_0", (400, 200)),
        ("img_0_4x", (100, 50)),
        ("img_0_4x_r1_c1", (400, 200)),
        ("img_0_r1_c1_4x", (100, 50)),
    ],
)
def test_split_restores_only_direct_super_resolution_overview(
    worker_class,
    image_name,
    expected_overview_size,
):
    worker = object.__new__(worker_class)
    response = worker.generate(
        {
            "image_dict": {
                image_name: {
                    "image_bytes": _image_bytes(),
                    "patch_size": 200,
                }
            }
        }
    )

    assert response["status"] == "success"
    image_result = response["results"][image_name]
    assert _decoded_size(image_result["overview"]) == expected_overview_size

    patches = list(image_result["patches"].values())
    assert [
        (patch["offset_x"], patch["offset_y"], patch["width"], patch["height"])
        for patch in patches
    ] == [
        (0, 0, 240, 200),
        (160, 0, 240, 200),
    ]
    assert [_decoded_size(patch["image_bytes"]) for patch in patches] == [
        (240, 200),
        (240, 200),
    ]


def test_split_failed_trajectory_dimensions_restore_overview_only():
    worker = object.__new__(ConcurrentSplitImageIntoPatches)
    response = worker.generate(
        {
            "image_dict": {
                "img_0_4x": {
                    "image_bytes": _image_bytes((2956, 1536)),
                    "patch_size": 400,
                }
            }
        }
    )

    assert response["status"] == "success"
    image_result = response["results"]["img_0_4x"]
    assert _decoded_size(image_result["overview"]) == (739, 384)
    assert len(image_result["patches"]) == 32
    assert max(patch["width"] for patch in image_result["patches"].values()) > 400
    assert max(patch["height"] for patch in image_result["patches"].values()) > 400
