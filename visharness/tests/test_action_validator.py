from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from visharness.agent_loop.action_parser import ParsedAction
from visharness.agent_loop.action_validator import validate_and_prepare_tool_call
from visharness.agent_loop.trajectory_state import VisionTrajectoryState

AVAILABLE_TOOLS = {
    "PhraseToPoint",
    "PhraseToBoxMask",
    "PointToBoxMask",
    "SplitImageIntoPatches",
    "SuperResolution",
    "MergeBoxMask",
}


def make_state() -> VisionTrajectoryState:
    return VisionTrajectoryState.from_initial_image(Image.new("RGB", (32, 24), "white"))


def assert_jpeg(image_bytes: bytes, expected_size: tuple[int, int] = (32, 24)) -> None:
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        assert image.size == expected_size


def tool_action(tool_name: str, **arguments) -> ParsedAction:
    return ParsedAction(action_type="tool_call", tool_name=tool_name, arguments=arguments)


def test_phrase_tool_builds_tool_parameters_without_mutating_state():
    state = make_state()

    result = validate_and_prepare_tool_call(
        tool_action("PhraseToPoint", images=["img_0"], phrase="the person on the left"),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid
    assert result.call.tool_name == "PhraseToPoint"
    assert result.call.tool_parameters["phrase"] == "the person on the left"
    assert_jpeg(result.call.tool_parameters["image_dict"]["img_0"])
    assert set(state.images["img_0"]) == {"image"}


def test_unknown_tool_is_rejected():
    result = validate_and_prepare_tool_call(tool_action("UnknownTool"), make_state(), AVAILABLE_TOOLS)

    assert not result.is_valid
    assert "not in available tool list" in result.error


def test_missing_image_is_rejected():
    result = validate_and_prepare_tool_call(
        tool_action("SuperResolution", images=["missing"]),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "specified image missing does not exist" in result.error


def test_point_to_box_mask_requires_previous_points():
    result = validate_and_prepare_tool_call(
        tool_action("PointToBoxMask", images=["img_0"], mode="confidence"),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "must have already been obtained valid points" in result.error


def test_point_to_box_mask_requires_explicit_mode():
    state = make_state()
    state.images["img_0"]["points"] = np.array([[3, 4], [10, 12]])

    result = validate_and_prepare_tool_call(
        tool_action("PointToBoxMask", images=["img_0"]),
        state,
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "'mode' is required" in result.error


@pytest.mark.parametrize("mode", [123, True, ["confidence"], {"value": "confidence"}])
def test_point_to_box_mask_rejects_non_string_mode_without_raising(mode):
    result = validate_and_prepare_tool_call(
        tool_action("PointToBoxMask", images=["img_0"], mode=mode),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "'mode' must be a string" in result.error


def test_point_to_box_mask_normalizes_string_mode():
    state = make_state()
    state.images["img_0"]["points"] = np.array([[3, 4]])

    result = validate_and_prepare_tool_call(
        tool_action("PointToBoxMask", images=["img_0"], mode=" AREA "),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid
    assert result.call.tool_parameters["mode"] == "area"


@pytest.mark.parametrize("phrase", [None, 123, True, [], {}, " ", "\t"])
def test_phrase_tool_rejects_non_string_or_blank_phrase_without_raising(phrase):
    result = validate_and_prepare_tool_call(
        tool_action("PhraseToPoint", images=["img_0"], phrase=phrase),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "'phrase' is required and must be a non-empty string" in result.error


def test_tool_call_rejects_non_dictionary_arguments_without_raising():
    action = ParsedAction(
        action_type="tool_call",
        tool_name="PointToBoxMask",
        arguments=None,
    )

    result = validate_and_prepare_tool_call(action, make_state(), AVAILABLE_TOOLS)

    assert not result.is_valid
    assert "tool arguments must be a JSON object" in result.error


def test_tool_call_rejects_non_string_tool_name_without_raising():
    action = ParsedAction(
        action_type="tool_call",
        tool_name=["PointToBoxMask"],
        arguments={},
    )

    result = validate_and_prepare_tool_call(action, make_state(), AVAILABLE_TOOLS)

    assert not result.is_valid
    assert "tool name must be a non-empty string" in result.error


def test_split_image_tool_builds_image_dict():
    result = validate_and_prepare_tool_call(
        tool_action("SplitImageIntoPatches", patch_configs={"img_0": 16}),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert result.is_valid
    image_data = result.call.tool_parameters["image_dict"]["img_0"]
    assert image_data["patch_size"] == 16
    assert_jpeg(image_data["image_bytes"])


def test_split_image_tool_rejects_empty_patch_configs():
    result = validate_and_prepare_tool_call(
        tool_action("SplitImageIntoPatches", patch_configs={}),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "must contain at least one image name" in result.error


def test_split_image_tool_accepts_exactly_64_patches():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (64, 64), "white"))

    result = validate_and_prepare_tool_call(
        tool_action("SplitImageIntoPatches", patch_configs={"img_0": 8}),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid


def test_split_image_tool_rejects_single_image_with_more_than_64_patches_before_execution():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (64, 64), "white"))

    result = validate_and_prepare_tool_call(
        tool_action("SplitImageIntoPatches", patch_configs={"img_0": 7}),
        state,
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "image img_0 would be split into 81 patches" in result.error
    assert "single image can be split into at most 64 patches" in result.error
    assert "Increase the patch size for this image" in result.error


def test_split_image_tool_applies_patch_limit_across_all_images():
    state = make_state()
    state.images["img_1"] = {"image": Image.new("RGB", (32, 24), "black")}

    result = validate_and_prepare_tool_call(
        tool_action(
            "SplitImageIntoPatches",
            patch_configs={"img_0": 4, "img_1": 4},
        ),
        state,
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "requested images would create 96 patches in total" in result.error
    assert "img_0: 48" in result.error
    assert "img_1: 48" in result.error
    assert "one image per call" in result.error
    assert "Each image can be split into at most 64 patches" in result.error


def test_split_image_tool_accepts_multiple_images_with_exactly_64_patches_in_total():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (64, 32), "white"))
    state.images["img_1"] = {"image": Image.new("RGB", (64, 32), "black")}

    result = validate_and_prepare_tool_call(
        tool_action(
            "SplitImageIntoPatches",
            patch_configs={"img_0": 8, "img_1": 8},
        ),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid


@pytest.mark.parametrize("patch_size", [0, -1, 1.5, True])
def test_split_image_tool_rejects_invalid_patch_size(patch_size):
    result = validate_and_prepare_tool_call(
        tool_action("SplitImageIntoPatches", patch_configs={"img_0": patch_size}),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "positive integer" in result.error


def test_merge_box_mask_encodes_selected_results_and_image_offsets():
    state = make_state()
    state.images["img_0"].update(
        {
            "bboxes": np.array([[1, 2, 10, 12]]),
            "masks": np.ones((1, 24, 32), dtype=np.uint8),
        }
    )
    state.images["img_0_r1_c1"] = {
        "image": Image.new("RGB", (16, 12), "black"),
        "parent_image": "img_0",
        "creator_tool": "SplitImageIntoPatches",
        "lineage_turns": [1],
        "transform_to_img0": np.array([[1, 0, 8], [0, 1, 6], [0, 0, 1]], dtype=float),
        "offset_x": 8,
        "offset_y": 6,
    }

    result = validate_and_prepare_tool_call(
        tool_action("MergeBoxMask", images=["img_0"]),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid
    image_dict = result.call.tool_parameters["image_dict"]
    selected = image_dict["img_0"]
    assert selected["bboxes"] == [[1, 2, 10, 12]]
    decoded = mask_utils.decode(selected["masks"])
    assert decoded.shape == (24, 32, 1)
    assert selected["has_split_ancestor"] is False
    assert image_dict["img_0_r1_c1"] == {
        "width": 16,
        "height": 12,
        "offset_x": 8.0,
        "offset_y": 6.0,
        "transform_to_img0": [[1.0, 0.0, 8.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]],
        "depth": 1,
        "has_split_ancestor": True,
    }


def test_merge_box_mask_requires_bboxes_and_masks():
    result = validate_and_prepare_tool_call(
        tool_action("MergeBoxMask", images=["img_0"]),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "must have already been obtained valid bboxes and masks" in result.error


def test_merge_box_mask_marks_super_resolved_patch_as_split_descendant():
    state = make_state()
    state.images["img_0_r1_c1"] = {
        "image": Image.new("RGB", (16, 12), "black"),
        "parent_image": "img_0",
        "creator_tool": "SplitImageIntoPatches",
        "lineage_turns": [1],
        "transform_to_img0": np.array([[1, 0, 8], [0, 1, 6], [0, 0, 1]], dtype=float),
    }
    state.images["img_0_r1_c1_4x"] = {
        "image": Image.new("RGB", (64, 48), "black"),
        "parent_image": "img_0_r1_c1",
        "creator_tool": "SuperResolution",
        "lineage_turns": [1, 2],
        "transform_to_img0": np.array([[0.25, 0, 8], [0, 0.25, 6], [0, 0, 1]], dtype=float),
        "bboxes": np.array([[4, 4, 20, 20]]),
        "masks": np.ones((1, 48, 64), dtype=np.uint8),
    }

    result = validate_and_prepare_tool_call(
        tool_action("MergeBoxMask", images=["img_0_r1_c1_4x"]),
        state,
        AVAILABLE_TOOLS,
    )

    assert result.is_valid
    selected = result.call.tool_parameters["image_dict"]["img_0_r1_c1_4x"]
    assert selected["has_split_ancestor"] is True
    assert selected["depth"] == 2
    assert selected["transform_to_img0"] == [
        [0.25, 0.0, 8.0],
        [0.0, 0.25, 6.0],
        [0.0, 0.0, 1.0],
    ]


def test_merge_box_mask_empty_images_tells_model_to_answer_directly():
    result = validate_and_prepare_tool_call(
        tool_action("MergeBoxMask", images=[]),
        make_state(),
        AVAILABLE_TOOLS,
    )

    assert not result.is_valid
    assert "directly output the final answer instead of calling MergeBoxMask" in result.error


def test_non_tool_action_is_rejected():
    action = ParsedAction(action_type="answer", answer="done")

    result = validate_and_prepare_tool_call(action, make_state(), AVAILABLE_TOOLS)

    assert not result.is_valid
    assert "Only a parsed tool_call action" in result.error
