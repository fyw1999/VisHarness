from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from visharness.agent_loop.tool_response_processor import (
    append_tool_response_guidance,
    archive_visible_images,
    build_error_observation,
    normalize_tool_response_closing_spacing,
    process_tool_response,
)
from visharness.agent_loop.trajectory_state import VisionTrajectoryState
from visharness.prompts import VISION_TOOL_PROMPT


def image_bytes(color="white", size=(16, 12)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_phrase_to_point_updates_state_and_returns_visual_observation():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    result = {
        "img_0": {
            "visual_image": image_bytes("red"),
            "text_response": "Found one point.",
            "points": [[4, 5]],
        },
    }

    processed = process_tool_response("PhraseToPoint", result, state)

    assert state.images["img_0"]["points"].tolist() == [[4, 5]]
    assert processed.image_names == ["img_0_PhraseToPoint_visual"]
    assert len(processed.images) == 1
    assert processed.message["role"] == "user"
    assert processed.message["content"][-1]["text"] == "</tool_response>"
    assert processed.result_summary["per_image"]["img_0"]["points"] == [[4.0, 5.0]]


def test_phrase_to_point_empty_result_clears_stale_points():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    state.images["img_0"]["points"] = np.asarray([[4, 5]])
    result = {
        "img_0": {
            "visual_image": image_bytes("red"),
            "text_response": "Found no points.",
            "points": [],
        },
    }

    processed = process_tool_response("PhraseToPoint", result, state, turn_index=2)

    assert "points" not in state.images["img_0"]
    assert processed.result_summary["per_image"]["img_0"]["points"] == []


def test_box_mask_response_keeps_only_valid_bbox_mask_pairs():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    masks = np.zeros((12, 16, 2), dtype=np.uint8)
    masks[2:6, 1:5, 1] = 1
    encoded = mask_utils.encode(np.asfortranarray(masks))
    result = {
        "img_0": {
            "visual_image": image_bytes("green"),
            "text_response": "Returned one empty and one valid mask.",
            "bboxes": [[0, 0, 0, 0], [1, 2, 4, 5]],
            "masks": encoded,
        },
    }

    process_tool_response("PointToBoxMask", result, state)

    assert state.images["img_0"]["bboxes"].tolist() == [[1.0, 2.0, 4.0, 5.0]]
    assert state.images["img_0"]["masks"].shape == (1, 12, 16)
    assert state.images["img_0"]["masks"][0].sum() == 16


def test_box_mask_response_clears_stale_state_when_no_valid_pairs():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    state.images["img_0"]["bboxes"] = np.asarray([[1, 1, 4, 4]])
    state.images["img_0"]["masks"] = np.ones((1, 12, 16), dtype=bool)
    masks = np.zeros((12, 16, 1), dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(masks))
    result = {
        "img_0": {
            "visual_image": image_bytes("yellow"),
            "text_response": "Returned only an empty mask.",
            "bboxes": [[0, 0, 0, 0]],
            "masks": encoded,
        },
    }

    process_tool_response("PointToBoxMask", result, state)

    assert "bboxes" not in state.images["img_0"]
    assert "masks" not in state.images["img_0"]


def test_tool_observations_end_exactly_at_closing_tag():
    error_observation = build_error_observation("Invalid tool arguments.")

    assert error_observation.message["content"][-1]["text"].endswith("</tool_response>")
    assert not error_observation.message["content"][-1]["text"].endswith("</tool_response>\n")


def test_append_tool_response_guidance_places_prompt_inside_wrapper():
    observation = build_error_observation("Tool result.")

    append_tool_response_guidance(
        observation,
        f"{VISION_TOOL_PROMPT}\n\n",
    )

    final_text = observation.message["content"][-1]["text"]
    normalized_prompt = VISION_TOOL_PROMPT.rstrip("\r\n")
    assert normalized_prompt in final_text
    assert final_text.endswith(
        f"{normalized_prompt}\n</tool_response>"
    )
    assert not final_text.endswith(
        f"{normalized_prompt}\n\n</tool_response>"
    )
    assert final_text.index(normalized_prompt) < final_text.index(
        "</tool_response>"
    )


def test_split_image_adds_patches_to_trajectory_state():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    result = {
        "img_0": {
            "overview": image_bytes("blue"),
            "patches": {
                "img_0_r1_c1": {
                    "image_bytes": image_bytes("green", (8, 8)),
                    "offset_x": 3,
                    "offset_y": 4,
                }
            },
        },
    }

    processed = process_tool_response("SplitImageIntoPatches", result, state, turn_index=3)

    assert state.images["img_0_r1_c1"]["offset_x"] == 3
    assert state.images["img_0_r1_c1"]["offset_y"] == 4
    assert state.images["img_0_r1_c1"]["created_by_turn"] == 3
    assert state.images["img_0_r1_c1"]["lineage_turns"] == [3]
    assert state.images["img_0_r1_c1"]["transform_to_img0"].tolist() == [
        [1.0, 0.0, 3.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ]
    assert processed.result_summary["created_images"]["img_0_r1_c1"]["created_by_turn"] == 3
    assert processed.image_names == ["img_0_split_overview", "img_0_r1_c1"]


def test_final_image_has_no_blank_line_before_tool_response_close():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    result = {
        "img_0": {
            "overview": image_bytes("blue"),
            "patches": {
                "img_0_r1_c1": {
                    "image_bytes": image_bytes("green", (8, 8)),
                    "offset_x": 0,
                    "offset_y": 0,
                }
            },
        },
    }

    processed = process_tool_response("SplitImageIntoPatches", result, state)
    normalize_tool_response_closing_spacing(processed)

    content = processed.message["content"]
    assert content[-3]["type"] == "image"
    assert content[-2] == {"type": "text", "text": "\n"}
    assert content[-1] == {"type": "text", "text": "</tool_response>"}


def test_image_keeps_blank_line_before_following_guidance():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    result = {
        "img_0": {
            "visual_image": image_bytes("red"),
            "text_response": "Found one point.",
            "points": [[4, 5]],
        },
    }

    processed = process_tool_response("PhraseToPoint", result, state)
    append_tool_response_guidance(processed, VISION_TOOL_PROMPT)
    normalize_tool_response_closing_spacing(processed)

    content = processed.message["content"]
    image_index = next(index for index, part in enumerate(content) if part["type"] == "image")
    assert content[image_index + 1] == {"type": "text", "text": "\n\n"}
    normalized_prompt = VISION_TOOL_PROMPT.rstrip("\r\n")
    assert content[-2]["text"].endswith(
        f"{normalized_prompt}\n"
    )
    assert content[-1] == {"type": "text", "text": "</tool_response>"}


def test_merge_result_is_saved_with_json_safe_rle_counts():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    mask = np.ones((12, 16, 1), dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask))
    result = {"final_bboxes": [[0, 0, 15, 11]], "final_masks": encoded, "count": 1}

    processed = process_tool_response("MergeBoxMask", result, state)

    assert processed.final_results["count"] == 1
    assert isinstance(processed.final_results["final_masks"][0]["counts"], str)
    assert state.final_results == processed.final_results


def test_merge_result_asserts_non_empty_final_results_before_saving():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    result = {"final_bboxes": [], "final_masks": [], "count": 0}

    with pytest.raises(AssertionError, match="no final objects"):
        process_tool_response("MergeBoxMask", result, state)

    assert state.final_results is None


def test_merge_result_asserts_non_empty_final_masks_before_saving():
    state = VisionTrajectoryState.from_initial_image(Image.new("RGB", (16, 12), "white"))
    mask = np.zeros((12, 16, 1), dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask))
    result = {"final_bboxes": [[0, 0, 15, 11]], "final_masks": encoded, "count": 1}

    with pytest.raises(AssertionError, match="final mask 0 is empty"):
        process_tool_response("MergeBoxMask", result, state)

    assert state.final_results is None


def test_archive_visible_images_replaces_structured_image_items():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look: "},
                {"type": "image", "image": Image.new("RGB", (2, 2)), "image_name": "img_0"},
            ],
        }
    ]

    archive_visible_images(messages, ["img_0"])

    assert all(item["type"] != "image" for item in messages[0]["content"])
    assert "img_0" in messages[0]["content"][1]["text"]


def test_archive_visible_images_ignores_image_marker_in_assistant_text():
    assistant_content = (
        "<think>The tool returned no objects.</think> <image> "
        '<tool_call>{"name": "PhraseToPoint"}</tool_call>'
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool visualization: "},
                {
                    "type": "image",
                    "image": Image.new("RGB", (2, 2)),
                    "image_name": "img_0_PhraseToPoint_visual",
                },
            ],
        },
        {"role": "assistant", "content": assistant_content},
    ]

    archive_visible_images(messages, ["img_0_PhraseToPoint_visual"])

    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "[System: Visualization image has been archived to save memory]",
    }
    assert messages[1]["content"] == assistant_content
