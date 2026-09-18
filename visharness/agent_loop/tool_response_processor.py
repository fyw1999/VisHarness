"""Convert visual-tool worker results into the next VisHarness observation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from .trajectory_state import VisionTrajectoryState

_VISUAL_RESULT_TOOLS = {"PhraseToPoint", "PhraseToBoxMask", "PointToBoxMask"}
_PER_IMAGE_TOOLS = _VISUAL_RESULT_TOOLS | {"SplitImageIntoPatches", "SuperResolution"}


@dataclass
class ProcessedToolResponse:
    """Observation and state updates produced by one visual-tool call."""

    message: dict[str, Any]
    images: list[Image.Image]
    image_names: list[str]
    final_results: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None
    artifact_events: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _archive_placeholder(image_name: str) -> str:
    if "visual" in image_name or "split_overview" in image_name:
        return "[System: Visualization image has been archived to save memory]"
    return f"[System: Image {image_name} has been archived to save memory]"


def archive_visible_images(messages: list[dict[str, Any]], image_names: list[str]) -> None:
    """Replace structured, currently visible images with archival text placeholders.

    Initial user images and visual-tool results are represented by structured
    ``{"type": "image", ...}`` content items. Assistant messages contain only
    generated text, so a literal ``<image>`` in assistant output must not be
    interpreted as a real image or consume an entry from ``image_names``.
    """
    remaining_names = iter(image_names)
    archived = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                try:
                    image_name = next(remaining_names)
                except StopIteration as exc:
                    raise ValueError("Messages contain more visible images than image_names") from exc
                new_content.append({"type": "text", "text": _archive_placeholder(image_name)})
                archived += 1
            else:
                new_content.append(item)
        message["content"] = new_content

    if archived != len(image_names):
        raise ValueError(f"Archived {archived} visible images, expected {len(image_names)}")


def build_error_observation(message: str) -> ProcessedToolResponse:
    """Return a recoverable model-format or tool-argument error to the agent."""
    return ProcessedToolResponse(
        message={
            "role": "user",
            "content": [
                {"type": "text", "text": f"<tool_response>\n{message}\n</tool_response>"},
            ],
        },
        images=[],
        image_names=[],
        is_error=True,
    )


def append_tool_response_guidance(
    observation: ProcessedToolResponse,
    guidance: str,
) -> None:
    """Append guidance inside the observation's outer tool-response wrapper."""

    guidance = guidance.rstrip("\r\n")
    if not guidance:
        return
    content = observation.message.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Tool observation content must be a non-empty list")
    final_part = content[-1]
    if not isinstance(final_part, dict) or final_part.get("type") != "text":
        raise ValueError("Tool observation must end with a text content part")
    final_text = final_part.get("text")
    if not isinstance(final_text, str) or not final_text.endswith("</tool_response>"):
        raise ValueError("Tool observation must end with </tool_response>")

    if final_text == "</tool_response>":
        content.insert(-1, {"type": "text", "text": f"{guidance}\n"})
        return

    body = final_text[: -len("</tool_response>")]
    separator = "" if not body or body.endswith(("\n", "\r")) else "\n"
    final_part["text"] = f"{body}{separator}{guidance}\n</tool_response>"


def normalize_tool_response_closing_spacing(
    observation: ProcessedToolResponse,
) -> None:
    """Keep exactly one line break immediately before ``</tool_response>``.

    Image results use two line breaks before subsequent explanatory text so
    that the rendered prompt contains one blank line.  The closing XML tag is
    structural rather than explanatory content, so a final image followed
    directly by the closing tag must use only one line break.
    """

    content = observation.message.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Tool observation content must be a non-empty list")

    final_part = content[-1]
    if not isinstance(final_part, dict) or final_part.get("type") != "text":
        raise ValueError("Tool observation must end with a text content part")
    final_text = final_part.get("text")
    if not isinstance(final_text, str) or not final_text.endswith("</tool_response>"):
        raise ValueError("Tool observation must end with </tool_response>")

    closing_tag = "</tool_response>"
    if final_text != closing_tag:
        body = final_text[: -len(closing_tag)].rstrip("\r\n")
        final_part["text"] = f"{body}\n{closing_tag}"
        return

    if len(content) == 1:
        content.insert(0, {"type": "text", "text": "\n"})
        return

    previous_part = content[-2]
    if isinstance(previous_part, dict) and previous_part.get("type") == "text":
        previous_text = previous_part.get("text")
        if not isinstance(previous_text, str):
            raise ValueError("Tool observation text content must be a string")
        if previous_text.strip("\r\n"):
            normalized_text = previous_text.rstrip("\r\n")
            previous_part["text"] = f"{normalized_text}\n"
        else:
            previous_part["text"] = "\n"
        return

    if isinstance(previous_part, dict) and previous_part.get("type") == "image":
        content.insert(-1, {"type": "text", "text": "\n"})
        return

    raise ValueError("Tool observation must contain text or image content before </tool_response>")


def _append_text(content: list[dict[str, Any]], text: str) -> None:
    content.append({"type": "text", "text": text})


def _append_image(
    content: list[dict[str, Any]],
    images: list[Image.Image],
    image_names: list[str],
    image: Image.Image,
    image_name: str,
) -> None:
    content.append({"type": "image", "image": image, "image_name": image_name})
    images.append(image)
    image_names.append(image_name)


def _json_safe_final_results(tool_response: dict[str, Any]) -> dict[str, Any]:
    final_results = {
        "final_bboxes": copy.deepcopy(tool_response["final_bboxes"]),
        "final_masks": copy.deepcopy(tool_response["final_masks"]),
        "count": tool_response["count"],
    }
    for rle in final_results["final_masks"]:
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
    return final_results


def _json_safe_rles(rles: Any) -> list[dict[str, Any]]:
    if isinstance(rles, dict):
        rles = [rles]
    normalized = copy.deepcopy(list(rles or []))
    for rle in normalized:
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        rle["size"] = [int(value) for value in rle["size"]]
    return normalized


def _encode_json_safe_masks(masks_nhw: np.ndarray) -> list[dict[str, Any]]:
    masks_nhw = np.asarray(masks_nhw, dtype=np.uint8)
    if masks_nhw.ndim != 3 or masks_nhw.shape[0] == 0:
        return []
    masks_hwn = np.transpose(masks_nhw, (1, 2, 0))
    return _json_safe_rles(mask_utils.encode(np.asfortranarray(masks_hwn)))


def _image_summary(image_state: dict[str, Any]) -> dict[str, Any]:
    width, height = image_state["image"].size
    transform = np.asarray(image_state.get("transform_to_img0", np.eye(3)), dtype=float)
    return {
        "width": int(width),
        "height": int(height),
        "parent_image": image_state.get("parent_image"),
        "created_by_turn": image_state.get("created_by_turn"),
        "creator_tool": image_state.get("creator_tool"),
        "lineage_turns": [int(value) for value in image_state.get("lineage_turns", [])],
        "transform_to_img0": transform.tolist(),
    }


def _created_image_state(
    *,
    image: Image.Image,
    parent_name: str,
    parent_state: dict[str, Any],
    turn_index: int | None,
    creator_tool: str,
    transform_to_parent: np.ndarray,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict[str, Any]:
    parent_transform = np.asarray(parent_state.get("transform_to_img0", np.eye(3)), dtype=float)
    lineage_turns = list(parent_state.get("lineage_turns", []))
    if turn_index is not None:
        lineage_turns.append(int(turn_index))
    return {
        "image": image,
        "parent_image": parent_name,
        "created_by_turn": turn_index,
        "creator_tool": creator_tool,
        "lineage_turns": lineage_turns,
        "transform_to_img0": parent_transform @ np.asarray(transform_to_parent, dtype=float),
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def _decode_json_safe_rles(rles: list[dict[str, Any]]) -> np.ndarray:
    decode_ready_rles = copy.deepcopy(rles)
    for rle in decode_ready_rles:
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
    masks = mask_utils.decode(decode_ready_rles)
    if masks.ndim == 2:
        masks = masks[:, :, np.newaxis]
    return masks


def _assert_valid_final_results(final_results: dict[str, Any]) -> None:
    """Assert that MergeBoxMask produced non-empty, valid bbox/mask pairs."""
    final_bboxes = final_results.get("final_bboxes")
    final_masks = final_results.get("final_masks")
    count = final_results.get("count")

    assert isinstance(final_bboxes, list), "MergeBoxMask final_bboxes must be a list"
    assert isinstance(final_masks, list), "MergeBoxMask final_masks must be a list"
    assert isinstance(count, int), "MergeBoxMask count must be an integer"
    assert count > 0, "MergeBoxMask returned no final objects; final_results should not be empty"
    assert len(final_bboxes) == count, (
        f"MergeBoxMask final_bboxes length mismatch: len={len(final_bboxes)}, count={count}"
    )
    assert len(final_masks) == count, (
        f"MergeBoxMask final_masks length mismatch: len={len(final_masks)}, count={count}"
    )

    for index, bbox in enumerate(final_bboxes):
        bbox_array = np.asarray(bbox, dtype=float)
        assert bbox_array.shape == (4,), f"MergeBoxMask final bbox {index} must have shape (4,), got {bbox_array.shape}"
        assert np.all(np.isfinite(bbox_array)), f"MergeBoxMask final bbox {index} contains non-finite values"
        x0, y0, x1, y1 = bbox_array
        assert x1 > x0 and y1 > y0, f"MergeBoxMask final bbox {index} has non-positive area: {bbox}"

    for index, rle in enumerate(final_masks):
        assert isinstance(rle, dict), f"MergeBoxMask final mask {index} must be a COCO RLE dict"
        size = rle.get("size")
        assert isinstance(size, (list, tuple)) and len(size) == 2, (
            f"MergeBoxMask final mask {index} must contain size=[height, width]"
        )
        height, width = int(size[0]), int(size[1])
        assert height > 0 and width > 0, f"MergeBoxMask final mask {index} has invalid size: {size}"
        assert "counts" in rle, f"MergeBoxMask final mask {index} is missing RLE counts"

    masks = _decode_json_safe_rles(final_masks)
    assert masks.ndim == 3, f"MergeBoxMask decoded final masks must be 3D, got shape={masks.shape}"
    assert masks.shape[0] > 0 and masks.shape[1] > 0, (
        f"MergeBoxMask decoded final masks have invalid spatial shape: {masks.shape}"
    )
    assert masks.shape[2] == count, (
        f"MergeBoxMask decoded final mask count mismatch: decoded={masks.shape[2]}, count={count}"
    )
    for index in range(count):
        assert np.any(masks[:, :, index]), f"MergeBoxMask final mask {index} is empty"


def _store_valid_box_masks(image_state: dict[str, Any], result: dict[str, Any]) -> None:
    """Store only non-empty bbox/mask pairs from a box-mask visual tool result."""
    image_state.pop("bboxes", None)
    image_state.pop("masks", None)

    raw_bboxes = result.get("bboxes") or []
    raw_masks = result.get("masks") or []
    if len(raw_bboxes) == 0 or len(raw_masks) == 0:
        return

    bboxes = np.asarray(raw_bboxes, dtype=float)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        return

    decoded_masks = mask_utils.decode(raw_masks)
    if decoded_masks.ndim == 2:
        decoded_masks = decoded_masks[:, :, np.newaxis]
    masks_nhw = np.transpose(decoded_masks, (2, 0, 1))

    valid_bboxes = []
    valid_masks = []
    for bbox, mask in zip(bboxes, masks_nhw, strict=False):
        x0, y0, x1, y1 = bbox
        if not np.all(np.isfinite(bbox)):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        if not np.any(mask):
            continue
        valid_bboxes.append(bbox)
        valid_masks.append(mask.astype(bool))

    if valid_bboxes:
        image_state["bboxes"] = np.asarray(valid_bboxes, dtype=float)
        image_state["masks"] = np.asarray(valid_masks, dtype=bool)


def process_tool_response(
    tool_name: str,
    tool_response: Any,
    state: VisionTrajectoryState,
    turn_index: int | None = None,
) -> ProcessedToolResponse:
    """Update one trajectory and build the observation used by the next model turn."""
    content: list[dict[str, Any]] = [{"type": "text", "text": "<tool_response>\n"}]
    new_images: list[Image.Image] = []
    new_image_names: list[str] = []
    final_results = None
    result_summary: dict[str, Any] | None = None
    artifact_events: list[dict[str, Any]] = []

    if isinstance(tool_response, str):
        _append_text(content, f"{tool_response}\n")
    else:
        _append_text(content, f"The {tool_name} tool has returned the following text and visual results:\n")

        if tool_name in _PER_IMAGE_TOOLS:
            result_summary = {"type": tool_name, "per_image": {}, "created_images": {}}
            for image_name, result in tool_response.items():
                original_image_name = re.sub(r"_[24]x$", "", image_name) if tool_name == "SuperResolution" else image_name
                _append_text(
                    content,
                    f"The execution results of the {tool_name} tool on image {original_image_name} are as follows:\n",
                )

                if tool_name in _VISUAL_RESULT_TOOLS:
                    visual_image = _bytes_to_pil(result["visual_image"])
                    visual_name = f"{original_image_name}_{tool_name}_visual"
                    _append_text(content, result["text_response"])
                    _append_text(
                        content,
                        f" The visualization results of the tool {tool_name} on image "
                        f"{original_image_name} are shown below:\n",
                    )
                    _append_image(content, new_images, new_image_names, visual_image, visual_name)
                    _append_text(content, "\n\n")

                    image_state = state.images[original_image_name]
                    if tool_name == "PhraseToPoint":
                        image_state.pop("points", None)
                        if len(result["points"]) > 0:
                            image_state["points"] = np.asarray(result["points"])
                        points = np.asarray(image_state.get("points", np.empty((0, 2))), dtype=float)
                        result_summary["per_image"][original_image_name] = {
                            **_image_summary(image_state),
                            "result_type": "points",
                            "points": points.tolist(),
                        }
                    else:
                        _store_valid_box_masks(image_state, result)
                        bboxes = np.asarray(image_state.get("bboxes", np.empty((0, 4))), dtype=float)
                        masks = np.asarray(
                            image_state.get(
                                "masks",
                                np.empty(
                                    (0, image_state["image"].height, image_state["image"].width),
                                    dtype=bool,
                                ),
                            ),
                            dtype=bool,
                        )
                        result_summary["per_image"][original_image_name] = {
                            **_image_summary(image_state),
                            "result_type": "box_mask",
                            "bboxes": bboxes.tolist(),
                            "masks": _encode_json_safe_masks(masks),
                        }

                elif tool_name == "SplitImageIntoPatches":
                    patches = result["patches"]
                    _append_text(
                        content,
                        f"The image {original_image_name} has been divided into {len(patches)} patches:\n"
                        "- Visualization of the Overall Partitioning Scheme:\n",
                    )
                    overview = _bytes_to_pil(result["overview"])
                    _append_image(
                        content,
                        new_images,
                        new_image_names,
                        overview,
                        f"{original_image_name}_split_overview",
                    )
                    _append_text(content, "\n\n")

                    for patch_name, patch_data in patches.items():
                        patch_image = _bytes_to_pil(patch_data["image_bytes"])
                        _append_text(content, f"- Patch Image {patch_name}:\n")
                        _append_image(content, new_images, new_image_names, patch_image, patch_name)
                        _append_text(content, "\n\n")
                        offset_x = float(patch_data.get("offset_x", 0))
                        offset_y = float(patch_data.get("offset_y", 0))
                        transform_to_parent = np.array(
                            [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]],
                            dtype=float,
                        )
                        state.images[patch_name] = _created_image_state(
                            image=patch_image,
                            parent_name=original_image_name,
                            parent_state=state.images[original_image_name],
                            turn_index=turn_index,
                            creator_tool=tool_name,
                            transform_to_parent=transform_to_parent,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        )
                        created_summary = _image_summary(state.images[patch_name])
                        result_summary["created_images"][patch_name] = created_summary
                        artifact_events.append(
                            {
                                "artifact_type": "image",
                                "artifact_name": patch_name,
                                "producer_turn": turn_index,
                                **created_summary,
                            }
                        )

                elif tool_name == "SuperResolution":
                    enhanced_image = _bytes_to_pil(result)
                    parent_state = state.images[original_image_name]
                    parent_width, parent_height = parent_state["image"].size
                    enhanced_width, enhanced_height = enhanced_image.size
                    transform_to_parent = np.array(
                        [
                            [parent_width / enhanced_width, 0.0, 0.0],
                            [0.0, parent_height / enhanced_height, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=float,
                    )
                    state.images[image_name] = _created_image_state(
                        image=enhanced_image,
                        parent_name=original_image_name,
                        parent_state=parent_state,
                        turn_index=turn_index,
                        creator_tool=tool_name,
                        transform_to_parent=transform_to_parent,
                    )
                    created_summary = _image_summary(state.images[image_name])
                    result_summary["created_images"][image_name] = created_summary
                    artifact_events.append(
                        {
                            "artifact_type": "image",
                            "artifact_name": image_name,
                            "producer_turn": turn_index,
                            **created_summary,
                        }
                    )
                    _append_text(
                        content,
                        f"The super-resolved image {image_name} of image {original_image_name} has been obtained. "
                        f"To save GPU memory, {image_name} is not displayed here, but it has been stored in memory "
                        "and can be used by subsequent tool calls.\n",
                    )

        elif tool_name == "MergeBoxMask":
            final_results = _json_safe_final_results(tool_response)
            _assert_valid_final_results(final_results)
            state.final_results = final_results
            result_summary = {
                "type": tool_name,
                "final_results": copy.deepcopy(final_results),
            }
            _append_text(
                content,
                "The MergeBoxMask tool has merged the detection/segmentation results from all patches and "
                "returned the following text:\n"
                "After deduplicating and merging the detection/segmentation results from all patches, the final "
                f"number of objects in the entire image is {final_results['count']}.\n",
            )
        else:
            raise ValueError(f"Unsupported visual tool response: {tool_name}")

    # Qwen3-VL identifies multi-step tool observations with a strict
    # ``content.endswith("</tool_response>")`` check. A trailing newline here
    # makes it treat the observation as a new user query and strip historical
    # assistant reasoning when rendering the next turn.
    _append_text(content, "</tool_response>")
    return ProcessedToolResponse(
        message={"role": "user", "content": content},
        images=new_images,
        image_names=new_image_names,
        final_results=final_results,
        result_summary=result_summary,
        artifact_events=artifact_events,
    )
