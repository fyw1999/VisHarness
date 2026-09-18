"""Validate parsed VisHarness actions and build visual-tool parameters."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils

from .action_parser import ParsedAction
from .trajectory_state import VisionTrajectoryState

IMAGE_LIST_TOOLS = {
    "PhraseToPoint",
    "PhraseToBoxMask",
    "SuperResolution",
    "PointToBoxMask",
    "MergeBoxMask",
}
PHRASE_TOOLS = {"PhraseToPoint", "PhraseToBoxMask"}
SUPPORTED_VISUAL_TOOLS = IMAGE_LIST_TOOLS | {"SplitImageIntoPatches"}

MAX_PATCHES_PER_IMAGE = 64
MAX_PATCHES_PER_SPLIT_CALL = 64
_SPLIT_MIN_USAGE_RATIO = 0.3


@dataclass(frozen=True)
class ValidatedToolCall:
    """A tool call whose parameters are ready for a visual-tool worker."""

    tool_name: str
    tool_parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCallValidationResult:
    """Result of validating one parsed tool call against trajectory state."""

    call: ValidatedToolCall | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.call is not None


def _failure(message: str) -> ToolCallValidationResult:
    return ToolCallValidationResult(error=message)


def _image_to_jpeg_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def _as_list(value: Any) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Expected a list-like value, got {type(value).__name__}")


def _has_split_ancestor(state: VisionTrajectoryState, image_name: str) -> bool:
    """Return whether an image was created by, or descends from, a split."""

    visited: set[str] = set()
    current_name: str | None = image_name
    while current_name is not None and current_name not in visited:
        visited.add(current_name)
        image_state = state.images.get(current_name)
        if image_state is None:
            return False
        if image_state.get("creator_tool") == "SplitImageIntoPatches":
            return True
        parent_name = image_state.get("parent_image")
        current_name = parent_name if isinstance(parent_name, str) and parent_name else None
    return False


def _merge_image_metadata(
    state: VisionTrajectoryState,
    image_name: str,
    image_state: dict[str, Any],
) -> dict[str, Any]:
    """Build geometry metadata consumed internally by MergeBoxMask."""

    width, height = image_state["image"].size
    transform = np.asarray(image_state.get("transform_to_img0", np.eye(3)), dtype=float)
    lineage_turns = [int(value) for value in image_state.get("lineage_turns", [])]
    return {
        "width": int(width),
        "height": int(height),
        "offset_x": float(image_state.get("offset_x", 0)),
        "offset_y": float(image_state.get("offset_y", 0)),
        "transform_to_img0": transform.tolist(),
        "depth": len(lineage_turns),
        "has_split_ancestor": _has_split_ancestor(state, image_name),
    }


def _count_split_segments(
    total_length: int,
    patch_size: int,
    *,
    min_usage_ratio: float = _SPLIT_MIN_USAGE_RATIO,
) -> int:
    """Count segments exactly as SplitImageIntoPatches.get_segments does."""

    if patch_size >= total_length:
        return 1

    complete_segments, remainder = divmod(total_length, patch_size)
    if remainder == 0 or remainder < patch_size * min_usage_ratio:
        return complete_segments
    return complete_segments + 1


def _get_image_state(
    state: VisionTrajectoryState,
    image_name: str,
) -> tuple[dict[str, Any] | None, ToolCallValidationResult | None]:
    if image_name not in state.images:
        return None, _failure(
            f"Incorrect tool call format: the specified image {image_name} does not exist, "
            "please check if the image has been properly obtained from previous tool executions."
        )
    image_state = state.images[image_name]
    return image_state, None


def _validate_image_names(tool_name: str, arguments: dict[str, Any]) -> tuple[list[str] | None, str | None]:
    images = arguments.get("images")
    if images is None:
        return None, (
            f"Incorrect tool call format: invalid tool call parameters, 'images' is required for tool: {tool_name}"
        )
    if not isinstance(images, list):
        return None, (
            f"Incorrect tool call format: invalid tool call parameters, 'images' must be a list of image names "
            f"for tool: {tool_name}"
        )
    if len(images) == 0:
        if tool_name == "MergeBoxMask":
            return None, (
                "Incorrect tool call: MergeBoxMask requires at least one image that already has valid bboxes and "
                "masks from previous tool executions. If there is no target object in the image, directly output "
                "the final answer instead of calling MergeBoxMask."
            )
        return None, (
            f"Incorrect tool call format: invalid tool call parameters, 'images' must be a non-empty list of image "
            f"names for tool: {tool_name}"
        )
    if not all(isinstance(name, str) and name for name in images):
        return None, (
            f"Incorrect tool call format: invalid tool call parameters, 'images' must be a list of image names "
            f"for tool: {tool_name}"
        )
    return images, None


def _prepare_regular_image_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: VisionTrajectoryState,
) -> ToolCallValidationResult:
    tool_parameters: dict[str, Any] = {}

    if tool_name in PHRASE_TOOLS:
        phrase = arguments.get("phrase", None)
        if not isinstance(phrase, str) or not phrase.strip():
            return _failure(
                "Incorrect tool call format: invalid tool call parameters, 'phrase' is required and must be a "
                f"non-empty string for tool: {tool_name}"
            )
        tool_parameters["phrase"] = phrase.strip()

    if tool_name == "PointToBoxMask":
        raw_mode = arguments.get("mode")
        if raw_mode is None:
            return _failure(
                "Incorrect tool call format: invalid tool call parameters, 'mode' is required for tool: "
                f"{tool_name}, and its value should be either 'confidence' or 'area'"
            )
        if not isinstance(raw_mode, str):
            return _failure(
                "Incorrect tool call format: invalid tool call parameters, 'mode' must be a string for tool: "
                f"{tool_name}, and its value should be either 'confidence' or 'area'"
            )
        mode = raw_mode.strip().lower()
        if mode not in {"confidence", "area"}:
            return _failure(
                "Incorrect tool call format: invalid tool call parameters, 'mode' is required for tool: "
                f"{tool_name}, and its value should be either 'confidence' or 'area'"
            )
        tool_parameters["mode"] = mode

    image_names, image_error = _validate_image_names(tool_name, arguments)
    if image_error is not None:
        return _failure(image_error)

    image_dict: dict[str, Any] = {}
    for image_name in image_names:
        image_state, state_error = _get_image_state(state, image_name)
        if state_error is not None:
            return state_error

        if tool_name == "SuperResolution" and image_name.endswith("4x"):
            return _failure(
                "Incorrect tool call: do not apply SuperResolution to the same image twice. "
                f"The specified image {image_name} appears to have already been super-resolved because its name "
                "ends with '4x'. Applying SuperResolution repeatedly can create an excessively high-resolution "
                "image and should be avoided. Please use the existing enhanced image or choose another tool."
            )

        image_bytes = _image_to_jpeg_bytes(image_state["image"])

        if tool_name == "PointToBoxMask":
            points = image_state.get("points")
            if points is None or len(points) == 0:
                return _failure(
                    "Incorrect tool call: for PointToBoxMask tool, the specified image "
                    f"{image_name} must have already been obtained valid points from PhraseToPoint tool execution"
                )

            image_dict[image_name] = {"image_bytes": image_bytes, "points": _as_list(points)}

        elif tool_name == "MergeBoxMask":
            bboxes = image_state.get("bboxes", None)
            masks = image_state.get("masks", None)
            if bboxes is None or len(bboxes) == 0 or masks is None or len(masks) == 0:
                return _failure(
                    "Incorrect tool call: for MergeBoxMask tool, the specified image "
                    f"{image_name} must have already been obtained valid bboxes and masks results from previous "
                    "tool executions"
                )

            masks_array = np.asarray(masks)
            masks_hwn = np.transpose(masks_array, (1, 2, 0))
            encoded_masks = mask_utils.encode(np.asfortranarray(masks_hwn.astype(np.uint8)))
            image_dict[image_name] = {
                **_merge_image_metadata(state, image_name, image_state),
                "image_bytes": image_bytes,
                "bboxes": _as_list(bboxes),
                "masks": encoded_masks,
            }

        else:
            image_dict[image_name] = image_bytes

    if tool_name == "MergeBoxMask":
        original_state, state_error = _get_image_state(state, "img_0")
        if state_error is not None:
            return state_error
        if "img_0" not in image_dict:
            image_dict["img_0"] = {
                **_merge_image_metadata(state, "img_0", original_state),
                "image_bytes": _image_to_jpeg_bytes(original_state["image"]),
            }

        for image_name, image_state in state.images.items():
            if image_name not in image_dict:
                image_dict[image_name] = _merge_image_metadata(state, image_name, image_state)

    tool_parameters["image_dict"] = image_dict
    return ToolCallValidationResult(call=ValidatedToolCall(tool_name, tool_parameters))


def _prepare_split_image_tool(
    arguments: dict[str, Any],
    state: VisionTrajectoryState,
) -> ToolCallValidationResult:
    patch_configs = arguments.get("patch_configs", None)
    if patch_configs is None:
        return _failure(
            "Incorrect tool call format: invalid tool call parameters, 'patch_configs' is required for tool: "
            "SplitImageIntoPatches"
        )
    if not isinstance(patch_configs, dict):
        return _failure(
            "Incorrect tool call format: invalid tool call parameters, 'patch_configs' must be a dictionary, "
            "where each key represents the image name, and the corresponding value specifies the patch size used "
            "when invoking the SplitImageIntoPatches tool."
        )
    if not patch_configs:
        return _failure(
            "Incorrect tool call format: invalid tool call parameters, 'patch_configs' must contain at least one "
            "image name and patch size for tool: SplitImageIntoPatches"
        )

    validated_configs: list[tuple[str, int, dict[str, Any]]] = []
    patch_counts: dict[str, int] = {}
    for image_name, patch_size in patch_configs.items():
        if not isinstance(image_name, str) or not image_name:
            return _failure(
                "Incorrect tool call format: every key in 'patch_configs' must be a non-empty image name."
            )
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
            return _failure(
                "Incorrect tool call format: every patch size in 'patch_configs' must be a positive integer."
            )
        image_state, state_error = _get_image_state(state, image_name)
        if state_error is not None:
            return state_error

        image_width, image_height = image_state["image"].size
        row_count = _count_split_segments(image_height, patch_size)
        column_count = _count_split_segments(image_width, patch_size)
        patch_counts[image_name] = row_count * column_count
        validated_configs.append((image_name, patch_size, image_state))

    total_patch_count = sum(patch_counts.values())
    if len(patch_counts) == 1 and total_patch_count > MAX_PATCHES_PER_IMAGE:
        image_name, patch_count = next(iter(patch_counts.items()))
        return _failure(
            f"Incorrect tool call: image {image_name} would be split into {patch_count} patches. "
            f"A single image can be split into at most {MAX_PATCHES_PER_IMAGE} patches. "
            "Increase the patch size for this image and regenerate the tool call."
        )

    if len(patch_counts) > 1 and total_patch_count > MAX_PATCHES_PER_SPLIT_CALL:
        per_image_counts = ", ".join(
            f"{image_name}: {patch_count}" for image_name, patch_count in patch_counts.items()
        )
        return _failure(
            f"Incorrect tool call: the requested images would create {total_patch_count} patches in total "
            f"({per_image_counts}), exceeding the maximum of {MAX_PATCHES_PER_SPLIT_CALL} patches allowed "
            "in one SplitImageIntoPatches call. Split this operation into multiple tool calls and process "
            f"one image per call. Each image can be split into at most {MAX_PATCHES_PER_IMAGE} patches; "
            "increase its patch size if needed."
        )

    image_dict: dict[str, dict[str, Any]] = {}
    for image_name, patch_size, image_state in validated_configs:
        image_dict[image_name] = {
            "image_bytes": _image_to_jpeg_bytes(image_state["image"]),
            "patch_size": patch_size,
        }

    return ToolCallValidationResult(
        call=ValidatedToolCall("SplitImageIntoPatches", {"image_dict": image_dict})
    )


def validate_and_prepare_tool_call(
    action: ParsedAction,
    state: VisionTrajectoryState,
    available_tools: Iterable[str],
) -> ToolCallValidationResult:
    """Validate a parsed action and construct parameters for the selected visual tool."""
    if action.action_type != "tool_call":
        return _failure("Only a parsed tool_call action can be validated as a visual-tool call.")
    if not isinstance(action.tool_name, str) or not action.tool_name.strip():
        return _failure("Incorrect tool call format: tool name must be a non-empty string.")
    if not isinstance(action.arguments, dict):
        return _failure("Incorrect tool call format: tool arguments must be a JSON object (dictionary).")

    available_tool_names = set(available_tools)
    if action.tool_name not in available_tool_names:
        return _failure(
            f"Incorrect tool call format: tool name {action.tool_name} not in available tool list "
            f"{sorted(available_tool_names)}"
        )

    if action.tool_name == "SplitImageIntoPatches":
        return _prepare_split_image_tool(action.arguments, state)
    return _prepare_regular_image_tool(action.tool_name, action.arguments, state)
