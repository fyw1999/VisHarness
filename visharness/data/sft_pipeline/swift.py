"""Convert normalized SFT snapshots to the ms-swift conversation format."""

from __future__ import annotations

import copy
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from PIL import Image

from visharness.prompts import TOOLS_LIST
from visharness.trajectory_runner.openai_action_parser import (
    SUBMIT_FINAL_ANSWER_TOOL,
)

from .io import atomic_text_writer, iter_jsonl
from .schema import validate_snapshot


@dataclass(frozen=True, slots=True)
class Qwen3VLVisualBudget:
    """Qwen3-VL image preprocessing limits used by the SFT recipe."""

    image_max_token_num: int = 2048
    max_total_raw_image_patches: int = 56_000
    image_min_token_num: int = 4
    patch_size: int = 16
    spatial_merge_size: int = 2
    max_aspect_ratio: float = 200.0

    def __post_init__(self) -> None:
        integer_fields = (
            "image_max_token_num",
            "max_total_raw_image_patches",
            "image_min_token_num",
            "patch_size",
            "spatial_merge_size",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{field_name} must be a positive integer, got {value!r}"
                )
        if self.image_max_token_num < self.image_min_token_num:
            raise ValueError(
                "image_max_token_num must be greater than or equal to "
                "image_min_token_num"
            )
        if self.max_aspect_ratio <= 0:
            raise ValueError("max_aspect_ratio must be positive")

    @property
    def resize_factor(self) -> int:
        return self.patch_size * self.spatial_merge_size

    @property
    def min_pixels(self) -> int:
        return self.image_min_token_num * self.resize_factor**2

    @property
    def max_pixels(self) -> int:
        return self.image_max_token_num * self.resize_factor**2


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _qwen3vl_resized_shape(
    height: int,
    width: int,
    budget: Qwen3VLVisualBudget,
) -> tuple[int, int]:
    """Match Qwen3-VL/qwen-vl-utils smart_resize for still images."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if max(height, width) / min(height, width) > budget.max_aspect_ratio:
        raise ValueError(
            "Image aspect ratio exceeds the Qwen3-VL limit: "
            f"{width}x{height}, max_ratio={budget.max_aspect_ratio}"
        )

    factor = budget.resize_factor
    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    resized_pixels = resized_height * resized_width
    if resized_pixels > budget.max_pixels:
        scale = math.sqrt((height * width) / budget.max_pixels)
        resized_height = _floor_by_factor(height / scale, factor)
        resized_width = _floor_by_factor(width / scale, factor)
    elif resized_pixels < budget.min_pixels:
        scale = math.sqrt(budget.min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * scale, factor)
        resized_width = _ceil_by_factor(width * scale, factor)
    return resized_height, resized_width


def _image_patch_cost(
    image_path: Path,
    budget: Qwen3VLVisualBudget,
) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(
            f"Failed to inspect training image {image_path}: {exc}"
        ) from exc

    resized_height, resized_width = _qwen3vl_resized_shape(height, width, budget)
    raw_patches = (resized_height // budget.patch_size) * (
        resized_width // budget.patch_size
    )
    merge_unit = budget.spatial_merge_size**2
    if raw_patches % merge_unit:
        raise AssertionError(
            f"Qwen3-VL raw patch count is not divisible by merge unit: {raw_patches}"
        )
    return raw_patches, raw_patches // merge_unit


def _visual_budget_decision(
    *,
    snapshot_id: str,
    line_number: int,
    image_paths: Sequence[str],
    budget: Qwen3VLVisualBudget,
    cost_cache: dict[Path, tuple[int, int]],
) -> dict[str, Any]:
    total_raw_patches = 0
    total_merged_tokens = 0
    for image_path_value in image_paths:
        image_path = Path(image_path_value)
        cost = cost_cache.get(image_path)
        if cost is None:
            cost = _image_patch_cost(image_path, budget)
            cost_cache[image_path] = cost
        # Count every occurrence. Repeated image paths are encoded repeatedly.
        total_raw_patches += cost[0]
        total_merged_tokens += cost[1]

    accepted = total_raw_patches <= budget.max_total_raw_image_patches
    return {
        "snapshot_id": snapshot_id,
        "input_line_number": line_number,
        "image_count": len(image_paths),
        "total_raw_image_patches": total_raw_patches,
        "total_merged_image_tokens": total_merged_tokens,
        "max_total_raw_image_patches": budget.max_total_raw_image_patches,
        "accepted": accepted,
        "reason": None if accepted else "total_raw_image_patches_exceeds_limit",
    }


def _content_to_text(content: Any, *, snapshot_id: str, message_index: int) -> str:
    if not isinstance(content, list):
        raise ValueError(
            f"Snapshot {snapshot_id} message {message_index} must be postprocessed "
            "to list-form content before Swift conversion"
        )
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError(
                f"Snapshot {snapshot_id} message {message_index} has a non-object content part"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Snapshot {snapshot_id} message {message_index} has invalid text content"
                )
            chunks.append(text)
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if not isinstance(image_url, dict) or not isinstance(
                image_url.get("url"), str
            ):
                raise ValueError(
                    f"Snapshot {snapshot_id} message {message_index} has invalid image content"
                )
            chunks.append("<image>")
        else:
            raise ValueError(
                f"Snapshot {snapshot_id} message {message_index} has unsupported content type {part_type!r}"
            )
    return "".join(chunks)


def _validated_tool_call(
    tool_call: Any,
    *,
    snapshot_id: str,
    message_index: int,
) -> str:
    if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
        raise ValueError(
            f"Snapshot {snapshot_id} message {message_index} has a non-function tool call"
        )
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError(
            f"Snapshot {snapshot_id} message {message_index} has an invalid function call"
        )
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Snapshot {snapshot_id} message {message_index} has an empty tool name"
        )
    if name == SUBMIT_FINAL_ANSWER_TOOL:
        raise ValueError(f"Snapshot {snapshot_id} leaked {SUBMIT_FINAL_ANSWER_TOOL}")
    if not isinstance(arguments, dict):
        raise ValueError(
            f"Snapshot {snapshot_id} message {message_index} tool arguments must be an object"
        )
    return json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
    )


def _resolved_image_path(root: Path, relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path.replace("\\", "/"))
    if posix_path.is_absolute() or any(
        part in {"", ".", ".."} for part in posix_path.parts
    ):
        raise ValueError(
            f"Image path must be a safe relative path, got {relative_path!r}"
        )
    path = root.joinpath(*posix_path.parts).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"Image path escapes image root: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def convert_snapshot_to_swift(
    snapshot: dict[str, Any],
    *,
    image_root_dir: str | Path,
    tools_definition: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one snapshot and put loss only on its explicit target turn."""

    target_index = validate_snapshot(snapshot)
    snapshot_id = str(snapshot["id"])
    if SUBMIT_FINAL_ANSWER_TOOL in json.dumps(snapshot, ensure_ascii=False):
        raise ValueError(f"Snapshot {snapshot_id} leaked {SUBMIT_FINAL_ANSWER_TOOL}")

    target_message = snapshot["messages"][target_index]
    target_tool_calls = target_message.get("tool_calls") or []
    if not isinstance(target_tool_calls, list):
        raise ValueError(f"Snapshot {snapshot_id} target has invalid tool_calls")
    target_action_type = snapshot.get("target_action_type")
    if target_action_type is None:
        target_action_type = "tool_call" if target_tool_calls else "answer"
    if target_action_type == "tool_call":
        if len(target_tool_calls) != 1:
            raise ValueError(
                f"Snapshot {snapshot_id} tool-call target must contain exactly one tool call"
            )
    elif target_action_type == "answer":
        if target_tool_calls:
            raise ValueError(
                f"Snapshot {snapshot_id} answer target must not contain tool calls"
            )
    else:
        raise ValueError(
            f"Snapshot {snapshot_id} has unsupported target action {target_action_type!r}"
        )

    image_root = Path(image_root_dir)
    images = [
        str(_resolved_image_path(image_root, relative_path))
        for relative_path in snapshot["images"]
    ]
    output: dict[str, Any] = {
        "messages": [],
        "images": images,
    }
    selected_tools = TOOLS_LIST if tools_definition is None else tools_definition
    if selected_tools:
        output["tools"] = copy.deepcopy(list(selected_tools))

    target_output_indices: list[int] = []
    for message_index, message in enumerate(snapshot["messages"]):
        role = message["role"]
        content = _content_to_text(
            message["content"],
            snapshot_id=snapshot_id,
            message_index=message_index,
        )
        emitted: list[dict[str, Any]] = []
        if role in {"system", "user"}:
            emitted.append({"role": role, "content": content})
        elif role == "tool":
            emitted.append({"role": "tool_response", "content": content})
        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError(
                    f"Snapshot {snapshot_id} message {message_index} has invalid tool_calls"
                )
            if tool_calls and content and not content.endswith("\n"):
                raise ValueError(
                    f"Snapshot {snapshot_id} message {message_index} assistant content "
                    "must end with a newline before its tool call"
                )
            if content:
                emitted.append({"role": "assistant", "content": content})
            for tool_call in tool_calls:
                emitted.append(
                    {
                        "role": "tool_call",
                        "content": _validated_tool_call(
                            tool_call,
                            snapshot_id=snapshot_id,
                            message_index=message_index,
                        ),
                    }
                )

        start_index = len(output["messages"])
        for emitted_message in emitted:
            emitted_message["loss"] = message_index == target_index
            output["messages"].append(emitted_message)
        if message_index == target_index:
            target_output_indices.extend(range(start_index, len(output["messages"])))

    if not target_output_indices:
        raise ValueError(
            f"Snapshot {snapshot_id} target turn produced no Swift messages"
        )
    if any(
        message["loss"] != (index in target_output_indices)
        for index, message in enumerate(output["messages"])
    ):
        raise AssertionError("Swift loss-mask construction is inconsistent")
    return output


def convert_to_swift_format(
    input_file: str | Path,
    output_file: str | Path,
    image_root_dir: str | Path | None = None,
    tools_definition: Sequence[dict[str, Any]] | None = None,
    visual_budget: Qwen3VLVisualBudget | None = None,
    visual_budget_decisions_file: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a normalized JSONL dataset atomically."""

    input_path = Path(input_file)
    output_path = Path(output_file)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Swift output must not overwrite its input file")
    image_root = input_path.parent if image_root_dir is None else Path(image_root_dir)
    if visual_budget is None and visual_budget_decisions_file is not None:
        raise ValueError("visual_budget_decisions_file requires visual_budget")
    decisions_path = (
        None
        if visual_budget is None
        else Path(visual_budget_decisions_file)
        if visual_budget_decisions_file is not None
        else output_path.with_name("visual_budget_decisions.jsonl")
    )
    if decisions_path is not None and decisions_path.resolve() in {
        input_path.resolve(),
        output_path.resolve(),
    }:
        raise ValueError("Visual-budget decisions must use a separate output file")

    input_count = 0
    accepted_count = 0
    rejected_count = 0
    cost_cache: dict[Path, tuple[int, int]] = {}
    decisions_context = (
        atomic_text_writer(decisions_path) if decisions_path is not None else None
    )
    with (
        atomic_text_writer(output_path) as output,
        (
            decisions_context if decisions_context is not None else nullcontext(None)
        ) as decisions,
    ):
        for line_number, snapshot in iter_jsonl(input_path):
            input_count += 1
            converted = convert_snapshot_to_swift(
                snapshot,
                image_root_dir=image_root,
                tools_definition=tools_definition,
            )
            if visual_budget is not None:
                assert decisions is not None
                decision = _visual_budget_decision(
                    snapshot_id=str(snapshot["id"]),
                    line_number=line_number,
                    image_paths=converted["images"],
                    budget=visual_budget,
                    cost_cache=cost_cache,
                )
                decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                if not decision["accepted"]:
                    rejected_count += 1
                    continue
            output.write(json.dumps(converted, ensure_ascii=False) + "\n")
            accepted_count += 1
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "image_root_dir": str(image_root),
        "input_snapshots": input_count,
        "converted_snapshots": accepted_count,
        "rejected_visual_budget_snapshots": rejected_count,
    }
    if visual_budget is not None:
        report["visual_budget"] = {
            **asdict(visual_budget),
            "decisions_path": str(decisions_path),
        }
    return report
