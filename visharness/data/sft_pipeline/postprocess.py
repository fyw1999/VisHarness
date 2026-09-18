"""Normalize filtered trajectory snapshots before cross-model merging."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from visharness.prompts import (
    PHRASE_TO_BOXMASK_PROMPT,
    PHRASE_TO_POINT_PROMPT,
    POINT_TO_BOXMASK_PROMPT,
    SPLIT_PROMPT,
    SR_PROMPT,
    TRAIN_TEST_SYSTEM_PROMPT,
    VISION_TOOL_PROMPT,
)
from visharness.trajectory_runner.openai_action_parser import (
    SUBMIT_FINAL_ANSWER_TOOL,
)
from visharness.trajectory_runner.serializer import (
    sanitize_submit_final_answer_reference,
)

from .io import atomic_text_writer, iter_jsonl
from .schema import validate_snapshot


_REMOVED_GUIDANCE_PROMPTS = (
    PHRASE_TO_BOXMASK_PROMPT,
    PHRASE_TO_POINT_PROMPT,
    POINT_TO_BOXMASK_PROMPT,
    SPLIT_PROMPT,
    SR_PROMPT,
)
_CONTINUE_GUIDANCE_PREFIX = (
    "Continue the reasoning process to answer the original question:"
)
_THINK_PATTERN = re.compile(r"<think>.+?</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>.+?</answer>", re.DOTALL)


def _clean_guidance_text(text: str) -> str:
    preserves_vision_guidance = VISION_TOOL_PROMPT in text
    for prompt in _REMOVED_GUIDANCE_PROMPTS:
        text = text.replace(prompt, "")
    if _CONTINUE_GUIDANCE_PREFIX in text:
        text = text.split(_CONTINUE_GUIDANCE_PREFIX, 1)[0]
    if preserves_vision_guidance:
        # The model chat template adds the newline before </tool_response>.
        # Avoid storing another trailing newline in normalized SFT content.
        text = text.rstrip("\r\n")
    return text


def _normalize_text_content(content: Any, *, clean_guidance: bool) -> list[dict[str, Any]]:
    if isinstance(content, str):
        parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        parts = copy.deepcopy(content)
    else:
        raise ValueError(f"Message content must be string or list, got {type(content).__name__}")

    normalized: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Message content parts must be JSON objects")
        if part.get("type") != "text":
            normalized.append(part)
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Text content parts must contain a string text field")
        if clean_guidance:
            text = _clean_guidance_text(text)
        text = sanitize_submit_final_answer_reference(text)
        if text:
            normalized.append({"type": "text", "text": text})
    return normalized


def _ensure_tool_call_separator(content: list[dict[str, Any]]) -> None:
    """End non-empty assistant content with one newline before a tool call."""
    if not content:
        return
    last_part = content[-1]
    if last_part.get("type") == "text":
        text = last_part["text"]
        last_part["text"] = text.rstrip("\r\n") + "\n"
        return
    content.append({"type": "text", "text": "\n"})


def postprocess_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    processed = copy.deepcopy(snapshot)
    target_index = validate_snapshot(processed)
    messages = processed["messages"]

    for message in messages:
        role = message["role"]
        if role == "system":
            message["content"] = [
                {"type": "text", "text": TRAIN_TEST_SYSTEM_PROMPT.strip()}
            ]
        elif role == "tool":
            message["content"] = _normalize_text_content(
                message["content"],
                clean_guidance=True,
            )
        elif role == "assistant":
            message["content"] = _normalize_text_content(
                message["content"],
                clean_guidance=False,
            )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                _ensure_tool_call_separator(message["content"])

    target_message = messages[target_index]
    target_text = "".join(
        part.get("text", "")
        for part in target_message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    if _THINK_PATTERN.search(target_text) is None:
        raise ValueError(
            f"SFT target in snapshot {processed['id']} must contain non-empty <think>...</think> reasoning"
        )

    target_action_type = processed.get("target_action_type")
    tool_calls = target_message.get("tool_calls") or []
    if target_action_type is None:
        target_action_type = "tool_call" if tool_calls else "answer"
        processed["target_action_type"] = target_action_type
    if target_action_type == "tool_call":
        if len(tool_calls) != 1:
            raise ValueError(
                f"Tool-call SFT target {processed['id']} must contain exactly one tool call"
            )
    elif target_action_type == "answer":
        if tool_calls or _ANSWER_PATTERN.search(target_text) is None:
            raise ValueError(
                f"Answer SFT target {processed['id']} must contain one <answer> block and no tool call"
            )
    else:
        raise ValueError(
            f"SFT target {processed['id']} has unsupported action type {target_action_type!r}"
        )

    if SUBMIT_FINAL_ANSWER_TOOL in json.dumps(processed, ensure_ascii=False):
        raise ValueError(
            f"Postprocessed snapshot {processed['id']} leaked {SUBMIT_FINAL_ANSWER_TOOL}"
        )
    processed["target_message_index"] = target_index
    validate_snapshot(processed)
    return processed


def postprocess_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Postprocessing output must not overwrite its input file")

    count = 0
    with atomic_text_writer(output_path) as output_file:
        for line_number, snapshot in iter_jsonl(input_path):
            processed = postprocess_snapshot(snapshot)
            output_file.write(json.dumps(processed, ensure_ascii=False) + "\n")
            count += 1
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "processed_snapshots": count,
    }
