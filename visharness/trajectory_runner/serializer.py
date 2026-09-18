"""Thread-safe result and trajectory serialization."""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import threading
from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

from .item import TrajectoryItem
from .errors import TrajectoryPersistenceError
from .model_client import openai_message_to_assistant_message
from .openai_action_parser import (
    SUBMIT_FINAL_ANSWER_TOOL,
    openai_message_reasoning,
)


_TOOL_RESPONSE_OPEN = "<tool_response>"
_TOOL_RESPONSE_CLOSE = "</tool_response>"


def sanitize_submit_final_answer_reference(text: str) -> str:
    """Remove references to the data-generation-only final-answer tool."""

    return re.sub(
        rf"(?:\b(?:call|use)\s+)?{re.escape(SUBMIT_FINAL_ANSWER_TOOL)}",
        "provide the final answer",
        text,
    )


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def strip_tool_response_wrapper(content: Any) -> Any | None:
    """Remove only the outer Qwen tool-response tags from one observation.

    ``None`` means that ``content`` is not a wrapped environment observation.
    """
    if isinstance(content, str):
        if not content.startswith(_TOOL_RESPONSE_OPEN) or not content.endswith(_TOOL_RESPONSE_CLOSE):
            return None
        stripped = content[len(_TOOL_RESPONSE_OPEN) :]
        if stripped.startswith("\r\n"):
            stripped = stripped[2:]
        elif stripped.startswith("\n"):
            stripped = stripped[1:]
        return stripped[: -len(_TOOL_RESPONSE_CLOSE)]

    if not isinstance(content, list):
        return None

    copied_content = copy.deepcopy(content)
    text_indices = [
        index
        for index, part in enumerate(copied_content)
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    ]
    if not text_indices:
        return None

    first_index = text_indices[0]
    last_index = text_indices[-1]
    first_text = copied_content[first_index]["text"]
    last_text = copied_content[last_index]["text"]
    if not first_text.startswith(_TOOL_RESPONSE_OPEN) or not last_text.endswith(_TOOL_RESPONSE_CLOSE):
        return None

    first_text = first_text[len(_TOOL_RESPONSE_OPEN) :]
    if first_text.startswith("\r\n"):
        first_text = first_text[2:]
    elif first_text.startswith("\n"):
        first_text = first_text[1:]
    copied_content[first_index]["text"] = first_text

    # The opening and closing tags can be stored in the same text part.
    last_text = copied_content[last_index]["text"]
    copied_content[last_index]["text"] = last_text[: -len(_TOOL_RESPONSE_CLOSE)]

    return [
        part
        for part in copied_content
        if not (
            isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text") == ""
        )
    ]


def _result_with_legacy_final_visual_image(result: dict[str, Any]) -> dict[str, Any]:
    """Add the latest merged visualization to persisted ``final_results`` only.

    The shared rollout state intentionally keeps only boxes, masks, and count.
    The old trajectory-runner ckpt format also stored ``final_visual_image``
    inside ``final_results``. Recover it from the latest raw MergeBoxMask
    response without mutating the runtime result or shared trajectory state.
    """
    final_results = result.get("final_results")
    if not isinstance(final_results, dict) or "final_visual_image" in final_results:
        return result

    tool_responses = result.get("tool_response")
    if not isinstance(tool_responses, list):
        return result

    final_visual_image = None
    for tool_response in reversed(tool_responses):
        if not isinstance(tool_response, dict):
            continue
        if not {"final_bboxes", "final_masks", "count"}.issubset(tool_response):
            continue
        if tool_response.get("final_visual_image") is None:
            continue
        final_visual_image = tool_response["final_visual_image"]
        break

    if final_visual_image is None:
        return result

    result_for_storage = dict(result)
    enriched_final_results = copy.deepcopy(final_results)
    enriched_final_results["final_visual_image"] = copy.deepcopy(final_visual_image)
    result_for_storage["final_results"] = enriched_final_results
    return result_for_storage


class TrajectorySerializer:
    def __init__(
        self,
        save_path: str | os.PathLike | None,
        save_trajectory: bool = False,
    ):
        self.save_path = Path(save_path) if save_path else None
        self.save_trajectory = bool(save_trajectory)
        self._lock = threading.Lock()
        self.save_ckpt_path: Path | None = None
        self.save_trajectory_path: Path | None = None
        self.benchmark_summary_path: Path | None = None
        self.benchmark_history_path: Path | None = None
        self.resource_trace_path: Path | None = None

        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            save_base_name = self.save_path.name
            self.save_ckpt_path = self.save_path / f"{save_base_name}_ckpt.jsonl"
            self.benchmark_summary_path = self.save_path / "benchmark_run.json"
            self.benchmark_history_path = self.save_path / "benchmark_runs.jsonl"
            self.resource_trace_path = self.save_path / "resource_trace.jsonl"
            if self.save_trajectory:
                self.save_trajectory_path = self.save_path / f"{save_base_name}_trajectory.jsonl"
                (self.save_path / "images").mkdir(parents=True, exist_ok=True)

    def store_result(
        self,
        result: dict[str, Any],
    ) -> None:
        if self.save_ckpt_path is None:
            raise TrajectoryPersistenceError(
                "Checkpoint save path is not configured."
            )
        try:
            result_for_storage = _result_with_legacy_final_visual_image(result)
            self._append_jsonl(
                self.save_ckpt_path,
                self._json_safe(result_for_storage),
            )
        except TrajectoryPersistenceError:
            raise
        except Exception as exc:
            raise TrajectoryPersistenceError(
                f"Failed to save trajectory checkpoint to {self.save_ckpt_path}"
            ) from exc

    def save_item_trajectory(
        self,
        item: TrajectoryItem,
        step_index: int | None = None,
        *,
        data_generation: bool = False,
    ) -> None:
        if self.save_trajectory_path is None or self.save_path is None:
            return
        try:
            self._save_item_trajectory(
                item,
                step_index=step_index,
                data_generation=data_generation,
            )
        except TrajectoryPersistenceError:
            raise
        except Exception as exc:
            raise TrajectoryPersistenceError(
                f"Failed to save SFT trajectory to {self.save_trajectory_path}"
            ) from exc

    def _save_item_trajectory(
        self,
        item: TrajectoryItem,
        step_index: int | None = None,
        *,
        data_generation: bool = False,
    ) -> None:
        assert self.save_trajectory_path is not None
        assert self.save_path is not None
        item_id = item.meta_data.get("id", item.trajectory_uid)
        if step_index is None:
            step_index = item.current_round
        snapshot_id = f"{item_id}_step_{step_index}"
        image_dir = self.save_path / "images" / snapshot_id

        messages = (
            self._data_generation_sft_messages(item)
            if data_generation
            else item.conversation
        )
        if not messages or _message_role(messages[-1]) != "assistant":
            raise ValueError(
                "An SFT snapshot must end with the assistant message used as its training target."
            )
        matching_turns = [
            turn
            for turn in item.turn_records
            if int(turn.get("turn_index", -1)) == int(step_index)
        ]
        if len(matching_turns) != 1:
            raise ValueError(
                f"Expected exactly one turn record for SFT step {step_index}, got {len(matching_turns)}."
            )
        target_turn = matching_turns[0]
        snapshot = {
            "schema_version": 2,
            "id": snapshot_id,
            "trajectory_id": str(item_id),
            "target_turn_index": int(step_index),
            "target_message_index": len(messages) - 1,
            "target_action_type": target_turn.get("action_type"),
            "images": [],
            "messages": [],
        }
        for message in messages:
            snapshot["messages"].append(self._serialize_message(message, image_dir, snapshot))

        if data_generation and SUBMIT_FINAL_ANSWER_TOOL in json.dumps(
            snapshot,
            ensure_ascii=False,
        ):
            raise ValueError(
                "SFT serialization leaked the data-generation-only SubmitFinalAnswer tool."
            )

        self._append_jsonl(self.save_trajectory_path, snapshot)

    @staticmethod
    def _project_data_generation_assistant(
        message: Any,
        turn_record: dict[str, Any],
    ) -> dict[str, Any]:
        is_valid_final_answer = (
            turn_record.get("submit_final_answer_attempt") is True
            and turn_record.get("action_type") == "answer"
            and turn_record.get("sft_eligible") is True
        )
        if not is_valid_final_answer:
            assistant_message = openai_message_to_assistant_message(message)
            assistant_message["content"] = sanitize_submit_final_answer_reference(
                assistant_message.get("content", "")
            )
            # Data-generation APIs may omit ``type`` or expose a provider-
            # specific value even though the payload is an ordinary function
            # call. Runtime parsing deliberately accepts those variants, but
            # the SFT schema must remain stable for downstream chat templates.
            for tool_call in assistant_message.get("tool_calls", []):
                tool_call["type"] = "function"
            return assistant_message

        reasoning = openai_message_reasoning(message)
        answer = turn_record.get("answer")
        if not reasoning or not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "A valid final-answer SFT turn must contain non-empty reasoning and answer text."
            )
        reasoning = sanitize_submit_final_answer_reference(reasoning)
        answer = sanitize_submit_final_answer_reference(answer.strip())
        return {
            "role": "assistant",
            "content": (
                f"<think>\n{reasoning}\n</think>\n"
                f"<answer>\n{answer}\n</answer>"
            ),
        }

    def _data_generation_sft_messages(self, item: TrajectoryItem) -> list[Any]:
        records_by_assistant_index: dict[int, dict[str, Any]] = {}
        excluded_indices: set[int] = set()

        for turn_record in item.turn_records:
            if not isinstance(turn_record, dict):
                continue
            assistant_index = turn_record.get("assistant_message_index")
            if isinstance(assistant_index, int):
                records_by_assistant_index[assistant_index] = turn_record
            if turn_record.get("exclude_from_sft_history") is True:
                if isinstance(assistant_index, int):
                    excluded_indices.add(assistant_index)
                excluded_indices.update(
                    index
                    for index in turn_record.get("observation_message_indices", [])
                    if isinstance(index, int)
                )

        messages: list[Any] = []
        for message_index, message in enumerate(item.conversation):
            if message_index in excluded_indices:
                continue
            turn_record = records_by_assistant_index.get(message_index)
            if turn_record is not None and _message_role(message) == "assistant":
                messages.append(
                    self._project_data_generation_assistant(message, turn_record)
                )
            else:
                messages.append(copy.deepcopy(message))
        return messages

    def store_benchmark_summary(self, summary: dict[str, Any]) -> None:
        """Persist both the latest run summary and append-only run history."""

        if (
            self.benchmark_summary_path is None
            or self.benchmark_history_path is None
        ):
            return
        try:
            safe_summary = self._json_safe(summary)
            serialized = json.dumps(
                safe_summary,
                ensure_ascii=False,
                indent=2,
            )
            with self._lock:
                temporary_path = self.benchmark_summary_path.with_suffix(
                    ".json.tmp"
                )
                with temporary_path.open("w", encoding="utf-8") as file:
                    file.write(serialized)
                    file.write("\n")
                os.replace(temporary_path, self.benchmark_summary_path)
                with self.benchmark_history_path.open("a", encoding="utf-8") as file:
                    file.write(
                        json.dumps(safe_summary, ensure_ascii=False) + "\n"
                    )
        except TrajectoryPersistenceError:
            raise
        except Exception as exc:
            raise TrajectoryPersistenceError(
                "Failed to save trajectory-runner benchmark summary to "
                f"{self.benchmark_summary_path}"
            ) from exc

    def _serialize_message(
        self,
        message: Any,
        image_dir: Path,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(message, dict) and _message_role(message) == "assistant":
            message = openai_message_to_assistant_message(message)
        elif not isinstance(message, dict):
            raise TypeError(
                "Trajectory messages must be dict or ChatCompletionMessage, "
                f"got {type(message).__name__}"
            )

        clean_message = {"role": message.get("role")}
        for key in ("tool_call_id", "name"):
            if key in message:
                clean_message[key] = message[key]
        if "tool_calls" in message:
            clean_message["tool_calls"] = self._json_safe(message["tool_calls"])

        content = message.get("content")
        if isinstance(content, str):
            clean_message["content"] = content
            return clean_message
        if not isinstance(content, list):
            clean_message["content"] = self._json_safe(content)
            return clean_message

        clean_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image_name = part.get("image_name", f"image_{len(snapshot['images'])}")
                rel_path = f"images/{snapshot['id']}/{image_name}.jpg"
                snapshot["images"].append(rel_path)
                clean_content.append({"type": "image_url", "image_url": {"url": rel_path}})
                with self._lock:
                    image_dir.mkdir(parents=True, exist_ok=True)
                    part["image"].save(self.save_path / rel_path)
            elif isinstance(part, dict) and part.get("type") == "text":
                clean_content.append({"type": "text", "text": part.get("text", "")})
            else:
                clean_content.append(self._json_safe(part))
        clean_message["content"] = clean_content
        return clean_message

    def _append_jsonl(self, path: Path, item: dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False)
        with self._lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        from openai.types.chat import ChatCompletionMessage

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, ChatCompletionMessage):
            return cls._json_safe(value.model_dump(exclude_none=True))
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return base64.b64encode(value).decode("utf-8")
        if isinstance(value, Image.Image):
            buffer = BytesIO()
            value.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                mask_fortran = np.asfortranarray(np.transpose(value, (1, 2, 0)).astype(np.uint8))
                rles = mask_util.encode(mask_fortran)
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")
                return rles
            return value.tolist()
        if is_dataclass(value):
            return cls._json_safe(asdict(value))
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        return repr(value)
