"""Asynchronous multi-turn rollout loop for VisHarness."""

from __future__ import annotations

import copy
import json
import os
import pickle
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    ToolListWrap,
)
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
from verl.workers.rollout.replica import TokenOutput
from visharness.prompts import VISION_RESULT_TOOLS, VISION_TOOL_PROMPT
from visharness.tools.errors import ToolOOMRetriesExhaustedError

from .action_parser import parse_action
from .action_validator import validate_and_prepare_tool_call
from .tool_response_processor import (
    ProcessedToolResponse,
    append_tool_response_guidance,
    archive_visible_images,
    build_error_observation,
    normalize_tool_response_closing_spacing,
    process_tool_response,
)
from .trajectory_state import VisionTrajectoryState


def _positive_int(value: int | str | None, fallback: int | float, name: str) -> int | float:
    if value is None:
        value = fallback
    if value == float("inf"):
        return value
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


_AGENT_LOOP_DEBUGGER_ATTACHED = False


class _RolloutPromptOverlong(Exception):
    """Raised inside one trajectory when the multimodal rollout prompt is too long."""

    def __init__(self, *, stage: str, turn_index: int, message: str):
        super().__init__(message)
        self.stage = stage
        self.turn_index = turn_index
        self.message = message


def _is_rollout_max_prompt_overlong_error(exc: ValueError) -> bool:
    message = str(exc)
    return ("rollout.prompt_length" in message or "rollout_max_prompt_length" in message) and "exceeding" in message


def _restore_implicit_think_start(completion: str) -> str:
    """Restore the opening think tag when the chat template prefills it.

    Some thinking-model chat templates put ``<think>\n`` in the assistant
    generation prompt, so vLLM returns only the reasoning body and the closing
    ``</think>`` as generated tokens. We restore the string stored in chat
    history without changing the generated token ids or rollout logprobs.
    """

    if completion.startswith("<think>"):
        return completion
    separator = "" if completion.startswith(("\n", "\r")) else "\n"
    return f"<think>{separator}{completion}"


def _load_rgb_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.copy().convert("RGB")
    if isinstance(image_value, (str, Path)):
        image_path = str(image_value)
        if image_path.startswith("file://"):
            image_path = image_path[7:]
        with Image.open(image_path) as image:
            return image.convert("RGB")
    if isinstance(image_value, bytes):
        with Image.open(BytesIO(image_value)) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image value type for VisHarness initial image: {type(image_value).__name__}")


def _iter_prompt_content_items(messages: list[dict[str, Any]]):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            yield from content


def _load_initial_tool_image(messages: list[dict[str, Any]], extra_info: dict[str, Any]) -> Image.Image:
    """Load the tool-coordinate image without running the model processor.

    The model side may smart-resize images when building pixel values. Visual
    tools and ground-truth masks must stay in the dataset image coordinate
    system, matching the original Swift plugin behavior.
    """

    image_path = extra_info.get("image_path")
    if image_path:
        return _load_rgb_image(image_path)

    image_items = []
    for item in _iter_prompt_content_items(messages):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "image":
            image_items.append(item)
        elif item_type in {"video", "audio"}:
            raise ValueError("VisHarness currently supports image inputs only")

    if len(image_items) != 1:
        raise ValueError(f"VisHarness expects exactly one initial image, got {len(image_items)}")

    image_item = image_items[0]
    if "image" in image_item:
        return _load_rgb_image(image_item["image"])
    if "bytes" in image_item:
        return _load_rgb_image(image_item["bytes"])
    if "image_url" in image_item:
        return _load_rgb_image(image_item["image_url"])
    raise ValueError("Initial image item must contain one of 'image', 'bytes', or 'image_url'")


def _maybe_wait_for_agent_loop_debugger() -> None:
    """Optionally pause a Ray AgentLoopWorker for VSCode attach debugging."""

    global _AGENT_LOOP_DEBUGGER_ATTACHED
    if os.getenv("VISHARNESS_DEBUG_AGENT_LOOP", "0") != "1":
        return
    if _AGENT_LOOP_DEBUGGER_ATTACHED and os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_ONCE", "1") != "0":
        return

    host = os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_HOST", "0.0.0.0")
    port = int(os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_PORT", "5682"))
    wait = os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_WAIT", "1") != "0"

    try:
        import debugpy
    except ImportError:
        print("[debug] debugpy is not installed in the AgentLoopWorker process; continuing without debugger.")
        return

    try:
        debugpy.listen((host, port))
        _AGENT_LOOP_DEBUGGER_ATTACHED = True
        print(f"[debug] VisHarness AgentLoopWorker waiting for VSCode attach on {host}:{port}, pid={os.getpid()}")
    except RuntimeError as exc:
        print(f"[debug] debugpy.listen({host}:{port}) in AgentLoopWorker failed: {exc}")
        _AGENT_LOOP_DEBUGGER_ATTACHED = True

    if wait:
        debugpy.wait_for_client()
    debugpy.breakpoint()


def _summarize_pil_image(image: Image.Image) -> dict[str, Any]:
    return {
        "__type__": "PIL.Image",
        "mode": image.mode,
        "size": list(image.size),
    }


def _summarize_ndarray(array: Any) -> dict[str, Any]:
    np_array = np.asarray(array)
    summary: dict[str, Any] = {
        "__type__": "ndarray",
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype),
    }
    if np_array.size <= 32:
        summary["values"] = np_array.tolist()
    else:
        if np.issubdtype(np_array.dtype, np.number) or np_array.dtype == np.bool_:
            summary["min"] = float(np.nanmin(np_array)) if np_array.size else None
            summary["max"] = float(np.nanmax(np_array)) if np_array.size else None
            summary["nonzero"] = int(np.count_nonzero(np_array))
    return summary


def _json_safe_summary(value: Any, *, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return f"<max_depth:{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__type__": "bytes", "len": len(value)}
    if isinstance(value, Image.Image):
        return _summarize_pil_image(value)
    if isinstance(value, np.ndarray):
        return _summarize_ndarray(value)
    if isinstance(value, VisionTrajectoryState):
        return _summarize_trajectory_state(value, depth=depth + 1, max_depth=max_depth)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_summary(val, depth=depth + 1, max_depth=max_depth)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 80 and all(isinstance(item, int) for item in value):
            return {
                "__type__": "int_list",
                "len": len(value),
                "head": value[:20],
                "tail": value[-20:],
            }
        if len(value) > 40:
            return {
                "__type__": type(value).__name__,
                "len": len(value),
                "head": [_json_safe_summary(item, depth=depth + 1, max_depth=max_depth) for item in value[:20]],
                "tail": [_json_safe_summary(item, depth=depth + 1, max_depth=max_depth) for item in value[-5:]],
            }
        return [_json_safe_summary(item, depth=depth + 1, max_depth=max_depth) for item in value]
    return repr(value)


def _summarize_image_state(image_state: dict[str, Any], *, depth: int, max_depth: int) -> dict[str, Any]:
    summary = {}
    for key, value in image_state.items():
        if key == "image" and isinstance(value, Image.Image):
            summary[key] = _summarize_pil_image(value)
        elif isinstance(value, np.ndarray):
            summary[key] = _summarize_ndarray(value)
        else:
            summary[key] = _json_safe_summary(value, depth=depth + 1, max_depth=max_depth)
    return summary


def _summarize_trajectory_state(
    state: VisionTrajectoryState,
    *,
    depth: int = 0,
    max_depth: int = 6,
) -> dict[str, Any]:
    return {
        "finished": bool(state.finished),
        "num_images": len(state.images),
        "images": {
            image_name: _summarize_image_state(image_state, depth=depth + 1, max_depth=max_depth)
            for image_name, image_state in state.images.items()
        },
        "num_turns": len(state.turns),
        "turns": _json_safe_summary(state.turns, depth=depth + 1, max_depth=max_depth),
        "final_results": _json_safe_summary(state.final_results, depth=depth + 1, max_depth=max_depth),
    }


class VisHarnessAgentLoop(AgentLoopBase):
    """Run one stateful visual-agent trajectory."""

    def __init__(
        self,
        *args,
        tools: ToolListWrap | None = None,
        model_family: str = "qwen3_vl",
        archive_previous_images: bool = True,
        per_turn_max_response_length: int | None = None,
        rollout_max_prompt_length: int | None = None,
        validation_max_prompt_length: int | None = None,
        max_trajectory_length: int | None = None,
        max_agent_turns: int | None = None,
        validation_max_response_length: int | None = None,
        validation_sampling_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.model_family = model_family
        self.archive_previous_images = archive_previous_images
        self.response_length = self.rollout_config.response_length
        self.rollout_placeholder_response_length = self.response_length
        self.rollout_max_prompt_length = _positive_int(
            rollout_max_prompt_length,
            self.rollout_config.prompt_length,
            "rollout_max_prompt_length",
        )
        model_max_length = int(
            _positive_int(
                getattr(self.rollout_config, "max_model_len", None),
                int(self.rollout_max_prompt_length) + 1,
                "max_model_len",
            )
        )
        self.validation_max_prompt_length = int(
            _positive_int(
                validation_max_prompt_length,
                max(model_max_length - 1, 1),
                "validation_max_prompt_length",
            )
        )
        if self.validation_max_prompt_length >= model_max_length:
            raise ValueError(
                "validation_max_prompt_length must leave at least one token for generation: "
                f"got validation_max_prompt_length={self.validation_max_prompt_length}, "
                f"max_model_len={model_max_length}"
            )
        self.per_turn_max_response_length = _positive_int(
            per_turn_max_response_length,
            self.response_length,
            "per_turn_max_response_length",
        )
        # Deprecated compatibility knob. The training prompt length is enforced
        # when per-turn samples are tensorized, not during rollout.
        self.deprecated_max_trajectory_length = max_trajectory_length
        self.max_agent_turns = _positive_int(
            max_agent_turns,
            self.rollout_config.multi_turn.max_assistant_turns or float("inf"),
            "max_agent_turns",
        )
        self.validation_max_response_length = _positive_int(
            validation_max_response_length,
            self.per_turn_max_response_length,
            "validation_max_response_length",
        )
        raw_validation_sampling = dict(validation_sampling_params or {})
        self.validation_sampling_params = {
            key: (
                int(value)
                if key in {"top_k", "seed"}
                else float(value)
            )
            for key, value in raw_validation_sampling.items()
        }
        self._visharness_error_context: dict[str, Any] = {}

    def _update_rollout_error_context(self, **updates: Any) -> None:
        self._visharness_error_context.update(updates)

    def _dump_rollout_exception(
        self,
        exc: BaseException,
        *,
        sampling_params: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        context = dict(getattr(self, "_visharness_error_context", {}) or {})
        trajectory_uid = context.get("request_id") or "unknown"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_root = Path(os.getenv("VISHARNESS_ROLLOUT_ERROR_DUMP_DIR", "outputs/visharness_rollout_errors"))
        dump_dir = dump_root / f"{timestamp}_pid{os.getpid()}_{trajectory_uid}"

        try:
            dump_dir.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_exc:
            print(f"[VisHarness rollout error dump] failed to create {dump_dir}: {mkdir_exc}", flush=True)
            return

        exception_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        pickle_payload = {
            "exception": exception_info,
            "sampling_params": sampling_params,
            "kwargs": kwargs,
            "context": context,
        }
        try:
            with (dump_dir / "snapshot.pkl").open("wb") as file:
                pickle.dump(pickle_payload, file)
        except Exception as pickle_exc:
            print(f"[VisHarness rollout error dump] failed to write snapshot.pkl: {pickle_exc}", flush=True)

        summary = {
            "exception": exception_info,
            "trajectory_uid": trajectory_uid,
            "dump_dir": str(dump_dir),
            "context_keys": sorted(context.keys()),
            "sampling_params": _json_safe_summary(sampling_params),
            "raw_prompt": _json_safe_summary(kwargs.get("raw_prompt")),
            "extra_info": _json_safe_summary(kwargs.get("extra_info")),
            "ground_truth": _json_safe_summary(kwargs.get("ground_truth")),
            "messages": _json_safe_summary(context.get("messages")),
            "state": _json_safe_summary(context.get("state")),
            "current_image_names": _json_safe_summary(context.get("current_image_names")),
            "last_turn_record": _json_safe_summary(context.get("last_turn_record")),
            "last_tool_name": _json_safe_summary(context.get("last_tool_name")),
            "last_tool_parameters": _json_safe_summary(context.get("last_tool_parameters")),
            "last_tool_result": _json_safe_summary(context.get("last_tool_result")),
        }
        try:
            with (dump_dir / "summary.json").open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
        except Exception as json_exc:
            print(f"[VisHarness rollout error dump] failed to write summary.json: {json_exc}", flush=True)

        print(f"[VisHarness rollout error dump] saved trajectory snapshot to {dump_dir}", flush=True)

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        remove_system_prompt: bool = False,
        max_prompt_length: int | None = None,
    ) -> list[int]:
        """Tokenize a VisHarness rollout prompt with a separate real prompt limit.

        In verl's base agent loop, ``rollout.prompt_length`` is both the real
        generation prompt limit and the padded width of the raw rollout tensor.
        VisHarness returns placeholder raw tensors and trains from ``turn_records``,
        so we keep ``rollout.prompt_length`` small while enforcing the real
        multimodal rollout limit with ``self.rollout_max_prompt_length``.
        """

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        effective_max_prompt_length = int(
            self.rollout_max_prompt_length
            if max_prompt_length is None
            else max_prompt_length
        )
        if effective_max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        if len(prompt_ids) > effective_max_prompt_length:
            if images or videos or audios:
                raise ValueError(
                    f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                    "visharness "
                    f"rollout_max_prompt_length={effective_max_prompt_length}. Truncating multimodal token "
                    f"sequences corrupts vision/audio feature alignment, so this trajectory is aborted. "
                    f"Reduce the multimodal input size or increase VISHARNESS_ROLLOUT_MAX_PROMPT_LENGTH."
                )
            prompt_ids = prompt_ids[-effective_max_prompt_length:]

        return prompt_ids

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # _maybe_wait_for_agent_loop_debugger()
        self._visharness_error_context = {
            "sampling_params": sampling_params,
            "raw_prompt": kwargs.get("raw_prompt"),
            "extra_info": kwargs.get("extra_info"),
            "ground_truth": kwargs.get("ground_truth"),
        }
        try:
            return await self._run_impl(sampling_params, **kwargs)
        except Exception as exc:
            self._dump_rollout_exception(exc, sampling_params=sampling_params, kwargs=kwargs)
            raise

    async def _run_impl(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        trajectory_started = time.perf_counter()
        sample_extra_info = kwargs.get("extra_info") or {}
        validation_rollout = bool(sample_extra_info.get("validation_sample", False))
        effective_sampling_params = dict(sampling_params)
        if validation_rollout:
            effective_sampling_params.update(
                dict(getattr(self, "validation_sampling_params", {}) or {})
            )
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        initial_image = _load_initial_tool_image(messages, kwargs.get("extra_info") or {})

        state = VisionTrajectoryState.from_initial_image(initial_image)
        current_images = [initial_image]
        current_image_names = ["img_0"]

        self._update_rollout_error_context(
            messages=messages,
            state=state,
            current_images=current_images,
            current_image_names=current_image_names,
        )

        if not self.tools:
            raise ValueError("VisHarnessAgentLoop requires the visual tools declared in tools.yaml")

        request_id = uuid4().hex
        mm_processor_kwargs = self._get_mm_processor_kwargs()
        self._update_rollout_error_context(
            request_id=request_id,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        initial_prompt_ids: list[int] | None = None
        trajectory_response_token_count = 0
        output_extra_fields: dict[str, Any] = {}
        assistant_turns = len(state.turns)
        user_observation_turns = sum(1 for turn in state.turns if turn.get("observation_ids"))
        tool_call_count = 0
        generate_time = 0.0
        tool_time = 0.0
        num_preempted = 0
        rollout_abort: _RolloutPromptOverlong | None = None
        invalid_trajectory: dict[str, Any] | None = None

        # vLLM stops at <|im_end|> before emitting the following chat-template newline.
        turn_separator_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if not turn_separator_ids:
            raise ValueError("Tokenizer produced no token ids for the chat turn separator")

        def rollout_placeholder_prompt_length() -> int:
            return int(getattr(getattr(self, "rollout_config", None), "prompt_length", 1) or 1)

        def real_rollout_max_prompt_length() -> int:
            if validation_rollout:
                return int(
                    getattr(
                        self,
                        "validation_max_prompt_length",
                        getattr(
                            self,
                            "rollout_max_prompt_length",
                            rollout_placeholder_prompt_length(),
                        ),
                    )
                )
            return int(
                getattr(
                    self,
                    "rollout_max_prompt_length",
                    rollout_placeholder_prompt_length(),
                )
            )

        def placeholder_prompt_ids() -> list[int]:
            for attr_name in ("eos_token_id", "bos_token_id", "pad_token_id"):
                token_id = getattr(self.tokenizer, attr_name, None)
                if token_id is not None:
                    return [int(token_id)]
            return [0]

        def safe_text_prompt_ids(text: str) -> list[int]:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            max_length = max(rollout_placeholder_prompt_length(), 1)
            if token_ids and len(token_ids) <= max_length:
                return token_ids
            if token_ids:
                return token_ids[:max_length]
            fallback_id = getattr(self.tokenizer, "eos_token_id", None)
            if fallback_id is None:
                fallback_id = getattr(self.tokenizer, "pad_token_id", None)
            return [int(fallback_id if fallback_id is not None else 0)]

        def safe_abort_prompt_ids() -> list[int]:
            return placeholder_prompt_ids()

        def safe_placeholder_prompt_ids() -> list[int]:
            return placeholder_prompt_ids()

        def placeholder_response_ids() -> list[int]:
            for attr_name in ("eos_token_id", "pad_token_id"):
                token_id = getattr(self.tokenizer, attr_name, None)
                if token_id is not None:
                    return [int(token_id)]
            return [0]

        def mark_rollout_prompt_overlong(stage: str, turn_index: int, exc: ValueError) -> _RolloutPromptOverlong:
            message = str(exc)
            if state.turns:
                state.turns[-1]["rollout_prompt_overlong"] = True
                state.turns[-1]["rollout_prompt_overlong_stage"] = stage
                state.turns[-1]["rollout_prompt_overlong_error"] = message
            return _RolloutPromptOverlong(stage=stage, turn_index=turn_index, message=message)

        def prepend_missing_turn_separator(token_ids: list[int]) -> list[int]:
            return turn_separator_ids + token_ids

        async def append_observation(observation: ProcessedToolResponse, turn_record: dict[str, Any]) -> None:
            nonlocal current_images, current_image_names, trajectory_response_token_count, user_observation_turns

            normalize_tool_response_closing_spacing(observation)
            is_error_observation = bool(getattr(observation, "is_error", False))
            should_archive_previous = self.archive_previous_images and current_image_names and not is_error_observation
            if should_archive_previous:
                archive_visible_images(messages, current_image_names)

            messages.append(observation.message)
            try:
                observation_ids = await self.apply_chat_template(
                    [observation.message],
                    images=observation.images or None,
                    remove_system_prompt=True,
                    max_prompt_length=real_rollout_max_prompt_length(),
                )
            except ValueError as exc:
                if _is_rollout_max_prompt_overlong_error(exc):
                    raise mark_rollout_prompt_overlong(
                        "observation",
                        int(turn_record.get("turn_index", assistant_turns)),
                        exc,
                    ) from exc
                raise
            observation_ids = prepend_missing_turn_separator(observation_ids)
            base_prompt_length = len(initial_prompt_ids) if initial_prompt_ids is not None else 0
            raw_observation_length = len(observation_ids)
            stored_observation_ids = observation_ids
            trajectory_response_token_count += raw_observation_length

            if self.archive_previous_images:
                if not is_error_observation:
                    current_images = observation.images
                    current_image_names = observation.image_names
            else:
                current_images = current_images + observation.images
                current_image_names = current_image_names + observation.image_names
            user_observation_turns += 1
            turn_record["observation_ids"] = stored_observation_ids
            turn_record["observation_token_length"] = len(stored_observation_ids)
            turn_record["raw_observation_token_length"] = raw_observation_length
            turn_record["observation_truncated_by_trajectory_limit"] = False
            turn_record["observation_truncated_by_rollout_response_buffer"] = False
            turn_record["visible_image_names_after_step"] = list(current_image_names)
            turn_record["trajectory_response_length_after_observation"] = trajectory_response_token_count
            turn_record["trajectory_token_length_after_observation"] = base_prompt_length + trajectory_response_token_count
            self._update_rollout_error_context(
                messages=messages,
                state=state,
                current_images=current_images,
                current_image_names=current_image_names,
                last_turn_record=turn_record,
            )

        while assistant_turns < self.max_agent_turns:
            try:
                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    images=current_images or None,
                    mm_processor_kwargs=mm_processor_kwargs,
                    max_prompt_length=real_rollout_max_prompt_length(),
                )
            except ValueError as exc:
                if _is_rollout_max_prompt_overlong_error(exc):
                    rollout_abort = mark_rollout_prompt_overlong(
                        "generation_prompt",
                        assistant_turns + 1,
                        exc,
                    )
                    break
                raise
            if initial_prompt_ids is None:
                initial_prompt_ids = list(prompt_ids)

            turn_max_tokens = int(
                getattr(self, "validation_max_response_length", self.per_turn_max_response_length)
                if validation_rollout
                else self.per_turn_max_response_length
            )
            if turn_max_tokens <= 0:
                break

            turn_sampling_params = dict(effective_sampling_params)
            turn_sampling_params["max_tokens"] = turn_max_tokens

            started = time.perf_counter()
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=current_images or None,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            generate_time += time.perf_counter() - started

            response_ids = list(output.token_ids[:turn_max_tokens])
            if not response_ids:
                break
            raw_completion = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            completion = _restore_implicit_think_start(raw_completion)
            messages.append({"role": "assistant", "content": completion})
            assistant_turns += 1

            trajectory_response_length_before_turn = trajectory_response_token_count
            trajectory_response_token_count += len(response_ids)
            output_extra_fields.update(output.extra_fields)
            num_preempted += output.num_preempted or 0

            backend_stop_reason = output.stop_reason
            turn_truncated_by_length = (
                backend_stop_reason == "length" or len(response_ids) >= turn_max_tokens
            )
            effective_stop_reason = "length" if turn_truncated_by_length else backend_stop_reason
            action = parse_action(
                completion,
                model_family=self.model_family,
                finish_reason=effective_stop_reason,
                max_output_tokens=turn_max_tokens,
            )
            base_prompt_length = len(initial_prompt_ids) if initial_prompt_ids is not None else len(prompt_ids)
            turn_record = {
                "turn_index": assistant_turns,
                "prompt_ids": list(prompt_ids),
                "response_ids": response_ids,
                "response_logprobs": list(output.log_probs[: len(response_ids)]) if output.log_probs else None,
                "raw_completion": raw_completion,
                "completion": completion,
                "backend_stop_reason": backend_stop_reason,
                "stop_reason": effective_stop_reason,
                "turn_max_tokens": turn_max_tokens,
                "turn_response_length": len(response_ids),
                "per_turn_max_response_length": self.per_turn_max_response_length,
                "rollout_max_prompt_length": real_rollout_max_prompt_length(),
                "rollout_placeholder_prompt_length": rollout_placeholder_prompt_length(),
                "rollout_placeholder_response_length": getattr(
                    self, "rollout_placeholder_response_length", self.response_length
                ),
                "prompt_token_length": len(prompt_ids),
                "generation_context_length_before_turn": len(prompt_ids),
                "generation_context_length_after_turn": len(prompt_ids) + len(response_ids),
                "trajectory_prompt_length": base_prompt_length,
                "trajectory_response_length_before_turn": trajectory_response_length_before_turn,
                "trajectory_response_length_after_turn": trajectory_response_token_count,
                "trajectory_token_length_before_turn": base_prompt_length + trajectory_response_length_before_turn,
                "trajectory_token_length_after_turn": base_prompt_length + trajectory_response_token_count,
                "turn_truncated_by_length": turn_truncated_by_length,
                "action_type": action.action_type,
                "tool_name": action.tool_name,
                "tool_arguments": copy.deepcopy(action.arguments) if action.arguments is not None else None,
                "output_format_success": action.action_type != "invalid",
                "tool_args_success": False,
                "tool_execution_success": False,
                "tool_execution_error_type": None,
                "tool_execution_error": None,
                "tool_oom_attempts": 0,
                "tool_oom_worker_history": [],
                "action_error": action.error,
                "tool_args_error": None,
                "visible_image_names": list(current_image_names),
                "multi_modal_data": {"images": copy.deepcopy(current_images)},
                "mm_processor_kwargs": copy.deepcopy(mm_processor_kwargs),
                "final_results_before_turn": copy.deepcopy(state.final_results),
            }
            state.turns.append(turn_record)
            self._update_rollout_error_context(
                messages=messages,
                state=state,
                current_images=current_images,
                current_image_names=current_image_names,
                last_turn_record=turn_record,
                last_tool_name=action.tool_name,
                last_tool_parameters=None,
                last_tool_result=None,
            )

            if action.action_type == "answer":
                turn_record["tool_args_success"] = True
                state.finished = True
                break

            if action.action_type == "invalid":
                try:
                    await append_observation(build_error_observation(action.error), turn_record)
                except _RolloutPromptOverlong as exc:
                    rollout_abort = exc
                    break
                continue

            validation = validate_and_prepare_tool_call(
                action=action,
                state=state,
                available_tools=self.tools,
            )
            if not validation.is_valid:
                turn_record["tool_args_error"] = validation.error
                try:
                    await append_observation(build_error_observation(validation.error), turn_record)
                except _RolloutPromptOverlong as exc:
                    rollout_abort = exc
                    break
                continue

            turn_record["tool_args_success"] = True
            validated_call = validation.call
            tool = self.tools[validated_call.tool_name]
            self._update_rollout_error_context(
                last_tool_name=validated_call.tool_name,
                last_tool_parameters=validated_call.tool_parameters,
                last_tool_result=None,
            )

            started = time.perf_counter()
            tool_call_count += 1
            try:
                _, _, tool_extra = await tool.execute(
                    instance_id=request_id,
                    parameters=validated_call.tool_parameters,
                )
            except ToolOOMRetriesExhaustedError as exc:
                tool_time += time.perf_counter() - started
                turn_record["tool_execution_error_type"] = "tool_oom_retry_exhausted"
                turn_record["tool_execution_error"] = str(exc)
                turn_record["tool_oom_attempts"] = int(exc.oom_attempts)
                turn_record["tool_oom_worker_history"] = copy.deepcopy(exc.worker_history)
                self._update_rollout_error_context(
                    last_tool_name=validated_call.tool_name,
                    last_tool_parameters=validated_call.tool_parameters,
                    last_tool_result=copy.deepcopy(exc.response),
                )
                invalid_trajectory = {
                    "reason": "tool_oom_retry_exhausted",
                    "stage": "tool_execution",
                    "turn_index": turn_record["turn_index"],
                    "error": str(exc),
                    "tool_name": exc.tool_name,
                    "oom_attempts": int(exc.oom_attempts),
                    "worker_history": copy.deepcopy(exc.worker_history),
                }
                break
            tool_time += time.perf_counter() - started
            self._update_rollout_error_context(
                last_tool_name=validated_call.tool_name,
                last_tool_parameters=validated_call.tool_parameters,
                last_tool_result=tool_extra.get("tool_result") if isinstance(tool_extra, dict) else tool_extra,
            )

            observation = process_tool_response(
                tool_name=validated_call.tool_name,
                tool_response=tool_extra["tool_result"],
                state=state,
                turn_index=turn_record["turn_index"],
            )
            if validated_call.tool_name in VISION_RESULT_TOOLS:
                append_tool_response_guidance(
                    observation,
                    VISION_TOOL_PROMPT,
                )
            turn_record["tool_execution_success"] = True
            turn_record["tool_result_summary"] = copy.deepcopy(observation.result_summary)
            turn_record["artifact_events"] = copy.deepcopy(observation.artifact_events)
            try:
                await append_observation(observation, turn_record)
            except _RolloutPromptOverlong as exc:
                rollout_abort = exc
                break

        if initial_prompt_ids is None:
            if rollout_abort is not None:
                initial_prompt_ids = safe_abort_prompt_ids()
            else:
                try:
                    initial_prompt_ids = await self.apply_chat_template(
                        messages,
                        tools=self.tool_schemas,
                        images=current_images or None,
                        mm_processor_kwargs=mm_processor_kwargs,
                        max_prompt_length=real_rollout_max_prompt_length(),
                    )
                except ValueError as exc:
                    if _is_rollout_max_prompt_overlong_error(exc):
                        rollout_abort = mark_rollout_prompt_overlong("initial_prompt", assistant_turns + 1, exc)
                        initial_prompt_ids = safe_abort_prompt_ids()
                    else:
                        raise

        output_extra_fields.update(
            {
                "trajectory_uid": request_id,
                "turn_records": state.turns,
                "final_results": state.final_results,
                "original_image_size": {
                    "width": int(state.images["img_0"]["image"].width),
                    "height": int(state.images["img_0"]["image"].height),
                },
                "trajectory_finished": state.finished,
                "tool_call_count": tool_call_count,
                "per_turn_max_response_length": self.per_turn_max_response_length,
                "rollout_max_prompt_length": real_rollout_max_prompt_length(),
                "rollout_placeholder_prompt_length": rollout_placeholder_prompt_length(),
                "rollout_placeholder_response_length": getattr(
                    self, "rollout_placeholder_response_length", self.response_length
                ),
                "max_agent_turns": self.max_agent_turns,
                "max_agent_turns_reached": assistant_turns >= self.max_agent_turns and not state.finished,
                "validation_rollout": validation_rollout,
                "effective_sampling_params": copy.deepcopy(effective_sampling_params),
                "assistant_response_token_count": sum(
                    int(turn.get("turn_response_length") or 0) for turn in state.turns
                ),
                "observation_token_count": sum(
                    int(turn.get("observation_token_length") or 0) for turn in state.turns
                ),
                "trajectory_response_length": trajectory_response_token_count,
                "trajectory_response_tensor_length": 0,
                "trajectory_response_placeholder": True,
                "trajectory_token_length": len(initial_prompt_ids) + trajectory_response_token_count,
                "generate_time_seconds": float(generate_time),
                "tool_time_seconds": float(tool_time),
                "trajectory_elapsed_seconds": float(time.perf_counter() - trajectory_started),
            }
        )
        if invalid_trajectory is not None:
            output_extra_fields.update(
                {
                    "rollout_prompt_overlong": False,
                    "tool_oom_retry_exhausted": True,
                    "tool_oom_attempts": int(invalid_trajectory["oom_attempts"]),
                    "tool_oom_worker_history": copy.deepcopy(invalid_trajectory["worker_history"]),
                    "trajectory_invalid": True,
                    "invalid_reason": invalid_trajectory["reason"],
                    "invalid_stage": invalid_trajectory["stage"],
                    "invalid_turn_index": invalid_trajectory["turn_index"],
                    "invalid_error": invalid_trajectory["error"],
                    "trajectory_aborted": True,
                    "abort_reason": invalid_trajectory["reason"],
                    "abort_stage": invalid_trajectory["stage"],
                    "abort_turn_index": invalid_trajectory["turn_index"],
                    "abort_error": invalid_trajectory["error"],
                }
            )
            return AgentLoopOutput(
                prompt_ids=safe_abort_prompt_ids(),
                response_ids=placeholder_response_ids(),
                response_mask=[0],
                response_logprobs=None,
                # Training drops invalid trajectories before optimization. For
                # validation, defer scoring so the reward worker still emits a
                # complete, index-aligned metric record for this failed sample.
                reward_score=None if validation_rollout else 0.0,
                num_turns=1 + assistant_turns + user_observation_turns,
                metrics=AgentLoopMetrics(
                    generate_sequences=generate_time,
                    tool_calls=tool_time,
                    num_preempted=num_preempted,
                ),
                extra_fields=output_extra_fields,
            )

        if rollout_abort is not None:
            output_extra_fields.update(
                {
                    "rollout_prompt_overlong": True,
                    "tool_oom_retry_exhausted": False,
                    "tool_oom_attempts": 0,
                    "tool_oom_worker_history": [],
                    "trajectory_invalid": True,
                    "invalid_reason": "rollout_prompt_overlong",
                    "invalid_stage": rollout_abort.stage,
                    "invalid_turn_index": rollout_abort.turn_index,
                    "invalid_error": rollout_abort.message,
                    "trajectory_aborted": True,
                    "abort_reason": "rollout_prompt_overlong",
                    "abort_stage": rollout_abort.stage,
                    "abort_turn_index": rollout_abort.turn_index,
                    "abort_error": rollout_abort.message,
                }
            )
            return AgentLoopOutput(
                prompt_ids=safe_abort_prompt_ids(),
                response_ids=placeholder_response_ids(),
                response_mask=[0],
                response_logprobs=None,
                reward_score=None if validation_rollout else 0.0,
                num_turns=1 + assistant_turns + user_observation_turns,
                metrics=AgentLoopMetrics(
                    generate_sequences=generate_time,
                    tool_calls=tool_time,
                    num_preempted=num_preempted,
                ),
                extra_fields=output_extra_fields,
            )

        output_extra_fields.update(
            {
                "rollout_prompt_overlong": False,
                "tool_oom_retry_exhausted": False,
                "tool_oom_attempts": 0,
                "tool_oom_worker_history": [],
                "trajectory_invalid": False,
                "invalid_reason": None,
                "invalid_stage": None,
                "invalid_turn_index": None,
                "invalid_error": None,
                "trajectory_aborted": False,
                "abort_reason": None,
                "abort_stage": None,
                "abort_turn_index": None,
                "abort_error": None,
            }
        )
        response_placeholder_ids = placeholder_response_ids()
        return AgentLoopOutput(
            prompt_ids=safe_placeholder_prompt_ids(),
            response_ids=response_placeholder_ids,
            response_mask=[0] * len(response_placeholder_ids),
            response_logprobs=None,
            num_turns=1 + assistant_turns + user_observation_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=generate_time,
                tool_calls=tool_time,
                num_preempted=num_preempted,
            ),
            extra_fields=output_extra_fields,
        )
