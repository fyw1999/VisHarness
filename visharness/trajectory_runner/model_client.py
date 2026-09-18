"""OpenAI-compatible model client used by the trajectory runner."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage
from PIL import Image

from .config import as_plain_dict
from .errors import (
    TrajectoryContextLengthExceededError,
    TrajectoryModelRequestTimeoutError,
    TrajectoryVisionEncoderCacheExceededError,
)
from visharness.prompts import (
    DATA_GENERATION_SYSTEM_PROMPT,
    SubmitFinalAnswer,
    TOOLS_LIST,
    TRAIN_TEST_SYSTEM_PROMPT,
)

from .openai_action_parser import (
    openai_message_reasoning,
    openai_message_tool_calls,
)

logger = logging.getLogger(__name__)


def pil_to_base64(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclass
class ModelTurnResponse:
    # Only inference mode has a canonical, token-decoded text completion.
    # Data-generation mode keeps the provider's structured API message intact.
    completion: str | None = None
    finish_reason: str | None = None
    stop_reason: Any = None
    api_message: Any = None
    assistant_message: dict[str, Any] | None = None
    token_ids: list[int] | None = None
    used_token_completion: bool = False
    turn_max_tokens: int | None = None
    model_latency_seconds: float | None = None
    prompt_token_count: int | None = None
    visual_token_count: int | None = None
    completion_token_count: int | None = None

    @property
    def effective_stop_reason(self) -> str | None:
        """Return the normalized stop reason used by the environment.

        ``finish_reason`` and ``stop_reason`` preserve the provider's raw
        OpenAI/vLLM response.  The environment additionally treats a response
        that reaches the configured token limit as length-truncated even when
        the backend fails to report ``"length"`` explicitly.
        """
        if self.truncated_by_length:
            return "length"
        if self.finish_reason is not None:
            return str(self.finish_reason)
        if self.stop_reason is not None:
            return str(self.stop_reason)
        return None

    @property
    def parse_finish_reason(self) -> str | None:
        """Compatibility alias for callers passing the reason to parsers."""

        return self.effective_stop_reason

    @property
    def truncated_by_length(self) -> bool:
        if _is_length_reason(self.finish_reason) or _is_length_reason(self.stop_reason):
            return True
        if self.turn_max_tokens is None or self.turn_max_tokens <= 0:
            return False
        token_count = self.completion_token_count
        if token_count is None and self.token_ids is not None:
            token_count = len(self.token_ids)
        return token_count is not None and token_count >= self.turn_max_tokens


def _is_length_reason(reason: Any) -> bool:
    return isinstance(reason, str) and reason.lower() == "length"


def _restore_implicit_think_start(completion: str) -> str:
    """Match the training rollout decode path for Qwen-style thinking output."""
    if not completion:
        return completion
    if completion.startswith("<think>"):
        return completion
    separator = "" if completion.startswith(("\n", "\r")) else "\n"
    return f"<think>{separator}{completion}"


def _loads_arguments(raw_arguments: Any) -> Any:
    if isinstance(raw_arguments, str):
        return json.loads(raw_arguments)
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return raw_arguments


def _content_to_text(raw_content: Any) -> str:
    if isinstance(raw_content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )
    return str(raw_content or "")


def openai_message_to_assistant_message(message: Any) -> dict[str, Any]:
    """Project a structured message without deciding whether it is valid.

    Runtime data-generation code must keep and parse the original API message.
    This helper is only suitable for serialization of non-final-answer history;
    in particular, it deliberately does not turn ``SubmitFinalAnswer`` into an
    ``<answer>`` block.
    """
    role = (
        message.get("role", "assistant")
        if isinstance(message, dict)
        else getattr(message, "role", "assistant")
    )
    raw_content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    ) or ""
    content = _content_to_text(raw_content).strip()
    reasoning = openai_message_reasoning(message)

    content_parts = []
    if reasoning:
        content_parts.append(f"<think>\n{reasoning}\n</think>")
    elif content:
        content_parts.append(content)

    formatted_tool_calls = []
    for tool_call in openai_message_tool_calls(message):
        function_name = tool_call.function_name or ""
        raw_arguments = tool_call.raw_arguments
        try:
            arguments = _loads_arguments(raw_arguments)
        except Exception:
            arguments = raw_arguments
        formatted_tool_calls.append(
            {
                "id": tool_call.call_id,
                "type": tool_call.call_type,
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                },
            }
        )

    content_text = "\n".join(content_parts).strip()
    if formatted_tool_calls and content_text:
        # The structured call is serialized as a separate message downstream.
        # Keep one explicit separator so the rendered training sequence is
        # ``</think>\n<tool_call>`` instead of joining the two tags together.
        content_text = content_text.rstrip("\r\n") + "\n"

    assistant_message = {"role": role, "content": content_text}
    if formatted_tool_calls:
        assistant_message["tool_calls"] = formatted_tool_calls
    return assistant_message


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        return model_extra.get(name, default)
    return default


_CONTEXT_LENGTH_ERROR_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "max_context_length_exceeded",
    "prompt_too_long",
}
_CONTEXT_LENGTH_MESSAGE_PATTERNS = (
    "maximum context length",
    "context length exceeded",
    "context window exceeded",
    "context_length_exceeded",
    "max context length",
    "prompt is too long",
    "input is too long",
)

_VISION_ENCODER_CACHE_MODALITY_PATTERNS = (
    "vision_chunk item",
    "image item",
    "video item",
)


def _provider_error_text(error: Exception) -> str:
    """Collect provider error text from the exception and common OpenAI bodies."""

    parts = [str(error)]
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body_message = body.get("message")
        if isinstance(body_message, str):
            parts.append(body_message)

        nested_error = body.get("error")
        if isinstance(nested_error, dict):
            nested_message = nested_error.get("message")
            if isinstance(nested_message, str):
                parts.append(nested_message)

    return "\n".join(parts).lower()


def _is_vision_encoder_cache_exceeded_error(error: Exception) -> bool:
    """Conservatively classify explicit per-item multimodal cache overflow."""

    message = _provider_error_text(error)
    return (
        "exceeds the pre-allocated encoder cache size" in message
        and "--limit-mm-per-prompt" in message
        and any(
            pattern in message
            for pattern in _VISION_ENCODER_CACHE_MODALITY_PATTERNS
        )
    )


def _is_context_length_exceeded_error(error: Exception) -> bool:
    """Conservatively classify provider errors that explicitly report context overflow."""

    error_objects: list[Any] = [error, getattr(error, "body", None)]
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_objects.append(body.get("error"))

    for error_object in error_objects:
        for field_name in ("code", "type"):
            value = _get_field(error_object, field_name)
            if isinstance(value, str) and value.strip().lower() in _CONTEXT_LENGTH_ERROR_CODES:
                return True

    message = str(error).lower()
    if isinstance(body, dict):
        message = f"{message}\n{body!r}".lower()
    return any(pattern in message for pattern in _CONTEXT_LENGTH_MESSAGE_PATTERNS)


def _normalize_token_ids(raw_token_ids: Any) -> list[int] | None:
    if raw_token_ids is None:
        return None
    if not isinstance(raw_token_ids, (list, tuple)):
        return None
    token_ids: list[int] = []
    for token_id in raw_token_ids:
        if isinstance(token_id, int):
            token_ids.append(token_id)
            continue
        if isinstance(token_id, str) and token_id.startswith("token_id:"):
            token_ids.append(int(token_id.split(":", 1)[1]))
            continue
        return None
    return token_ids


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


class OnlineVllmModelClient:
    """OpenAI-compatible client that keeps the legacy model API shape."""

    def __init__(self, mode: str, **kwargs):
        from openai import OpenAI

        self.mode = str(mode).lower()
        self.model_name = kwargs.get("model_name", "")
        self.use_tools = bool(kwargs.get("use_tools", True))
        self.prefer_token_completion = bool(
            kwargs.get("prefer_token_completion", self.mode == "inference")
        )
        self.tokenizer_path = (
            kwargs.get("tokenizer_path")
            or kwargs.get("model_path")
            or kwargs.get("local_model_path")
            or self.model_name
        )
        self.trust_remote_code = bool(kwargs.get("trust_remote_code", True))
        self.tokenizer = kwargs.get("tokenizer")
        self._tokenizer_load_error: str | None = None
        self._image_token_id: int | None = None
        self._image_token_id_resolved = False
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        api_key = kwargs.get("api_key", "EMPTY")
        http_limits = httpx.Limits(
            max_connections=int(kwargs.get("max_connections", 64)),
            max_keepalive_connections=int(kwargs.get("max_keepalive_connections", 32)),
        )
        http_timeout = httpx.Timeout(
            timeout=float(kwargs.get("timeout", 300.0)),
            connect=float(kwargs.get("connect_timeout", 15.0)),
        )
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(limits=http_limits, timeout=http_timeout, trust_env=False),
            max_retries=int(kwargs.get("max_retries", 2)),
        )
        self.generation_config: dict[str, Any] = {}
        if self.mode == "inference":
            if not self.prefer_token_completion:
                raise ValueError(
                    "Inference mode requires prefer_token_completion=True so output parsing matches training rollout."
                )
            self._get_tokenizer()
        elif self.mode == "data_generation":
            if self.prefer_token_completion:
                raise ValueError(
                    "Data-generation mode requires prefer_token_completion=False so the original structured "
                    "OpenAI message remains authoritative."
                )
        else:
            raise ValueError("mode should be either data_generation or inference")

    def set_generation_config(self, generation_config: Any = None) -> None:
        self.generation_config = as_plain_dict(generation_config)

    def _ensure_return_token_ids(self, api_kwargs: dict[str, Any]) -> None:
        if not self.prefer_token_completion:
            return
        extra_body = dict(api_kwargs.get("extra_body") or {})
        if "return_token_ids" in api_kwargs:
            extra_body["return_token_ids"] = bool(api_kwargs.pop("return_token_ids"))
        extra_body["return_token_ids"] = True
        api_kwargs["extra_body"] = extra_body

    def _get_tokenizer(self) -> Any | None:
        if self.tokenizer is not None:
            return self.tokenizer
        if self._tokenizer_load_error is not None:
            return None
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
            )
            return self.tokenizer
        except Exception as tokenizer_exc:
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=self.trust_remote_code,
                )
                self.tokenizer = getattr(processor, "tokenizer", None)
                if self.tokenizer is not None:
                    return self.tokenizer
                raise RuntimeError("AutoProcessor did not expose a tokenizer")
            except Exception as processor_exc:
                self._tokenizer_load_error = f"{type(processor_exc).__name__}: {processor_exc}"
                if self.mode == "inference":
                    raise RuntimeError(
                        "Inference mode requires a loadable tokenizer so returned token IDs can be decoded exactly "
                        f"like training rollout, but loading from {self.tokenizer_path!r} failed. "
                        f"AutoTokenizer error: {type(tokenizer_exc).__name__}: {tokenizer_exc}; "
                        f"AutoProcessor error: {self._tokenizer_load_error}"
                    ) from processor_exc
                logger.warning(
                    "Failed to load tokenizer from %s; falling back to parsed OpenAI message. "
                    "AutoTokenizer error was %s: %s; AutoProcessor error was %s",
                    self.tokenizer_path,
                    type(tokenizer_exc).__name__,
                    tokenizer_exc,
                    self._tokenizer_load_error,
                )
                return None

    def _decode_token_completion(self, token_ids: list[int] | None) -> str | None:
        if not token_ids:
            if self.mode == "inference":
                raise RuntimeError(
                    "Inference mode requires the OpenAI-compatible server to return non-empty output token IDs. "
                    "Ensure return_token_ids is supported and enabled by the server."
                )
            return None
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            if self.mode == "inference":
                raise RuntimeError(
                    "Inference mode requires a tokenizer to decode returned token IDs exactly like training rollout."
                )
            return None
        raw_completion = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _restore_implicit_think_start(raw_completion)

    def _resolve_image_token_id(self) -> int | None:
        """Resolve Qwen-style visual patch tokens without hard-coding a vocab id."""

        if getattr(self, "_image_token_id_resolved", False):
            return getattr(self, "_image_token_id", None)
        self._image_token_id_resolved = True
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if not isinstance(token_id, int) or token_id < 0:
            return None
        try:
            if tokenizer.convert_ids_to_tokens(token_id) != "<|image_pad|>":
                return None
        except Exception:
            pass
        self._image_token_id = token_id
        return token_id

    def _count_visual_tokens(
        self,
        prompt_token_ids: list[int] | None,
    ) -> int | None:
        if prompt_token_ids is None:
            return None
        image_token_id = self._resolve_image_token_id()
        if image_token_id is None:
            return None
        return sum(token_id == image_token_id for token_id in prompt_token_ids)

    def generate_conversation_fn(self, text: str, image: Image.Image) -> list[dict[str, Any]]:
        system_prompt = DATA_GENERATION_SYSTEM_PROMPT if self.mode == "data_generation" else TRAIN_TEST_SYSTEM_PROMPT
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "image_name": "img_0"},
                    {
                        "type": "text",
                        "text": f"{text} The height and width of the image are {image.height} and {image.width}, respectively.\n",
                    },
                ],
            },
        ]

    def form_messages_from_item(self, item: Any) -> list[dict[str, Any]]:
        openai_messages = []
        for message in item.conversation:
            if isinstance(message, ChatCompletionMessage):
                if "kimi" in self.model_name.lower() or "qwen" in self.model_name.lower():
                    openai_messages.append(message)
                else:
                    openai_messages.append(message.model_dump(exclude_none=True))
                continue

            if not isinstance(message, dict):
                raise TypeError(
                    "Trajectory conversation messages must be dict or ChatCompletionMessage, "
                    f"got {type(message).__name__}"
                )

            role = message.get("role")
            content = message.get("content")
            if isinstance(content, list):
                openai_content = []
                for part in content:
                    if not isinstance(part, dict):
                        raise
                    if part.get("type") == "text":
                        openai_content.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image":
                        image_base64 = pil_to_base64(part["image"])
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            }
                        )
                    elif part.get("type") == "image_url":
                        openai_content.append(part)
                openai_message = {"role": role, "content": openai_content}
            else:
                openai_message = {"role": role, "content": content or ""}
            for key in ("tool_call_id", "name"):
                if key in message:
                    openai_message[key] = message[key]
            if "tool_calls" in message:
                openai_message["tool_calls"] = message["tool_calls"]
            openai_messages.append(openai_message)
        return openai_messages

    def _tools_for_mode(self) -> list[dict[str, Any]] | None:
        if not self.use_tools:
            return None
        if self.mode == "data_generation":
            return TOOLS_LIST + [SubmitFinalAnswer]
        return TOOLS_LIST

    def generate_one_item(self, item: Any) -> ModelTurnResponse:
        api_kwargs = dict(self.generation_config or {})
        self._ensure_return_token_ids(api_kwargs)
        raw_turn_max_tokens = api_kwargs.get("max_tokens")
        turn_max_tokens = int(raw_turn_max_tokens) if raw_turn_max_tokens is not None else None
        tools = self._tools_for_mode()
        if tools is not None:
            api_kwargs["tools"] = tools
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.form_messages_from_item(item),
                **api_kwargs,
            )
        except APITimeoutError as exc:
            configured_timeout = api_kwargs.get("timeout")
            timeout_suffix = (
                f" after {configured_timeout} seconds per attempt"
                if configured_timeout is not None
                else ""
            )
            raise TrajectoryModelRequestTimeoutError(
                f"Model API request timed out{timeout_suffix}."
            ) from exc
        except Exception as exc:
            if _is_vision_encoder_cache_exceeded_error(exc):
                raise TrajectoryVisionEncoderCacheExceededError(str(exc)) from exc
            if _is_context_length_exceeded_error(exc):
                raise TrajectoryContextLengthExceededError(str(exc)) from exc
            raise
        choice = response.choices[0]
        message = choice.message
        token_ids = _normalize_token_ids(_get_field(choice, "token_ids"))
        prompt_token_ids = _normalize_token_ids(
            _get_field(response, "prompt_token_ids")
        )
        usage = _get_field(response, "usage")
        raw_prompt_token_count = _get_field(usage, "prompt_tokens")
        raw_completion_token_count = _get_field(usage, "completion_tokens")
        prompt_token_count = _optional_nonnegative_int(raw_prompt_token_count)
        if prompt_token_count is None and prompt_token_ids is not None:
            prompt_token_count = len(prompt_token_ids)
        visual_token_count = self._count_visual_tokens(prompt_token_ids)
        completion_tokens_from_usage = _optional_nonnegative_int(
            raw_completion_token_count
        )
        completion_token_count = (
            len(token_ids)
            if token_ids is not None
            else completion_tokens_from_usage
        )
        if self.mode == "inference":
            token_completion = self._decode_token_completion(token_ids)
            assistant_message = {"role": "assistant", "content": token_completion}
            completion = token_completion
            used_token_completion = True
        elif self.mode == "data_generation":
            assistant_message = None
            completion = None
            used_token_completion = False
        else:
            raise ValueError("mode should be either data_generation or inference")

        #############debug##############
        # prompt_token_ids = _get_field(response, "prompt_token_ids")
        # output_token_ids = token_ids

        # print("=" * 80)
        # print("Prompt token num:", len(prompt_token_ids) if prompt_token_ids else None)
        # print("Output token num:", len(output_token_ids) if output_token_ids else None)

        # print("=" * 80)
        # print("Decoded prompt:")
        # if prompt_token_ids is not None:
        #     print(self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False))

        # print("=" * 80)
        # print("Decoded output:")
        # if output_token_ids is not None:
        #     print(self.tokenizer.decode(output_token_ids, skip_special_tokens=False))

        # print("=" * 80)
        # print("Normal response:")
        # print(response.choices[0].message)
        # os.system("cls" if os.name == "nt" else "clear")
        #######################################
        return ModelTurnResponse(
            completion=completion,
            finish_reason=getattr(choice, "finish_reason", None),
            stop_reason=_get_field(choice, "stop_reason"),
            api_message=message,
            assistant_message=assistant_message,
            token_ids=token_ids,
            used_token_completion=used_token_completion,
            turn_max_tokens=turn_max_tokens,
            prompt_token_count=prompt_token_count,
            visual_token_count=visual_token_count,
            completion_token_count=completion_token_count,
        )

    def eval(self) -> "OnlineVllmModelClient":
        return self
