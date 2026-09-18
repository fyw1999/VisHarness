"""Parse structured OpenAI-compatible assistant messages for data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from visharness.agent_loop.action_parser import ParsedAction


SUBMIT_FINAL_ANSWER_TOOL = "SubmitFinalAnswer"
_STRUCTURED_REASONING_ERROR = (
    "Incorrect output format: The response must include a non-empty reasoning "
    "process before invoking a tool or submitting the final answer."
)
_STRUCTURED_TOOL_CALL_ENVELOPE_ERROR = (
    "Incorrect tool call format: At each step, you must invoke exactly one tool "
    "through the provided tool-calling interface. The response must contain "
    "exactly one structured function call."
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    attribute = getattr(value, name, None)
    if attribute is not None:
        return attribute
    model_extra = getattr(value, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        return model_extra.get(name, default)
    return default


def openai_message_reasoning(message: Any) -> str:
    """Return reasoning exposed either as a typed field or provider extra."""

    for field_name in ("reasoning_content", "reasoning"):
        reasoning = _field(message, field_name)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    return ""


@dataclass(frozen=True)
class OpenAIToolCall:
    """The fields needed to validate and persist one structured tool call."""

    call_id: str | None
    call_type: str | None
    function_name: str | None
    raw_arguments: Any


def openai_message_tool_calls(message: Any) -> list[OpenAIToolCall]:
    raw_tool_calls = _field(message, "tool_calls", []) or []
    if not isinstance(raw_tool_calls, (list, tuple)):
        return []

    tool_calls: list[OpenAIToolCall] = []
    for raw_tool_call in raw_tool_calls:
        function = _field(raw_tool_call, "function")
        function_name = _field(function, "name") if function is not None else None
        call_id = _field(raw_tool_call, "id")
        raw_call_type = _field(raw_tool_call, "type")
        tool_calls.append(
            OpenAIToolCall(
                call_id=str(call_id) if call_id is not None else None,
                call_type=(
                    raw_call_type
                    if isinstance(raw_call_type, str) and raw_call_type
                    else None
                ),
                function_name=(
                    str(function_name).strip()
                    if isinstance(function_name, str) and function_name.strip()
                    else None
                ),
                raw_arguments=(
                    _field(function, "arguments")
                    if function is not None
                    else None
                ),
            )
        )
    return tool_calls


def _invalid(error: str) -> ParsedAction:
    return ParsedAction(action_type="invalid", error=error)


def _truncated_action(max_output_tokens: int | None) -> ParsedAction:
    if max_output_tokens is None:
        length_limit = "the maximum output length"
    else:
        length_limit = f"the {int(max_output_tokens)}-token limit"
    return _invalid(
        f"Incorrect output format: The response reached {length_limit} and was truncated. "
        "Retry with concise reasoning and one complete tool call or final answer."
    )


def parse_openai_action(
    model_response: Any,
    *,
    finish_reason: str | None = None,
    max_output_tokens: int | None = None,
) -> ParsedAction:
    """Parse one data-generation turn from structured API response fields.

    Unlike the rollout parser, this parser never interprets textual
    ``<tool_call>`` or ``<answer>`` envelopes. A final answer is accepted only
    through a valid ``SubmitFinalAnswer`` function call.
    """

    if finish_reason == "length":
        return _truncated_action(max_output_tokens)
    if model_response is None:
        return _invalid("Incorrect output: MLLM does not response any text.")

    if not openai_message_reasoning(model_response):
        return _invalid(_STRUCTURED_REASONING_ERROR)

    tool_calls = openai_message_tool_calls(model_response)
    if len(tool_calls) != 1:
        return _invalid(_STRUCTURED_TOOL_CALL_ENVELOPE_ERROR)

    tool_call = tool_calls[0]
    if tool_call.function_name is None:
        return _invalid(
            "Incorrect tool call format: No valid tool name was detected. Please check the format and invoke "
            "the tool again according to the tool call format."
        )

    raw_arguments = tool_call.raw_arguments
    if raw_arguments is None or raw_arguments == "":
        return _invalid(
            "Incorrect tool call format: No valid tool arguments were detected. Please check the format and "
            "invoke the tool again according to the tool call format."
        )
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (TypeError, ValueError, json.JSONDecodeError):
        return _invalid("Incorrect tool call format: The tool arguments cannot be parsed as valid JSON.")
    if not isinstance(arguments, dict):
        return _invalid("Incorrect tool call format: 'arguments' must be a JSON object (dictionary).")

    if tool_call.function_name == SUBMIT_FINAL_ANSWER_TOOL:
        final_answer = arguments.get("final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            return _invalid(
                "Incorrect final answer: The final answer must be a non-empty string. "
                "Please provide a valid final answer."
            )
        return ParsedAction(action_type="answer", answer=final_answer.strip())

    return ParsedAction(
        action_type="tool_call",
        tool_name=tool_call.function_name,
        arguments=arguments,
    )
