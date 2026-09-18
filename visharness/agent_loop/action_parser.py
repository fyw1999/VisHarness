"""Strict action parsing for VisHarness model responses."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


_SUPPORTED_MODEL_FAMILIES = {
    "qwen3_vl": "qwen3_vl",
    "qwen3-vl": "qwen3_vl",
    "qwen3.5": "qwen3_5",
    "qwen3_5": "qwen3_5",
    "qwen3-5": "qwen3_5",
}

_TOOL_CALL_ENVELOPE_ERROR = (
    "Incorrect tool call format: At each step, you may invoke exactly one tool, "
    "and the tool call must be enclosed by exactly one correctly ordered "
    "<tool_call>...</tool_call> pair, with no text outside the tags."
)
_ANSWER_ENVELOPE_ERROR = (
    "Incorrect answer format: At each step, you may provide exactly one final answer, "
    "and the answer must be enclosed by exactly one correctly ordered "
    "<answer>...</answer> pair, with no text outside the tags."
)
_MIXED_ACTION_ERROR = (
    "Incorrect output format: You cannot invoke a tool and provide the final answer "
    "in the same step. Output exactly one <tool_call>...</tool_call> or one "
    "<answer>...</answer>."
)
_MISSING_ACTION_ERROR = (
    "Incorrect output format: At each step, you must either invoke exactly one tool "
    "using <tool_call>...</tool_call> or provide exactly one final answer using "
    "<answer>...</answer>."
)


@dataclass
class ParsedAction:
    """Structured result of parsing one assistant turn."""

    action_type: Literal["tool_call", "answer", "invalid"]
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    error: str | None = None


def _invalid(error: str) -> ParsedAction:
    return ParsedAction(action_type="invalid", error=error)


def _extract_action_body(
    completion: str,
    finish_reason: str | None,
    max_output_tokens: int | None,
) -> str | ParsedAction:
    if finish_reason == "length":
        if max_output_tokens is None:
            length_limit = "the maximum output length"
        else:
            length_limit = f"the {int(max_output_tokens)}-token limit"
        return _invalid(
            f"Incorrect output format: The response reached {length_limit} and was truncated. "
            "Retry with concise reasoning and one complete tool call or final answer."
        )

    # Token-decoding callers must restore a chat-template-prefilled opening
    # <think> before parsing. Keep this parser strict so malformed prefixes are
    # not silently accepted as part of an otherwise valid action.
    think_match = re.match(r"\s*<think>(.*?)</think>", completion, re.DOTALL)
    if think_match is not None:
        return completion[think_match.end() :].strip()

    return _invalid(
        "Incorrect output format: Missing or unclosed <think> tag. "
        "You must output <think>...</think> first."
    )


def _contains_envelope_tag_syntax(action_body: str, tag_name: str) -> bool:
    """Detect complete or malformed attempts to use one protocol tag."""

    return re.search(rf"</?{re.escape(tag_name)}\b", action_body, re.IGNORECASE) is not None


def _match_single_envelope(action_body: str, tag_name: str) -> re.Match[str] | None:
    """Match exactly one ordered tag pair covering the entire action body."""

    opening_tag = f"<{tag_name}>"
    closing_tag = f"</{tag_name}>"
    if action_body.count(opening_tag) != 1 or action_body.count(closing_tag) != 1:
        return None

    return re.fullmatch(
        rf"{re.escape(opening_tag)}(.*?){re.escape(closing_tag)}",
        action_body,
        re.DOTALL,
    )


def _parse_action_envelope(action_body: str) -> tuple[Literal["tool_call", "answer"], str] | ParsedAction:
    action_body = action_body.strip()
    has_tool_syntax = _contains_envelope_tag_syntax(action_body, "tool_call")
    has_answer_syntax = _contains_envelope_tag_syntax(action_body, "answer")

    if has_tool_syntax and has_answer_syntax:
        return _invalid(_MIXED_ACTION_ERROR)

    if has_tool_syntax:
        tool_match = _match_single_envelope(action_body, "tool_call")
        if tool_match is None:
            return _invalid(_TOOL_CALL_ENVELOPE_ERROR)
        return "tool_call", tool_match.group(1).strip()

    if has_answer_syntax:
        answer_match = _match_single_envelope(action_body, "answer")
        if answer_match is None:
            return _invalid(_ANSWER_ENVELOPE_ERROR)
        return "answer", answer_match.group(1)

    return _invalid(_MISSING_ACTION_ERROR)


def _parse_qwen3_vl_tool_call(tool_content: str) -> ParsedAction:
    try:
        tool_data = json.loads(tool_content)
    except json.JSONDecodeError:
        return _invalid(
            "Incorrect tool call format: The content inside <tool_call> cannot be correctly parsed as a valid "
            "JSON object."
        )

    if not isinstance(tool_data, dict):
        return _invalid("Incorrect tool call format: The content inside <tool_call> must be a JSON object.")
    if "name" not in tool_data:
        return _invalid(
            "Incorrect tool call format: No valid tool name was detected. Please check the format and invoke "
            "the tool again according to the tool call format."
        )
    if "arguments" not in tool_data:
        return _invalid(
            "Incorrect tool call format: No valid tool arguments were detected. Please check the format and "
            "invoke the tool again according to the tool call format."
        )
    if not isinstance(tool_data["name"], str) or not tool_data["name"].strip():
        return _invalid("Incorrect tool call format: 'name' must be a non-empty string.")
    if not isinstance(tool_data["arguments"], dict):
        return _invalid("Incorrect tool call format: 'arguments' must be a JSON object (dictionary).")

    return ParsedAction(
        action_type="tool_call",
        tool_name=tool_data["name"].strip(),
        arguments=tool_data["arguments"],
    )


def _parse_qwen3_5_parameter(name: str, value: str) -> Any | ParsedAction:
    value = value.strip()
    if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return _invalid(
                f"Incorrect tool call format: The content of <parameter={name}> must be a valid JSON string."
            )
    return value


def _parse_qwen3_5_tool_call(tool_content: str) -> ParsedAction:
    function_match = re.search(r"<function=([^>]+)>(.*?)</function>", tool_content, re.DOTALL)
    if function_match is None:
        return _invalid(
            "Incorrect tool call format: Missing or malformed <function=Name>...</function> tag inside "
            "<tool_call>."
        )

    function_name = function_match.group(1).strip()
    if not function_name:
        return _invalid("Incorrect tool call format: The function name must not be empty.")

    arguments: dict[str, Any] = {}
    parameter_count = 0
    for parameter_match in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", function_match.group(2), re.DOTALL):
        parameter_count += 1
        parameter_name = parameter_match.group(1).strip()
        parsed_value = _parse_qwen3_5_parameter(parameter_name, parameter_match.group(2))
        if isinstance(parsed_value, ParsedAction):
            return parsed_value
        arguments[parameter_name] = parsed_value

    if parameter_count == 0:
        return _invalid(
            "Incorrect tool call format: Missing or malformed <parameter=Name>...</parameter> tag inside "
            "<function>."
        )

    return ParsedAction(action_type="tool_call", tool_name=function_name, arguments=arguments)


def parse_action(
    completion: str,
    model_family: str = "qwen3_vl",
    finish_reason: str | None = None,
    max_output_tokens: int | None = None,
) -> ParsedAction:
    """Parse one complete assistant turn using the VisHarness output protocol.

    Callers decoding generated token IDs must first restore any opening
    ``<think>`` supplied by the chat template. Parsing only validates the
    assistant response structure. Tool availability, required parameters, and
    trajectory-dependent image state are validated by the tool execution layer.
    """
    normalized_family = _SUPPORTED_MODEL_FAMILIES.get(model_family.lower())
    if normalized_family is None:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_FAMILIES))
        return _invalid(f"Unsupported model family {model_family!r}. Supported values: {supported}.")

    action_body = _extract_action_body(
        completion,
        finish_reason,
        max_output_tokens=max_output_tokens,
    )
    if isinstance(action_body, ParsedAction):
        return action_body

    envelope = _parse_action_envelope(action_body)
    if isinstance(envelope, ParsedAction):
        return envelope

    action_type, content = envelope
    if action_type == "answer":
        return ParsedAction(action_type="answer", answer=content)
    if normalized_family == "qwen3_vl":
        return _parse_qwen3_vl_tool_call(content)
    return _parse_qwen3_5_tool_call(content)
