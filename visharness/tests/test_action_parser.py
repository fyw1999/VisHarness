import pytest

from visharness.agent_loop.action_parser import parse_action


def test_parse_qwen3_vl_tool_call():
    completion = """
<think>I should locate the object first.</think>
<tool_call>
{"name": "PhraseToPoint", "arguments": {"images": ["img_0"], "phrase": "the person on the left"}}
</tool_call>
"""

    action = parse_action(completion)

    assert action.action_type == "tool_call"
    assert action.tool_name == "PhraseToPoint"
    assert action.arguments == {"images": ["img_0"], "phrase": "the person on the left"}
    assert action.error is None


def test_parse_rejects_missing_opening_think_after_caller_normalization():
    completion = """
I should locate the object first.</think>
<tool_call>
{"name": "PhraseToPoint", "arguments": {"images": ["img_0"], "phrase": "the person on the left"}}
</tool_call>
"""

    action = parse_action(completion)

    assert action.action_type == "invalid"
    assert "Missing or unclosed <think>" in action.error


def test_parse_rejects_non_whitespace_before_opening_think():
    action = parse_action("prefix<think>x</think><answer>done</answer>")

    assert action.action_type == "invalid"
    assert "Missing or unclosed <think>" in action.error


def test_parse_qwen3_5_tool_call():
    completion = """
<think>I should split the image.</think>
<tool_call>
<function=SplitImageIntoPatches>
<parameter=patch_configs>{"img_0": 320}</parameter>
</function>
</tool_call>
"""

    action = parse_action(completion, model_family="qwen3_5")

    assert action.action_type == "tool_call"
    assert action.tool_name == "SplitImageIntoPatches"
    assert action.arguments == {"patch_configs": {"img_0": 320}}


def test_parse_answer():
    action = parse_action("<think>The target is absent.</think><answer>No matching object was found.</answer>")

    assert action.action_type == "answer"
    assert action.answer == "No matching object was found."
    assert action.error is None


@pytest.mark.parametrize(
    ("completion", "error_fragment"),
    [
        ("<answer>done</answer>", "Missing or unclosed <think>"),
        ("reasoning without a closing think tag<answer>done</answer>", "Missing or unclosed <think>"),
        ("<think>x</think>plain text", "must either invoke exactly one tool"),
        (
            "<think>x</think><answer>done</answer><tool_call>{}</tool_call>",
            "cannot invoke a tool and provide the final answer",
        ),
        (
            "<think>x</think><tool_call>{}</tool_call><tool_call>{}</tool_call>",
            "may invoke exactly one tool",
        ),
        (
            '<think>x</think><tool_call>{"name": "Tool", "arguments": []}</tool_call>',
            "'arguments' must be a JSON object",
        ),
    ],
)
def test_invalid_qwen3_vl_outputs(completion: str, error_fragment: str):
    action = parse_action(completion)

    assert action.action_type == "invalid"
    assert error_fragment in action.error


@pytest.mark.parametrize(
    "action_body",
    [
        '<tool_call>{"name": "Tool", "arguments": {}}',
        '{"name": "Tool", "arguments": {}}</tool_call>',
        '</tool_call><tool_call>{"name": "Tool", "arguments": {}}',
        (
            '<tool_call>{"name": "Tool", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "Tool", "arguments": {}}</tool_call>'
        ),
        'prefix<tool_call>{"name": "Tool", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "Tool", "arguments": {}}</tool_call>suffix',
        '<tool_call >{"name": "Tool", "arguments": {}}</tool_call>',
    ],
)
def test_tool_call_requires_one_exact_envelope(action_body: str):
    action = parse_action(f"<think>x</think>{action_body}")

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect tool call format: At each step, you may invoke exactly one tool, "
        "and the tool call must be enclosed by exactly one correctly ordered "
        "<tool_call>...</tool_call> pair, with no text outside the tags."
    )


@pytest.mark.parametrize(
    "action_body",
    [
        "<answer>done",
        "done</answer>",
        "</answer><answer>done",
        "<answer>first</answer><answer>second</answer>",
        "prefix<answer>done</answer>",
        "<answer>done</answer>suffix",
        "<answer >done</answer>",
    ],
)
def test_answer_requires_one_exact_envelope(action_body: str):
    action = parse_action(f"<think>x</think>{action_body}")

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect answer format: At each step, you may provide exactly one final answer, "
        "and the answer must be enclosed by exactly one correctly ordered "
        "<answer>...</answer> pair, with no text outside the tags."
    )


@pytest.mark.parametrize(
    "action_body",
    [
        "<tool_call>{}</tool_call><answer>done</answer>",
        "<tool_call>{}</tool_call><answer>",
        "<tool_call><answer>done</answer>",
    ],
)
def test_tool_call_and_answer_syntax_cannot_appear_together(action_body: str):
    action = parse_action(f"<think>x</think>{action_body}")

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect output format: You cannot invoke a tool and provide the final answer "
        "in the same step. Output exactly one <tool_call>...</tool_call> or one "
        "<answer>...</answer>."
    )


def test_length_truncation_is_invalid():
    action = parse_action(
        "<think>unfinished",
        finish_reason="length",
        max_output_tokens=2048,
    )

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect output format: The response reached the 2048-token limit and was truncated. "
        "Retry with concise reasoning and one complete tool call or final answer."
    )


def test_qwen3_5_json_parameters_and_plain_text_parameters():
    completion = """
<think>Use one tool.</think>
<tool_call>
<function=PhraseToPoint>
<parameter=images>["img_0"]</parameter>
<parameter=phrase>the person on the left</parameter>
</function>
</tool_call>
"""

    action = parse_action(completion, model_family="qwen3.5")

    assert action.action_type == "tool_call"
    assert action.arguments == {"images": ["img_0"], "phrase": "the person on the left"}


def test_qwen3_5_rejects_invalid_json_parameter():
    completion = """
<think>Use one tool.</think>
<tool_call>
<function=PhraseToPoint>
<parameter=images>["img_0",]</parameter>
</function>
</tool_call>
"""

    action = parse_action(completion, model_family="qwen3_5")

    assert action.action_type == "invalid"
    assert "must be a valid JSON string" in action.error


def test_unsupported_model_family_is_invalid():
    action = parse_action("<think>x</think><answer>done</answer>", model_family="other")

    assert action.action_type == "invalid"
    assert "Unsupported model family" in action.error
