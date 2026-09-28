import asyncio
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest

from verl.workers.rollout.replica import TokenOutput
from visharness.agent_loop.visharness_agent_loop import VisHarnessAgentLoop, _restore_implicit_think_start
from visharness.prompts import VISION_TOOL_PROMPT
from visharness.tools.errors import ToolOOMRetriesExhaustedError


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    completions = {
        11: (
            '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
            '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
        ),
        12: "<think>done</think><answer>Completed.</answer>",
        13: "<think>bad</think>I should retry this turn.",
        14: (
            '<think>enhance</think><tool_call>{"name":"SuperResolution",'
            '"arguments":{"images":["img_0"]}}</tool_call>'
        ),
    }

    def decode(self, token_ids, **kwargs):
        return self.completions[token_ids[0]]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if text == "\n":
            return [89]
        return [98]


class FakeServerManager:
    def __init__(self):
        self.outputs = [
            TokenOutput(token_ids=[11], log_probs=[-0.1]),
            TokenOutput(token_ids=[12], log_probs=[-0.2]),
        ]
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class FakePhraseToPointTool:
    name = "PhraseToPoint"
    tool_schema = SimpleNamespace(model_dump=lambda **kwargs: {"type": "function"})

    def __init__(self):
        self.parameters = None

    async def execute(self, instance_id, parameters, **kwargs):
        self.parameters = parameters
        return (
            None,
            0.0,
            {
                "tool_result": {
                    "img_0": {
                        "visual_image": image_bytes(),
                        "text_response": "Found one point.",
                        "points": [[2, 3]],
                    }
                }
            },
        )


class FakeSuperResolutionTool:
    name = "SuperResolution"
    tool_schema = SimpleNamespace(model_dump=lambda **kwargs: {"type": "function"})

    async def execute(self, instance_id, parameters, **kwargs):
        return (
            None,
            0.0,
            {"tool_result": {"img_0_2x": image_bytes()}},
        )


class FakeOOMPhraseToPointTool(FakePhraseToPointTool):
    async def execute(self, instance_id, parameters, **kwargs):
        raise ToolOOMRetriesExhaustedError(
            {
                "status": "error",
                "error_type": "tool_oom_retry_exhausted",
                "tool_name": self.name,
                "oom_attempts": 5,
                "oom_worker_history": [
                    {"worker_name": "PointToBoxMask_h20_2", "worker_addr": "http://localhost:8103"}
                ],
                "message": "five consecutive OOM responses",
            }
        )


class FakeNonOOMPhraseToPointTool(FakePhraseToPointTool):
    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("non-OOM tool implementation bug")


def make_loop():
    loop = VisHarnessAgentLoop.__new__(VisHarnessAgentLoop)
    tool = FakePhraseToPointTool()
    loop.tools = {"PhraseToPoint": tool}
    loop.tool_schemas = []
    loop.config = {"visharness": {}}
    loop.model_family = "qwen3_vl"
    loop.archive_previous_images = True
    loop.max_agent_turns = 4
    loop.per_turn_max_response_length = 4
    loop.rollout_max_prompt_length = 100
    loop.validation_max_prompt_length = 200
    loop.max_trajectory_length = 32
    loop.response_length = 32
    loop.server_manager = FakeServerManager()
    loop.tokenizer = FakeTokenizer()
    loop.mm_processor_kwargs = {}
    loop.apply_chat_template_calls = []

    async def process_multi_modal_info(messages):
        return {"images": [Image.new("RGB", (8, 8), "white")]}

    async def apply_chat_template(messages, **kwargs):
        loop.apply_chat_template_calls.append(kwargs)
        if kwargs.get("remove_system_prompt"):
            return [90, 91]
        return [1, 2, 3]

    loop.process_multi_modal_info = process_multi_modal_info
    loop.apply_chat_template = apply_chat_template
    loop._get_mm_processor_kwargs = lambda audio_data=None: {}
    return loop, tool


def test_agent_loop_runs_tool_then_answer_and_builds_training_mask():
    loop, tool = make_loop()
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert output.response_ids == [99]
    assert output.response_mask == [0]
    assert output.response_logprobs is None
    assert output.extra_fields["turn_records"][0]["observation_ids"] == [89, 90, 91]
    assert output.extra_fields["turn_records"][0]["response_ids"] == [11]
    assert output.extra_fields["turn_records"][1]["response_ids"] == [12]
    assert tool.parameters["phrase"] == "person"
    assert "image_dict" in tool.parameters
    assert output.extra_fields["tool_call_count"] == 1
    assert output.extra_fields["trajectory_finished"] is True
    assert output.extra_fields["trajectory_uid"]
    assert len(output.extra_fields["turn_records"]) == 2
    assert output.extra_fields["turn_records"][0]["mm_processor_kwargs"] == {}
    assert loop.server_manager.calls[0]["sampling_params"]["max_tokens"] == 4
    assert loop.server_manager.calls[1]["sampling_params"]["max_tokens"] == 4
    assert output.extra_fields["turn_records"][0]["turn_max_tokens"] == 4
    assert output.extra_fields["turn_records"][0]["turn_response_length"] == 1
    assert output.extra_fields["turn_records"][0]["tool_arguments"] == {
        "images": ["img_0"],
        "phrase": "person",
    }
    assert output.extra_fields["turn_records"][0]["tool_execution_success"] is True
    assert output.extra_fields["turn_records"][0]["tool_result_summary"]["per_image"]["img_0"]["points"] == [
        [2.0, 3.0]
    ]
    assert output.extra_fields["original_image_size"] == {"width": 8, "height": 8}
    assert output.extra_fields["trajectory_response_placeholder"] is True
    assert output.extra_fields["trajectory_response_tensor_length"] == 0
    assert output.extra_fields["trajectory_token_length"] == 8
    assert output.extra_fields["assistant_response_token_count"] == 2
    assert output.extra_fields["observation_token_count"] == 3
    assert output.extra_fields["generate_time_seconds"] >= 0
    assert output.extra_fields["tool_time_seconds"] >= 0
    assert output.extra_fields["trajectory_elapsed_seconds"] >= 0
    observation_text = "".join(
        part.get("text", "")
        for message in loop._visharness_error_context["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )
    assert VISION_TOOL_PROMPT.rstrip("\r\n") in observation_text
    assert all(
        call["max_prompt_length"] == 100
        for call in loop.apply_chat_template_calls
    )


def test_validation_sample_uses_greedy_sampling_parameters_and_token_limit():
    loop, _ = make_loop()
    loop.validation_max_response_length = 6
    loop.validation_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(
        loop.run(
            {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
            raw_prompt=raw_prompt,
            extra_info={"validation_sample": True},
        )
    )

    for call in loop.server_manager.calls:
        params = call["sampling_params"]
        assert params["max_tokens"] == 6
        assert params["temperature"] == 0.0
        assert params["top_p"] == 1.0
        assert params["top_k"] == -1
        assert params["presence_penalty"] == 0.0
        assert params["min_p"] == 0.0
        assert params["repetition_penalty"] == 1.0
    assert output.extra_fields["validation_rollout"] is True
    assert all(turn["turn_max_tokens"] == 6 for turn in output.extra_fields["turn_records"])
    assert all(
        call["max_prompt_length"] == 200
        for call in loop.apply_chat_template_calls
    )


def test_agent_loop_restores_length_stop_reason_and_uses_dynamic_limit_in_error():
    loop, _ = make_loop()
    loop.per_turn_max_response_length = 7
    loop.server_manager.outputs = [
        TokenOutput(
            token_ids=[13] * 7,
            log_probs=[-0.3] * 7,
            stop_reason="completed",
        ),
        TokenOutput(token_ids=[12], log_probs=[-0.2], stop_reason="completed"),
    ]
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert len(loop.server_manager.calls) == 2
    assert loop.server_manager.calls[0]["sampling_params"]["max_tokens"] == 7
    first_turn = output.extra_fields["turn_records"][0]
    assert first_turn["backend_stop_reason"] == "completed"
    assert first_turn["stop_reason"] == "length"
    assert first_turn["turn_truncated_by_length"] is True
    assert first_turn["action_type"] == "invalid"
    assert first_turn["action_error"] == (
        "Incorrect output format: The response reached the 7-token limit and was truncated. "
        "Retry with concise reasoning and one complete tool call or final answer."
    )
    assert output.extra_fields["turn_records"][1]["action_type"] == "answer"
    assert output.extra_fields["trajectory_finished"] is True


def test_agent_loop_marks_trajectory_invalid_after_tool_oom_limit():
    loop, _ = make_loop()
    oom_tool = FakeOOMPhraseToPointTool()
    loop.tools = {"PhraseToPoint": oom_tool}
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert len(loop.server_manager.calls) == 1
    assert output.reward_score == 0.0
    assert output.response_mask == [0]
    assert output.extra_fields["trajectory_invalid"] is True
    assert output.extra_fields["trajectory_aborted"] is True
    assert output.extra_fields["invalid_reason"] == "tool_oom_retry_exhausted"
    assert output.extra_fields["tool_oom_attempts"] == 5
    assert output.extra_fields["tool_call_count"] == 1
    turn = output.extra_fields["turn_records"][0]
    assert turn["tool_execution_success"] is False
    assert turn["tool_execution_error_type"] == "tool_oom_retry_exhausted"
    assert turn["tool_oom_attempts"] == 5


def test_agent_loop_does_not_convert_non_oom_tool_error_into_invalid_trajectory():
    loop, _ = make_loop()
    loop.tools = {"PhraseToPoint": FakeNonOOMPhraseToPointTool()}
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    with pytest.raises(RuntimeError, match="non-OOM tool implementation bug"):
        asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))


def test_validation_tool_oom_defers_scoring_to_emit_complete_metrics():
    loop, _ = make_loop()
    loop.tools = {"PhraseToPoint": FakeOOMPhraseToPointTool()}
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(
        loop.run(
            {"temperature": 1.0},
            raw_prompt=raw_prompt,
            extra_info={"validation_sample": True},
        )
    )

    assert output.reward_score is None
    assert output.extra_fields["trajectory_invalid"] is True
    assert output.extra_fields["invalid_reason"] == "tool_oom_retry_exhausted"
    assert output.extra_fields["assistant_response_token_count"] == 1


def test_initial_tool_image_prefers_extra_info_image_path(tmp_path):
    loop, tool = make_loop()
    image_path = tmp_path / "dataset_image.jpg"
    Image.new("RGB", (13, 17), "blue").save(image_path)
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt, extra_info={"image_path": str(image_path)}))

    tool_image_bytes = tool.parameters["image_dict"]["img_0"]
    with Image.open(BytesIO(tool_image_bytes)) as tool_image:
        assert tool_image.size == (13, 17)
    assert loop.server_manager.calls[0]["image_data"][0].size == (13, 17)


def test_rollout_uses_placeholder_response_tensor_and_turn_records():
    loop, _ = make_loop()
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert output.response_ids == [99]
    assert output.response_mask == [0]
    assert output.response_logprobs is None
    assert output.extra_fields["turn_records"][0]["observation_ids"] == [89, 90, 91]
    assert output.extra_fields["turn_records"][0]["response_ids"] == [11]
    assert output.extra_fields["turn_records"][1]["response_ids"] == [12]
    assert output.extra_fields["trajectory_response_placeholder"] is True
    assert output.extra_fields["trajectory_response_tensor_length"] == 0
    assert output.extra_fields["trajectory_response_length"] == 5


def test_error_observation_keeps_current_visual_context():
    loop, _ = make_loop()
    loop.server_manager.outputs = [
        TokenOutput(token_ids=[13], log_probs=[-0.3]),
        TokenOutput(token_ids=[12], log_probs=[-0.2]),
    ]
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert output.extra_fields["turn_records"][0]["output_format_success"] is False
    assert output.extra_fields["turn_records"][0]["visible_image_names_after_step"] == ["img_0"]
    assert len(loop.server_manager.calls[0]["image_data"]) == 1
    assert len(loop.server_manager.calls[1]["image_data"]) == 1
    observation_text = "".join(
        part.get("text", "")
        for message in loop._visharness_error_context["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )
    assert VISION_TOOL_PROMPT.rstrip("\r\n") not in observation_text


def test_successful_text_only_tool_observation_archives_current_visual_context():
    loop, _ = make_loop()
    loop.tools = {"SuperResolution": FakeSuperResolutionTool()}
    loop.server_manager.outputs = [
        TokenOutput(token_ids=[14], log_probs=[-0.3]),
        TokenOutput(token_ids=[12], log_probs=[-0.2]),
    ]
    raw_prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "white")},
                {"type": "text", "text": "Find the person."},
            ],
        },
    ]

    output = asyncio.run(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    assert output.extra_fields["turn_records"][0]["tool_name"] == "SuperResolution"
    assert output.extra_fields["turn_records"][0]["tool_args_success"] is True
    assert output.extra_fields["turn_records"][0]["visible_image_names_after_step"] == []
    assert output.extra_fields["turn_records"][1]["visible_image_names"] == []
    assert len(loop.server_manager.calls[0]["image_data"]) == 1
    assert loop.server_manager.calls[1]["image_data"] is None
    observation_text = "".join(
        part.get("text", "")
        for message in loop._visharness_error_context["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )
    assert VISION_TOOL_PROMPT.rstrip("\r\n") not in observation_text


def test_restore_implicit_think_start_from_completion_shape():
    implicit = "reasoning body</think><answer>done</answer>"

    assert _restore_implicit_think_start(implicit) == (
        "<think>\nreasoning body</think><answer>done</answer>"
    )
    assert _restore_implicit_think_start("\nreasoning body</think><answer>done</answer>") == (
        "<think>\nreasoning body</think><answer>done</answer>"
    )
    assert _restore_implicit_think_start("<think>x</think><answer>done</answer>") == (
        "<think>x</think><answer>done</answer>"
    )
