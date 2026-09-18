import base64
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from PIL import Image

from visharness.agent_loop.tool_response_processor import build_error_observation
from visharness.prompts import (
    PHRASE_TO_BOXMASK_PROMPT,
    PHRASE_TO_POINT_PROMPT,
    POINT_TO_BOXMASK_PROMPT,
    SPLIT_PROMPT,
    SR_PROMPT,
    VISION_TOOL_PROMPT,
)
from visharness.trajectory_runner.benchmark import (
    _calculate_kv_cache_capacity,
    _parse_prometheus_metric_labels,
)
from visharness.trajectory_runner.dataset import TrajectoryDataset
from visharness.trajectory_runner.inferencer import (
    BaseTrajectoryInferencer,
    SyncToolCaller,
)
from visharness.trajectory_runner.errors import (
    TrajectoryContextLengthExceededError,
    TrajectoryModelRequestTimeoutError,
    TrajectoryPersistenceError,
    TrajectoryVisionEncoderCacheExceededError,
)
from visharness.trajectory_runner.model_client import (
    ModelTurnResponse,
    OnlineVllmModelClient,
    _is_context_length_exceeded_error,
    _is_vision_encoder_cache_exceeded_error,
    openai_message_to_assistant_message,
)
from visharness.trajectory_runner.openai_action_parser import parse_openai_action
from visharness.trajectory_runner.serializer import (
    TrajectorySerializer,
    strip_tool_response_wrapper,
)
from visharness.tools.errors import ToolOOMRetriesExhaustedError


def image_bytes(color="red"):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def assert_only_initial_user_message_is_unwrapped(conversation):
    user_messages = [
        message
        for message in conversation
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    assert user_messages
    assert strip_tool_response_wrapper(user_messages[0].get("content")) is None
    for message in user_messages[1:]:
        assert strip_tool_response_wrapper(message.get("content")) is not None


def submit_final_answer_message(arguments, reasoning_content="ready to answer"):
    return ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content=reasoning_content,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_final",
                type="function",
                function=Function(
                    name="SubmitFinalAnswer",
                    arguments=json.dumps(arguments),
                ),
            )
        ],
    )


def structured_tool_message(
    name,
    arguments,
    *,
    call_id="call_tool",
    reasoning_content="use the visual tool",
):
    return ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content=reasoning_content,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id=call_id,
                type="function",
                function=Function(
                    name=name,
                    arguments=(
                        json.dumps(arguments)
                        if isinstance(arguments, dict)
                        else arguments
                    ),
                ),
            )
        ],
    )


def structured_response(message, **kwargs):
    return ModelTurnResponse(
        completion=None,
        api_message=message,
        assistant_message=None,
        **kwargs,
    )


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_visible_names = []

    def generate_conversation_fn(self, text, image):
        return [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "image_name": "img_0"},
                    {"type": "text", "text": text},
                ],
            },
        ]

    def generate_one_item(self, item):
        self.seen_visible_names.append(list(item.current_image_names))
        response = self.responses.pop(0)
        if response.api_message is None:
            response.used_token_completion = True
        return response


class FakeToolCaller:
    available_tools = [
        "PhraseToPoint",
        "PhraseToBoxMask",
        "PointToBoxMask",
        "SplitImageIntoPatches",
        "SuperResolution",
        "MergeBoxMask",
    ]

    def call(self, tool_name, tool_parameters):
        assert tool_name == "PhraseToPoint"
        return {
            "img_0": {
                "visual_image": image_bytes("blue"),
                "text_response": "Found one point.",
                "points": [[2, 3]],
            }
        }


class FakeTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded
        self.decode_kwargs = None
        self.decode_calls = []

    def decode(self, token_ids, **kwargs):
        self.decode_kwargs = {"token_ids": list(token_ids), **kwargs}
        self.decode_calls.append(self.decode_kwargs)
        return self.decoded

    def convert_tokens_to_ids(self, token):
        return 151655 if token == "<|image_pad|>" else None

    def convert_ids_to_tokens(self, token_id):
        return "<|image_pad|>" if token_id == 151655 else None


def make_inferencer(responses, **kwargs):
    return BaseTrajectoryInferencer(
        tp_model=FakeModel(responses),
        max_rounds=len(responses),
        mode=kwargs.pop("mode", "inference"),
        serializer=kwargs.pop("serializer", TrajectorySerializer(None)),
        tool_caller=kwargs.pop("tool_caller", FakeToolCaller()),
        **kwargs,
    )


def make_item(inferencer):
    return inferencer._make_item(
        {
            "id": "sample-0",
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
    )


def observation_text(observation):
    return "".join(
        part.get("text", "")
        for part in observation.message["content"]
        if isinstance(part, dict)
    )


@pytest.mark.parametrize(
    ("tool_name", "specific_prompt"),
    [
        ("PhraseToBoxMask", PHRASE_TO_BOXMASK_PROMPT),
        ("PhraseToPoint", PHRASE_TO_POINT_PROMPT),
        ("PointToBoxMask", POINT_TO_BOXMASK_PROMPT),
    ],
)
def test_data_generation_adds_vision_and_specific_tool_guidance(
    tool_name,
    specific_prompt,
):
    inferencer = make_inferencer(
        [ModelTurnResponse("unused")],
        mode="data_generation",
    )
    observation = build_error_observation("Tool result.")

    inferencer._add_tool_result_guidance(
        observation,
        tool_name=tool_name,
        original_prompt="Find the person.",
    )

    text = observation_text(observation)
    assert VISION_TOOL_PROMPT.rstrip("\r\n") in text
    assert specific_prompt.rstrip("\r\n") in text
    assert "Continue the reasoning process to answer the original question:" in text
    api_message = inferencer._message_for_observation(
        observation,
        {"tool_call_id": "call_1"},
    )
    api_text = "".join(
        part.get("text", "")
        for part in api_message["content"]
        if isinstance(part, dict)
    )
    assert not api_text.endswith(("\n", "\r"))


@pytest.mark.parametrize(
    ("tool_name", "specific_prompt"),
    [
        ("SplitImageIntoPatches", SPLIT_PROMPT),
        ("SuperResolution", SR_PROMPT),
    ],
)
def test_data_generation_adds_only_specific_guidance_for_image_transform_tools(
    tool_name,
    specific_prompt,
):
    inferencer = make_inferencer(
        [ModelTurnResponse("unused")],
        mode="data_generation",
    )
    observation = build_error_observation("Tool result.")

    inferencer._add_tool_result_guidance(
        observation,
        tool_name=tool_name,
        original_prompt="Find the person.",
    )

    text = observation_text(observation)
    assert specific_prompt.rstrip("\r\n") in text
    assert VISION_TOOL_PROMPT.rstrip("\r\n") not in text
    assert "Continue the reasoning process to answer the original question:" in text
    api_message = inferencer._message_for_observation(
        observation,
        {"tool_call_id": "call_1"},
    )
    api_text = "".join(
        part.get("text", "")
        for part in api_message["content"]
        if isinstance(part, dict)
    )
    assert not api_text.endswith(("\n", "\r"))


@pytest.mark.parametrize(
    "tool_name",
    ["PhraseToBoxMask", "PhraseToPoint", "PointToBoxMask"],
)
def test_inference_adds_only_vision_guidance_for_result_tools(tool_name):
    inferencer = make_inferencer([ModelTurnResponse("unused")], mode="inference")
    observation = build_error_observation("Tool result.")

    inferencer._add_tool_result_guidance(
        observation,
        tool_name=tool_name,
        original_prompt="Find the person.",
    )

    text = observation_text(observation)
    assert VISION_TOOL_PROMPT.rstrip("\r\n") in text
    assert "Continue the reasoning process" not in text
    assert PHRASE_TO_BOXMASK_PROMPT.rstrip("\r\n") not in text
    assert PHRASE_TO_POINT_PROMPT.rstrip("\r\n") not in text
    assert POINT_TO_BOXMASK_PROMPT.rstrip("\r\n") not in text


@pytest.mark.parametrize(
    "tool_name",
    ["SplitImageIntoPatches", "SuperResolution", "MergeBoxMask"],
)
def test_inference_does_not_add_guidance_for_other_tools(tool_name):
    inferencer = make_inferencer([ModelTurnResponse("unused")], mode="inference")
    observation = build_error_observation("Tool result.")

    inferencer._add_tool_result_guidance(
        observation,
        tool_name=tool_name,
        original_prompt="Find the person.",
    )

    text = observation_text(observation)
    assert VISION_TOOL_PROMPT.rstrip("\r\n") not in text
    assert "Continue the reasoning process" not in text


def test_format_error_observation_keeps_current_visual_context():
    inferencer = make_inferencer(
        [
            ModelTurnResponse("<think>bad</think>I should retry."),
            ModelTurnResponse("<think>done</think><answer>Done.</answer>"),
        ]
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    assert item.turn_records[0]["output_format_success"] is False
    assert item.turn_records[0]["visible_image_names_after_step"] == ["img_0"]
    assert inferencer.tp_model.seen_visible_names == [["img_0"], ["img_0"]]
    assert item.conversation[1]["content"][0]["type"] == "image"


def test_tool_argument_error_after_success_keeps_last_tool_visual_visible():
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
            ),
            ModelTurnResponse(
                '<think>box</think><tool_call>{"name":"PointToBoxMask",'
                '"arguments":{"images":["img_0"],"mode":"bad"}}</tool_call>'
            ),
            ModelTurnResponse("<think>done</think><answer>Done.</answer>"),
        ]
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    assert item.turn_records[0]["tool_execution_success"] is True
    assert item.turn_records[0]["visible_image_names_after_step"] == ["img_0_PhraseToPoint_visual"]
    assert item.turn_records[1]["tool_args_success"] is False
    assert item.turn_records[1]["visible_image_names_after_step"] == ["img_0_PhraseToPoint_visual"]
    assert inferencer.tp_model.seen_visible_names == [
        ["img_0"],
        ["img_0_PhraseToPoint_visual"],
        ["img_0_PhraseToPoint_visual"],
    ]


def test_successful_tool_observation_archives_previous_image():
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
            ),
            ModelTurnResponse("<think>done</think><answer>Done.</answer>"),
        ]
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    first_user_content = item.conversation[1]["content"]
    assert first_user_content[0]["type"] == "text"
    assert "archived" in first_user_content[0]["text"]
    assert item.turn_records[0]["visible_image_names_after_step"] == ["img_0_PhraseToPoint_visual"]


def test_full_history_mode_keeps_all_visual_images_visible():
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
            ),
            ModelTurnResponse("<think>done</think><answer>Done.</answer>"),
        ],
        archive_previous_images=False,
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.conversation[1]["content"][0]["type"] == "image"
    assert inferencer.tp_model.seen_visible_names == [
        ["img_0"],
        ["img_0", "img_0_PhraseToPoint_visual"],
    ]
    assert item.turn_records[0]["visible_image_names_after_step"] == [
        "img_0",
        "img_0_PhraseToPoint_visual",
    ]


def test_data_generation_saves_only_the_corrected_valid_turn(tmp_path):
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    invalid_message = ChatCompletionMessage(
        role="assistant",
        content="I should retry.",
        reasoning_content="bad",
    )
    valid_message = submit_final_answer_message({"final_answer": "Done."})
    inferencer = make_inferencer(
        [
            structured_response(invalid_message),
            structured_response(valid_message),
        ],
        mode="data_generation",
        serializer=serializer,
        tool_caller=FakeToolCaller(),
    )
    item = make_item(inferencer)

    inferencer.process_single_trajectory(item)

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    lines = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    step_2 = lines[0]
    assert set(step_2.keys()) == {
        "schema_version",
        "id",
        "trajectory_id",
        "target_turn_index",
        "target_message_index",
        "target_action_type",
        "images",
        "messages",
    }
    assert step_2["schema_version"] == 2
    assert step_2["trajectory_id"] == "sample-0"
    assert step_2["target_turn_index"] == 2
    assert step_2["target_message_index"] == len(step_2["messages"]) - 1
    assert step_2["target_action_type"] == "answer"
    assert step_2["id"] == "sample-0_step_2"
    image_parts = [
        part
        for message in step_2["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if part.get("type") == "image_url"
    ]
    assert image_parts == [
        {
            "type": "image_url",
            "image_url": {"url": "images/sample-0_step_2/img_0.jpg"},
        }
    ]
    error_messages = [
        message
        for message in step_2["messages"]
        if message["role"] == "tool"
        and message.get("tool_call_id") is None
    ]
    assert len(error_messages) == 1
    assert strip_tool_response_wrapper(error_messages[0]["content"]) is None
    error_text = "".join(
        part.get("text", "")
        for part in error_messages[0]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "Incorrect tool call format" in error_text
    assert "Continue the reasoning process" not in error_text


def test_ckpt_storage_restores_latest_final_visual_image_without_mutating_runtime_state(tmp_path):
    serializer = TrajectorySerializer(tmp_path / "run")
    first_visual = b"\xfffirst-merged-visual"
    latest_visual = b"\xfflatest-merged-visual"
    runtime_final_results = {
        "final_bboxes": [[1, 2, 3, 4]],
        "final_masks": [{"size": [8, 8], "counts": "encoded"}],
        "count": 1,
    }
    result = {
        "conversation": [],
        "turn_records": [],
        "tool_response": [
            {
                "final_bboxes": [[0, 0, 1, 1]],
                "final_masks": [{"size": [8, 8], "counts": "old"}],
                "count": 1,
                "final_visual_image": first_visual,
            },
            {"text_response": "an unrelated visual-tool response"},
            {
                "final_bboxes": [[1, 2, 3, 4]],
                "final_masks": [{"size": [8, 8], "counts": "encoded"}],
                "count": 1,
                "final_visual_image": latest_visual,
            },
        ],
        "final_results": runtime_final_results,
    }

    serializer.store_result(result)

    assert "final_visual_image" not in runtime_final_results
    assert "final_visual_image" not in result["final_results"]
    stored_result = json.loads(
        (tmp_path / "run" / "run_ckpt.jsonl").read_text(encoding="utf-8").strip()
    )
    assert stored_result["final_results"]["final_visual_image"] == base64.b64encode(
        latest_visual
    ).decode("utf-8")
    assert stored_result["final_results"]["final_bboxes"] == [[1, 2, 3, 4]]
    assert stored_result["final_results"]["count"] == 1


def test_dataset_task_names_string_is_not_split_into_characters(tmp_path):
    selected = [{"id": "keep", "question": "q", "image_path": "keep.jpg"}]
    skipped = [{"id": "skip", "question": "q", "image_path": "skip.jpg"}]
    (tmp_path / "Dense200_QA_test.json").write_text(
        json.dumps(selected),
        encoding="utf-8",
    )
    (tmp_path / "Dense201_QA_test.json").write_text(
        json.dumps(skipped),
        encoding="utf-8",
    )

    dataset = TrajectoryDataset(
        {
            "dataset_path": str(tmp_path),
            "task_names": "Dense200",
            "split": "test",
            "shuffle": False,
        }
    )

    assert [item["id"] for item in dataset.full_data] == ["keep"]


def test_dataset_split_matches_exact_manifest_name(tmp_path):
    full_records = [{"id": "full", "question": "q", "image_path": "full.jpg"}]
    subset_records = [{"id": "subset", "question": "q", "image_path": "subset.jpg"}]
    (tmp_path / "GRES_QA_val.json").write_text(
        json.dumps(full_records),
        encoding="utf-8",
    )
    (tmp_path / "GRES_QA_val-subset200.json").write_text(
        json.dumps(subset_records),
        encoding="utf-8",
    )

    full_dataset = TrajectoryDataset(
        {
            "dataset_path": str(tmp_path),
            "task_names": "GRES",
            "split": "val",
        }
    )
    subset_dataset = TrajectoryDataset(
        {
            "dataset_path": str(tmp_path),
            "task_names": "GRES",
            "split": "val-subset200",
        }
    )

    assert [item["id"] for item in full_dataset.full_data] == ["full"]
    assert [item["id"] for item in subset_dataset.full_data] == ["subset"]


def test_directory_dataset_requires_task_names_and_split(tmp_path):
    with pytest.raises(
        ValueError,
        match="both task_names and split are required",
    ):
        TrajectoryDataset({"dataset_path": str(tmp_path)})


def test_directory_dataset_fails_when_exact_manifest_is_missing(tmp_path):
    (tmp_path / "GRES_QA_val-subset200.json").write_text("[]", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError,
        match="GRES_QA_val.json",
    ):
        TrajectoryDataset(
            {
                "dataset_path": str(tmp_path),
                "task_names": "GRES",
                "split": "val",
            }
        )


def test_dataset_rejects_duplicate_ids_before_inference(tmp_path):
    duplicate_records = [
        {"id": "duplicate", "question": "first", "image_path": "first.jpg"},
        {"id": "duplicate", "question": "second", "image_path": "second.jpg"},
    ]
    (tmp_path / "GRES_QA_val.json").write_text(
        json.dumps(duplicate_records),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"1 duplicate IDs.*duplicate",
    ):
        TrajectoryDataset(
            {
                "dataset_path": str(tmp_path),
                "task_names": "GRES",
                "split": "val",
            }
        )


def test_dataset_fails_when_configured_resume_checkpoint_is_missing(tmp_path):
    (tmp_path / "GRES_QA_val.json").write_text(
        json.dumps([{"id": "sample", "question": "q", "image_path": "sample.jpg"}]),
        encoding="utf-8",
    )
    missing_checkpoint = tmp_path / "missing_ckpt.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match=r"Configured resume checkpoint does not exist: .*missing_ckpt\.jsonl",
    ):
        TrajectoryDataset(
            {
                "dataset_path": str(tmp_path),
                "task_names": "GRES",
                "split": "val",
                "resume_from_ckpt": [str(missing_checkpoint)],
            }
        )


def test_dataset_existing_resume_checkpoint_filters_processed_ids(tmp_path):
    (tmp_path / "GRES_QA_val.json").write_text(
        json.dumps(
            [
                {"id": "processed", "question": "q", "image_path": "first.jpg"},
                {"id": "pending", "question": "q", "image_path": "second.jpg"},
            ]
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "run_ckpt.jsonl"
    checkpoint.write_text(
        json.dumps({"meta_data": {"id": "processed"}}) + "\n",
        encoding="utf-8",
    )

    dataset = TrajectoryDataset(
        {
            "dataset_path": str(tmp_path),
            "task_names": "GRES",
            "split": "val",
            "resume_from_ckpt": str(checkpoint),
        }
    )

    assert [item["id"] for item in dataset.meta_data] == ["pending"]


def test_dataset_task_names_folded_multiline_scalar_selects_every_task(tmp_path):
    for task_name in ("ReasonSeg", "GRES", "REC8K"):
        records = [{"id": task_name, "question": "q", "image_path": f"{task_name}.jpg"}]
        (tmp_path / f"{task_name}_QA_train.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )
    (tmp_path / "Other_QA_train.json").write_text(
        json.dumps([{"id": "Other", "question": "q", "image_path": "Other.jpg"}]),
        encoding="utf-8",
    )

    dataset = TrajectoryDataset(
        {
            "dataset_path": str(tmp_path),
            # This is the string produced by PyYAML for the legacy multiline
            # scalar syntax used in the data-generation configs.
            "task_names": "ReasonSeg GRES REC8K",
            "split": "train",
            "shuffle": False,
        }
    )

    assert {item["id"] for item in dataset.full_data} == {"ReasonSeg", "GRES", "REC8K"}


def test_structured_openai_tool_call_is_preserved_and_parsed(tmp_path):
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="This text must not appear after the thinking block in SFT data.",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    assistant_message = openai_message_to_assistant_message(raw_message)
    inferencer = make_inferencer(
        [
            structured_response(raw_message),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
        serializer=serializer,
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    assert item.turn_records[0]["tool_name"] == "PhraseToPoint"
    assert item.turn_records[0]["completion"] is None
    assert item.conversation[2] is raw_message
    assert item.conversation[2].content == (
        "This text must not appear after the thinking block in SFT data."
    )
    assert set(item.result_dict().keys()) == {
        "max_rounds",
        "current_round",
        "status",
        "trajectory_finished",
        "max_rounds_reached",
        "termination_reason",
        "answer",
        "error",
        "trajectory_invalid",
        "invalid_reason",
        "invalid_stage",
        "invalid_turn_index",
        "invalid_error",
        "trajectory_uid",
        "efficiency_metrics",
        "turn_records",
        "meta_data",
        "conversation",
        "tool_response",
        "images",
        "final_results",
    }

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    first_snapshot = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])
    assistant_messages = [message for message in first_snapshot["messages"] if message["role"] == "assistant"]
    assert assistant_messages[0]["content"] == "<think>\nlocate\n</think>\n"
    assert assistant_messages[0]["tool_calls"] == assistant_message["tool_calls"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"final_answer": None},
        {"final_answer": ""},
        {"final_answer": "   "},
        {"final_answer": 123},
        {"final_answer": []},
        {"final_answer": {}},
    ],
)
def test_structured_final_answer_rejects_missing_empty_or_non_string_value(arguments):
    action = parse_openai_action(submit_final_answer_message(arguments))

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect final answer: The final answer must be a non-empty string. "
        "Please provide a valid final answer."
    )
    assert "SubmitFinalAnswer" not in action.error


def test_data_generation_does_not_parse_direct_answer_tags():
    message = ChatCompletionMessage(
        role="assistant",
        content="<answer>Done.</answer>",
        reasoning_content="ready",
    )

    action = parse_openai_action(message)

    assert action.action_type == "invalid"
    assert "invoke exactly one tool" in action.error


def test_structured_action_requires_exposed_reasoning_without_content_fallback():
    action = parse_openai_action(
        {
            "role": "assistant",
            "reasoning_content": "",
            "content": "This content must not be treated as reasoning.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Tool", "arguments": "{}"},
                }
            ],
        }
    )

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect output format: The response must include a non-empty reasoning "
        "process before invoking a tool or submitting the final answer."
    )


@pytest.mark.parametrize(
    "tool_calls",
    [
        [],
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "Tool", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "Tool", "arguments": "{}"},
            },
        ],
    ],
)
def test_structured_action_requires_exactly_one_tool_call(tool_calls):
    action = parse_openai_action(
        {
            "role": "assistant",
            "reasoning_content": "reasoning",
            "content": "",
            "tool_calls": tool_calls,
        }
    )

    assert action.action_type == "invalid"
    assert action.error == (
        "Incorrect tool call format: At each step, you must invoke exactly one tool "
        "through the provided tool-calling interface. The response must contain "
        "exactly one structured function call."
    )


def test_data_generation_multiple_tool_calls_return_tool_error_without_id_or_wrapper():
    invalid_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="I should use both tools.",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            ),
            ChatCompletionMessageFunctionToolCall(
                id="call_2",
                type="function",
                function=Function(
                    name="SuperResolution",
                    arguments='{"images":["img_0"]}',
                ),
            ),
        ],
    )
    inferencer = make_inferencer(
        [
            structured_response(invalid_message),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    turn_record = item.turn_records[0]
    assert turn_record["action_type"] == "invalid"
    assert turn_record["api_tool_call_count"] == 2
    assert turn_record["tool_call_id"] is None
    error_message = item.conversation[3]
    assert error_message["role"] == "tool"
    assert "tool_call_id" in error_message
    assert error_message["tool_call_id"] is None
    assert strip_tool_response_wrapper(error_message["content"]) is None


@pytest.mark.parametrize("call_type", [None, "custom"])
def test_structured_action_uses_extractable_function_regardless_of_call_type(call_type):
    action = parse_openai_action(
        {
            "role": "assistant",
            "reasoning_content": "reasoning",
            "content": "This text is not used to determine the structured action.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": call_type,
                    "function": {
                        "name": "Tool",
                        "arguments": '{"images": ["img_0"]}',
                    },
                }
            ],
        }
    )

    assert action.action_type == "tool_call"
    assert action.tool_name == "Tool"
    assert action.arguments == {"images": ["img_0"]}


@pytest.mark.parametrize("call_type", [None, "custom"])
def test_data_generation_sft_normalizes_accepted_call_type(call_type):
    message = {
        "role": "assistant",
        "reasoning_content": "reasoning",
        "content": "provider-specific content",
        "tool_calls": [
            {
                "id": "call_1",
                "type": call_type,
                "function": {
                    "name": "Tool",
                    "arguments": '{"images": ["img_0"]}',
                },
            }
        ],
    }

    sft_message = TrajectorySerializer._project_data_generation_assistant(
        message,
        {
            "action_type": "tool_call",
            "sft_eligible": True,
            "submit_final_answer_attempt": False,
        },
    )
    assert sft_message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["type"] == call_type


def test_data_generation_tool_call_without_id_is_fatal():
    message_without_id = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "locate",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "PhraseToPoint",
                    "arguments": '{"images":["img_0"],"phrase":"person"}',
                },
            }
        ],
    }
    inferencer = make_inferencer(
        [structured_response(message_without_id)],
        mode="data_generation",
    )

    with pytest.raises(ValueError, match="tool_call_id"):
        inferencer.process_single_trajectory(make_item(inferencer))


def test_invalid_structured_final_answer_returns_feedback_and_can_be_corrected():
    invalid_message = submit_final_answer_message({"final_answer": "   "})
    valid_message = submit_final_answer_message(
        {"final_answer": "  Detection complete. Three dogs were found.  "}
    )
    invalid_assistant = openai_message_to_assistant_message(invalid_message)
    valid_assistant = openai_message_to_assistant_message(valid_message)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=invalid_assistant["content"],
                api_message=invalid_message,
                assistant_message=invalid_assistant,
            ),
            ModelTurnResponse(
                completion=valid_assistant["content"],
                api_message=valid_message,
                assistant_message=valid_assistant,
            ),
        ],
        mode="data_generation",
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.turn_records[0]["action_type"] == "invalid"
    assert item.turn_records[0]["tool_call_id"] == "call_final"
    assert item.conversation[3]["role"] == "tool"
    assert item.conversation[3]["tool_call_id"] == "call_final"
    feedback_content = item.conversation[3]["content"]
    assert strip_tool_response_wrapper(feedback_content) is None
    feedback_text = "".join(
        part.get("text", "")
        for part in feedback_content
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "The final answer must be a non-empty string" in feedback_text
    assert "SubmitFinalAnswer" not in feedback_text
    assert item.turn_records[1]["action_type"] == "answer"
    assert item.answer == "Detection complete. Three dogs were found."
    assert item.trajectory_finished is True
    assert_only_initial_user_message_is_unwrapped(item.conversation)


def test_valid_structured_final_answer_is_saved_as_think_and_answer_without_tool_call(tmp_path):
    raw_message = submit_final_answer_message(
        {"final_answer": "Detection complete. Three dogs were found."},
        reasoning_content="The merged result contains three dogs.",
    )
    assistant_message = openai_message_to_assistant_message(raw_message)
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=assistant_message["content"],
                api_message=raw_message,
                assistant_message=assistant_message,
            )
        ],
        mode="data_generation",
        serializer=serializer,
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.answer == "Detection complete. Three dogs were found."
    snapshot = json.loads(
        (tmp_path / "run" / "run_trajectory.jsonl").read_text(encoding="utf-8").strip()
    )
    saved_assistant = [
        message for message in snapshot["messages"] if message["role"] == "assistant"
    ][0]
    assert saved_assistant["content"] == (
        "<think>\nThe merged result contains three dogs.\n</think>\n"
        "<answer>\nDetection complete. Three dogs were found.\n</answer>"
    )
    assert "tool_calls" not in saved_assistant


def test_invalid_submit_and_feedback_are_removed_from_later_sft_snapshot(tmp_path):
    invalid_message = submit_final_answer_message({"final_answer": "   "})
    valid_message = submit_final_answer_message(
        {"final_answer": "Detection complete. Three dogs were found."},
        reasoning_content="The correction contains a complete final answer.",
    )
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [structured_response(invalid_message), structured_response(valid_message)],
        mode="data_generation",
        serializer=serializer,
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.conversation[2] is invalid_message
    assert item.conversation[3]["role"] == "tool"
    snapshot_text = (tmp_path / "run" / "run_trajectory.jsonl").read_text(
        encoding="utf-8"
    )
    assert "SubmitFinalAnswer" not in snapshot_text
    snapshot = json.loads(snapshot_text)
    assistant_messages = [
        message for message in snapshot["messages"] if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert "<answer>" in assistant_messages[0]["content"]
    assert not any(message["role"] == "tool" for message in snapshot["messages"])


def test_result_metadata_distinguishes_answer_from_max_rounds(tmp_path):
    serializer = TrajectorySerializer(tmp_path / "run")
    answered_inferencer = make_inferencer(
        [ModelTurnResponse("<think>done</think><answer>Done.</answer>")],
        serializer=serializer,
    )
    answered_item = answered_inferencer.process_single_trajectory(make_item(answered_inferencer))
    answered_result = answered_item.result_dict()

    assert answered_result["status"] == "finished"
    assert answered_result["trajectory_finished"] is True
    assert answered_result["max_rounds_reached"] is False
    assert answered_result["termination_reason"] == "answer"
    assert answered_result["answer"] == "Done."
    assert answered_result["error"] is None
    assert answered_result["trajectory_uid"] == answered_item.trajectory_uid
    assert answered_result["turn_records"] == answered_item.turn_records

    serializer.store_result(answered_result)
    stored_result = json.loads(
        (tmp_path / "run" / "run_ckpt.jsonl").read_text(encoding="utf-8").strip()
    )
    assert stored_result["trajectory_finished"] is True
    assert stored_result["max_rounds_reached"] is False
    assert stored_result["termination_reason"] == "answer"
    assert stored_result["answer"] == "Done."
    assert stored_result["trajectory_uid"] == answered_item.trajectory_uid
    assert len(stored_result["turn_records"]) == 1

    exhausted_inferencer = make_inferencer(
        [ModelTurnResponse("<think>retry</think>No valid action.")]
    )
    exhausted_item = exhausted_inferencer.process_single_trajectory(make_item(exhausted_inferencer))
    exhausted_result = exhausted_item.result_dict()

    assert exhausted_result["status"] == "finished"
    assert exhausted_result["trajectory_finished"] is False
    assert exhausted_result["max_rounds_reached"] is True
    assert exhausted_result["termination_reason"] == "max_rounds_reached"
    assert exhausted_result["answer"] is None
    assert exhausted_result["error"] is None


def test_non_oom_tool_failure_is_fatal():
    class FailingToolCaller(FakeToolCaller):
        def call(self, tool_name, tool_parameters):
            raise RuntimeError("synthetic tool failure")

    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
            )
        ],
        tool_caller=FailingToolCaller(),
    )
    item = make_item(inferencer)
    with pytest.raises(RuntimeError, match="synthetic tool failure"):
        inferencer.process_single_trajectory(item)

    assert len(item.turn_records) == 1
    assert item.turn_records[0]["tool_execution_error_type"] == "RuntimeError"
    assert item.turn_records[0]["tool_execution_error"] == "synthetic tool failure"


def test_tool_oom_retry_exhaustion_aborts_only_the_trajectory():
    class OOMToolCaller(FakeToolCaller):
        def call(self, tool_name, tool_parameters):
            raise ToolOOMRetriesExhaustedError(
                {
                    "status": "error",
                    "error_type": "tool_oom_retry_exhausted",
                    "message": "five consecutive OOM responses",
                    "tool_name": tool_name,
                    "oom_attempts": 5,
                    "oom_worker_history": [
                        {"worker_name": f"worker-{index}", "worker_addr": "local"}
                        for index in range(5)
                    ],
                }
            )

    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
            )
        ],
        tool_caller=OOMToolCaller(),
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))
    result = item.result_dict()

    assert result["status"] == "failed"
    assert result["trajectory_invalid"] is True
    assert result["invalid_reason"] == "tool_oom_retry_exhausted"
    assert result["invalid_stage"] == "tool_execution"
    assert result["invalid_turn_index"] == 1
    assert result["termination_reason"] == "failed"
    assert item.turn_records[0]["tool_execution_error_type"] == (
        "tool_oom_retry_exhausted"
    )
    assert item.turn_records[0]["tool_oom_attempts"] == 5
    assert len(item.turn_records[0]["tool_oom_worker_history"]) == 5


def test_sync_tool_caller_preserves_exhausted_oom_error_type():
    response = {
        "status": "error",
        "error_type": "tool_oom_retry_exhausted",
        "message": "five consecutive OOM responses",
        "tool_name": "PhraseToPoint",
        "oom_attempts": 5,
        "oom_worker_history": [],
    }
    caller = object.__new__(SyncToolCaller)
    caller.manager = SimpleNamespace(
        dynamic_call_tool=lambda tool_name, payload: response
    )

    with pytest.raises(ToolOOMRetriesExhaustedError) as exc_info:
        caller.call("PhraseToPoint", {"images": ["img_0"]})

    assert exc_info.value.oom_attempts == 5


def test_context_length_error_aborts_only_the_trajectory():
    class ContextLengthModel(FakeModel):
        def generate_one_item(self, item):
            raise TrajectoryContextLengthExceededError(
                "maximum context length exceeded"
            )

    inferencer = make_inferencer([ModelTurnResponse("unused")])
    inferencer.tp_model = ContextLengthModel([])

    item = inferencer.process_single_trajectory(make_item(inferencer))
    result = item.result_dict()

    assert result["status"] == "failed"
    assert result["trajectory_invalid"] is True
    assert result["invalid_reason"] == "context_length_exceeded"
    assert result["invalid_stage"] == "model_generation"
    assert result["invalid_turn_index"] == 1
    assert result["termination_reason"] == "failed"


def test_context_length_classifier_does_not_hide_other_api_errors():
    context_error = RuntimeError(
        "This model's maximum context length is 32768 tokens."
    )
    unrelated_error = RuntimeError("model server returned HTTP 500")

    assert _is_context_length_exceeded_error(context_error) is True
    assert _is_context_length_exceeded_error(unrelated_error) is False


def test_online_client_translates_openai_timeout():
    provider_error = APITimeoutError(
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    )

    class FailingCompletions:
        @staticmethod
        def create(**kwargs):
            raise provider_error

    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "data_generation"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = False
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client.generation_config = {"timeout": 1000}

    with pytest.raises(TrajectoryModelRequestTimeoutError) as exc_info:
        client.generate_one_item(SimpleNamespace(conversation=[]))

    assert "1000 seconds per attempt" in str(exc_info.value)
    assert exc_info.value.__cause__ is provider_error


def test_model_request_timeout_aborts_only_the_trajectory():
    class TimeoutModel(FakeModel):
        def generate_one_item(self, item):
            raise TrajectoryModelRequestTimeoutError("Model API request timed out.")

    inferencer = make_inferencer([ModelTurnResponse("unused")])
    inferencer.tp_model = TimeoutModel([])

    item = inferencer.process_single_trajectory(make_item(inferencer))
    result = item.result_dict()

    assert result["status"] == "failed"
    assert result["trajectory_invalid"] is True
    assert result["invalid_reason"] == "model_request_timeout"
    assert result["invalid_stage"] == "model_generation"
    assert result["invalid_turn_index"] == 1
    assert result["termination_reason"] == "failed"
    assert item.turn_records == []


@pytest.mark.parametrize(
    "message",
    [
        (
            "The decoder prompt contains a(n) vision_chunk item with length 4230, "
            "which exceeds the pre-allocated encoder cache size 4225. Please reduce "
            "the input size or increase the encoder cache size by setting "
            "--limit-mm-per-prompt at startup."
        ),
        (
            "The decoder prompt contains a(n) image item with 4230 embedding tokens, "
            "which exceeds the pre-allocated encoder cache size 4225. Please reduce "
            "the input size or increase the encoder cache size by setting "
            "--limit-mm-per-prompt at startup."
        ),
    ],
)
def test_vision_encoder_cache_classifier_accepts_explicit_vllm_errors(message):
    error = RuntimeError("OpenAI-compatible request failed")
    error.body = {"error": {"message": message}}

    assert _is_vision_encoder_cache_exceeded_error(error) is True


@pytest.mark.parametrize(
    "message",
    [
        "model server returned HTTP 400",
        "vision_chunk item was invalid",
        "request exceeds the pre-allocated encoder cache size",
        "set --limit-mm-per-prompt at startup",
    ],
)
def test_vision_encoder_cache_classifier_does_not_hide_other_errors(message):
    assert _is_vision_encoder_cache_exceeded_error(RuntimeError(message)) is False


def test_online_client_translates_vision_encoder_cache_error():
    provider_error = RuntimeError("OpenAI-compatible request failed")
    provider_error.body = {
        "error": {
            "message": (
                "The decoder prompt contains a(n) vision_chunk item with length 4230, "
                "which exceeds the pre-allocated encoder cache size 4225. Please "
                "reduce the input size or increase the encoder cache size by setting "
                "--limit-mm-per-prompt at startup."
            )
        }
    }

    class FailingCompletions:
        @staticmethod
        def create(**kwargs):
            raise provider_error

    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "data_generation"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = False
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    client.generation_config = {}

    with pytest.raises(TrajectoryVisionEncoderCacheExceededError) as exc_info:
        client.generate_one_item(SimpleNamespace(conversation=[]))

    assert exc_info.value.__cause__ is provider_error


def test_vision_encoder_cache_error_aborts_only_the_trajectory():
    class VisionEncoderCacheModel(FakeModel):
        def generate_one_item(self, item):
            raise TrajectoryVisionEncoderCacheExceededError(
                "vision_chunk length 4230 exceeds cache size 4225"
            )

    inferencer = make_inferencer([ModelTurnResponse("unused")])
    inferencer.tp_model = VisionEncoderCacheModel([])

    item = inferencer.process_single_trajectory(make_item(inferencer))
    result = item.result_dict()

    assert result["status"] == "failed"
    assert result["trajectory_invalid"] is True
    assert result["invalid_reason"] == "vision_encoder_cache_exceeded"
    assert result["invalid_stage"] == "model_generation"
    assert result["invalid_turn_index"] == 1
    assert result["termination_reason"] == "failed"


def test_inference_uses_token_decoded_text_even_when_api_message_is_present(tmp_path):
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_raw",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    decoded_completion = (
        '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
        '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
    )
    assistant_message = {"role": "assistant", "content": decoded_completion}
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=decoded_completion,
                api_message=raw_message,
                assistant_message=assistant_message,
                used_token_completion=True,
            ),
            ModelTurnResponse("<think>done</think><answer>Done.</answer>"),
        ],
        serializer=serializer,
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    assert item.conversation[2] == assistant_message
    assert item.conversation[2] is not raw_message
    assert item.turn_records[0]["tool_call_id"] == "call_raw"
    assert item.conversation[3]["role"] == "user"

    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.model_name = "qwen-test"
    assert client.form_messages_from_item(SimpleNamespace(conversation=[raw_message]))[0] is raw_message

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    first_snapshot = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])
    assistant_messages = [message for message in first_snapshot["messages"] if message["role"] == "assistant"]
    assert assistant_messages[0]["content"] == decoded_completion
    assert "tool_calls" not in assistant_messages[0]
    second_snapshot = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[1])
    persisted_observations = [
        message
        for message in second_snapshot["messages"]
        if message["role"] == "user"
        and isinstance(message.get("content"), list)
        and any(
            part.get("type") == "text" and "<tool_response>" in part.get("text", "")
            for part in message["content"]
            if isinstance(part, dict)
        )
    ]
    assert len(persisted_observations) == 1


def test_serializer_preserves_visible_tool_image_before_error(tmp_path):
    serializer = TrajectorySerializer(
        tmp_path / "run",
        save_trajectory=True,
    )
    inferencer = make_inferencer(
        [
            structured_response(
                structured_tool_message(
                    "PhraseToPoint",
                    {"images": ["img_0"], "phrase": "person"},
                    call_id="call_locate",
                    reasoning_content="locate",
                )
            ),
            structured_response(
                structured_tool_message(
                    "PointToBoxMask",
                    {"images": ["img_0"], "mode": "bad"},
                    call_id="call_bad_args",
                    reasoning_content="bad args",
                )
            ),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
        serializer=serializer,
    )
    item = make_item(inferencer)

    inferencer.process_single_trajectory(item)

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    snapshots = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    snapshot = snapshots[-1]
    assert [item["id"] for item in snapshots] == ["sample-0_step_1", "sample-0_step_3"]
    image_urls = [
        part["image_url"]["url"]
        for message in snapshot["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if part.get("type") == "image_url"
    ]
    assert "images/sample-0_step_3/img_0_PhraseToPoint_visual.jpg" in image_urls


def test_data_generation_keeps_runtime_messages_but_saves_legacy_tool_protocol(tmp_path):
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_raw",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    assistant_message = openai_message_to_assistant_message(raw_message)
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=assistant_message["content"],
                api_message=raw_message,
                assistant_message=assistant_message,
            ),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
        serializer=serializer,
    )
    item = inferencer.parallel_batch_inference(
        [
            {
                "id": "sample-0",
                "question": "Find the person.",
                "image": Image.new("RGB", (8, 8), "white"),
            }
        ]
    )[0]

    assert item.conversation[2] is raw_message
    assert item.conversation[3]["role"] == "tool"
    assert item.conversation[3]["tool_call_id"] == "call_raw"
    runtime_tool_text = "".join(
        part.get("text", "")
        for part in item.conversation[3]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "<tool_response>" not in runtime_tool_text
    assert "</tool_response>" not in runtime_tool_text
    assert "Found one point." in runtime_tool_text
    assert item.turn_records[0]["tool_call_id"] == "call_raw"
    assert_only_initial_user_message_is_unwrapped(item.conversation)

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    snapshots = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    first_snapshot = snapshots[0]
    assistant_messages = [message for message in first_snapshot["messages"] if message["role"] == "assistant"]
    assert assistant_messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "images": ["img_0"],
        "phrase": "person",
    }
    tool_messages = [message for message in snapshots[1]["messages"] if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_raw"
    persisted_tool_text = "".join(
        part.get("text", "")
        for part in tool_messages[0]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "<tool_response>" not in persisted_tool_text
    assert "</tool_response>" not in persisted_tool_text
    assert "Found one point." in persisted_tool_text

    ckpt_path = tmp_path / "run" / "run_ckpt.jsonl"
    stored_result = json.loads(ckpt_path.read_text(encoding="utf-8").strip())
    stored_tool_messages = [
        message for message in stored_result["conversation"] if message["role"] == "tool"
    ]
    assert len(stored_tool_messages) == 1
    assert stored_tool_messages[0]["tool_call_id"] == "call_raw"
    stored_tool_text = "".join(
        part.get("text", "")
        for part in stored_tool_messages[0]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "<tool_response>" not in stored_tool_text
    assert "</tool_response>" not in stored_tool_text
    assert "Found one point." in stored_tool_text


def test_data_generation_invalid_structured_output_preserves_tool_call_id(tmp_path):
    invalid_raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_invalid",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    invalid_assistant_message = openai_message_to_assistant_message(invalid_raw_message)
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=invalid_assistant_message["content"],
                api_message=invalid_raw_message,
                assistant_message=invalid_assistant_message,
            ),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
        serializer=serializer,
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.turn_records[0]["action_type"] == "invalid"
    assert item.turn_records[0]["tool_call_id"] == "call_invalid"
    assert item.turn_records[0]["observation_appended"] is True
    assert item.conversation[3]["role"] == "tool"
    assert item.conversation[3]["tool_call_id"] == "call_invalid"
    assert strip_tool_response_wrapper(item.conversation[3]["content"]) is None
    assert_only_initial_user_message_is_unwrapped(item.conversation)

    trajectory_path = tmp_path / "run" / "run_trajectory.jsonl"
    snapshots = trajectory_path.read_text(encoding="utf-8").splitlines()
    assert len(snapshots) == 1
    second_snapshot = json.loads(snapshots[0])
    assert second_snapshot["id"] == "sample-0_step_2"
    tool_messages = [
        message for message in second_snapshot["messages"] if message["role"] == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_invalid"


def test_data_generation_parameter_error_uses_openai_tool_message():
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="convert point",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_bad_args",
                type="function",
                function=Function(
                    name="PointToBoxMask",
                    arguments='{"images":["img_0"],"mode":"confidence"}',
                ),
            )
        ],
    )
    assistant_message = openai_message_to_assistant_message(raw_message)
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=assistant_message["content"],
                api_message=raw_message,
                assistant_message=assistant_message,
            ),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    error_message = item.conversation[3]
    assert error_message["role"] == "tool"
    assert error_message["tool_call_id"] == "call_bad_args"
    assert strip_tool_response_wrapper(error_message["content"]) is None
    error_text = "".join(
        part.get("text", "")
        for part in error_message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "must have already been obtained valid points" in error_text
    assert "Based on the returned text and visualized results" not in error_text
    assert not error_text.endswith(("\n", "\r"))
    assert "bounding boxes shown in the visualization" not in error_text
    assert "Continue the reasoning process" not in error_text
    assert item.turn_records[0]["tool_args_success"] is False
    assert_only_initial_user_message_is_unwrapped(item.conversation)


def test_checkpoint_preserves_observation_ids_for_valid_and_invalid_tool_calls(tmp_path):
    successful_raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_success",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    invalid_raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_rejected",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )
    successful_assistant = openai_message_to_assistant_message(successful_raw_message)
    invalid_assistant = openai_message_to_assistant_message(invalid_raw_message)
    serializer = TrajectorySerializer(tmp_path / "run")
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=successful_assistant["content"],
                api_message=successful_raw_message,
                assistant_message=successful_assistant,
            ),
            ModelTurnResponse(
                completion=invalid_assistant["content"],
                api_message=invalid_raw_message,
                assistant_message=invalid_assistant,
            ),
            structured_response(
                submit_final_answer_message({"final_answer": "Done."})
            ),
        ],
        mode="data_generation",
        serializer=serializer,
    )

    item = inferencer.parallel_batch_inference(
        [
            {
                "id": "sample-0",
                "question": "Find the person.",
                "image": Image.new("RGB", (8, 8), "white"),
            }
        ]
    )[0]

    runtime_environment_messages = [
        message
        for message in item.conversation
        if isinstance(message, dict)
        and (
            message.get("role") == "tool"
            or (
                message.get("role") == "user"
                and strip_tool_response_wrapper(message.get("content")) is not None
            )
        )
    ]
    assert [message["role"] for message in runtime_environment_messages] == ["tool", "tool"]
    assert [message["tool_call_id"] for message in runtime_environment_messages] == [
        "call_success",
        "call_rejected",
    ]
    assert_only_initial_user_message_is_unwrapped(item.conversation)

    stored_result = json.loads(
        (tmp_path / "run" / "run_ckpt.jsonl").read_text(encoding="utf-8").strip()
    )
    stored_tool_messages = [
        message for message in stored_result["conversation"] if message["role"] == "tool"
    ]
    assert [message.get("tool_call_id") for message in stored_tool_messages] == [
        "call_success",
        "call_rejected",
    ]


def test_online_client_prefers_returned_token_ids_for_completion():
    decoded_completion = (
        '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
        '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
    )
    tokenizer = FakeTokenizer(decoded_completion)
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_structured",
                type="function",
                function=Function(
                    name="PointToBoxMask",
                    arguments='{"images":["img_0"],"points":[[1,2]]}',
                ),
            )
        ],
    )

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=raw_message,
                        finish_reason="tool_calls",
                        model_extra={"token_ids": [11, 12], "stop_reason": None},
                    )
                ]
            )

    completions = FakeCompletions()
    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "inference"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = True
    client.tokenizer = tokenizer
    client.tokenizer_path = "unused"
    client.trust_remote_code = True
    client._tokenizer_load_error = None
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.generation_config = {"extra_body": {"top_k": 20}}

    response = client.generate_one_item(SimpleNamespace(conversation=[]))

    assert completions.kwargs["extra_body"]["return_token_ids"] is True
    assert completions.kwargs["extra_body"]["top_k"] == 20
    assert tokenizer.decode_calls[0] == {
        "token_ids": [11, 12],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    assert response.used_token_completion is True
    assert response.completion == decoded_completion
    assert response.assistant_message == {"role": "assistant", "content": decoded_completion}
    assert response.completion_token_count == 2
    assert not hasattr(response, "raw_message")


def test_online_client_defaults_to_structured_outputs_for_data_generation():
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="",
        reasoning_content="locate",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_structured",
                type="function",
                function=Function(
                    name="PhraseToPoint",
                    arguments='{"images":["img_0"],"phrase":"person"}',
                ),
            )
        ],
    )

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=raw_message,
                        finish_reason="tool_calls",
                        model_extra={"token_ids": [11, 12], "stop_reason": None},
                    )
                ]
            )

    completions = FakeCompletions()
    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "data_generation"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = False
    client.tokenizer = FakeTokenizer("<think>wrong branch</think>")
    client.tokenizer_path = "unused"
    client.trust_remote_code = True
    client._tokenizer_load_error = None
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.generation_config = {"extra_body": {"top_k": 20}}

    response = client.generate_one_item(SimpleNamespace(conversation=[]))

    assert "return_token_ids" not in completions.kwargs["extra_body"]
    assert response.used_token_completion is False
    assert response.token_ids == [11, 12]
    assert response.api_message is raw_message
    assert response.assistant_message is None
    assert response.completion is None
    assert not any(
        call.get("skip_special_tokens") is True
        for call in client.tokenizer.decode_calls
    )
    assert not hasattr(response, "raw_message")


def test_online_client_mode_defaults():
    inference_client = OnlineVllmModelClient(
        "inference",
        model_name="VisionAgent",
        tokenizer=FakeTokenizer("<think>done</think><answer>Done.</answer>"),
    )
    data_generation_client = OnlineVllmModelClient("data_generation", model_name="VisionAgent")

    assert inference_client.prefer_token_completion is True
    assert data_generation_client.prefer_token_completion is False


def test_inference_client_fails_immediately_when_tokenizer_cannot_be_loaded(monkeypatch):
    def fail_to_load_tokenizer(self):
        raise RuntimeError("synthetic tokenizer load failure")

    monkeypatch.setattr(OnlineVllmModelClient, "_get_tokenizer", fail_to_load_tokenizer)

    with pytest.raises(RuntimeError, match="synthetic tokenizer load failure"):
        OnlineVllmModelClient("inference", model_name="VisionAgent")


def test_inference_client_rejects_missing_output_token_ids():
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="<think>done</think><answer>Done.</answer>",
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=raw_message,
                        finish_reason="stop",
                        model_extra={"stop_reason": None},
                    )
                ]
            )

    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "inference"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = True
    client.tokenizer = FakeTokenizer("<think>done</think><answer>Done.</answer>")
    client.tokenizer_path = "unused"
    client.trust_remote_code = True
    client._tokenizer_load_error = None
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client.generation_config = {"max_tokens": 2048}

    with pytest.raises(RuntimeError, match="return non-empty output token IDs"):
        client.generate_one_item(SimpleNamespace(conversation=[]))


def test_stop_reason_length_overrides_tool_call_finish_reason():
    valid_tool_completion = (
        '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
        '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
    )
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=valid_tool_completion,
                finish_reason="tool_calls",
                stop_reason="length",
                assistant_message={"role": "assistant", "content": valid_tool_completion},
                used_token_completion=True,
                token_ids=[1, 2, 3],
            )
        ]
    )
    item = make_item(inferencer)

    item = inferencer.process_single_trajectory(item)

    assert item.turn_records[0]["action_type"] == "invalid"
    assert item.turn_records[0]["turn_truncated_by_length"] is True
    assert item.turn_records[0]["finish_reason"] == "tool_calls"
    assert item.turn_records[0]["backend_stop_reason"] == "length"
    assert item.turn_records[0]["stop_reason"] == "length"
    assert item.turn_records[0]["tool_execution_success"] is False


def test_token_count_at_limit_overrides_tool_call_finish_reason():
    valid_tool_completion = (
        '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
        '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>'
    )
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=valid_tool_completion,
                finish_reason="tool_calls",
                stop_reason=None,
                assistant_message={"role": "assistant", "content": valid_tool_completion},
                used_token_completion=True,
                token_ids=[1, 2, 3],
                turn_max_tokens=3,
                completion_token_count=3,
            )
        ]
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    assert item.turn_records[0]["action_type"] == "invalid"
    assert item.turn_records[0]["turn_truncated_by_length"] is True
    assert item.turn_records[0]["finish_reason"] == "tool_calls"
    assert item.turn_records[0]["backend_stop_reason"] is None
    assert item.turn_records[0]["stop_reason"] == "length"
    assert item.turn_records[0]["turn_max_tokens"] == 3
    assert item.turn_records[0]["turn_response_length"] == 3
    assert item.turn_records[0]["tool_execution_success"] is False
    error_text = "".join(
        part.get("text", "")
        for part in item.conversation[-1]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "3-token limit" in error_text


def test_numeric_backend_stop_reason_is_preserved_but_normalized_for_environment():
    completion = "<think>done</think><answer>Done.</answer>"
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                completion=completion,
                finish_reason="stop",
                stop_reason=163586,
                assistant_message={"role": "assistant", "content": completion},
                used_token_completion=True,
                token_ids=[1, 2, 3],
            )
        ]
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    turn_record = item.turn_records[0]
    assert turn_record["action_type"] == "answer"
    assert turn_record["finish_reason"] == "stop"
    assert turn_record["backend_stop_reason"] == 163586
    assert turn_record["stop_reason"] == "stop"
    assert turn_record["turn_truncated_by_length"] is False


def test_data_generation_truncation_uses_structured_path_and_is_not_saved(tmp_path):
    raw_message = structured_tool_message(
        "PhraseToPoint",
        {"images": ["img_0"], "phrase": "person"},
        call_id="call_truncated",
        reasoning_content="locate",
    )
    serializer = TrajectorySerializer(tmp_path / "run", save_trajectory=True)
    inferencer = make_inferencer(
        [
            structured_response(
                raw_message,
                finish_reason="tool_calls",
                stop_reason="length",
                turn_max_tokens=3,
                completion_token_count=3,
            )
        ],
        mode="data_generation",
        serializer=serializer,
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))

    turn_record = item.turn_records[0]
    assert turn_record["action_type"] == "invalid"
    assert turn_record["turn_truncated_by_length"] is True
    assert turn_record["tool_call_id"] == "call_truncated"
    assert turn_record["sft_eligible"] is False
    assert item.conversation[-1]["role"] == "tool"
    assert item.conversation[-1]["tool_call_id"] == "call_truncated"
    error_text = "".join(
        part.get("text", "")
        for part in item.conversation[-1]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert "3-token limit" in error_text
    assert not (tmp_path / "run" / "run_trajectory.jsonl").exists()


def test_trajectory_efficiency_metrics_accumulate_all_model_turns():
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>',
                prompt_token_count=100,
                visual_token_count=40,
                completion_token_count=10,
            ),
            ModelTurnResponse(
                "<think>done</think><answer>Done.</answer>",
                prompt_token_count=180,
                visual_token_count=60,
                completion_token_count=12,
            ),
        ]
    )

    item = inferencer.process_single_trajectory(make_item(inferencer))
    metrics = item.efficiency_metrics

    assert metrics["model_call_count"] == 2
    assert metrics["successful_model_call_count"] == 2
    assert metrics["tool_call_count"] == 1
    assert metrics["successful_tool_call_count"] == 1
    assert metrics["cumulative_prompt_tokens"] == 280
    assert metrics["cumulative_visual_tokens"] == 100
    assert metrics["cumulative_completion_tokens"] == 22
    assert metrics["max_visual_tokens_per_call"] == 60
    assert metrics["trajectory_elapsed_seconds"] >= 0.0
    assert metrics["llm_generate_seconds"] >= 0.0
    assert metrics["tool_seconds"] >= 0.0
    assert item.turn_records[0]["prompt_token_count"] == 100
    assert item.turn_records[0]["visual_token_count"] == 40
    assert item.turn_records[0]["model_latency_seconds"] >= 0.0
    assert item.turn_records[0]["tool_latency_seconds"] >= 0.0


def test_online_client_counts_qwen_visual_prompt_tokens():
    raw_message = ChatCompletionMessage(
        role="assistant",
        content="<think>done</think><answer>Done.</answer>",
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=raw_message,
                        finish_reason="stop",
                        model_extra={
                            "token_ids": [11, 12],
                            "stop_reason": None,
                        },
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=7,
                    completion_tokens=2,
                ),
                model_extra={
                    "prompt_token_ids": [
                        1,
                        151655,
                        151655,
                        2,
                        151655,
                        3,
                        4,
                    ]
                },
            )

    client = OnlineVllmModelClient.__new__(OnlineVllmModelClient)
    client.mode = "inference"
    client.model_name = "VisionAgent"
    client.use_tools = False
    client.prefer_token_completion = True
    client.tokenizer = FakeTokenizer(
        "<think>done</think><answer>Done.</answer>"
    )
    client.tokenizer_path = "unused"
    client.trust_remote_code = True
    client._tokenizer_load_error = None
    client._image_token_id = None
    client._image_token_id_resolved = False
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    client.generation_config = {"max_tokens": 2048}

    response = client.generate_one_item(SimpleNamespace(conversation=[]))

    assert response.prompt_token_count == 7
    assert response.visual_token_count == 3
    assert response.completion_token_count == 2


def test_parallel_inference_persists_run_level_benchmark_summary(tmp_path):
    serializer = TrajectorySerializer(tmp_path / "run")
    inferencer = make_inferencer(
        [
            ModelTurnResponse(
                "<think>done</think><answer>Done.</answer>",
                prompt_token_count=100,
                visual_token_count=40,
                completion_token_count=10,
            )
        ],
        serializer=serializer,
        batch_size=1,
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": "sample-0",
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
    ]

    results = inferencer.parallel_batch_inference(dataset)

    assert len(results) == 1
    summary = json.loads(
        (tmp_path / "run" / "benchmark_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["processed_trajectories"] == 1
    assert summary["archive_previous_images"] is True
    assert summary["rollout_wall_seconds"] > 0.0
    assert (
        summary["rollout_throughput_trajectories_per_minute"] > 0.0
    )
    assert (
        summary["trajectory_metrics"]["cumulative_visual_tokens"]["mean"]
        == 40.0
    )
    assert summary["resource_metrics"]["enabled"] is False
    history_lines = (
        tmp_path / "run" / "benchmark_runs.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1


def test_thread_pool_continues_after_allowlisted_trajectory_abort(tmp_path):
    class MixedOutcomeModel(FakeModel):
        def __init__(self):
            super().__init__([])

        def generate_one_item(self, item):
            if item.meta_data["id"] == "oom-sample":
                return ModelTurnResponse(
                    '<think>locate</think><tool_call>{"name":"PhraseToPoint",'
                    '"arguments":{"images":["img_0"],"phrase":"person"}}</tool_call>',
                    used_token_completion=True,
                )
            return ModelTurnResponse(
                "<think>done</think><answer>Done.</answer>",
                used_token_completion=True,
            )

    class OOMToolCaller(FakeToolCaller):
        def call(self, tool_name, tool_parameters):
            raise ToolOOMRetriesExhaustedError(
                {
                    "status": "error",
                    "error_type": "tool_oom_retry_exhausted",
                    "message": "five consecutive OOM responses",
                    "tool_name": tool_name,
                    "oom_attempts": 5,
                    "oom_worker_history": [],
                }
            )

    serializer = TrajectorySerializer(tmp_path / "run")
    inferencer = BaseTrajectoryInferencer(
        tp_model=MixedOutcomeModel(),
        max_rounds=1,
        batch_size=2,
        serializer=serializer,
        tool_caller=OOMToolCaller(),
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": sample_id,
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
        for sample_id in ("oom-sample", "valid-sample")
    ]

    results = inferencer.parallel_batch_inference(dataset)

    results_by_id = {item.meta_data["id"]: item for item in results}
    assert set(results_by_id) == {"oom-sample", "valid-sample"}
    assert results_by_id["oom-sample"].trajectory_invalid is True
    assert results_by_id["oom-sample"].invalid_reason == (
        "tool_oom_retry_exhausted"
    )
    assert results_by_id["valid-sample"].trajectory_finished is True
    stored_results = [
        json.loads(line)
        for line in (tmp_path / "run" / "run_ckpt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(stored_results) == 2


def test_thread_pool_continues_after_vision_encoder_cache_abort(tmp_path):
    class MixedOutcomeModel(FakeModel):
        def __init__(self):
            super().__init__([])

        def generate_one_item(self, item):
            if item.meta_data["id"] == "vision-cache-sample":
                raise TrajectoryVisionEncoderCacheExceededError(
                    "vision_chunk length 4230 exceeds cache size 4225"
                )
            return ModelTurnResponse(
                "<think>done</think><answer>Done.</answer>",
                used_token_completion=True,
            )

    serializer = TrajectorySerializer(tmp_path / "run")
    inferencer = BaseTrajectoryInferencer(
        tp_model=MixedOutcomeModel(),
        max_rounds=1,
        batch_size=2,
        serializer=serializer,
        tool_caller=FakeToolCaller(),
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": sample_id,
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
        for sample_id in ("vision-cache-sample", "valid-sample")
    ]

    results = inferencer.parallel_batch_inference(dataset)

    results_by_id = {item.meta_data["id"]: item for item in results}
    assert set(results_by_id) == {"vision-cache-sample", "valid-sample"}
    failed_item = results_by_id["vision-cache-sample"]
    assert failed_item.trajectory_invalid is True
    assert failed_item.invalid_reason == "vision_encoder_cache_exceeded"
    assert results_by_id["valid-sample"].trajectory_finished is True

    stored_results = [
        json.loads(line)
        for line in (tmp_path / "run" / "run_ckpt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(stored_results) == 2
    stored_by_id = {result["meta_data"]["id"]: result for result in stored_results}
    assert stored_by_id["vision-cache-sample"]["status"] == "failed"
    assert (
        stored_by_id["vision-cache-sample"]["invalid_reason"]
        == "vision_encoder_cache_exceeded"
    )

    summary = json.loads(
        (tmp_path / "run" / "benchmark_run.json").read_text(encoding="utf-8")
    )
    assert summary["invalid_reason_counts"] == {
        "vision_encoder_cache_exceeded": 1
    }


def test_thread_pool_continues_after_model_request_timeout(tmp_path):
    class MixedOutcomeModel(FakeModel):
        def __init__(self):
            super().__init__([])

        def generate_one_item(self, item):
            if item.meta_data["id"] == "timeout-sample":
                raise TrajectoryModelRequestTimeoutError(
                    "Model API request timed out after 1000 seconds per attempt."
                )
            return ModelTurnResponse(
                "<think>done</think><answer>Done.</answer>",
                used_token_completion=True,
            )

    serializer = TrajectorySerializer(tmp_path / "run")
    inferencer = BaseTrajectoryInferencer(
        tp_model=MixedOutcomeModel(),
        max_rounds=1,
        batch_size=2,
        serializer=serializer,
        tool_caller=FakeToolCaller(),
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": sample_id,
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
        for sample_id in ("timeout-sample", "valid-sample")
    ]

    results = inferencer.parallel_batch_inference(dataset)

    results_by_id = {item.meta_data["id"]: item for item in results}
    assert set(results_by_id) == {"timeout-sample", "valid-sample"}
    failed_item = results_by_id["timeout-sample"]
    assert failed_item.trajectory_invalid is True
    assert failed_item.invalid_reason == "model_request_timeout"
    assert results_by_id["valid-sample"].trajectory_finished is True

    stored_results = [
        json.loads(line)
        for line in (tmp_path / "run" / "run_ckpt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(stored_results) == 2
    stored_by_id = {result["meta_data"]["id"]: result for result in stored_results}
    assert stored_by_id["timeout-sample"]["status"] == "failed"
    assert (
        stored_by_id["timeout-sample"]["invalid_reason"]
        == "model_request_timeout"
    )

    summary = json.loads(
        (tmp_path / "run" / "benchmark_run.json").read_text(encoding="utf-8")
    )
    assert summary["invalid_reason_counts"] == {"model_request_timeout": 1}


def test_thread_pool_propagates_non_allowlisted_trajectory_error(tmp_path):
    class FatalModel(FakeModel):
        def __init__(self):
            super().__init__([])

        def generate_one_item(self, item):
            raise RuntimeError("fatal model service failure")

    inferencer = BaseTrajectoryInferencer(
        tp_model=FatalModel(),
        max_rounds=1,
        batch_size=2,
        serializer=TrajectorySerializer(tmp_path / "run"),
        tool_caller=FakeToolCaller(),
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": "sample-0",
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
    ]

    with pytest.raises(RuntimeError, match="fatal model service failure"):
        inferencer.parallel_batch_inference(dataset)


def test_thread_pool_propagates_checkpoint_persistence_error(tmp_path):
    class FailingCheckpointSerializer(TrajectorySerializer):
        def store_result(self, result):
            raise TrajectoryPersistenceError("checkpoint unavailable")

    inferencer = make_inferencer(
        [ModelTurnResponse("<think>done</think><answer>Done.</answer>")],
        serializer=FailingCheckpointSerializer(tmp_path / "run"),
        batch_size=1,
        benchmark_config={"enabled": False},
    )
    dataset = [
        {
            "id": "sample-0",
            "question": "Find the person.",
            "image": Image.new("RGB", (8, 8), "white"),
        }
    ]

    with pytest.raises(
        TrajectoryPersistenceError,
        match="checkpoint unavailable",
    ):
        inferencer.parallel_batch_inference(dataset)


def test_sft_persistence_error_is_not_converted_to_failed_item(tmp_path):
    class FailingSFTSerializer(TrajectorySerializer):
        def save_item_trajectory(self, *args, **kwargs):
            raise TrajectoryPersistenceError("SFT storage unavailable")

    serializer = FailingSFTSerializer(
        tmp_path / "run",
        save_trajectory=True,
    )
    inferencer = make_inferencer(
        [structured_response(submit_final_answer_message({"final_answer": "Done."}))],
        mode="data_generation",
        serializer=serializer,
    )

    with pytest.raises(
        TrajectoryPersistenceError,
        match="SFT storage unavailable",
    ):
        inferencer.process_single_trajectory(make_item(inferencer))


def test_vllm_cache_config_parsing_and_qwen_kv_capacity(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "dtype": "bfloat16",
                "text_config": {
                    "dtype": "bfloat16",
                    "hidden_size": 4096,
                    "head_dim": 128,
                    "num_attention_heads": 32,
                    "num_hidden_layers": 36,
                    "num_key_value_heads": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics_text = (
        '# HELP vllm:cache_config_info cache config\n'
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
        'kv_cache_memory_bytes="None",num_gpu_blocks="50196"} 1.0\n'
    )

    cache_config = _parse_prometheus_metric_labels(
        metrics_text,
        "vllm:cache_config_info",
    )
    capacity = _calculate_kv_cache_capacity(
        cache_config,
        model_config_path=model_dir,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    assert capacity["num_gpu_blocks"] == 50196
    assert capacity["block_size_tokens"] == 16
    assert capacity["capacity_tokens"] == 803136
    assert capacity["kv_heads_per_gpu"] == 4
    assert capacity["bytes_per_token_per_gpu"] == 73728
    assert capacity["pool_gib_per_gpu"] == pytest.approx(
        55.14697265625
    )
