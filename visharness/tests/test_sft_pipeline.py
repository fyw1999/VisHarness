import json
from pathlib import Path

import pytest
from PIL import Image

from visharness.data.sft_pipeline.__main__ import main as pipeline_main
from visharness.data.sft_pipeline.filter import filter_generated_trajectories
from visharness.data.sft_pipeline.merge import SFTSource, merge_sft_datasets
from visharness.data.sft_pipeline.pipeline import (
    FilteredSFTSource,
    build_sft_dataset,
)
from visharness.data.sft_pipeline.postprocess import postprocess_snapshot
from visharness.data.sft_pipeline.scoring import TrajectoryAcceptanceScorer
from visharness.data.sft_pipeline.swift import (
    Qwen3VLVisualBudget,
    convert_snapshot_to_swift,
    convert_to_swift_format,
)
from visharness.evaluate.common import CompactPrediction
from visharness.prompts import (
    PHRASE_TO_BOXMASK_PROMPT,
    PHRASE_TO_POINT_PROMPT,
    POINT_TO_BOXMASK_PROMPT,
    SPLIT_PROMPT,
    SR_PROMPT,
    VISION_TOOL_PROMPT,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def answer_snapshot(
    snapshot_id: str,
    *,
    trajectory_id: str,
    answer: str = "done",
    image_path: str | None = None,
) -> dict:
    user_content = [{"type": "text", "text": "question"}]
    images: list[str] = []
    if image_path is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_path}})
        images.append(image_path)
    return {
        "schema_version": 2,
        "id": snapshot_id,
        "trajectory_id": trajectory_id,
        "target_turn_index": 1,
        "target_message_index": 2,
        "target_action_type": "answer",
        "images": images,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"<think>reason</think><answer>{answer}</answer>",
                    }
                ],
            },
        ],
    }


def test_postprocess_is_strict_only_for_target_and_removes_submit_tool():
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_2",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_message_index"] = 4
    snapshot["target_turn_index"] = 2
    snapshot["messages"].insert(
        2,
        {
            "role": "assistant",
            "content": "bad historical output mentioning SubmitFinalAnswer",
        },
    )
    snapshot["messages"].insert(
        3,
        {
            "role": "tool",
            "content": "error feedback\nContinue the reasoning process to answer the original question: question",
        },
    )

    processed = postprocess_snapshot(snapshot)

    assert processed["target_message_index"] == 4
    assert processed["messages"][2]["content"] == [
        {
            "type": "text",
            "text": "bad historical output mentioning provide the final answer",
        }
    ]
    assert processed["messages"][3]["content"] == [
        {"type": "text", "text": "error feedback\n"}
    ]
    assert "SubmitFinalAnswer" not in json.dumps(processed)


@pytest.mark.parametrize(
    "tool_guidance",
    [
        PHRASE_TO_BOXMASK_PROMPT,
        PHRASE_TO_POINT_PROMPT,
        POINT_TO_BOXMASK_PROMPT,
    ],
)
def test_postprocess_preserves_vision_guidance_and_removes_specific_guidance(
    tool_guidance,
):
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_2",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_message_index"] = 3
    snapshot["messages"].insert(
        2,
        {
            "role": "tool",
            "content": (
                f"tool result\n{VISION_TOOL_PROMPT}{tool_guidance}"
                "Continue the reasoning process to answer the original question: question"
            ),
        },
    )

    processed = postprocess_snapshot(snapshot)

    assert processed["messages"][2]["content"] == [
        {
            "type": "text",
            "text": "tool result\n" + VISION_TOOL_PROMPT.rstrip("\r\n"),
        }
    ]


@pytest.mark.parametrize(
    "generation_guidance",
    [SPLIT_PROMPT, SR_PROMPT, ""],
)
def test_postprocess_removes_other_generation_guidance(
    generation_guidance,
):
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_2",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_message_index"] = 3
    snapshot["messages"].insert(
        2,
        {
            "role": "tool",
            "content": (
                f"tool result\n{generation_guidance}"
                "Continue the reasoning process to answer the original question: question"
            ),
        },
    )

    processed = postprocess_snapshot(snapshot)

    assert processed["messages"][2]["content"] == [
        {"type": "text", "text": "tool result\n"}
    ]
    assert VISION_TOOL_PROMPT not in json.dumps(
        processed["messages"][2],
        ensure_ascii=False,
    )


def test_postprocess_does_not_add_vision_prompt_to_error_feedback():
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_2",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_message_index"] = 4
    snapshot["target_turn_index"] = 2
    snapshot["messages"].insert(
        2,
        {"role": "assistant", "content": "invalid historical output"},
    )
    snapshot["messages"].insert(
        3,
        {"role": "tool", "content": "Incorrect tool call parameters."},
    )

    processed = postprocess_snapshot(snapshot)

    assert processed["messages"][3]["content"] == [
        {"type": "text", "text": "Incorrect tool call parameters."}
    ]
    assert VISION_TOOL_PROMPT not in json.dumps(
        processed["messages"][3],
        ensure_ascii=False,
    )


def test_postprocess_adds_newline_before_structured_tool_call():
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_action_type"] = "tool_call"
    snapshot["messages"][-1] = {
        "role": "assistant",
        "content": "<think>locate it</think>",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "PhraseToBoxMask",
                    "arguments": {"images": ["img_0"], "phrase": "cat"},
                },
            }
        ],
    }

    processed = postprocess_snapshot(snapshot)

    assert processed["messages"][-1]["content"] == [
        {"type": "text", "text": "<think>locate it</think>\n"}
    ]


def test_filter_keeps_latest_snapshot_for_successful_empty_rec8k(tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"demo.jpg": {"target": {"points": []}}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "run_ckpt.jsonl"
    write_jsonl(
        checkpoint,
        [
            {
                "meta_data": {"id": "REC8K-demo-target"},
                "status": "finished",
                "trajectory_finished": True,
                "termination_reason": "answer",
                "answer": "none",
                "current_round": 2,
                "max_rounds": 10,
            },
            {
                "meta_data": {"id": "REC8K-failed-target"},
                "status": "failed",
                "termination_reason": "failed",
            },
        ],
    )
    trajectory = tmp_path / "run_trajectory.jsonl"
    write_jsonl(
        trajectory,
        [
            answer_snapshot(
                "REC8K-demo-target_step_2",
                trajectory_id="REC8K-demo-target",
                answer="old",
            ),
            answer_snapshot(
                "REC8K-demo-target_step_2",
                trajectory_id="REC8K-demo-target",
                answer="latest",
            ),
        ],
    )

    report = filter_generated_trajectories(
        checkpoint_paths=checkpoint,
        trajectory_paths=trajectory,
        output_dir=tmp_path / "filtered",
        rec8k_annotations_path=annotations,
    )

    assert report["accepted_trajectories"] == 1
    assert report["accepted_checkpoint_records"] == 1
    assert report["rejected_checkpoint_records"] == 1
    assert report["thresholds"] == {
        "gres_iou": 0.7,
        "reasonseg_iou": 0.7,
        "rec8k_relative_count_error": 0.3,
        "aspect_ratio_tolerance": 0.05,
    }
    assert report["datasets"]["rec8k_annotations_path"] == str(annotations)
    accepted_checkpoint = Path(report["accepted_checkpoint_path"])
    assert accepted_checkpoint.name == "accepted_checkpoints.jsonl"
    assert Path(report["rejected_checkpoint_path"]).name == "rejected_checkpoints.jsonl"
    assert Path(report["accepted_sft_snapshots_path"]).name == (
        "accepted_sft_snapshots.jsonl"
    )
    assert json.loads(accepted_checkpoint.read_text())["meta_data"]["id"] == (
        "REC8K-demo-target"
    )
    output_records = [
        json.loads(line)
        for line in Path(report["accepted_sft_snapshots_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(output_records) == 1
    assert "latest" in output_records[0]["messages"][-1]["content"][0]["text"]
    accepted_ids = (tmp_path / "filtered" / "accepted_ids.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(accepted_ids)["id"] == "REC8K-demo-target"


def test_filter_requires_answer_but_allows_answer_on_last_turn(tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"demo.jpg": {"target": {"points": [[0, 0]]}}}),
        encoding="utf-8",
    )
    scorer = TrajectoryAcceptanceScorer(
        rec8k_annotations_path=annotations,
    )
    visual_result = {
        "final_bboxes": [[0.0, 0.0, 1.0, 1.0]],
        "final_masks": [{"size": [1, 1], "counts": "1"}],
        "count": 1,
    }

    exhausted = CompactPrediction(
        item_id="REC8K-demo-target",
        final_bboxes=visual_result["final_bboxes"],
        final_masks=visual_result["final_masks"],
        count=visual_result["count"],
        status="finished",
        termination_reason="max_rounds_reached",
        trajectory_finished=False,
        max_rounds_reached=True,
        current_round=12,
        max_rounds=12,
    )
    exhausted_decision = scorer.score(exhausted)

    assert exhausted_decision.accepted is False
    assert exhausted_decision.reason == "trajectory_max_rounds_reached"
    assert exhausted_decision.metric is None

    answered_on_last_turn = CompactPrediction(
        item_id="REC8K-demo-target",
        final_bboxes=visual_result["final_bboxes"],
        final_masks=visual_result["final_masks"],
        count=visual_result["count"],
        status="finished",
        termination_reason="answer",
        trajectory_finished=True,
        max_rounds_reached=False,
        current_round=12,
        max_rounds=12,
    )
    answered_decision = scorer.score(answered_on_last_turn)

    assert answered_decision.accepted is True
    assert answered_decision.reason == "passed"
    assert answered_decision.metric == "relative_count_error"
    assert answered_decision.value == 0.0


def test_end_to_end_pipeline_consumes_explicit_filter_result(tmp_path):
    source_root = tmp_path / "generated"
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"demo.jpg": {"target": {"points": []}}}),
        encoding="utf-8",
    )
    checkpoint = source_root / "run_ckpt.jsonl"
    write_jsonl(
        checkpoint,
        [
            {
                "meta_data": {"id": "REC8K-demo-target"},
                "status": "finished",
                "trajectory_finished": True,
                "termination_reason": "answer",
                "answer": "none",
            }
        ],
    )
    relative_image = "images/REC8K-demo-target_step_1/img_0.jpg"
    image_path = source_root / relative_image
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), "white").save(image_path)
    trajectory = source_root / "run_trajectory.jsonl"
    write_jsonl(
        trajectory,
        [
            answer_snapshot(
                "REC8K-demo-target_step_1",
                trajectory_id="REC8K-demo-target",
                image_path=relative_image,
            )
        ],
    )

    filter_report = filter_generated_trajectories(
        checkpoint_paths=checkpoint,
        trajectory_paths=trajectory,
        output_dir=source_root / "filtered",
        rec8k_annotations_path=annotations,
    )
    filtered_sft_path = Path(filter_report["accepted_sft_snapshots_path"])

    # Build must consume the explicit filter result, not silently score or
    # rescan the raw runner outputs a second time.
    checkpoint.unlink()
    trajectory.unlink()

    output_dir = tmp_path / "training"
    report = build_sft_dataset(
        [
            FilteredSFTSource(
                alias="model-a",
                filtered_sft_path=filtered_sft_path,
                image_root=source_root,
            )
        ],
        output_dir,
    )

    assert report["schema_version"] == 2
    assert report["sources"]["model-a"]["filtered_sft_path"] == str(filtered_sft_path)
    assert "filter" not in report["sources"]["model-a"]
    assert report["merge"]["merged_snapshots"] == 1
    assert report["swift"]["converted_snapshots"] == 1
    assert report["swift"]["rejected_visual_budget_snapshots"] == 0
    assert report["swift"]["visual_budget"]["image_max_token_num"] == 2048
    assert report["swift"]["visual_budget"]["max_total_raw_image_patches"] == 56_000
    assert (
        output_dir / "postprocessed" / "model-a" / "sft_postprocessed.jsonl"
    ).is_file()
    assert not (output_dir / "sources").exists()
    assert (output_dir / "merged_sft_data.jsonl").is_file()
    assert (output_dir / "merged_sft_data_swift.jsonl").is_file()
    assert (output_dir / "visual_budget_decisions.jsonl").is_file()
    assert (output_dir / "images" / "REC8K-demo-target_step_1" / "img_0.jpg").is_file()


def test_build_cli_uses_explicit_filtered_sources(tmp_path, monkeypatch):
    from visharness.data.sft_pipeline import pipeline

    captured = {}

    def fake_build(sources, output_dir, **kwargs):
        captured["sources"] = sources
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(pipeline, "build_sft_dataset", fake_build)

    assert (
        pipeline_main(
            [
                "build",
                "--source",
                "kimi=generated/accepted_sft_snapshots.jsonl",
                "--image-root",
                "kimi=generated",
                "--output-dir",
                "training",
                "--merged-name",
                "merged.jsonl",
                "--swift-name",
                "swift.jsonl",
            ]
        )
        == 0
    )
    source = captured["sources"][0]
    assert source.alias == "kimi"
    assert source.filtered_sft_path == "generated/accepted_sft_snapshots.jsonl"
    assert source.image_root == "generated"
    assert captured["output_dir"] == "training"
    assert captured["kwargs"] == {
        "merged_filename": "merged.jsonl",
        "swift_filename": "swift.jsonl",
        "image_max_token_num": 2048,
        "max_total_raw_image_patches": 56_000,
    }


def test_build_cli_requires_matching_image_roots():
    with pytest.raises(ValueError, match="required for source aliases"):
        pipeline_main(
            [
                "build",
                "--source",
                "kimi=kimi.jsonl",
                "--source",
                "qwen=qwen.jsonl",
                "--image-root",
                "kimi=generated/kimi",
                "--output-dir",
                "training",
            ]
        )


def test_merge_preserves_disjoint_ids_and_images_and_swift_masks_target(tmp_path):
    sources: list[SFTSource] = []
    for alias, color in (("kimi", "red"), ("qwen", "blue")):
        source_root = tmp_path / alias
        trajectory_id = f"REC8K-{alias}-target"
        snapshot_id = f"{trajectory_id}_step_1"
        relative_image = f"images/{snapshot_id}/image.jpg"
        image_path = source_root / relative_image
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 3), color).save(image_path)
        snapshot = answer_snapshot(
            snapshot_id,
            trajectory_id=trajectory_id,
            answer=alias,
            image_path=relative_image,
        )
        source_jsonl = source_root / "processed.jsonl"
        write_jsonl(source_jsonl, [snapshot])
        sources.append(SFTSource(alias=alias, path=source_jsonl))

    output_dir = tmp_path / "merged"
    report = merge_sft_datasets(sources, output_dir)

    assert report["merged_snapshots"] == 2
    merged_path = output_dir / "merged_sft_data.jsonl"
    merged = [json.loads(line) for line in merged_path.read_text().splitlines()]
    assert [item["id"] for item in merged] == [
        "REC8K-kimi-target_step_1",
        "REC8K-qwen-target_step_1",
    ]
    assert merged[0]["trajectory_id"] == "REC8K-kimi-target"
    assert merged[1]["trajectory_id"] == "REC8K-qwen-target"
    assert merged[0]["images"] == ["images/REC8K-kimi-target_step_1/image.jpg"]
    assert merged[1]["images"] == ["images/REC8K-qwen-target_step_1/image.jpg"]
    assert (output_dir / merged[0]["images"][0]).is_file()
    assert (output_dir / merged[1]["images"][0]).is_file()

    swift_path = output_dir / "swift.jsonl"
    swift_report = convert_to_swift_format(
        merged_path,
        swift_path,
        image_root_dir=output_dir,
        tools_definition=[],
    )
    assert swift_report["converted_snapshots"] == 2
    swift_records = [json.loads(line) for line in swift_path.read_text().splitlines()]
    for item in swift_records:
        assert sum(message["loss"] for message in item["messages"]) == 1
        assert item["messages"][-1]["role"] == "assistant"
        assert item["messages"][-1]["loss"] is True
        assert Path(item["images"][0]).is_absolute()


def test_swift_filters_aggregate_qwen3vl_raw_image_patch_budget(tmp_path):
    relative_image = "images/demo/image.png"
    image_path = tmp_path / relative_image
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(image_path)

    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
        image_path=relative_image,
    )
    # Repeating an image path still incurs another image encoding and must not
    # be deduplicated by the visual-budget calculation.
    snapshot["images"].append(relative_image)
    snapshot["messages"][1]["content"].append(
        {"type": "image_url", "image_url": {"url": relative_image}}
    )
    input_path = tmp_path / "merged.jsonl"
    output_path = tmp_path / "swift.jsonl"
    decisions_path = tmp_path / "visual_budget_decisions.jsonl"
    write_jsonl(input_path, [snapshot])

    report = convert_to_swift_format(
        input_path,
        output_path,
        image_root_dir=tmp_path,
        tools_definition=[],
        visual_budget=Qwen3VLVisualBudget(
            image_max_token_num=4,
            max_total_raw_image_patches=31,
        ),
        visual_budget_decisions_file=decisions_path,
    )

    assert output_path.read_text(encoding="utf-8") == ""
    assert report["input_snapshots"] == 1
    assert report["converted_snapshots"] == 0
    assert report["rejected_visual_budget_snapshots"] == 1
    decision = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert decision == {
        "snapshot_id": "REC8K-demo-target_step_1",
        "input_line_number": 1,
        "image_count": 2,
        "total_raw_image_patches": 32,
        "total_merged_image_tokens": 8,
        "max_total_raw_image_patches": 31,
        "accepted": False,
        "reason": "total_raw_image_patches_exceeds_limit",
    }


def test_swift_visual_budget_accepts_limit_and_matches_qwen3vl_resize(tmp_path):
    relative_image = "images/demo/image.png"
    image_path = tmp_path / relative_image
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2048, 1024), "white").save(image_path)
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
        image_path=relative_image,
    )
    input_path = tmp_path / "merged.jsonl"
    output_path = tmp_path / "swift.jsonl"
    decisions_path = tmp_path / "visual_budget_decisions.jsonl"
    write_jsonl(input_path, [snapshot])

    report = convert_to_swift_format(
        input_path,
        output_path,
        image_root_dir=tmp_path,
        tools_definition=[],
        visual_budget=Qwen3VLVisualBudget(
            image_max_token_num=2048,
            max_total_raw_image_patches=8192,
        ),
        visual_budget_decisions_file=decisions_path,
    )

    assert report["converted_snapshots"] == 1
    decision = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert decision["total_raw_image_patches"] == 8192
    assert decision["total_merged_image_tokens"] == 2048
    assert decision["accepted"] is True


def test_merge_rejects_trajectory_overlap_between_filtered_sources(tmp_path):
    sources: list[SFTSource] = []
    for alias in ("kimi", "qwen"):
        source_root = tmp_path / alias
        source_jsonl = source_root / "processed.jsonl"
        write_jsonl(
            source_jsonl,
            [
                answer_snapshot(
                    f"REC8K-demo-target_step_{1 if alias == 'kimi' else 2}",
                    trajectory_id="REC8K-demo-target",
                )
            ],
        )
        sources.append(SFTSource(alias=alias, path=source_jsonl))

    with pytest.raises(ValueError, match="must contain disjoint trajectories"):
        merge_sft_datasets(sources, tmp_path / "merged")


def test_swift_tool_target_marks_reasoning_and_call_for_loss(tmp_path):
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_action_type"] = "tool_call"
    snapshot["messages"][-1] = {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>locate it</think>\n"}],
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "PhraseToBoxMask",
                    "arguments": {"images": ["img_0"], "phrase": "cat"},
                },
            }
        ],
    }

    converted = convert_snapshot_to_swift(
        snapshot,
        image_root_dir=tmp_path,
        tools_definition=[],
    )

    assert [message["role"] for message in converted["messages"][-2:]] == [
        "assistant",
        "tool_call",
    ]
    assert [message["loss"] for message in converted["messages"][-2:]] == [True, True]


def test_swift_rejects_missing_newline_before_tool_call(tmp_path):
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_action_type"] = "tool_call"
    snapshot["messages"][-1] = {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>locate it</think>"}],
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "PhraseToBoxMask",
                    "arguments": {"images": ["img_0"], "phrase": "cat"},
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="must end with a newline"):
        convert_snapshot_to_swift(
            snapshot,
            image_root_dir=tmp_path,
            tools_definition=[],
        )


def test_swift_rejects_non_normalized_tool_call_type(tmp_path):
    snapshot = answer_snapshot(
        "REC8K-demo-target_step_1",
        trajectory_id="REC8K-demo-target",
    )
    snapshot["target_action_type"] = "tool_call"
    snapshot["messages"][-1]["content"] = [
        {"type": "text", "text": "<think>reason</think>\n"}
    ]
    snapshot["messages"][-1]["tool_calls"] = [
        {
            "type": "provider_specific",
            "function": {"name": "PhraseToBoxMask", "arguments": {}},
        }
    ]

    with pytest.raises(ValueError, match="non-function tool call"):
        convert_snapshot_to_swift(
            snapshot,
            image_root_dir=tmp_path,
            tools_definition=[],
        )
