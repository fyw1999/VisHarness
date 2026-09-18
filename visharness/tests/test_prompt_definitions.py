"""Tests for the shared VisHarness prompt and tool-definition source."""

from __future__ import annotations

from visharness import prompts
from visharness.trajectory_runner import legacy_prompts


EXPECTED_TOOL_NAMES = {
    "PhraseToBoxMask",
    "PhraseToPoint",
    "PointToBoxMask",
    "SplitImageIntoPatches",
    "SuperResolution",
    "MergeBoxMask",
}


def test_canonical_tool_definitions_are_complete_and_unique():
    tool_names = [tool["function"]["name"] for tool in prompts.TOOLS_LIST]

    assert set(tool_names) == EXPECTED_TOOL_NAMES
    assert len(tool_names) == len(set(tool_names))
    assert prompts.SubmitFinalAnswer["function"]["name"] == "SubmitFinalAnswer"
    assert prompts.VISION_RESULT_TOOLS == {
        "PhraseToPoint",
        "PhraseToBoxMask",
        "PointToBoxMask",
    }


def test_trajectory_runner_compatibility_module_reexports_canonical_objects():
    assert legacy_prompts.TRAIN_TEST_SYSTEM_PROMPT is prompts.TRAIN_TEST_SYSTEM_PROMPT
    assert legacy_prompts.DATA_GENERATION_SYSTEM_PROMPT is prompts.DATA_GENERATION_SYSTEM_PROMPT
    assert legacy_prompts.PHRASE_TO_BOXMASK_PROMPT is prompts.PHRASE_TO_BOXMASK_PROMPT
    assert legacy_prompts.PHRASE_TO_POINT_PROMPT is prompts.PHRASE_TO_POINT_PROMPT
    assert legacy_prompts.POINT_TO_BOXMASK_PROMPT is prompts.POINT_TO_BOXMASK_PROMPT
    assert legacy_prompts.SPLIT_PROMPT is prompts.SPLIT_PROMPT
    assert legacy_prompts.SR_PROMPT is prompts.SR_PROMPT
    assert legacy_prompts.TOOLS_LIST is prompts.TOOLS_LIST
    assert legacy_prompts.SubmitFinalAnswer is prompts.SubmitFinalAnswer
