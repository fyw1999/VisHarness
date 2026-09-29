from __future__ import annotations

import pytest

from verl.utils.dataset.rl_dataset import RLHFDataset

from visharness.data.visharness_dataset import VisHarnessDataset, replace_system_prompt
from visharness.prompts import TRAIN_TEST_SYSTEM_PROMPT


class _MappedDataframe:
    def __init__(self, rows):
        self.rows = rows

    def map(self, function, *, fn_kwargs, desc):
        assert desc == "Injecting current VisHarness system prompt"
        return _MappedDataframe(
            [
                {
                    **row,
                    **function(row, **fn_kwargs),
                }
                for row in self.rows
            ]
        )


def test_replace_system_prompt_replaces_existing_message_without_mutating_input():
    messages = [
        {"role": "system", "content": "stale", "metadata": {"source": "parquet"}},
        {"role": "user", "content": "question"},
    ]

    updated = replace_system_prompt(messages, "current")

    assert messages[0]["content"] == "stale"
    assert updated == [
        {"role": "system", "content": "current", "metadata": {"source": "parquet"}},
        {"role": "user", "content": "question"},
    ]


def test_replace_system_prompt_prepends_message_when_missing():
    updated = replace_system_prompt(
        [{"role": "user", "content": "question"}],
        "current",
    )

    assert updated == [
        {"role": "system", "content": "current"},
        {"role": "user", "content": "question"},
    ]


def test_replace_system_prompt_rejects_multiple_system_messages():
    with pytest.raises(ValueError, match="found 2"):
        replace_system_prompt(
            [
                {"role": "system", "content": "one"},
                {"role": "system", "content": "two"},
                {"role": "user", "content": "question"},
            ],
            "current",
        )


def test_dataset_injects_current_prompt_before_parent_length_filter(monkeypatch):
    captured = {}

    def fake_parent_filter(self, dataframe):
        captured["messages"] = dataframe.rows[0]["prompt"]
        return dataframe

    monkeypatch.setattr(RLHFDataset, "maybe_filter_out_long_prompts", fake_parent_filter)
    dataset = VisHarnessDataset.__new__(VisHarnessDataset)
    dataset.config = {"dynamic_system_prompt": True}
    dataset.prompt_key = "prompt"
    dataframe = _MappedDataframe(
        [
            {
                "prompt": [
                    {"role": "system", "content": "stale"},
                    {"role": "user", "content": "question"},
                ]
            }
        ]
    )

    dataset.maybe_filter_out_long_prompts(dataframe)

    assert captured["messages"][0]["content"] == TRAIN_TEST_SYSTEM_PROMPT.strip()
    assert dataset.system_prompt_sha256 is not None


def test_dataset_can_disable_dynamic_system_prompt(monkeypatch):
    captured = {}

    def fake_parent_filter(self, dataframe):
        captured["messages"] = dataframe.rows[0]["prompt"]
        return dataframe

    monkeypatch.setattr(RLHFDataset, "maybe_filter_out_long_prompts", fake_parent_filter)
    dataset = VisHarnessDataset.__new__(VisHarnessDataset)
    dataset.config = {"dynamic_system_prompt": False}
    dataset.prompt_key = "prompt"
    dataframe = _MappedDataframe(
        [{"prompt": [{"role": "system", "content": "stored"}]}]
    )

    dataset.maybe_filter_out_long_prompts(dataframe)

    assert captured["messages"][0]["content"] == "stored"
    assert dataset.system_prompt_sha256 is None
