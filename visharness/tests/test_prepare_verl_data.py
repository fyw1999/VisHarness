from __future__ import annotations

import sys

from visharness.data.prepare_verl_data import build_prompt, parse_args
from visharness.prompts import TRAIN_TEST_SYSTEM_PROMPT


def _required_cli_args() -> list[str]:
    return [
        "--source-root",
        "/datasets/visionagent-4k",
        "--rec8k-anno-path",
        "/datasets/rec8k/annotations.json",
        "--gres-data-root",
        "/datasets/gres",
        "--reasonseg-data-root",
        "/datasets/reasonseg/train",
    ]


def test_build_prompt_omits_system_prompt_by_default():
    assert build_prompt("question") == [
        {"role": "user", "content": "question"},
    ]


def test_build_prompt_can_embed_current_system_prompt():
    assert build_prompt("question", embed_system_prompt=True) == [
        {"role": "system", "content": TRAIN_TEST_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": "question"},
    ]


def test_prepare_verl_data_cli_does_not_embed_prompt_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_verl_data.py", *_required_cli_args()])

    assert parse_args().embed_system_prompt is False


def test_prepare_verl_data_cli_can_embed_prompt(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_verl_data.py", *_required_cli_args(), "--embed-system-prompt"],
    )

    assert parse_args().embed_system_prompt is True
