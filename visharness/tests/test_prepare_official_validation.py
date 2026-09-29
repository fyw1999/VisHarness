import random
import sys

from visharness.data.prepare_official_validation import (
    _largest_remainder_quotas,
    _legacy_rec_frame_filter,
    _sample_strata,
    parse_args,
)


def test_largest_remainder_quotas_are_exact_and_deterministic():
    assert _largest_remainder_quotas({"a": 1, "b": 1, "c": 1}, 5) == {
        "a": 1,
        "b": 2,
        "c": 2,
    }


def test_rec_frame_filter_matches_consecutive_phrase_set_rule():
    grouped = {
        "0001.jpg": ["person"],
        "0002.jpg": ["person"],
        "0003.jpg": ["car"],
        "0004.jpg": ["car"],
        "0005.jpg": ["person"],
    }
    assert _legacy_rec_frame_filter(grouped) == [
        "0001.jpg",
        "0003.jpg",
        "0005.jpg",
    ]


def test_stratified_sampler_preserves_unique_images_and_zero_quota():
    candidates = [
        {"stratum": "a", "image_key": "image-1"},
        {"stratum": "a", "image_key": "image-2"},
        {"stratum": "b", "image_key": "image-3"},
    ]
    selected = _sample_strata(
        candidates,
        {"a": 2, "b": 0},
        rng=random.Random(42),
    )
    assert len(selected) == 2
    assert {item["image_key"] for item in selected} == {"image-1", "image-2"}


def test_official_validation_cli_does_not_embed_prompt_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_official_validation.py"])

    assert parse_args().embed_system_prompt is False


def test_official_validation_cli_can_embed_prompt(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_official_validation.py", "--embed-system-prompt"],
    )

    assert parse_args().embed_system_prompt is True
