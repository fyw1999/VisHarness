from pathlib import Path

import pytest

from visharness.rl_trainer.checkpoint_retention import (
    discover_resume_checkpoints,
    prune_resume_checkpoints,
)


def _make_resume_checkpoint(root: Path, step: int, role: str = "actor") -> Path:
    step_directory = root / f"global_step_{step}"
    role_directory = step_directory / role
    role_directory.mkdir(parents=True)
    (role_directory / "checkpoint.pt").write_text(str(step))
    (step_directory / "data.pt").write_text("data")
    (step_directory / "visharness_resume_state.pt").write_text("state")
    return role_directory


def test_pruning_after_resume_keeps_latest_three_role_checkpoints(tmp_path):
    role_paths = {
        step: _make_resume_checkpoint(tmp_path, step)
        for step in (10, 20, 30, 40)
    }
    archive = tmp_path / "archived" / "global_step_10"
    archive.mkdir(parents=True)
    (archive / "model.safetensors").write_text("archive")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("40")

    removed = prune_resume_checkpoints(
        str(tmp_path),
        role="actor",
        max_to_keep=3,
    )

    assert removed == [str(role_paths[10])]
    assert not role_paths[10].exists()
    assert (tmp_path / "global_step_10" / "data.pt").exists()
    assert all(role_paths[step].is_dir() for step in (20, 30, 40))
    assert (archive / "model.safetensors").exists()


def test_discovery_ignores_unpublished_and_incomplete_checkpoints(tmp_path):
    _make_resume_checkpoint(tmp_path, 10)
    _make_resume_checkpoint(tmp_path, 20)
    unpublished = _make_resume_checkpoint(tmp_path, 30)
    incomplete = tmp_path / "global_step_15" / "actor"
    incomplete.mkdir(parents=True)
    (incomplete / "checkpoint.pt").write_text("partial")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")

    checkpoints = discover_resume_checkpoints(str(tmp_path), role="actor")

    assert [checkpoint.step for checkpoint in checkpoints] == [10, 20]
    assert unpublished.is_dir()
    assert incomplete.is_dir()


def test_pruning_is_disabled_for_none_zero_or_negative_limit(tmp_path):
    role_path = _make_resume_checkpoint(tmp_path, 10)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")

    for limit in (None, 0, -1):
        assert (
            prune_resume_checkpoints(
                str(tmp_path),
                role="actor",
                max_to_keep=limit,
            )
            == []
        )
        assert role_path.is_dir()


def test_invalid_published_step_fails_loudly(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("not-a-step")

    with pytest.raises(ValueError, match="Invalid checkpoint tracker value"):
        discover_resume_checkpoints(str(tmp_path), role="actor")
