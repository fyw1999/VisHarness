"""VisHarness-owned checkpoint retention across trainer restarts."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass


_STEP_DIRECTORY_PATTERN = re.compile(r"global_step_(\d+)")


@dataclass(frozen=True)
class ResumeCheckpoint:
    """One published checkpoint role directory that can be resumed exactly."""

    step: int
    step_directory: str
    role_directory: str


def _read_published_step(checkpoint_root: str) -> int | None:
    tracker_path = os.path.join(
        checkpoint_root,
        "latest_checkpointed_iteration.txt",
    )
    if not os.path.isfile(tracker_path):
        return None
    with open(tracker_path, "rb") as tracker_file:
        tracker_value = tracker_file.read().decode().strip()
    try:
        return int(tracker_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid checkpoint tracker value at {tracker_path}: "
            f"{tracker_value!r}"
        ) from error


def discover_resume_checkpoints(
    checkpoint_root: str,
    *,
    role: str,
) -> list[ResumeCheckpoint]:
    """Find published VisHarness checkpoints with all driver-side resume state."""

    checkpoint_root = os.path.abspath(os.path.expanduser(checkpoint_root))
    published_step = _read_published_step(checkpoint_root)
    if published_step is None or not os.path.isdir(checkpoint_root):
        return []

    checkpoints: list[ResumeCheckpoint] = []
    for entry in os.scandir(checkpoint_root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        match = _STEP_DIRECTORY_PATTERN.fullmatch(entry.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > published_step:
            # A newer directory can be an interrupted or concurrent save. It
            # is not published and must never displace a resumable checkpoint.
            continue

        role_directory = os.path.join(entry.path, role)
        if not os.path.isdir(role_directory):
            continue
        if not os.path.isfile(os.path.join(entry.path, "data.pt")):
            continue
        if not os.path.isfile(
            os.path.join(entry.path, "visharness_resume_state.pt")
        ):
            continue
        checkpoints.append(
            ResumeCheckpoint(
                step=step,
                step_directory=os.path.abspath(entry.path),
                role_directory=os.path.abspath(role_directory),
            )
        )

    checkpoints.sort(key=lambda checkpoint: checkpoint.step)
    return checkpoints


def prune_resume_checkpoints(
    checkpoint_root: str,
    *,
    role: str,
    max_to_keep: int | None,
) -> list[str]:
    """Keep only the newest resumable role directories.

    The step directory and its small runtime-state files are retained, matching
    verl's normal role-level rotation. Inference archives live under
    ``checkpoint_root/archived`` and are never scanned or removed here.
    """

    if not (
        max_to_keep
        and isinstance(max_to_keep, int)
        and max_to_keep > 0
    ):
        return []

    checkpoints = discover_resume_checkpoints(
        checkpoint_root,
        role=role,
    )
    stale_checkpoints = checkpoints[:-max_to_keep]
    removed_paths: list[str] = []
    for checkpoint in stale_checkpoints:
        shutil.rmtree(checkpoint.role_directory)
        removed_paths.append(checkpoint.role_directory)
    return removed_paths
