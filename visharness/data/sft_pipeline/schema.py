"""Intermediate trajectory-SFT schema and validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LEGACY_SNAPSHOT_ID = re.compile(r"^(?P<trajectory_id>.+)_step_(?P<turn_index>\d+)$")


@dataclass(frozen=True, slots=True)
class SFTFilterDecision:
    trajectory_id: str
    accepted: bool
    task: str
    reason: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    termination_reason: str | None = None
    prediction_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_trajectory_id(snapshot: dict[str, Any]) -> str:
    trajectory_id = snapshot.get("trajectory_id")
    if isinstance(trajectory_id, str) and trajectory_id:
        return trajectory_id
    snapshot_id = snapshot.get("id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("SFT snapshot must contain a non-empty id")
    match = _LEGACY_SNAPSHOT_ID.fullmatch(snapshot_id)
    if match is None:
        return snapshot_id
    return match.group("trajectory_id")


def snapshot_target_message_index(snapshot: dict[str, Any]) -> int:
    messages = snapshot.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT snapshot messages must be a non-empty list")
    target_index = snapshot.get("target_message_index")
    if isinstance(target_index, int) and not isinstance(target_index, bool):
        if not 0 <= target_index < len(messages):
            raise ValueError(
                f"target_message_index={target_index} is outside messages[0:{len(messages)}]"
            )
        return target_index
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            return index
    raise ValueError("Legacy SFT snapshot does not contain an assistant target")


def validate_snapshot(
    snapshot: dict[str, Any],
    *,
    source: str | Path | None = None,
) -> int:
    location = f" in {source}" if source is not None else ""
    snapshot_id = snapshot.get("id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError(f"SFT snapshot{location} must contain a non-empty id")
    snapshot_trajectory_id(snapshot)

    images = snapshot.get("images")
    messages = snapshot.get("messages")
    if not isinstance(images, list) or not all(
        isinstance(path, str) and path for path in images
    ):
        raise ValueError(f"SFT snapshot {snapshot_id}{location} has invalid images")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT snapshot {snapshot_id}{location} has invalid messages")

    message_images: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"SFT snapshot {snapshot_id}{location} message {message_index} must be an object"
            )
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"SFT snapshot {snapshot_id}{location} message {message_index} has unsupported role {role!r}"
            )
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"SFT snapshot {snapshot_id}{location} message {message_index} has invalid content"
            )
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError(
                        f"SFT snapshot {snapshot_id}{location} contains a non-object content part"
                    )
                if part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    if not isinstance(image_url, dict) or not isinstance(
                        image_url.get("url"), str
                    ):
                        raise ValueError(
                            f"SFT snapshot {snapshot_id}{location} contains an invalid image_url"
                        )
                    message_images.append(image_url["url"])

    if images != message_images:
        raise ValueError(
            f"SFT snapshot {snapshot_id}{location} image order mismatch: "
            f"images={images!r}, messages={message_images!r}"
        )

    target_index = snapshot_target_message_index(snapshot)
    if messages[target_index].get("role") != "assistant":
        raise ValueError(
            f"SFT snapshot {snapshot_id}{location} target message must use role=assistant"
        )
    return target_index
