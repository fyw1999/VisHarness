"""Per-sample state for VisHarness trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from PIL import Image

from visharness.agent_loop.trajectory_state import VisionTrajectoryState


@dataclass
class TrajectoryItem:
    max_rounds: int
    current_round: int
    meta_data: dict[str, Any]
    conversation: list[Any]
    state: VisionTrajectoryState
    status: str = "pending"
    tool_response: list[Any] = field(default_factory=list)
    current_images: list[Image.Image] = field(default_factory=list)
    current_image_names: list[str] = field(default_factory=list)
    answer: str | None = None
    trajectory_uid: str = field(default_factory=lambda: uuid4().hex)
    error: str | None = None
    trajectory_invalid: bool = False
    invalid_reason: str | None = None
    invalid_stage: str | None = None
    invalid_turn_index: int | None = None
    invalid_error: str | None = None
    efficiency_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def images(self) -> dict[str, dict[str, Any]]:
        return self.state.images

    @property
    def turn_records(self) -> list[dict[str, Any]]:
        return self.state.turns

    @property
    def trajectory_finished(self) -> bool:
        """Whether the model explicitly ended the trajectory with an answer."""
        return bool(self.state.finished)

    @property
    def max_rounds_reached(self) -> bool:
        """Whether processing ended naturally because the turn budget was exhausted."""
        return (
            self.status == "finished"
            and not self.trajectory_finished
            and int(self.current_round) >= int(self.max_rounds)
        )

    @property
    def termination_reason(self) -> str | None:
        """Describe why processing stopped without changing lifecycle ``status`` semantics."""
        if self.status == "failed":
            return "failed"
        if self.trajectory_finished:
            return "answer"
        if self.max_rounds_reached:
            return "max_rounds_reached"
        return None

    def result_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "current_round": int(self.current_round),
            "status": self.status,
            "trajectory_finished": self.trajectory_finished,
            "max_rounds_reached": self.max_rounds_reached,
            "termination_reason": self.termination_reason,
            "answer": self.answer,
            "error": self.error,
            "trajectory_invalid": bool(self.trajectory_invalid),
            "invalid_reason": self.invalid_reason,
            "invalid_stage": self.invalid_stage,
            "invalid_turn_index": self.invalid_turn_index,
            "invalid_error": self.invalid_error,
            "trajectory_uid": self.trajectory_uid,
            "efficiency_metrics": self.efficiency_metrics,
            "turn_records": self.turn_records,
            "meta_data": self.meta_data,
            "conversation": self.conversation,
            "tool_response": self.tool_response,
            "images": self.state.images,
            "final_results": self.state.final_results,
        }
