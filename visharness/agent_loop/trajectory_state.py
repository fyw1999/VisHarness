"""Per-trajectory visual state used by the VisHarness agent loop."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionTrajectoryState:
    """State isolated to one asynchronous rollout trajectory."""

    images: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    final_results: dict[str, Any] | None = None
    finished: bool = False

    @classmethod
    def from_initial_image(cls, image: Any) -> "VisionTrajectoryState":
        return cls(images={"img_0": {"image": image}})
