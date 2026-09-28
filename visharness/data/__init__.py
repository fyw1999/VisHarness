"""Dataset and data-conversion utilities for VisHarness.

The RL dataset depends on ``verl`` while offline SFT conversion does not.  Load
the RL class only when it is requested so the SFT pipeline remains usable in a
lightweight inference/data-processing environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .visharness_dataset import VisHarnessDataset

__all__ = ["VisHarnessDataset"]


def __getattr__(name: str) -> Any:
    if name == "VisHarnessDataset":
        from .visharness_dataset import VisHarnessDataset

        return VisHarnessDataset
    raise AttributeError(name)
