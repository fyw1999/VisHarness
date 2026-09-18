"""Build strict Swift SFT datasets from VisHarness trajectory snapshots."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SFTSource",
    "FilteredSFTSource",
    "build_sft_dataset",
    "convert_to_swift_format",
    "filter_generated_trajectories",
    "merge_sft_datasets",
    "postprocess_jsonl",
]

_EXPORT_MODULES = {
    "SFTSource": ".merge",
    "FilteredSFTSource": ".pipeline",
    "build_sft_dataset": ".pipeline",
    "convert_to_swift_format": ".swift",
    "filter_generated_trajectories": ".filter",
    "merge_sft_datasets": ".merge",
    "postprocess_jsonl": ".postprocess",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)
