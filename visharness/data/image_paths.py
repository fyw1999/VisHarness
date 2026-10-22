"""Resolve COCO images after consolidating the dataset image directory."""

from __future__ import annotations

from pathlib import Path


def resolve_coco_image_path(
    image_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> Path:
    """Prefer images/train2014 while keeping old QA/checkpoint paths readable.

    Existing datasets and checkpoints may contain either relative
    train2014/<filename> paths or absolute paths to that directory.
    Only redirect COCO_train2014_* filenames in this exact directory layout,
    and only when the canonical image exists. Other dataset paths and
    missing-file errors stay unchanged.
    No filesystem entries or input records are modified.
    """

    path = Path(image_path)
    if not path.is_absolute() and dataset_root is not None:
        path = Path(dataset_root) / path
    if (
        path.name.startswith("COCO_train2014_")
        and path.parent.name == "train2014"
        and path.parent.parent.name != "images"
    ):
        canonical = path.parent.parent / "images" / "train2014" / path.name
        if canonical.is_file():
            return canonical
    return path
