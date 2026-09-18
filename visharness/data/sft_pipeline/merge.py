"""Merge disjoint postprocessed SFT sources without changing sample identity."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from uuid import uuid4

from .io import (
    atomic_text_writer,
    iter_jsonl,
    validate_output_filename,
    write_json_atomic,
)
from .schema import snapshot_trajectory_id, validate_snapshot


_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class SFTSource:
    """One model/source dataset participating in a merge."""

    alias: str
    path: str | Path
    image_root: str | Path | None = None


def validate_source_alias(alias: str) -> str:
    if not alias or _SAFE_ALIAS.fullmatch(alias) is None or alias in {".", ".."}:
        raise ValueError(
            f"Invalid source alias {alias!r}; use only letters, digits, '.', '_' or '-'"
        )
    return alias


def _relative_image_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Image path must be a safe relative path, got {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_image_strict(source: Path, target: Path) -> bool:
    """Copy one image atomically; return whether a new file was installed."""

    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _sha256(source) != _sha256(target):
            raise FileExistsError(
                f"Refusing to replace a different image at merge destination {target}"
            )
        return False

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def _copy_snapshot_images(
    snapshot: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], int, int]:
    copied_snapshot = copy.deepcopy(snapshot)
    copied = 0
    reused = 0
    for image_path in dict.fromkeys(copied_snapshot["images"]):
        relative_path = _relative_image_path(image_path)
        source_path = source_root.joinpath(*relative_path.parts)
        target_path = output_root.joinpath(*relative_path.parts)
        if _copy_image_strict(source_path, target_path):
            copied += 1
        else:
            reused += 1
    return copied_snapshot, copied, reused


def merge_sft_datasets(
    sources: Sequence[SFTSource],
    output_dir: str | Path,
    *,
    output_jsonl_name: str = "merged_sft_data.jsonl",
) -> dict[str, Any]:
    """Merge disjoint filtered sources and reject any cross-source overlap."""

    if not sources:
        raise ValueError("At least one SFT source is required")
    output_dir = Path(output_dir)
    output_jsonl_name = validate_output_filename(output_jsonl_name)
    output_path = output_dir / output_jsonl_name
    aliases = [validate_source_alias(source.alias) for source in sources]
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"SFT source aliases must be unique, got {aliases!r}")

    source_paths = [Path(source.path) for source in sources]
    if output_path.resolve() in {path.resolve() for path in source_paths}:
        raise ValueError("Merged output must not overwrite an input JSONL file")

    total_snapshots = 0
    copied_images = 0
    reused_images = 0
    source_counts: dict[str, int] = {}
    output_ids: set[str] = set()
    trajectory_owners: dict[str, str] = {}
    with atomic_text_writer(output_path) as output_file:
        for source, source_path, alias in zip(sources, source_paths, aliases):
            source_root = (
                source_path.parent
                if source.image_root is None
                else Path(source.image_root)
            )
            source_count = 0
            for line_number, snapshot in iter_jsonl(source_path):
                validate_snapshot(snapshot, source=f"{source_path}:{line_number}")
                source_snapshot_id = str(snapshot["id"])
                source_trajectory_id = snapshot_trajectory_id(snapshot)
                existing_owner = trajectory_owners.get(source_trajectory_id)
                if existing_owner is not None and existing_owner != alias:
                    raise ValueError(
                        "Filtered SFT sources must contain disjoint trajectories, but "
                        f"trajectory id {source_trajectory_id!r} appears in both "
                        f"{existing_owner!r} and {alias!r}"
                    )
                if source_snapshot_id in output_ids:
                    raise ValueError(
                        f"Duplicate snapshot id {source_snapshot_id!r} while merging {source_path}"
                    )
                trajectory_owners[source_trajectory_id] = alias
                output_ids.add(source_snapshot_id)

                merged_snapshot, newly_copied, already_present = _copy_snapshot_images(
                    snapshot,
                    source_root=source_root,
                    output_root=output_dir,
                )
                merged_snapshot["source_alias"] = alias
                merged_snapshot["source_snapshot_id"] = source_snapshot_id
                merged_snapshot["source_trajectory_id"] = source_trajectory_id
                validate_snapshot(
                    merged_snapshot,
                    source=f"{source_path}:{line_number}",
                )
                output_file.write(
                    json.dumps(merged_snapshot, ensure_ascii=False) + "\n"
                )

                source_count += 1
                total_snapshots += 1
                copied_images += newly_copied
                reused_images += already_present
            source_counts[alias] = source_count

    report = {
        "schema_version": 1,
        "output_path": str(output_path),
        "source_counts": source_counts,
        "merged_snapshots": total_snapshots,
        "copied_images": copied_images,
        "reused_images": reused_images,
    }
    write_json_atomic(output_dir / "merge_report.json", report)
    return report
