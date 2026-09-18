"""Strict streaming and atomic file helpers for the SFT pipeline."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO
from uuid import uuid4


def validate_output_filename(value: str) -> str:
    """Require a plain filename so output stays inside its output directory."""

    path = Path(value)
    if not value or value in {".", ".."} or path.name != value or path.is_absolute():
        raise ValueError(f"Output filename must be a plain filename, got {value!r}")
    return value


def iter_jsonl(path_value: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, item


@contextmanager
def atomic_text_writer(path_value: str | Path) -> Iterator[TextIO]:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(
    path_value: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    with atomic_text_writer(path_value) as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_atomic(path_value: str | Path, payload: Any) -> None:
    with atomic_text_writer(path_value) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
