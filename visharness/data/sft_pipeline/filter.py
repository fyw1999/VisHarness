"""Filter trajectory snapshots using compact, task-aware checkpoint scoring."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from visharness.evaluate.common import load_compact_predictions

from .io import (
    atomic_text_writer,
    iter_jsonl,
    validate_output_filename,
    write_json_atomic,
    write_jsonl_atomic,
)
from .schema import snapshot_trajectory_id, validate_snapshot
from .scoring import FilterThresholds, TrajectoryAcceptanceScorer


def _as_paths(values: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(values, (str, Path)):
        values = [values]
    paths = [Path(value) for value in values]
    if not paths:
        raise ValueError("At least one input path is required")
    return paths


def filter_generated_trajectories(
    *,
    checkpoint_paths: str | Path | Sequence[str | Path],
    trajectory_paths: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    rec8k_annotations_path: str | Path | None = None,
    gres_dataset_root: str | Path | None = None,
    reasonseg_dataset_root: str | Path | None = None,
    thresholds: FilterThresholds | None = None,
    accepted_sft_filename: str = "accepted_sft_snapshots.jsonl",
) -> dict[str, Any]:
    """Score latest checkpoints and retain latest snapshots for accepted IDs."""

    checkpoint_paths = _as_paths(checkpoint_paths)
    trajectory_paths = _as_paths(trajectory_paths)
    output_dir = Path(output_dir)
    accepted_sft_filename = validate_output_filename(accepted_sft_filename)
    accepted_sft_path = output_dir / accepted_sft_filename
    accepted_checkpoint_path = output_dir / "accepted_checkpoints.jsonl"
    rejected_checkpoint_path = output_dir / "rejected_checkpoints.jsonl"
    decisions_path = output_dir / "filter_decisions.jsonl"
    accepted_ids_path = output_dir / "accepted_ids.jsonl"
    report_path = output_dir / "filter_report.json"
    input_resolved = {
        path.resolve() for path in [*checkpoint_paths, *trajectory_paths]
    }
    colliding_outputs = [
        path
        for path in (
            accepted_sft_path,
            accepted_checkpoint_path,
            rejected_checkpoint_path,
            decisions_path,
            accepted_ids_path,
            report_path,
        )
        if path.resolve() in input_resolved
    ]
    if colliding_outputs:
        raise ValueError(
            "Filter outputs must not overwrite input files: "
            f"{[str(path) for path in colliding_outputs]}"
        )

    predictions, load_report = load_compact_predictions(
        checkpoint_paths,
        strict=True,
    )
    effective_thresholds = thresholds or FilterThresholds()
    scorer = TrajectoryAcceptanceScorer(
        rec8k_annotations_path=rec8k_annotations_path,
        gres_dataset_root=gres_dataset_root,
        reasonseg_dataset_root=reasonseg_dataset_root,
        thresholds=effective_thresholds,
    )
    decisions = [scorer.score(prediction) for prediction in predictions.values()]
    decisions_by_id = {
        decision.trajectory_id: decision for decision in decisions
    }
    accepted_ids = {
        decision.trajectory_id for decision in decisions if decision.accepted
    }

    # First pass stores only the last line position for each snapshot ID. This
    # preserves append/resume semantics without retaining full snapshots in RAM.
    latest_positions: dict[str, tuple[int, int]] = {}
    latest_trajectory_ids: dict[str, str] = {}
    total_snapshot_lines = 0
    for path_index, path in enumerate(trajectory_paths):
        for line_number, snapshot in iter_jsonl(path):
            validate_snapshot(snapshot, source=f"{path}:{line_number}")
            snapshot_id = str(snapshot["id"])
            latest_positions[snapshot_id] = (path_index, line_number)
            latest_trajectory_ids[snapshot_id] = snapshot_trajectory_id(snapshot)
            total_snapshot_lines += 1

    missing_snapshot_ids = sorted(
        accepted_ids - set(latest_trajectory_ids.values())
    )
    if missing_snapshot_ids:
        raise ValueError(
            f"{len(missing_snapshot_ids)} accepted trajectories have no SFT snapshot; "
            f"examples={missing_snapshot_ids[:10]}"
        )

    retained_snapshot_count = 0
    retained_trajectory_ids: set[str] = set()
    with atomic_text_writer(accepted_sft_path) as output_file:
        for path_index, path in enumerate(trajectory_paths):
            for line_number, snapshot in iter_jsonl(path):
                if latest_positions[str(snapshot["id"])] != (
                    path_index,
                    line_number,
                ):
                    continue
                trajectory_id = snapshot_trajectory_id(snapshot)
                if trajectory_id not in accepted_ids:
                    continue
                output_file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                retained_snapshot_count += 1
                retained_trajectory_ids.add(trajectory_id)

    if retained_trajectory_ids != accepted_ids:
        raise AssertionError("Filtered trajectory accounting is inconsistent")

    latest_checkpoint_positions = {
        (prediction.source_path, prediction.line_number): prediction.item_id
        for prediction in predictions.values()
    }
    accepted_checkpoint_count = 0
    rejected_checkpoint_count = 0
    with atomic_text_writer(accepted_checkpoint_path) as accepted_file:
        with atomic_text_writer(rejected_checkpoint_path) as rejected_file:
            for checkpoint_path in checkpoint_paths:
                for line_number, record in iter_jsonl(checkpoint_path):
                    item_id = latest_checkpoint_positions.get(
                        (str(checkpoint_path), line_number)
                    )
                    if item_id is None:
                        continue
                    decision = decisions_by_id[item_id]
                    destination = accepted_file if decision.accepted else rejected_file
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if decision.accepted:
                        accepted_checkpoint_count += 1
                    else:
                        rejected_checkpoint_count += 1

    write_jsonl_atomic(
        decisions_path,
        (decision.to_dict() for decision in decisions),
    )
    write_jsonl_atomic(
        accepted_ids_path,
        ({"id": item_id} for item_id in sorted(accepted_ids)),
    )
    reason_counts = Counter(decision.reason for decision in decisions)
    report = {
        "schema_version": 2,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "accepted_sft_snapshots_path": str(accepted_sft_path),
        "accepted_checkpoint_path": str(accepted_checkpoint_path),
        "rejected_checkpoint_path": str(rejected_checkpoint_path),
        "accepted_ids_path": str(accepted_ids_path),
        "filter_decisions_path": str(decisions_path),
        "datasets": {
            "rec8k_annotations_path": (
                str(rec8k_annotations_path) if rec8k_annotations_path else None
            ),
            "gres_dataset_root": (
                str(gres_dataset_root) if gres_dataset_root else None
            ),
            "reasonseg_dataset_root": (
                str(reasonseg_dataset_root) if reasonseg_dataset_root else None
            ),
        },
        "thresholds": asdict(effective_thresholds),
        "prediction_load_report": load_report.to_dict(),
        "evaluated_trajectories": len(decisions),
        "accepted_trajectories": len(accepted_ids),
        "rejected_trajectories": len(decisions) - len(accepted_ids),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "input_snapshot_lines": total_snapshot_lines,
        "unique_snapshot_ids": len(latest_positions),
        "retained_snapshots": retained_snapshot_count,
        "accepted_checkpoint_records": accepted_checkpoint_count,
        "rejected_checkpoint_records": rejected_checkpoint_count,
    }
    write_json_atomic(report_path, report)
    return report
