"""Build a Swift SFT dataset from explicitly filtered model sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .io import validate_output_filename, write_json_atomic
from .merge import SFTSource, merge_sft_datasets, validate_source_alias
from .postprocess import postprocess_jsonl
from .swift import Qwen3VLVisualBudget, convert_to_swift_format


@dataclass(frozen=True, slots=True)
class FilteredSFTSource:
    """Accepted SFT snapshots produced by one explicit filter run."""

    alias: str
    filtered_sft_path: str | Path
    image_root: str | Path


def build_sft_dataset(
    sources: Sequence[FilteredSFTSource],
    output_dir: str | Path,
    *,
    merged_filename: str = "merged_sft_data.jsonl",
    swift_filename: str = "merged_sft_data_swift.jsonl",
    image_max_token_num: int = 2048,
    max_total_raw_image_patches: int = 56_000,
) -> dict[str, Any]:
    """Run postprocess -> merge -> Swift conversion on filtered inputs."""

    if not sources:
        raise ValueError("At least one generated SFT source is required")
    merged_filename = validate_output_filename(merged_filename)
    swift_filename = validate_output_filename(swift_filename)
    if merged_filename == swift_filename:
        raise ValueError("Merged and Swift output filenames must be different")
    output_dir = Path(output_dir)
    postprocessed_dir = output_dir / "postprocessed"

    source_reports: dict[str, Any] = {}
    processed_sources: list[SFTSource] = []
    for source in sources:
        alias = validate_source_alias(source.alias)
        if alias in source_reports:
            raise ValueError(f"Duplicate generated SFT source alias {alias!r}")
        source_dir = postprocessed_dir / alias
        filtered_path = Path(source.filtered_sft_path)
        processed_path = source_dir / "sft_postprocessed.jsonl"
        postprocess_report = postprocess_jsonl(filtered_path, processed_path)
        processed_sources.append(
            SFTSource(
                alias=alias,
                path=processed_path,
                image_root=source.image_root,
            )
        )
        source_reports[alias] = {
            "filtered_sft_path": str(filtered_path),
            "image_root": str(source.image_root),
            "postprocess": postprocess_report,
        }

    merge_report = merge_sft_datasets(
        processed_sources,
        output_dir,
        output_jsonl_name=merged_filename,
    )
    swift_report = convert_to_swift_format(
        output_dir / merged_filename,
        output_dir / swift_filename,
        image_root_dir=output_dir,
        visual_budget=Qwen3VLVisualBudget(
            image_max_token_num=image_max_token_num,
            max_total_raw_image_patches=max_total_raw_image_patches,
        ),
        visual_budget_decisions_file=output_dir / "visual_budget_decisions.jsonl",
    )
    report = {
        "schema_version": 2,
        "sources": source_reports,
        "merge": merge_report,
        "swift": swift_report,
    }
    write_json_atomic(output_dir / "sft_pipeline_report.json", report)
    return report
