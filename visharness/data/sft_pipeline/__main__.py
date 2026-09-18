"""Command-line entry points for the trajectory-to-Swift SFT pipeline."""

from __future__ import annotations

import argparse
import json
from typing import Sequence


def _thresholds(args: argparse.Namespace):
    from .scoring import FilterThresholds

    return FilterThresholds(
        gres_iou=args.gres_iou,
        reasonseg_iou=args.reasonseg_iou,
        rec8k_relative_count_error=args.rec8k_relative_count_error,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
    )


def _add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gres-iou", type=float, default=0.7)
    parser.add_argument("--reasonseg-iou", type=float, default=0.7)
    parser.add_argument(
        "--rec8k-relative-count-error",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--aspect-ratio-tolerance",
        type=float,
        default=0.05,
    )


def _parse_key_value(values: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects ALIAS=PATH, got {value!r}")
        alias, path = value.split("=", 1)
        if not alias or not path or alias in result:
            raise ValueError(f"Invalid or duplicate {option} value {value!r}")
        result[alias] = path
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser(
        "filter", help="Filter generated trajectories"
    )
    filter_parser.add_argument("--checkpoint", action="append", required=True)
    filter_parser.add_argument("--trajectory", action="append", required=True)
    filter_parser.add_argument("--output-dir", required=True)
    filter_parser.add_argument("--rec8k-annotations")
    filter_parser.add_argument("--gres-dataset-root")
    filter_parser.add_argument("--reasonseg-dataset-root")
    _add_threshold_arguments(filter_parser)

    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Normalize filtered snapshots"
    )
    postprocess_parser.add_argument("input")
    postprocess_parser.add_argument("output")

    merge_parser = subparsers.add_parser("merge", help="Merge normalized sources")
    merge_parser.add_argument(
        "--source", action="append", required=True, help="ALIAS=JSONL"
    )
    merge_parser.add_argument(
        "--image-root", action="append", default=[], help="optional ALIAS=PATH"
    )
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--output-name", default="merged_sft_data.jsonl")

    swift_parser = subparsers.add_parser("swift", help="Convert merged data to Swift")
    swift_parser.add_argument("input")
    swift_parser.add_argument("output")
    swift_parser.add_argument("--image-root")

    build = subparsers.add_parser(
        "build",
        help="Postprocess and merge filtered sources, then convert them to Swift",
    )
    build.add_argument(
        "--source",
        action="append",
        required=True,
        help="filtered source as ALIAS=accepted_sft_snapshots.jsonl",
    )
    build.add_argument(
        "--image-root",
        action="append",
        required=True,
        help="source image root as ALIAS=PATH; required for every source",
    )
    build.add_argument("--output-dir", required=True)
    build.add_argument("--merged-name", default="merged_sft_data.jsonl")
    build.add_argument("--swift-name", default="merged_sft_data_swift.jsonl")
    build.add_argument(
        "--image-max-token-num",
        type=_positive_int,
        default=2048,
        help="Qwen3-VL per-image merged-token limit used by training",
    )
    build.add_argument(
        "--max-total-raw-image-patches",
        type=_positive_int,
        default=56_000,
        help="maximum pre-merge Qwen3-VL image patches allowed per SFT sample",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "filter":
        from .filter import filter_generated_trajectories

        report = filter_generated_trajectories(
            checkpoint_paths=args.checkpoint,
            trajectory_paths=args.trajectory,
            output_dir=args.output_dir,
            rec8k_annotations_path=args.rec8k_annotations,
            gres_dataset_root=args.gres_dataset_root,
            reasonseg_dataset_root=args.reasonseg_dataset_root,
            thresholds=_thresholds(args),
        )
    elif args.command == "postprocess":
        from .postprocess import postprocess_jsonl

        report = postprocess_jsonl(args.input, args.output)
    elif args.command == "merge":
        from .merge import SFTSource, merge_sft_datasets

        paths = _parse_key_value(args.source, option="--source")
        image_roots = _parse_key_value(args.image_root, option="--image-root")
        unknown_roots = set(image_roots) - set(paths)
        if unknown_roots:
            raise ValueError(
                f"Image roots supplied for unknown aliases: {sorted(unknown_roots)}"
            )
        report = merge_sft_datasets(
            [
                SFTSource(alias, path, image_roots.get(alias))
                for alias, path in paths.items()
            ],
            args.output_dir,
            output_jsonl_name=args.output_name,
        )
    elif args.command == "swift":
        from .swift import convert_to_swift_format

        report = convert_to_swift_format(
            args.input,
            args.output,
            image_root_dir=args.image_root,
        )
    else:
        from .pipeline import FilteredSFTSource, build_sft_dataset

        paths = _parse_key_value(args.source, option="--source")
        image_roots = _parse_key_value(args.image_root, option="--image-root")
        missing_roots = set(paths) - set(image_roots)
        unknown_roots = set(image_roots) - set(paths)
        if missing_roots:
            raise ValueError(
                f"Image roots are required for source aliases: {sorted(missing_roots)}"
            )
        if unknown_roots:
            raise ValueError(
                f"Image roots supplied for unknown aliases: {sorted(unknown_roots)}"
            )
        report = build_sft_dataset(
            [
                FilteredSFTSource(
                    alias=alias,
                    filtered_sft_path=path,
                    image_root=image_roots[alias],
                )
                for alias, path in paths.items()
            ],
            args.output_dir,
            merged_filename=args.merged_name,
            swift_filename=args.swift_name,
            image_max_token_num=args.image_max_token_num,
            max_total_raw_image_patches=args.max_total_raw_image_patches,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
