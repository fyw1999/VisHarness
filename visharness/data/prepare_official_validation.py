"""Build a fixed, diverse RL validation set from the official validation splits.

The resulting parquet contains exactly one rollout per selected image and uses
the same VisHarness prompt/ground-truth representation as RL training.  A
frozen manifest and an audit report are written next to the parquet so every
checkpoint is evaluated on the same examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from datasets import Dataset
from PIL import Image, ImageOps

from visharness.data.prepare_verl_data import (
    G_REFER,
    PROJECT_ROOT,
    build_ground_truth,
    build_row,
    parse_bool,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_data/GRPO/verl_visharness_official_val"

TASK_ORDER = ("ReasonSeg", "GRES", "REC8K")
REC_COUNT_BINS = ("1", "2-5", "6-10", "11-20", "21-50", "51-100", "101+")
# This is the rounded distribution agreed for a 100-sample REC-8K validation set.
REC_COUNT_QUOTAS_100 = {
    "1": 8,
    "2-5": 22,
    "6-10": 23,
    "11-20": 27,
    "21-50": 15,
    "51-100": 3,
    "101+": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rec8k-data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gres-data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reasonseg-data-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train-manifest-root",
        type=Path,
        required=True,
        help="Used only to audit and exclude train IDs/images.",
    )
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-long-edge", type=int, default=1920)
    parser.add_argument("--max-short-edge", type=int, default=1080)
    parser.add_argument(
        "--embed-system-prompt",
        action="store_true",
        help=(
            "Embed the current system prompt in parquet rows. By default only the user "
            "message is stored and VisHarnessDataset injects the prompt at runtime."
        ),
    )
    parser.add_argument("--overwrite", type=parse_bool, default=False, metavar="{true,false}")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("id") or not item.get("image_path"):
            raise ValueError(f"Invalid manifest item {index} in {path}")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate id {item_id!r} in {path}")
        seen.add(item_id)
        result.append(dict(item))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_image_key(image_path: str) -> str:
    name = Path(image_path).name
    for prefix in ("ReasonSeg-", "REC8K-", "GRES-", "GERS-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _normalize_rec_id(item_id: str) -> str:
    return item_id if item_id.startswith("REC8K-") else f"REC8K-{item_id}"


def _load_train_exclusions(root: Path) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    train_ids: set[str] = set()
    train_images: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in sorted(root.glob("*train.json")):
        manifest = _load_manifest(path)
        for item in manifest:
            train_ids.add(str(item["id"]))
            train_images.add(_normalized_image_key(str(item["image_path"])))
        sources.append({"path": str(path), "samples": len(manifest), "sha256": _sha256(path)})
    if not sources:
        raise FileNotFoundError(f"No *train.json files found in {root}")
    return train_ids, train_images, sources


def _largest_remainder_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0}
    weight_sum = sum(positive.values())
    if total and not positive:
        raise ValueError("Cannot allocate a positive total from empty weights")
    raw = {key: total * value / weight_sum for key, value in positive.items()}
    quotas = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    ranking = sorted(positive, key=lambda key: (raw[key] - quotas[key], key), reverse=True)
    for key in ranking[:remainder]:
        quotas[key] += 1
    return {key: quotas.get(key, 0) for key in weights}


def _quantile_bins(values: list[float], num_bins: int = 4) -> tuple[list[int], list[float]]:
    if not values:
        return [], []
    boundaries = [
        float(value)
        for value in np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, num_bins + 1)[1:-1])
    ]
    return [int(np.searchsorted(boundaries, value, side="right")) for value in values], boundaries


def _sample_strata(
    candidates: list[dict[str, Any]],
    quotas: dict[str, int],
    *,
    rng: random.Random,
    used_images: set[str] | None = None,
) -> list[dict[str, Any]]:
    used_images = used_images if used_images is not None else set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["stratum"])].append(candidate)
    for group in grouped.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    # Scarce strata are handled first so an image shared across strata cannot
    # consume a rare stratum's only candidate.
    stratum_order = sorted(
        quotas,
        key=lambda key: (
            len({item["image_key"] for item in grouped.get(key, [])}) / max(quotas[key], 1),
            key,
        ),
    )
    for stratum in stratum_order:
        need = quotas[stratum]
        if need <= 0:
            continue
        for candidate in grouped.get(stratum, []):
            if candidate["image_key"] in used_images:
                continue
            selected.append(candidate)
            used_images.add(candidate["image_key"])
            if sum(item["stratum"] == stratum for item in selected) >= need:
                break
        actual = sum(item["stratum"] == stratum for item in selected)
        if actual != need:
            raise RuntimeError(
                f"Could not satisfy stratum {stratum!r}: selected {actual}, required {need}, "
                f"while preserving unique images"
            )
    return selected


def _reasonseg_area_ratio(annotation_path: Path, image_path: Path) -> float:
    annotation = _load_json(annotation_path)
    with Image.open(image_path) as raw_image:
        width, height = ImageOps.exif_transpose(raw_image).size
    target_area = 0.0
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "")).lower()
        if label == "flag" or "ignore" in label:
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3:
            target_area += abs(float(cv2.contourArea(points)))
    return float(np.clip(target_area / max(width * height, 1), 0.0, 1.0))


def select_reasonseg(
    dataset_root: Path,
    sample_count: int,
    *,
    rng: random.Random,
    train_ids: set[str],
    train_images: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_root / "ReasonSeg_QA_val.json"
    manifest = _load_manifest(manifest_path)
    candidates: list[dict[str, Any]] = []
    for item in manifest:
        item_id = str(item["id"])
        image_key = _normalized_image_key(str(item["image_path"]))
        if item_id in train_ids or image_key in train_images:
            continue
        annotation_path = dataset_root / "val" / f"{item_id.removeprefix('ReasonSeg-')}.json"
        annotation = _load_json(annotation_path)
        candidates.append(
            {
                "item": item,
                "image_key": image_key,
                "is_sentence": bool(annotation.get("is_sentence", False)),
                "area_ratio": _reasonseg_area_ratio(
                    annotation_path,
                    dataset_root / str(item["image_path"]),
                ),
            }
        )

    category_counts = Counter("sentence" if item["is_sentence"] else "phrase" for item in candidates)
    category_quotas = _largest_remainder_quotas(dict(category_counts), sample_count)
    selected: list[dict[str, Any]] = []
    area_boundaries: dict[str, list[float]] = {}
    for category in ("sentence", "phrase"):
        category_candidates = [
            item for item in candidates if ("sentence" if item["is_sentence"] else "phrase") == category
        ]
        bins, boundaries = _quantile_bins([item["area_ratio"] for item in category_candidates])
        area_boundaries[category] = boundaries
        for item, bin_index in zip(category_candidates, bins, strict=True):
            item["stratum"] = f"{category}/area_q{bin_index + 1}"
        stratum_counts = Counter(item["stratum"] for item in category_candidates)
        stratum_quotas = _largest_remainder_quotas(dict(stratum_counts), category_quotas[category])
        selected.extend(_sample_strata(category_candidates, stratum_quotas, rng=rng))

    if len(selected) != sample_count:
        raise RuntimeError(f"ReasonSeg selected {len(selected)} samples, expected {sample_count}")
    rng.shuffle(selected)
    return [dict(item["item"]) for item in selected], {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_samples": len(manifest),
        "eligible_samples": len(candidates),
        "selected_samples": len(selected),
        "unique_images": len({item["image_key"] for item in selected}),
        "category_quotas": category_quotas,
        "category_counts": dict(Counter("sentence" if item["is_sentence"] else "phrase" for item in selected)),
        "area_quantile_boundaries": area_boundaries,
        "selected_area_quartiles": dict(Counter(item["stratum"] for item in selected)),
    }


def _gres_area_ratio(gref: G_REFER, ref: dict[str, Any]) -> float:
    image_info = gref.Imgs[int(ref["image_id"])]
    image_area = max(int(image_info["width"]) * int(image_info["height"]), 1)
    annotation_ids = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
    area = sum(float(gref.Anns[ann_id].get("area", 0.0)) for ann_id in annotation_ids if ann_id in gref.Anns)
    return float(np.clip(area / image_area, 0.0, 1.0))


def select_gres(
    dataset_root: Path,
    sample_count: int,
    *,
    gref: G_REFER,
    rng: random.Random,
    train_ids: set[str],
    train_images: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_root / "GRES_QA_val.json"
    manifest = _load_manifest(manifest_path)
    manifest_by_id = {str(item["id"]): item for item in manifest}
    candidates: list[dict[str, Any]] = []
    for ref_id in gref.getRefIds(split="val"):
        item_id = f"GRES-{int(ref_id)}"
        item = manifest_by_id.get(item_id)
        if item is None:
            raise KeyError(f"{item_id} is missing from {manifest_path}")
        ref = gref.loadRefs([ref_id])[0]
        image_key = _normalized_image_key(str(item["image_path"]))
        if item_id in train_ids or image_key in train_images:
            continue
        annotation_ids = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
        is_empty = bool(ref.get("no_target")) or not annotation_ids or annotation_ids == [-1]
        candidates.append(
            {
                "item": item,
                "image_key": image_key,
                "is_empty": is_empty,
                "area_ratio": 0.0 if is_empty else _gres_area_ratio(gref, ref),
                "long_expression": max(len(sentence["sent"].split()) for sentence in ref["sentences"]) >= 10,
                "target_count": 0 if is_empty else len(annotation_ids),
            }
        )

    empty_quota = round(sample_count * sum(item["is_empty"] for item in candidates) / len(candidates))
    present_quota = sample_count - empty_quota
    empty_candidates = [item for item in candidates if item["is_empty"]]
    for item in empty_candidates:
        item["stratum"] = "empty"
    used_images: set[str] = set()
    selected = _sample_strata(
        empty_candidates,
        {"empty": empty_quota},
        rng=rng,
        used_images=used_images,
    )

    present_candidates = [item for item in candidates if not item["is_empty"]]
    bins, boundaries = _quantile_bins([item["area_ratio"] for item in present_candidates])
    for item, bin_index in zip(present_candidates, bins, strict=True):
        item["stratum"] = f"present/area_q{bin_index + 1}"
    stratum_counts = Counter(item["stratum"] for item in present_candidates)
    present_quotas = _largest_remainder_quotas(dict(stratum_counts), present_quota)
    selected.extend(
        _sample_strata(
            present_candidates,
            present_quotas,
            rng=rng,
            used_images=used_images,
        )
    )

    if len(selected) != sample_count:
        raise RuntimeError(f"GRES selected {len(selected)} samples, expected {sample_count}")
    rng.shuffle(selected)
    return [dict(item["item"]) for item in selected], {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_samples": len(manifest),
        "eligible_samples": len(candidates),
        "selected_samples": len(selected),
        "unique_images": len({item["image_key"] for item in selected}),
        "empty_quota": empty_quota,
        "present_quota": present_quota,
        "empty_selected": sum(item["is_empty"] for item in selected),
        "present_selected": sum(not item["is_empty"] for item in selected),
        "present_area_quantile_boundaries": boundaries,
        "selected_area_quartiles": dict(Counter(item["stratum"] for item in selected)),
        "long_expression_selected": sum(item["long_expression"] for item in selected),
        "multi_target_selected": sum(item["target_count"] > 1 for item in selected),
    }


def _rec_count_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return "101+"


def _rec_source(image_name: str) -> str:
    parts = image_name.split("-")
    return parts[1] if len(parts) > 2 else "unknown"


def _rec_quotas(sample_count: int) -> dict[str, int]:
    if sample_count == 100:
        return dict(REC_COUNT_QUOTAS_100)
    return _largest_remainder_quotas(REC_COUNT_QUOTAS_100, sample_count)


def _legacy_rec_frame_filter(image_to_phrases: dict[str, list[str]]) -> list[str]:
    """Match tackle_rec-8k.py: retain the first sorted frame per phrase-set run."""
    kept: list[str] = []
    previous_phrases: tuple[str, ...] | None = None
    for image_name in sorted(image_to_phrases):
        phrases = tuple(sorted(image_to_phrases[image_name]))
        if phrases != previous_phrases:
            kept.append(image_name)
            previous_phrases = phrases
    return kept


def select_rec8k(
    dataset_root: Path,
    sample_count: int,
    *,
    annotations: dict[str, Any],
    rng: random.Random,
    train_ids: set[str],
    train_images: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_root / "REC8K_QA_val.json"
    manifest = _load_manifest(manifest_path)
    manifest_by_id = {_normalize_rec_id(str(item["id"])): item for item in manifest}
    splits = _load_json(dataset_root / "splits.json")
    split_items = splits.get("val")
    if not isinstance(split_items, list):
        raise ValueError("REC-8K splits.json has no list-valued 'val' split")

    image_to_phrases: dict[str, list[str]] = defaultdict(list)
    for image_name, phrase in split_items:
        image_to_phrases[str(image_name)].append(str(phrase))
    kept_images = _legacy_rec_frame_filter(image_to_phrases)

    candidates: list[dict[str, Any]] = []
    for image_name in kept_images:
        image_key = _normalized_image_key(image_name)
        if image_key in train_images:
            continue
        for phrase in image_to_phrases[image_name]:
            base_id = f"{Path(image_name).stem}-{'_'.join(phrase.split())}"
            item_id = _normalize_rec_id(base_id)
            if item_id in train_ids:
                continue
            source_item = manifest_by_id.get(item_id)
            if source_item is None:
                raise KeyError(f"{item_id} is missing from {manifest_path}")
            count = len(annotations[image_name][phrase].get("points", []))
            if count <= 0:
                raise ValueError(f"REC-8K validation sample {item_id} has no points")
            normalized_item = dict(source_item)
            normalized_item["id"] = item_id
            candidates.append(
                {
                    "item": normalized_item,
                    "image_key": image_key,
                    "image_name": image_name,
                    "source": _rec_source(image_name),
                    "count": count,
                    "stratum": _rec_count_bin(count),
                }
            )

    quotas = _rec_quotas(sample_count)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["stratum"]].append(candidate)
    random_tie_break = {id(candidate): rng.random() for candidate in candidates}

    eligible_images_by_source = Counter(
        {
            source: len({item["image_key"] for item in candidates if item["source"] == source})
            for source in {item["source"] for item in candidates}
        }
    )
    source_targets = _largest_remainder_quotas(dict(eligible_images_by_source), sample_count)
    source_selected: Counter[str] = Counter()
    used_images: set[str] = set()
    selected: list[dict[str, Any]] = []

    # Fill rare count bins first. Within a bin, prefer the source furthest below
    # its image-level target and use the seeded random value only as a tie break.
    bin_order = sorted(
        REC_COUNT_BINS,
        key=lambda key: (
            len({item["image_key"] for item in grouped[key]}) / max(quotas[key], 1),
            key,
        ),
    )
    for count_bin in bin_order:
        for _ in range(quotas[count_bin]):
            available = [item for item in grouped[count_bin] if item["image_key"] not in used_images]
            if not available:
                raise RuntimeError(
                    f"Could not satisfy REC-8K count bin {count_bin!r} while preserving unique images"
                )

            def candidate_rank(item: dict[str, Any]) -> tuple[float, int, float]:
                target = max(source_targets.get(item["source"], 0), 1)
                return (
                    source_selected[item["source"]] / target,
                    source_selected[item["source"]],
                    random_tie_break[id(item)],
                )

            chosen = min(available, key=candidate_rank)
            selected.append(chosen)
            used_images.add(chosen["image_key"])
            source_selected[chosen["source"]] += 1

    if len(selected) != sample_count:
        raise RuntimeError(f"REC-8K selected {len(selected)} samples, expected {sample_count}")
    rng.shuffle(selected)
    return [dict(item["item"]) for item in selected], {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_samples": len(manifest),
        "original_unique_images": len(image_to_phrases),
        "images_after_contiguous_frame_filter": len(kept_images),
        "removed_redundant_frames": len(image_to_phrases) - len(kept_images),
        "eligible_candidate_pairs": len(candidates),
        "selected_samples": len(selected),
        "unique_images": len({item["image_key"] for item in selected}),
        "count_bin_quotas": quotas,
        "count_bin_selected": dict(Counter(item["stratum"] for item in selected)),
        "source_targets": source_targets,
        "source_selected": dict(source_selected),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _original_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as raw_image:
        return ImageOps.exif_transpose(raw_image).size


def _build_validation_rows(
    selected_by_task: dict[str, list[dict[str, Any]]],
    *,
    rec8k_data_root: Path,
    gres_data_root: Path,
    reasonseg_data_root: Path,
    output_dir: Path,
    annotations: dict[str, Any],
    gref: G_REFER,
    max_long_edge: int,
    max_short_edge: int,
    embed_system_prompt: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    task_roots = {
        "ReasonSeg": reasonseg_data_root,
        "GRES": gres_data_root,
        "REC8K": rec8k_data_root,
    }
    reasonseg_root = reasonseg_data_root / "val"
    rows: list[dict[str, Any]] = []

    for task in TASK_ORDER:
        source_root = task_roots[task]
        images_dir = output_dir / "images" / task
        images_dir.mkdir(parents=True, exist_ok=True)
        for item in selected_by_task[task]:
            row = build_row(
                item=item,
                index=len(rows),
                source_root=source_root,
                images_dir=images_dir,
                rec8k_anno=annotations,
                gref=gref,
                reasonseg_root=reasonseg_root,
                max_long_edge=max_long_edge,
                max_short_edge=max_short_edge,
                embed_system_prompt=embed_system_prompt,
            )
            source_path = source_root / str(item["image_path"])
            original_width, original_height = _original_image_size(source_path)
            evaluation_ground_truth = build_ground_truth(
                str(item["id"]),
                annotations,
                gref,
                reasonseg_root,
                1.0,
                original_width,
                original_height,
            )
            row["uid"] = str(item["id"])
            row["extra_info"].update(
                {
                    "split": "val",
                    "source_split": "val",
                    "validation_sample": True,
                    "validation_task": task,
                    "source_image_path": str(source_path.resolve()),
                    "evaluation_ground_truth": json.dumps(evaluation_ground_truth, ensure_ascii=False),
                    "evaluation_original_width": original_width,
                    "evaluation_original_height": original_height,
                }
            )
            rows.append(row)

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index
    return rows


def _selected_overlap(
    selected: Iterable[dict[str, Any]],
    train_ids: set[str],
    train_images: set[str],
) -> dict[str, list[str]]:
    selected_ids = {str(item["id"]) for item in selected}
    selected_images = {_normalized_image_key(str(item["image_path"])) for item in selected}
    return {
        "ids": sorted(selected_ids & train_ids),
        "images": sorted(selected_images & train_images),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    output_dir = args.output_dir.resolve()
    parquet_path = output_dir / "val.parquet"
    report_path = output_dir / "selection_report.json"
    manifests_dir = output_dir / "manifests"

    if output_dir.exists() and not args.overwrite:
        existing = [path for path in (parquet_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Validation output already exists ({existing}); pass --overwrite true to replace it"
            )
    if args.overwrite:
        for path in (parquet_path, report_path):
            path.unlink(missing_ok=True)
        shutil.rmtree(manifests_dir, ignore_errors=True)
        shutil.rmtree(output_dir / "images", ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, train_images, train_sources = _load_train_exclusions(args.train_manifest_root)
    rec_root = args.rec8k_data_root
    annotations = _load_json(rec_root / "annotations.json")
    gref = G_REFER(str(args.gres_data_root), dataset="grefcoco", splitBy="unc")

    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    task_reports: dict[str, Any] = {}
    # Independent task RNGs make one task's future implementation changes
    # unable to perturb another task's frozen sample selection.
    for offset, task in enumerate(TASK_ORDER):
        task_rng = random.Random(args.seed + offset)
        if task == "ReasonSeg":
            selected, report = select_reasonseg(
                args.reasonseg_data_root,
                args.samples_per_task,
                rng=task_rng,
                train_ids=train_ids,
                train_images=train_images,
            )
        elif task == "GRES":
            selected, report = select_gres(
                args.gres_data_root,
                args.samples_per_task,
                gref=gref,
                rng=task_rng,
                train_ids=train_ids,
                train_images=train_images,
            )
        else:
            selected, report = select_rec8k(
                rec_root,
                args.samples_per_task,
                annotations=annotations,
                rng=task_rng,
                train_ids=train_ids,
                train_images=train_images,
            )
        overlap = _selected_overlap(selected, train_ids, train_images)
        if overlap["ids"] or overlap["images"]:
            raise RuntimeError(f"{task} validation selection overlaps training data: {overlap}")
        selected_by_task[task] = selected
        task_reports[task] = {**report, "train_overlap": overlap}

    manifest_names = {
        "ReasonSeg": f"ReasonSeg_QA_val_rl{args.samples_per_task}.json",
        "GRES": f"GRES_QA_val_rl{args.samples_per_task}.json",
        "REC8K": f"REC8K_QA_val_rl{args.samples_per_task}.json",
    }
    manifest_reports: dict[str, Any] = {}
    for task in TASK_ORDER:
        manifest_path = manifests_dir / manifest_names[task]
        _write_json(manifest_path, selected_by_task[task])
        manifest_reports[task] = {
            "path": str(manifest_path),
            "samples": len(selected_by_task[task]),
            "sha256": _sha256(manifest_path),
        }

    rows = _build_validation_rows(
        selected_by_task,
        rec8k_data_root=args.rec8k_data_root,
        gres_data_root=args.gres_data_root,
        reasonseg_data_root=args.reasonseg_data_root,
        output_dir=output_dir,
        annotations=annotations,
        gref=gref,
        max_long_edge=args.max_long_edge,
        max_short_edge=args.max_short_edge,
        embed_system_prompt=args.embed_system_prompt,
        rng=random.Random(args.seed),
    )
    expected_rows = len(TASK_ORDER) * args.samples_per_task
    if len(rows) != expected_rows or len({row["uid"] for row in rows}) != expected_rows:
        raise RuntimeError("Validation rows are not complete and uniquely identified")
    Dataset.from_list(rows).to_parquet(str(parquet_path))

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "total_samples": len(rows),
        "max_long_edge": args.max_long_edge,
        "max_short_edge": args.max_short_edge,
        "system_prompt_embedded": args.embed_system_prompt,
        "dataset_roots": {
            "REC8K": str(args.rec8k_data_root.resolve()),
            "GRES": str(args.gres_data_root.resolve()),
            "ReasonSeg": str(args.reasonseg_data_root.resolve()),
        },
        "parquet": {
            "path": str(parquet_path),
            "samples": len(rows),
            "sha256": _sha256(parquet_path),
        },
        "frozen_manifests": manifest_reports,
        "training_manifests": train_sources,
        "tasks": task_reports,
    }
    _write_json(report_path, report)

    print(f"Wrote {len(rows)} official validation samples to {parquet_path}")
    for task in TASK_ORDER:
        print(
            f"  {task}: {len(selected_by_task[task])} samples, "
            f"{task_reports[task]['unique_images']} unique images"
        )
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
