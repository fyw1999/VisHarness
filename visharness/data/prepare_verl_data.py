"""Build verl-compatible VisHarness parquet files from the raw annotations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import Dataset
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visharness.prompts import TRAIN_TEST_SYSTEM_PROMPT
from visharness.evaluate.grefer import G_REFER

DEFAULT_DATASETS_ROOT = Path("/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/datasets")
DEFAULT_SOURCE_ROOT = DEFAULT_DATASETS_ROOT / "VisionAgent-4K"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_data/GRPO/verl_visharness"


def parse_bool(value: str | bool) -> bool:
    """Parse an explicit command-line boolean for debugger-friendly arguments."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw VisionAgent-4K annotations into verl-compatible parquet files."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--rec8k-anno-path", type=Path, default=DEFAULT_DATASETS_ROOT / "REC-8K/annotations.json")
    parser.add_argument("--gres-data-root", type=Path, default=DEFAULT_DATASETS_ROOT / "GRES")
    parser.add_argument("--reasonseg-data-root", type=Path, default=DEFAULT_DATASETS_ROOT / "ReasonSeg/train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-pattern", default="*train.json")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--shuffle",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="Shuffle train/val rows after splitting. Set false to preserve deterministic source order.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit raw samples for a smoke test; -1 keeps all.")
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


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def get_rec8k_gt(item_id: str, rec8k_anno: dict[str, Any], scale: float = 1.0) -> dict[str, Any]:
    """Look up REC-8K point annotations and scale them with the resized image."""
    rest = item_id.removeprefix("REC8K-")
    target_key = None
    matched_phrase = None

    image_name, phrase = rest.rsplit("-", 1)
    phrase = " ".join(phrase.split("_"))
    for suffix in (".jpg", ".png"):
        candidate = image_name + suffix
        if candidate in rec8k_anno and phrase in rec8k_anno[candidate]:
            target_key = candidate
            matched_phrase = phrase
            break

    if target_key is None:
        parts = rest.split("-")
        for split_at in range(len(parts) - 2, 0, -1):
            image_name = "-".join(parts[:split_at])
            phrase = " ".join(("-".join(parts[split_at:])).split("_"))
            for suffix in (".jpg", ".png"):
                candidate = image_name + suffix
                if candidate in rec8k_anno and phrase in rec8k_anno[candidate]:
                    target_key = candidate
                    matched_phrase = phrase
                    break
            if target_key is not None:
                break

    if target_key is None or matched_phrase is None:
        raise ValueError(f"Could not match {item_id} in REC-8K annotations")

    points = rec8k_anno[target_key][matched_phrase]["points"]
    if not points:
        raise ValueError(f"REC-8K sample {item_id} has no annotated points")
    scaled_points = [[point[0] * scale, point[1] * scale] for point in points]
    return {"type": "point", "data": scaled_points}


def get_gres_gt(item_id: str, gref: G_REFER, scale: float = 1.0) -> dict[str, Any]:
    """Load a GRES mask, resize it with nearest-neighbor interpolation, and encode it as COCO RLE."""
    ref_id = int(item_id.split("-")[1])
    ref = gref.loadRefs(ref_id)[0]
    mask_info = gref.getMaskByRef(ref=ref, merge=True)
    if mask_info.get("empty", False):
        return {"type": "gres_rle_mask", "data": None}

    mask = mask_info["mask"]
    if scale != 1.0:
        height, width = mask.shape
        resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        mask = cv2.resize(mask, resized_size, interpolation=cv2.INTER_NEAREST)
    return {"type": "gres_rle_mask", "data": encode_mask(mask)}


def get_reasonseg_gt(
    item_id: str,
    reasonseg_root: Path,
    scale: float,
    new_width: int,
    new_height: int,
) -> dict[str, Any]:
    """Render ReasonSeg polygons on the resized canvas and encode target/ignore masks."""
    annotation_path = reasonseg_root / f"{item_id.removeprefix('ReasonSeg-')}.json"
    with annotation_path.open(encoding="utf-8") as file:
        shapes = json.load(file)["shapes"]

    polygons: list[tuple[int, str, list[list[float]]]] = []
    for shape in shapes:
        label = shape["label"]
        if label.lower() == "flag":
            continue

        points = [[x * scale, y * scale] for x, y in shape["points"]]
        polygon = np.array([points], dtype=np.int32)
        area_mask = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.polylines(area_mask, polygon, True, 1, 1)
        cv2.fillPoly(area_mask, polygon, 1)
        polygons.append((int(area_mask.sum()), label, points))

    gt_mask = np.zeros((new_height, new_width), dtype=np.uint8)
    for _, label, points in sorted(polygons, key=lambda item: item[0], reverse=True):
        label_value = 255 if "ignore" in label.lower() else 1
        polygon = np.array([points], dtype=np.int32)
        cv2.polylines(gt_mask, polygon, True, label_value, 1)
        cv2.fillPoly(gt_mask, polygon, label_value)

    return {
        "type": "reasonseg_rle_mask",
        "data": {
            "target_rle": encode_mask(gt_mask == 1),
            "ignore_rle": encode_mask(gt_mask == 255),
            "original_shape": [new_height, new_width],
        },
    }


def load_raw_items(source_root: Path, input_pattern: str, max_samples: int, seed: int) -> list[dict[str, Any]]:
    input_files = sorted(source_root.glob(input_pattern))
    if not input_files:
        raise FileNotFoundError(f"No input files matched {source_root / input_pattern}")

    items: list[dict[str, Any]] = []
    for input_file in input_files:
        with input_file.open(encoding="utf-8") as file:
            payload = json.load(file)
        file_items = payload if isinstance(payload, list) else [payload]
        items.extend(file_items)
        print(f"Loaded {len(file_items):5d} samples from {input_file.name}")

    if 0 < max_samples < len(items):
        items = random.Random(seed).sample(items, max_samples)
        print(f"Selected {len(items)} samples for this run")
    return items


def resize_and_save_image(
    source_path: Path,
    images_dir: Path,
    max_long_edge: int,
    max_short_edge: int,
) -> tuple[Path, float, int, int]:
    with Image.open(source_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
        width, height = image.size
        long_edge = max(width, height)
        short_edge = min(width, height)
        scale = min(max_long_edge / long_edge, max_short_edge / short_edge, 1.0)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        if scale != 1.0:
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        output_path = images_dir / source_path.name
        image.save(output_path)

    return output_path.resolve(), scale, new_width, new_height


def data_source_for(item_id: str) -> str:
    if item_id.startswith("REC8K"):
        return "visharness/rec8k"
    if item_id.startswith("GRES"):
        return "visharness/gres"
    if item_id.startswith("ReasonSeg"):
        return "visharness/reasonseg"
    raise ValueError(f"Unsupported sample id: {item_id}")


def build_ground_truth(
    item_id: str,
    rec8k_anno: dict[str, Any],
    gref: G_REFER,
    reasonseg_root: Path,
    scale: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    if item_id.startswith("REC8K"):
        return get_rec8k_gt(item_id, rec8k_anno, scale)
    if item_id.startswith("GRES"):
        return get_gres_gt(item_id, gref, scale)
    if item_id.startswith("ReasonSeg"):
        return get_reasonseg_gt(item_id, reasonseg_root, scale, width, height)
    raise ValueError(f"Unsupported sample id: {item_id}")


def build_prompt(user_content: str, *, embed_system_prompt: bool = False) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if embed_system_prompt:
        messages.append({"role": "system", "content": TRAIN_TEST_SYSTEM_PROMPT.strip()})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_row(
    item: dict[str, Any],
    index: int,
    source_root: Path,
    images_dir: Path,
    rec8k_anno: dict[str, Any],
    gref: G_REFER,
    reasonseg_root: Path,
    max_long_edge: int,
    max_short_edge: int,
    embed_system_prompt: bool = False,
) -> dict[str, Any]:
    item_id = item["id"]
    source_path = source_root / item["image_path"]
    if not source_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {source_path}")

    output_path, scale, width, height = resize_and_save_image(
        source_path, images_dir, max_long_edge, max_short_edge
    )
    ground_truth = build_ground_truth(item_id, rec8k_anno, gref, reasonseg_root, scale, width, height)
    data_source = data_source_for(item_id)
    question = item["question"]
    user_content = f"<image>{question} The height and width of the image are {height} and {width}, respectively.\n"

    return {
        "data_source": data_source,
        "agent_name": "visharness_agent",
        "prompt": build_prompt(
            user_content,
            embed_system_prompt=embed_system_prompt,
        ),
        "images": [{"image": str(output_path)}],
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        },
        "extra_info": {
            "index": index,
            "split": "train",
            "item_id": item_id,
            "question": question,
            "task_type": ground_truth["type"],
            "image_path": str(output_path),
            "image_height": height,
            "image_width": width,
            "system_prompt_embedded": embed_system_prompt,
        },
    }


def stratified_split(
    rows: list[dict[str, Any]], val_ratio: float, seed: int, shuffle: bool
) -> tuple[list[dict], list[dict]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    rng = random.Random(seed)
    if val_ratio == 0:
        train_rows = list(rows)
        if shuffle:
            rng.shuffle(train_rows)
        for row in train_rows:
            row["extra_info"]["split"] = "train"
        return train_rows, []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        if shuffle:
            rng.shuffle(group)
        val_count = min(len(group) - 1, max(1, round(len(group) * val_ratio))) if len(group) > 1 else 0
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])

    if shuffle:
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
    for row in train_rows:
        row["extra_info"]["split"] = "train"
    for row in val_rows:
        row["extra_info"]["split"] = "val"
    return train_rows, val_rows


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        print(f"Skipping empty split: {output_path.name}")
        return
    Dataset.from_list(rows).to_parquet(str(output_path))
    print(f"Wrote {len(rows):5d} samples to {output_path}")


def main() -> None:
    args = parse_args()
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    failure_path = args.output_dir / "failures.json"
    if not args.overwrite and (train_path.exists() or val_path.exists()):
        raise FileExistsError(f"{args.output_dir} already contains parquet output; pass --overwrite true to replace it")
    if args.overwrite:
        for output_path in (train_path, val_path, failure_path):
            output_path.unlink(missing_ok=True)

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    raw_items = load_raw_items(args.source_root, args.input_pattern, args.max_samples, args.seed)
    print("Loading REC-8K and GRES annotations...")
    with args.rec8k_anno_path.open(encoding="utf-8") as file:
        rec8k_anno = json.load(file)
    gref = G_REFER(str(args.gres_data_root), dataset="grefcoco", splitBy="unc")

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, item in enumerate(raw_items):
        try:
            rows.append(
                build_row(
                    item=item,
                    index=index,
                    source_root=args.source_root,
                    images_dir=images_dir,
                    rec8k_anno=rec8k_anno,
                    gref=gref,
                    reasonseg_root=args.reasonseg_data_root,
                    max_long_edge=args.max_long_edge,
                    max_short_edge=args.max_short_edge,
                    embed_system_prompt=args.embed_system_prompt,
                )
            )
        except Exception as error:
            failures.append((item.get("id", "<missing-id>"), str(error)))
            print(f"[skip] {failures[-1][0]}: {failures[-1][1]}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No samples were converted successfully")

    print(f"Embed system prompt in parquet: {args.embed_system_prompt}")
    print(f"Shuffle split rows: {args.shuffle}")
    train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed, args.shuffle)
    write_parquet(train_rows, train_path)
    write_parquet(val_rows, val_path)

    print(f"Converted {len(rows)} samples; skipped {len(failures)}")
    if failures:
        with failure_path.open("w", encoding="utf-8") as file:
            json.dump([{"item_id": item_id, "error": error} for item_id, error in failures], file, indent=2)
        print(f"Failure details: {failure_path}")


if __name__ == "__main__":
    main()
