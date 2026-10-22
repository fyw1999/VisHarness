"""Prepare evaluation QA manifests and the VisHarness-4K training seed pool."""

from __future__ import annotations

import argparse
import filecmp
import json
import random
import shutil
from pathlib import Path
from typing import Any


GRES_TRAIN_QUOTAS = {
    "long_sentence": 500,
    "multi_target": 500,
    "single_target": 500,
    "no_target": 100,
}
SEGMENTATION_TEMPLATES = (
    "Please segment the following object(s) in the image: {expression}.",
    "Could you find and mask '{expression}' in this image?",
    "Generate segmentation mask(s) for: {expression}.",
    "Locate '{expression}' in the provided image and output mask(s).",
    "What is the segmentation result for '{expression}'?",
    "I need the pixel-wise mask for the object(s) described as '{expression}'.",
    "Can you highlight '{expression}' with segmentation mask(s)?",
    "Find the region corresponding to '{expression}' and segment object(s).",
    "Identify and segment {expression}.",
)
REASONSEG_PREFIXES = (
    "Please segment the object in the image that matches the following description: ",
    "Based on the following reasoning, generate the segmentation mask for the target object: ",
    "Can you find and segment the region corresponding to this description: ",
    "Identify and segment the item in the picture described as follows: ",
    "Read the following text and provide a segmentation mask for the described object: ",
    "Locate the target mentioned in the sentence below and segment it: ",
    "Please output the segmentation mask for the object that fits this specific reasoning: ",
    "Analyze the image and segment the entity that aligns with this description: ",
)
COUNT_TEMPLATES = (
    "Count the number of {phrase} in the image.",
    "What is the total count of {phrase}?",
    "Count all instances of {phrase}.",
    "How many instances of {phrase} are visible?",
    "Count the {phrase}.",
    "Please tell me how many {phrase} are present.",
)
DETECTION_TEMPLATES = (
    "Detect the {phrase} in the image.",
    "Locate the {phrase}.",
    "Find all instances of {phrase}.",
    "Where can I find {phrase} in this picture?",
    "Show me the location of the {phrase}.",
    "Identify the position of every {phrase}.",
    "Find the {phrase}.",
)
DATASET_DIRS = {"gres": "GRES", "reasonseg": "ReasonSeg", "rec8k": "REC-8K", "dense200": "Dense200"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def _load_gres(root: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    refs = _read_json(root / "grefcoco/grefs(unc).json")
    instances = _read_json(root / "grefcoco/instances.json")
    filenames = {image["id"]: image["file_name"] for image in instances["images"]}
    return refs, filenames


def _finish_sentence(text: str, punctuation: str) -> str:
    text = text.rstrip()
    return text if text.endswith(tuple(punctuation)) else text + "."


def build_gres_manifests(root: Path) -> dict[str, list[dict[str, Any]]]:
    refs, filenames = _load_gres(root)
    manifests = {}
    for split in ("val", "testA", "testB"):
        records = []
        for ref in refs:
            if ref["split"] != split:
                continue
            question = "Please segment the following object(s) in the image: " + ref["sentences"][0]["sent"]
            records.append({
                "question": _finish_sentence(question, ".?!"),
                "image_path": f"images/train2014/{filenames[ref['image_id']]}",
                "id": f"GRES-{ref['ref_id']}",
            })
        manifests[f"GRES_QA_{split}.json"] = records
    return manifests


def _reasonseg_examples(root: Path, split: str):
    paths = sorted((root / split).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No ReasonSeg annotations found in {root / split}")
    for path in paths:
        texts = _read_json(path).get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"Missing or empty ReasonSeg descriptions: {path}")
        yield path.stem, texts


def build_reasonseg_manifests(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        f"ReasonSeg_QA_{split}.json": [
            {
                "id": f"ReasonSeg-{stem}",
                "image_path": f"{split}/{stem}.jpg",
                "question": _finish_sentence(REASONSEG_PREFIXES[0] + texts[0], ".?"),
            }
            for stem, texts in _reasonseg_examples(root, split)
        ]
        for split in ("val", "test")
    }


def _rec8k_groups(items: list, annotations: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for image, phrase in items:
        if image not in annotations or phrase not in annotations[image]:
            raise ValueError(f"Missing REC-8K annotation for {image}: {phrase}")
        grouped.setdefault(image, []).append(phrase)
    return grouped


def build_rec8k_manifests(root: Path) -> dict[str, list[dict[str, Any]]]:
    annotations = _read_json(root / "annotations.json")
    splits = _read_json(root / "splits.json")
    return {
        f"REC8K_QA_{split}.json": [
            {
                "question": COUNT_TEMPLATES[0].format(phrase=phrase),
                "image_path": f"images/{image}",
                "id": image.split(".")[0] + "-" + "_".join(phrase.split()),
            }
            for image, phrases in _rec8k_groups(splits[split], annotations).items()
            for phrase in phrases
        ]
        for split in ("val", "test")
    }


def build_dense200_manifests(root: Path) -> dict[str, list[dict[str, Any]]]:
    records = []
    with (root / "Dense200.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            for category in item["gt"]:
                records.append({
                    "id": f"Dense200-{Path(item['image_path']).stem}-{category.replace(' ', '_')}",
                    "image_path": item["image_path"],
                    "question": f"Detect the {category} in the image.",
                })
    return {"Dense200_QA_test.json": records}


def _gres_category(ref: dict[str, Any]) -> str:
    ann_ids = ref["ann_id"]
    if not isinstance(ann_ids, list):
        ann_ids = [ann_ids]
    if max(len(sentence["sent"].split()) for sentence in ref["sentences"]) >= 10:
        return "long_sentence"
    if ann_ids == [-1] or not ann_ids:
        return "no_target"
    return "multi_target" if len(ann_ids) > 1 else "single_target"


def _build_gres_seed(root: Path, seed: int) -> list[dict[str, Any]]:
    refs, filenames = _load_gres(root)
    refs = [ref for ref in refs if ref["split"] == "train"]
    rng = random.Random(seed)
    rng.shuffle(refs)
    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in GRES_TRAIN_QUOTAS}
    seen_refs: set[int] = set()
    seen_images: set[int] = set()
    for require_unique_images in (True, False):
        for ref in refs:
            category = _gres_category(ref)
            if len(selected[category]) >= GRES_TRAIN_QUOTAS[category] or ref["ref_id"] in seen_refs:
                continue
            if require_unique_images and ref["image_id"] in seen_images:
                continue
            selected[category].append(ref)
            seen_refs.add(ref["ref_id"])
            seen_images.add(ref["image_id"])
    for category, quota in GRES_TRAIN_QUOTAS.items():
        if len(selected[category]) != quota:
            raise ValueError(f"Insufficient GRES {category} references: {len(selected[category])}/{quota}")
    records = []
    for category, group in selected.items():
        for ref in group:
            expression = rng.choice(ref["sentences"])["sent"]
            records.append({
                "question": rng.choice(SEGMENTATION_TEMPLATES).format(expression=expression),
                "image_path": f"images/GRES-{filenames[ref['image_id']]}",
                "id": f"GRES-{ref['ref_id']}",
                "category": category,
            })
    return records


def _build_rec8k_seed(root: Path, seed: int) -> list[dict[str, Any]]:
    annotations = _read_json(root / "annotations.json")
    grouped = _rec8k_groups(_read_json(root / "splits.json")["train"], annotations)
    rng = random.Random(seed)
    records = []
    previous_phrases = None
    for image in sorted(grouped):
        # Preserve the original training-only consecutive phrase-set filter.
        phrase_set = tuple(sorted(grouped[image]))
        if phrase_set == previous_phrases:
            continue
        previous_phrases = phrase_set
        for phrase in grouped[image]:
            templates = COUNT_TEMPLATES if rng.random() < 0.5 else DETECTION_TEMPLATES
            records.append({
                "question": rng.choice(templates).format(phrase=phrase),
                "image_path": f"images/REC8K-{image}",
                "id": "REC8K-" + image.split(".")[0] + "-" + "_".join(phrase.split()),
            })
    return records


def _build_reasonseg_seed(root: Path, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for stem, texts in _reasonseg_examples(root, "train"):
        description = rng.choice(texts)
        records.append({
            "id": f"ReasonSeg-{stem}",
            "image_path": f"images/ReasonSeg-{stem}.jpg",
            "question": rng.choice(REASONSEG_PREFIXES) + description,
        })
    return records


def build_seed_manifests(data_root: Path, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    return {
        "GRES_QA_train.json": _build_gres_seed(data_root / "GRES", seed),
        "REC8K_QA_train.json": _build_rec8k_seed(data_root / "REC-8K", seed),
        "ReasonSeg_QA_train.json": _build_reasonseg_seed(data_root / "ReasonSeg", seed),
    }


def _seed_image_source(data_root: Path, relative_path: str) -> Path:
    name = Path(relative_path).name
    if name.startswith("GRES-"):
        return data_root / "GRES/images/train2014" / name.removeprefix("GRES-")
    if name.startswith("REC8K-"):
        return data_root / "REC-8K/images" / name.removeprefix("REC8K-")
    if name.startswith("ReasonSeg-"):
        return data_root / "ReasonSeg/train" / name.removeprefix("ReasonSeg-")
    raise ValueError(f"Unknown seed image prefix: {relative_path}")


def prepare_dataset(
    data_root: Path,
    dataset: str,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> list[Path]:
    """Validate inputs before writing; existing unequal outputs require opt-in."""
    if dataset == "seed":
        manifests = build_seed_manifests(data_root, seed)
        destination = data_root / "VisHarness-4K"
        source_root = None
    else:
        source_root = data_root / DATASET_DIRS[dataset]
        builder = {
            "gres": build_gres_manifests,
            "reasonseg": build_reasonseg_manifests,
            "rec8k": build_rec8k_manifests,
            "dense200": build_dense200_manifests,
        }[dataset]
        manifests = builder(source_root)
        destination = data_root / DATASET_DIRS[dataset]

    copy_jobs: dict[Path, Path] = {}
    for filename, records in manifests.items():
        if not records:
            raise ValueError(f"No examples for {filename}")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError(f"Duplicate example IDs in {filename}")
        output_path = destination / filename
        if output_path.exists() and _read_json(output_path) != records and not overwrite:
            raise FileExistsError(f"Different manifest already exists: {output_path}; pass --overwrite to replace it")
        for record in records:
            source = (
                _seed_image_source(data_root, record["image_path"])
                if source_root is None else source_root / record["image_path"]
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            if dataset == "seed":
                target = destination / record["image_path"]
                copy_jobs[target] = source

    for target, source in copy_jobs.items():
        if target.exists() and not filecmp.cmp(source, target, shallow=False) and not overwrite:
            raise FileExistsError(f"Different image already exists: {target}; pass --overwrite to replace it")
    for target, source in copy_jobs.items():
        if not target.exists() or (overwrite and not filecmp.cmp(source, target, shallow=False)):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, records in manifests.items():
        output_path = destination / filename
        if not output_path.exists() or _read_json(output_path) != records:
            output_path.write_text(json.dumps(records, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{output_path}: {len(records)} examples")
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Parent directory of GRES, ReasonSeg, REC-8K, and Dense200")
    parser.add_argument("--dataset", choices=(*DATASET_DIRS, "seed"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs only when their contents differ")
    args = parser.parse_args()
    prepare_dataset(args.data_root, args.dataset, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
