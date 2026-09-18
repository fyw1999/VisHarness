"""Dataset loading for VisHarness trajectory inference and data generation."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .config import as_plain_dict


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        return _read_jsonl(data_path)
    if data_path.suffix.lower() == ".json":
        with data_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, list) else [payload]
    if data_path.suffix.lower() == ".parquet":
        from datasets import Dataset as HFDataset

        return list(HFDataset.from_parquet(str(data_path)))
    raise ValueError(f"Unsupported dataset file: {data_path}")


def _load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _resize_like_legacy_eval(
    image: Image.Image,
    max_long_edge: int = 1920,
    max_short_edge: int = 1080,
) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    short_edge = min(width, height)
    scale = min(max_long_edge / long_edge, max_short_edge / short_edge, 1.0)
    if scale == 1.0:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)


def _normalize_task_names(task_names: Any) -> set[str] | None:
    if task_names is None:
        return None
    if isinstance(task_names, str):
        # Legacy YAML configs sometimes express multiple task names as an
        # indented plain scalar without list markers:
        #
        # task_names:
        #   ReasonSeg
        #   GRES
        #   REC8K
        #
        # PyYAML folds that value into "ReasonSeg GRES REC8K". Support both
        # this representation and the existing comma-separated form.
        return {item for item in re.split(r"[\s,]+", task_names.strip()) if item}
    return {str(item) for item in task_names}


def _select_manifest_files(
    dataset_path: str | Path,
    task_names: Any,
    split: Any,
) -> list[Path]:
    """Resolve directory-backed QA manifests by exact task/split names."""

    task_name_set = _normalize_task_names(task_names)
    split_name = str(split).strip() if split is not None else ""
    if not task_name_set or not split_name:
        raise ValueError(
            "When dataset_path is a directory, both task_names and split are required"
        )

    manifest_paths = [
        Path(dataset_path) / f"{task_name}_QA_{split_name}.json"
        for task_name in sorted(task_name_set)
    ]
    missing_paths = [path for path in manifest_paths if not path.is_file()]
    if missing_paths:
        available_files = sorted(path.name for path in Path(dataset_path).glob("*.json"))
        raise FileNotFoundError(
            "Expected dataset manifest file(s) do not exist: "
            f"{[str(path) for path in missing_paths]}. "
            f"Available JSON files: {available_files}"
        )
    return manifest_paths


def _validate_unique_record_ids(records: list[dict[str, Any]]) -> None:
    """Fail before inference when resume/output identity would be ambiguous."""

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset record {index} must be a JSON object")
        item_id = item.get("id")
        if item_id is None:
            raise ValueError(f"Dataset record {index} does not contain an id")
        item_id = str(item_id)
        if item_id in seen_ids:
            duplicate_ids.add(item_id)
        seen_ids.add(item_id)

    if duplicate_ids:
        examples = sorted(duplicate_ids)[:10]
        raise ValueError(
            f"Dataset contains {len(duplicate_ids)} duplicate IDs. "
            f"Examples: {examples}"
        )


class TrajectoryDataset(Dataset):
    """Load samples using the old ``tf_eval`` config shape."""

    def __init__(self, dataset_args: Any = None):
        self.dataset_args = as_plain_dict(dataset_args)
        self.full_data = self.load_data_function(self.dataset_args)
        self.meta_data = list(self.full_data)
        self._resume_from_ckpt(self.dataset_args.get("resume_from_ckpt"))

    def load_data_function(self, dataset_args: dict[str, Any]) -> list[dict[str, Any]]:
        dataset_path = dataset_args.get("dataset_path")
        if dataset_path is None:
            raise ValueError("dataset_args.dataset_path is required")

        dataset_path = str(dataset_path)
        num_samples = dataset_args.get("num_sample")
        if os.path.isdir(dataset_path):
            json_files = _select_manifest_files(
                dataset_path,
                dataset_args.get("task_names"),
                dataset_args.get("split"),
            )
            records = []
            for json_file in json_files:
                for item in _load_records(json_file):
                    image_file = item.get("image_path")
                    if image_file and not os.path.isabs(str(image_file)):
                        item = dict(item)
                        item["image_path"] = os.path.join(dataset_path, str(image_file))
                    records.append(item)
        else:
            records = _load_records(dataset_path)
            base_dir = str(Path(dataset_path).parent)
            for index, item in enumerate(records):
                item.setdefault("id", str(index))
                image_path = item.get("image_path") or item.get("image")
                if isinstance(image_path, str) and not os.path.isabs(image_path):
                    item["image_path"] = os.path.join(base_dir, image_path)

        _validate_unique_record_ids(records)
        if num_samples is not None:
            records = records[: int(num_samples)]
        if dataset_args.get("shuffle", False):
            random.Random(int(dataset_args.get("seed", 42))).shuffle(records)
        return records

    def _resume_from_ckpt(self, ckpt_paths: Any) -> None:
        if not ckpt_paths:
            return
        if isinstance(ckpt_paths, (str, Path)):
            ckpt_paths = [ckpt_paths]
        processed_ids = set()
        for ckpt_path_value in ckpt_paths:
            ckpt_path = Path(ckpt_path_value)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Configured resume checkpoint does not exist: {ckpt_path}"
                )
            for item in _read_jsonl(ckpt_path):
                item_id = item.get("id")
                if item_id is None and isinstance(item.get("meta_data"), dict):
                    item_id = item["meta_data"].get("id")
                if isinstance(item_id, str) and "_step_" in item_id:
                    item_id = item_id.split("_step_", 1)[0]
                if item_id is not None:
                    processed_ids.add(str(item_id))
        self.meta_data = [item for item in self.meta_data if str(item.get("id")) not in processed_ids]

    def __len__(self) -> int:
        return len(self.meta_data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.meta_data[index])
        if isinstance(item.get("extra_info"), dict):
            item.setdefault("id", item["extra_info"].get("item_id", item["extra_info"].get("index", index)))
            item.setdefault("question", item["extra_info"].get("question"))
            item.setdefault("image_path", item["extra_info"].get("image_path"))
        if isinstance(item.get("reward_model"), dict):
            item.setdefault("ground_truth", item["reward_model"].get("ground_truth"))
        if item.get("image_path") is None and isinstance(item.get("images"), list) and item["images"]:
            first_image = item["images"][0]
            if isinstance(first_image, dict):
                item["image_path"] = first_image.get("image") or first_image.get("path")
            elif isinstance(first_image, str):
                item["image_path"] = first_image
        if not item.get("question") and isinstance(item.get("prompt"), list):
            for message in reversed(item["prompt"]):
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if isinstance(content, str):
                    item["question"] = content.replace("<image>", "", 1).strip()
                elif isinstance(content, list):
                    texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    item["question"] = "".join(texts).strip()
                break
        image_value = item.get("image")
        if isinstance(image_value, Image.Image):
            image = ImageOps.exif_transpose(image_value).convert("RGB")
        else:
            image_path = item.get("image_path") or image_value
            if not image_path:
                raise ValueError("Dataset item must contain 'image' or 'image_path'")
            image = _load_rgb_image(image_path)
            item["image_path"] = str(image_path)
        image = _resize_like_legacy_eval(
            image,
            max_long_edge=int(self.dataset_args.get("max_long_edge", 1920)),
            max_short_edge=int(self.dataset_args.get("max_short_edge", 1080)),
        )
        item["image"] = image
        item.setdefault("id", str(index))
        item.setdefault("question", item.get("prompt") or item.get("text") or "")
        return item
