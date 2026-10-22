from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from PIL import Image

from visharness.data.image_paths import resolve_coco_image_path


@pytest.fixture
def gres_image(tmp_path):
    root = tmp_path / "GRES"
    canonical = root / "images" / "train2014" / "COCO_train2014_000000000001.jpg"
    canonical.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), "red").save(canonical)
    return root, canonical, root / "train2014" / canonical.name


def test_resolves_relative_and_absolute_legacy_paths(gres_image):
    root, canonical, legacy = gres_image
    assert not legacy.exists()
    assert resolve_coco_image_path(legacy) == canonical
    assert resolve_coco_image_path(
        Path("train2014") / canonical.name, dataset_root=root
    ) == canonical
    assert resolve_coco_image_path(canonical) == canonical
    assert resolve_coco_image_path(
        Path("images/train2014") / canonical.name, dataset_root=root
    ) == canonical


def test_prefers_canonical_image_even_before_duplicate_is_removed(gres_image):
    _, canonical, legacy = gres_image
    legacy.parent.mkdir()
    Image.new("RGB", (8, 6), "blue").save(legacy)
    assert resolve_coco_image_path(legacy) == canonical


def test_keeps_legacy_only_datasets_and_missing_file_errors(tmp_path):
    legacy = tmp_path / "GRES/train2014/a.jpg"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert resolve_coco_image_path(legacy) == legacy
    legacy.unlink()
    assert resolve_coco_image_path(legacy) == legacy


def test_does_not_redirect_other_images_or_double_prefix_canonical_paths(gres_image):
    root, canonical, _ = gres_image
    other = root / "val" / canonical.name
    assert resolve_coco_image_path(other) == other
    assert resolve_coco_image_path(canonical) == canonical


@pytest.mark.parametrize("filename", ["a.jpg", "REC8K-example.jpg"])
def test_does_not_redirect_non_coco_filenames(gres_image, filename):
    root, canonical, _ = gres_image
    (canonical.parent / filename).touch()
    legacy = root / "train2014" / filename
    assert resolve_coco_image_path(legacy) == legacy


@pytest.mark.parametrize("manifest_input", ["directory", "file"])
@pytest.mark.parametrize("path_kind", ["legacy_relative", "legacy_absolute", "canonical"])
def test_trajectory_dataset_reads_images_without_legacy_directory(
    gres_image, manifest_input, path_kind
):
    from visharness.trajectory_runner.dataset import TrajectoryDataset

    root, canonical, legacy = gres_image
    paths = {
        "legacy_relative": "train2014/" + canonical.name,
        "legacy_absolute": str(legacy),
        "canonical": "images/train2014/" + canonical.name,
    }
    records = [{
        "id": "GRES-1",
        "question": "Segment the object.",
        "image_path": paths[path_kind],
    }]
    manifest = root / "GRES_QA_val.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    dataset = TrajectoryDataset({
        "dataset_path": str(root if manifest_input == "directory" else manifest),
        "task_names": "GRES",
        "split": "val",
    })
    sample = dataset[0]
    assert sample["image_path"] == str(canonical)
    assert sample["image"].size == (8, 6)
    with Image.open(canonical) as image:
        assert sample["image"].tobytes() == image.convert("RGB").tobytes()
    assert json.loads(manifest.read_text()) == records


@pytest.mark.parametrize("absolute", [False, True])
def test_evaluation_reads_legacy_paths_without_legacy_directory(gres_image, absolute):
    from visharness.evaluate.common import (
        load_exif_transposed_image_size,
        resolve_image_path,
    )

    root, canonical, legacy = gres_image
    item = {
        "id": "GRES-1",
        "image_path": str(legacy) if absolute else "train2014/" + canonical.name,
    }
    assert resolve_image_path(
        item, dataset_root=root, manifest_path=root / "GRES_QA_val.json"
    ) == canonical
    assert load_exif_transposed_image_size(legacy) == (8, 6)


def test_evaluation_keeps_manifest_directory_fallback(gres_image, tmp_path):
    from visharness.evaluate.common import resolve_image_path

    root, canonical, _ = gres_image
    assert resolve_image_path(
        {"image_path": "train2014/" + canonical.name},
        dataset_root=tmp_path / "elsewhere",
        manifest_path=root / "GRES_QA_val.json",
    ) == canonical


def test_training_and_validation_builders_accept_old_manifests(
    gres_image, tmp_path, monkeypatch
):
    from visharness.data import prepare_official_validation, prepare_verl_data

    root, canonical, _ = gres_image
    ground_truth = {"type": "mask"}
    monkeypatch.setattr(
        prepare_verl_data, "build_ground_truth", lambda *args: ground_truth
    )
    monkeypatch.setattr(
        prepare_official_validation, "build_ground_truth", lambda *args: ground_truth
    )
    item = {
        "id": "GRES-1",
        "image_path": "train2014/" + canonical.name,
        "question": "Segment the object.",
    }
    images_dir = tmp_path / "training_images"
    images_dir.mkdir()
    row = prepare_verl_data.build_row(
        item=item, index=0, source_root=root, images_dir=images_dir,
        rec8k_anno={}, gref=None, reasonseg_root=tmp_path / "ReasonSeg/train",
        max_long_edge=1920, max_short_edge=1080,
    )
    assert Path(row["images"][0]["image"]).is_file()
    assert row["extra_info"]["image_width"] == 8

    rows = prepare_official_validation._build_validation_rows(
        selected_by_task={"ReasonSeg": [], "GRES": [item], "REC8K": []},
        rec8k_data_root=tmp_path / "REC-8K", gres_data_root=root,
        reasonseg_data_root=tmp_path / "ReasonSeg",
        output_dir=tmp_path / "validation", annotations={}, gref=None,
        max_long_edge=1920, max_short_edge=1080,
        embed_system_prompt=False, rng=random.Random(42),
    )
    assert len(rows) == 1
    assert rows[0]["extra_info"]["source_image_path"] == str(canonical)
    assert Path(rows[0]["images"][0]["image"]).is_file()
    assert item["image_path"].startswith("train2014/")


@pytest.mark.parametrize("file_url", [False, True])
def test_rollout_initial_image_reads_old_checkpoint_path(gres_image, file_url):
    from visharness.agent_loop.visharness_agent_loop import _load_rgb_image

    _, canonical, legacy = gres_image
    path = ("file://" if file_url else "") + str(legacy)
    loaded = _load_rgb_image(path)
    with Image.open(canonical) as image:
        assert loaded.tobytes() == image.convert("RGB").tobytes()
