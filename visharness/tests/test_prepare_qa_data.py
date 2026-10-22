import json
import random
import sys

import pytest

from visharness.data import prepare_qa_data as prep


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _image(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())


@pytest.fixture
def raw_data(tmp_path):
    root = tmp_path / "datasets"
    refs = []
    for index, (ann_ids, text) in enumerate([
        ([1], "the cat"),
        ([1, 2], "the cats"),
        ([-1], "the dog"),
        ([1], "one two three four five six seven eight nine ten"),
    ], 1):
        refs.append({
            "ref_id": index,
            "image_id": index,
            "ann_id": ann_ids,
            "split": "train",
            "sentences": [{"sent": text}, {"sent": text + "!"}],
        })
    for index, split in enumerate(("val", "testA", "testB"), 101):
        refs.append({
            "ref_id": index,
            "image_id": index,
            "ann_id": [1],
            "split": split,
            "sentences": [{"sent": "the cat! "}, {"sent": "not the first sentence"}],
        })
    images = [
        {"id": ref["image_id"], "file_name": f"COCO_train2014_{ref['image_id']:012d}.jpg"}
        for ref in refs
    ]
    _write_json(root / "GRES/grefcoco/grefs(unc).json", refs)
    _write_json(root / "GRES/grefcoco/instances.json", {"images": images})
    for image in images:
        _image(root / "GRES/images/train2014", image["file_name"])

    annotations = {"a.jpg": {"cat": {}}, "a2.jpg": {"cat": {}}, "b.jpg": {"car": {}}, "c.jpg": {"cat": {}}}
    annotations.update({"val1.jpg": {"cat": {}, "dog": {}}, "val2.jpg": {"cat": {}, "dog": {}}})
    splits = {
        "train": [[image, next(iter(annotations[image]))] for image in ("c.jpg", "a2.jpg", "b.jpg", "a.jpg")],
        "val": [[image, phrase] for image in ("val2.jpg", "val1.jpg") for phrase in ("dog", "cat")],
        "test": [["a.jpg", "cat"], ["a2.jpg", "cat"]],
    }
    _write_json(root / "REC-8K/annotations.json", annotations)
    _write_json(root / "REC-8K/splits.json", splits)
    for image in annotations:
        _image(root / "REC-8K/images", image)

    for split in ("train", "val", "test"):
        _write_json(root / "ReasonSeg" / split / "reason.json", {"text": ["the cat", "the dog"]})
        _image(root / "ReasonSeg" / split, "reason.jpg")
    dense = {"image_path": "dense200/one.jpg", "gt": {"red bird": [[0, 0, 1, 1]], "cat": []}}
    (root / "Dense200").mkdir()
    (root / "Dense200/Dense200.jsonl").write_text(json.dumps(dense) + "\n\n", encoding="utf-8")
    _image(root / "Dense200", "dense200/one.jpg")
    return root


@pytest.fixture
def tiny_quotas(monkeypatch):
    monkeypatch.setattr(prep, "GRES_TRAIN_QUOTAS", dict.fromkeys(prep.GRES_TRAIN_QUOTAS, 1))


def test_gres_evaluation_splits_first_sentence_and_canonical_path(raw_data):
    manifests = prep.build_gres_manifests(raw_data / "GRES")
    assert list(manifests) == ["GRES_QA_val.json", "GRES_QA_testA.json", "GRES_QA_testB.json"]
    assert manifests["GRES_QA_val.json"] == [{
        "question": "Please segment the following object(s) in the image: the cat!",
        "image_path": "images/train2014/COCO_train2014_000000000101.jpg",
        "id": "GRES-101",
    }]


def test_rec8k_evaluation_keeps_all_frames_phrases_and_split_order(raw_data):
    manifests = prep.build_rec8k_manifests(raw_data / "REC-8K")
    val = manifests["REC8K_QA_val.json"]
    assert len(val) == 4
    assert [record["id"] for record in val] == ["val2-dog", "val2-cat", "val1-dog", "val1-cat"]
    assert val[0]["question"] == "Count the number of dog in the image."
    assert len(manifests["REC8K_QA_test.json"]) == 2


def test_rec8k_missing_annotation_is_an_error(raw_data):
    annotations_path = raw_data / "REC-8K/annotations.json"
    annotations = json.loads(annotations_path.read_text())
    del annotations["val1.jpg"]["cat"]
    _write_json(annotations_path, annotations)
    with pytest.raises(ValueError, match="Missing REC-8K annotation"):
        prep.build_rec8k_manifests(raw_data / "REC-8K")


def test_reasonseg_evaluation_and_legacy_annotation_encoding(raw_data):
    annotation = raw_data / "ReasonSeg/val/reason.json"
    annotation.write_bytes(json.dumps({"text": "the café?"}, ensure_ascii=False).encode("cp1252"))
    records = prep.build_reasonseg_manifests(raw_data / "ReasonSeg")["ReasonSeg_QA_val.json"]
    assert records == [{
        "id": "ReasonSeg-reason",
        "image_path": "val/reason.jpg",
        "question": prep.REASONSEG_PREFIXES[0] + "the café?",
    }]


@pytest.mark.parametrize("value", [[], "", [" "]])
def test_reasonseg_rejects_empty_descriptions(raw_data, value):
    _write_json(raw_data / "ReasonSeg/val/reason.json", {"text": value})
    with pytest.raises(ValueError, match="Missing or empty ReasonSeg"):
        prep.build_reasonseg_manifests(raw_data / "ReasonSeg")


def test_dense200_produces_one_example_per_category(raw_data):
    records = prep.build_dense200_manifests(raw_data / "Dense200")["Dense200_QA_test.json"]
    assert [record["id"] for record in records] == ["Dense200-one-red_bird", "Dense200-one-cat"]
    assert records[0]["question"] == "Detect the red bird in the image."
    assert records[0]["image_path"] == "dense200/one.jpg"


def test_seed_sampling_is_deterministic_and_train_only(raw_data, tiny_quotas):
    state = random.getstate()
    manifests = prep.build_seed_manifests(raw_data)
    assert random.getstate() == state
    assert manifests == prep.build_seed_manifests(raw_data)
    assert [record["category"] for record in manifests["GRES_QA_train.json"]] == list(prep.GRES_TRAIN_QUOTAS)
    assert {record["id"] for record in manifests["GRES_QA_train.json"]} == {"GRES-1", "GRES-2", "GRES-3", "GRES-4"}
    # Only the middle adjacent identical phrase set (a2.jpg) is filtered.
    assert [record["id"] for record in manifests["REC8K_QA_train.json"]] == ["REC8K-a-cat", "REC8K-b-car", "REC8K-c-cat"]
    assert len(manifests["ReasonSeg_QA_train.json"]) == 1


def test_gres_seed_falls_back_to_repeated_images_but_not_repeated_refs(raw_data, tiny_quotas):
    path = raw_data / "GRES/grefcoco/grefs(unc).json"
    refs = json.loads(path.read_text())
    for ref in refs:
        if ref["split"] == "train":
            ref["image_id"] = 1
    _write_json(path, refs)
    manifests = prep.build_seed_manifests(raw_data)
    assert len(manifests["GRES_QA_train.json"]) == 4
    assert len({record["id"] for record in manifests["GRES_QA_train.json"]}) == 4


def test_gres_seed_fails_if_quotas_cannot_be_filled(raw_data):
    with pytest.raises(ValueError, match="Insufficient GRES"):
        prep.build_seed_manifests(raw_data)


@pytest.mark.parametrize("dataset", ["gres", "reasonseg", "rec8k", "dense200", "seed"])
def test_prepare_writes_manifests_and_is_safe_to_repeat(raw_data, tiny_quotas, dataset):
    outputs = prep.prepare_dataset(raw_data, dataset)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    assert prep.prepare_dataset(raw_data, dataset) == outputs
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs} == before
    for path in outputs:
        for record in json.loads(path.read_text()):
            assert (path.parent / record["image_path"]).is_file()
    if dataset == "seed":
        assert not (raw_data / "VisionAgent-4K/REC8K_QA_val.json").exists()
        assert not (raw_data / "VisionAgent-4K/images/REC8K-a2.jpg").exists()
        for path in outputs:
            for record in json.loads(path.read_text()):
                source = prep._seed_image_source(raw_data, record["image_path"])
                assert source.read_bytes() == (path.parent / record["image_path"]).read_bytes()


def test_prepare_does_not_overwrite_changed_manifest_without_flag(raw_data):
    path = raw_data / "GRES/GRES_QA_testA.json"
    _write_json(path, [{"id": "do-not-overwrite"}])
    with pytest.raises(FileExistsError):
        prep.prepare_dataset(raw_data, "gres")
    assert json.loads(path.read_text()) == [{"id": "do-not-overwrite"}]
    assert not (raw_data / "GRES/GRES_QA_val.json").exists()
    prep.prepare_dataset(raw_data, "gres", overwrite=True)
    assert json.loads(path.read_text())[0]["id"] == "GRES-102"


def test_prepare_checks_images_before_any_output_is_written(raw_data):
    (raw_data / "GRES/images/train2014/COCO_train2014_000000000103.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        prep.prepare_dataset(raw_data, "gres")
    assert not list((raw_data / "GRES").glob("*QA*.json"))


def test_seed_does_not_overwrite_changed_image_without_flag(raw_data, tiny_quotas):
    prep.prepare_dataset(raw_data, "seed")
    path = raw_data / "VisionAgent-4K/images/REC8K-a.jpg"
    path.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        prep.prepare_dataset(raw_data, "seed")
    assert path.read_bytes() == b"do-not-overwrite"
    prep.prepare_dataset(raw_data, "seed", overwrite=True)
    assert path.read_bytes() == (raw_data / "REC-8K/images/a.jpg").read_bytes()


def test_cli_accepts_readme_command(raw_data, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_qa_data", "--data-root", str(raw_data), "--dataset", "dense200"])
    prep.main()
    assert (raw_data / "Dense200/Dense200_QA_test.json").is_file()
