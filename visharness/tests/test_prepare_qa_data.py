import json
import random
import re
import shlex
import sys
from pathlib import Path

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


def _source_roots(root):
    return {
        "gres": root / "GRES",
        "reasonseg": root / "ReasonSeg",
        "rec8k": root / "REC-8K",
        "dense200": root / "Dense200",
    }


def _seed_roots(root):
    sources = _source_roots(root)
    return {f"{name}_data_root": sources[name] for name in ("gres", "rec8k", "reasonseg")}


def _prepare_paths(root, dataset):
    if dataset == "seed":
        return {**_seed_roots(root), "output_dir": root / "VisHarness-4K"}
    return {f"{dataset}_data_root": _source_roots(root)[dataset]}


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
    manifests = prep.build_seed_manifests(**_seed_roots(raw_data))
    assert random.getstate() == state
    assert manifests == prep.build_seed_manifests(**_seed_roots(raw_data))
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
    manifests = prep.build_seed_manifests(**_seed_roots(raw_data))
    assert len(manifests["GRES_QA_train.json"]) == 4
    assert len({record["id"] for record in manifests["GRES_QA_train.json"]}) == 4


def test_gres_seed_fails_if_quotas_cannot_be_filled(raw_data):
    with pytest.raises(ValueError, match="Insufficient GRES"):
        prep.build_seed_manifests(**_seed_roots(raw_data))


@pytest.mark.parametrize("dataset", ["gres", "reasonseg", "rec8k", "dense200", "seed"])
def test_prepare_writes_manifests_and_is_safe_to_repeat(raw_data, tiny_quotas, dataset):
    outputs = prep.prepare_dataset(dataset, **_prepare_paths(raw_data, dataset))
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    assert prep.prepare_dataset(dataset, **_prepare_paths(raw_data, dataset)) == outputs
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs} == before
    for path in outputs:
        for record in json.loads(path.read_text()):
            assert (path.parent / record["image_path"]).is_file()
    if dataset == "seed":
        assert not (raw_data / "VisHarness-4K/REC8K_QA_val.json").exists()
        assert not (raw_data / "VisHarness-4K/images/REC8K-a2.jpg").exists()
        for path in outputs:
            for record in json.loads(path.read_text()):
                source = prep._seed_image_source(_source_roots(raw_data), record["image_path"])
                assert source.read_bytes() == (path.parent / record["image_path"]).read_bytes()


def test_prepare_does_not_overwrite_changed_manifest_without_flag(raw_data):
    path = raw_data / "GRES/GRES_QA_testA.json"
    _write_json(path, [{"id": "do-not-overwrite"}])
    with pytest.raises(FileExistsError):
        prep.prepare_dataset("gres", gres_data_root=raw_data / "GRES")
    assert json.loads(path.read_text()) == [{"id": "do-not-overwrite"}]
    assert not (raw_data / "GRES/GRES_QA_val.json").exists()
    prep.prepare_dataset("gres", gres_data_root=raw_data / "GRES", overwrite=True)
    assert json.loads(path.read_text())[0]["id"] == "GRES-102"


def test_prepare_checks_images_before_any_output_is_written(raw_data):
    (raw_data / "GRES/images/train2014/COCO_train2014_000000000103.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        prep.prepare_dataset("gres", gres_data_root=raw_data / "GRES")
    assert not list((raw_data / "GRES").glob("*QA*.json"))


def test_seed_does_not_overwrite_changed_image_without_flag(raw_data, tiny_quotas):
    prep.prepare_dataset("seed", **_prepare_paths(raw_data, "seed"))
    path = raw_data / "VisHarness-4K/images/REC8K-a.jpg"
    path.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        prep.prepare_dataset("seed", **_prepare_paths(raw_data, "seed"))
    assert path.read_bytes() == b"do-not-overwrite"
    prep.prepare_dataset("seed", **_prepare_paths(raw_data, "seed"), overwrite=True)
    assert path.read_bytes() == (raw_data / "REC-8K/images/a.jpg").read_bytes()


def test_seed_output_uses_visharness_name(raw_data, tiny_quotas):
    outputs = prep.prepare_dataset("seed", **_prepare_paths(raw_data, "seed"))
    assert {path.parent for path in outputs} == {raw_data / "VisHarness-4K"}
    assert {path.name for path in outputs} == {
        "GRES_QA_train.json", "REC8K_QA_train.json", "ReasonSeg_QA_train.json",
    }
    assert not (raw_data / "VisionAgent-4K").exists()


@pytest.mark.parametrize("dataset", ["gres", "reasonseg", "rec8k", "dense200", "seed"])
def test_cli_accepts_readme_command(raw_data, tiny_quotas, monkeypatch, dataset):
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    commands = [
        shlex.split(block.replace("\\\n", " "))
        for block in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL)
        if block.startswith("python -m visharness.data.prepare_qa_data ")
    ]
    command = next(command for command in commands if command[command.index("--dataset") + 1] == dataset)
    assert command[:3] == ["python", "-m", "visharness.data.prepare_qa_data"]
    replacements = {
        "/path/to/GRES": raw_data / "GRES",
        "/path/to/REC-8K": raw_data / "REC-8K",
        "/path/to/ReasonSeg": raw_data / "ReasonSeg",
        "/path/to/Dense200": raw_data / "Dense200",
        "/path/to/VisHarness-4K": raw_data / "VisHarness-4K",
    }
    argv = [str(replacements.get(token, token)) for token in command[2:]]
    monkeypatch.setattr(sys, "argv", argv)
    prep.main()
    if dataset == "seed":
        assert (raw_data / "VisHarness-4K/GRES_QA_train.json").is_file()
    else:
        assert list(_source_roots(raw_data)[dataset].glob("*QA*.json"))


@pytest.mark.parametrize("dataset", ["gres", "reasonseg", "rec8k", "dense200"])
def test_evaluation_uses_exact_root_without_appending_dataset_name(raw_data, tmp_path, dataset):
    source = _source_roots(raw_data)[dataset]
    custom_root = tmp_path / f"{dataset}-storage" / "arbitrary-name"
    custom_root.parent.mkdir()
    source.rename(custom_root)
    outputs = prep.prepare_dataset(dataset, **{f"{dataset}_data_root": custom_root})
    assert outputs
    assert {path.parent for path in outputs} == {custom_root}
    for path in outputs:
        for record in json.loads(path.read_text()):
            assert (custom_root / record["image_path"]).is_file()


def test_seed_cli_supports_independent_source_roots_and_output(raw_data, tiny_quotas, tmp_path, monkeypatch):
    expected = prep.build_seed_manifests(**_seed_roots(raw_data))
    roots = {}
    argv = ["prepare_qa_data", "--dataset", "seed", "--seed", "42"]
    for name in ("gres", "rec8k", "reasonseg"):
        target = tmp_path / f"{name}-storage" / "custom-input-name"
        target.parent.mkdir()
        _source_roots(raw_data)[name].rename(target)
        roots[name] = target
        argv.extend([f"--{name}-data-root", str(target)])
    output = tmp_path / "output-storage" / "VisHarness-4K"
    argv.extend(["--output-dir", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    prep.main()
    for filename, records in expected.items():
        assert json.loads((output / filename).read_text()) == records
        for record in records:
            assert (output / record["image_path"]).read_bytes() == prep._seed_image_source(
                roots, record["image_path"],
            ).read_bytes()
    before = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    prep.main()
    assert {path: path.stat().st_mtime_ns for path in before} == before
    assert not list(raw_data.glob("**/*QA*.json"))


@pytest.mark.parametrize("dataset", ["gres", "reasonseg", "rec8k", "dense200", "seed"])
def test_cli_reports_required_dataset_roots_before_reading_data(monkeypatch, capsys, dataset):
    monkeypatch.setattr(sys, "argv", ["prepare_qa_data", "--dataset", dataset])
    with pytest.raises(SystemExit) as exc:
        prep.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    names = ("gres", "rec8k", "reasonseg") if dataset == "seed" else (dataset,)
    for name in names:
        assert f"--{name}-data-root" in error.split("error:")[-1]
    if dataset == "seed":
        assert "--output-dir" in error.split("error:")[-1]


@pytest.mark.parametrize("missing", ["gres_data_root", "rec8k_data_root", "reasonseg_data_root", "output_dir"])
def test_seed_rejects_missing_explicit_path_before_writing(raw_data, tiny_quotas, missing):
    paths = _prepare_paths(raw_data, "seed")
    del paths[missing]
    with pytest.raises(ValueError, match="--" + missing.replace("_", "-")):
        prep.prepare_dataset("seed", **paths)
    assert not (raw_data / "VisHarness-4K").exists()


def test_cli_rejects_legacy_shared_parent_argument(raw_data, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "prepare_qa_data", "--dataset", "gres", "--data-root", str(raw_data),
    ])
    with pytest.raises(SystemExit) as exc:
        prep.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --data-root" in capsys.readouterr().err


def test_cli_rejects_output_dir_for_evaluation(raw_data, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "prepare_qa_data", "--dataset", "gres", "--gres-data-root", str(raw_data / "GRES"),
        "--output-dir", str(raw_data / "unused"),
    ])
    with pytest.raises(SystemExit) as exc:
        prep.main()
    assert exc.value.code == 2
    assert "--output-dir is only supported with --dataset seed" in capsys.readouterr().err
    assert not list(raw_data.glob("**/*QA*.json"))


def test_seed_checks_all_source_images_before_copying(raw_data, tiny_quotas):
    (raw_data / "ReasonSeg/train/reason.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        prep.prepare_dataset("seed", **_prepare_paths(raw_data, "seed"))
    assert not (raw_data / "VisHarness-4K").exists()


def test_readme_qa_commands_use_explicit_dataset_roots():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    assert "prepare_qa_data --data-root" not in readme
    for name in prep.DATASETS:
        assert f"--dataset {name} --{name}-data-root " in readme
    assert "--output-dir /path/to/VisHarness-4K" in readme
