import hashlib
import json

import pytest
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from visharness.rl_trainer.checkpoint_export import compact_bf16_hf_checkpoint
from visharness.rl_trainer.visharness_trainer import VisHarnessTrainer


def _config(path, dtype="bfloat16"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"dtype": dtype}))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compaction_matches_bf16_inference_and_preserves_other_dtypes(tmp_path):
    _config(tmp_path)
    original = {
        "weight": torch.randn(128, 128),
        "bias": torch.randn(128),
        "bf16": torch.randn(4, dtype=torch.bfloat16),
        "fp16": torch.randn(4, dtype=torch.float16),
        "fp64": torch.randn(4, dtype=torch.float64),
        "integer": torch.tensor([1, 2]),
        "boolean": torch.tensor([True, False]),
    }
    path = tmp_path / "model.safetensors"
    save_file(original, str(path), metadata={"format": "pt"})
    path.chmod(0o640)
    old_size = path.stat().st_size

    assert compact_bf16_hf_checkpoint(str(tmp_path)) == 2
    loaded = load_file(str(path))
    for key, tensor in original.items():
        expected = tensor.bfloat16() if tensor.dtype == torch.float32 else tensor
        assert loaded[key].dtype == expected.dtype
        assert torch.equal(loaded[key], expected)
    inputs = torch.randn(2, 128, dtype=torch.bfloat16)
    expected_output = torch.nn.functional.linear(
        inputs, original["weight"].bfloat16(), original["bias"].bfloat16()
    )
    assert torch.equal(
        torch.nn.functional.linear(inputs, loaded["weight"], loaded["bias"]),
        expected_output,
    )
    assert original["weight"].dtype == torch.float32
    assert path.stat().st_size < old_size * 0.6
    assert path.stat().st_mode & 0o777 == 0o640
    with safe_open(path, framework="pt") as source:
        assert source.metadata() == {"format": "pt"}


@pytest.mark.parametrize("config", [{"dtype": "float32"}, {}, {"torch_dtype": "float16"}])
def test_non_bf16_exports_are_not_reduced(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"not inspected for non-BF16 exports")
    before = path.read_bytes()
    assert compact_bf16_hf_checkpoint(str(tmp_path)) == 0
    assert path.read_bytes() == before


def test_legacy_torch_dtype_config_is_supported(tmp_path):
    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(4)}, str(path))
    assert compact_bf16_hf_checkpoint(str(tmp_path)) == 1
    assert load_file(str(path))["weight"].dtype == torch.bfloat16


def test_sharded_index_is_updated_and_repeated_compaction_is_noop(tmp_path):
    _config(tmp_path)
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a": torch.ones(8)}, str(first))
    save_file({"b": torch.ones(4, dtype=torch.bfloat16)}, str(second))
    index = {
        "metadata": {"total_size": 40, "extra": "preserved"},
        "weight_map": {"a": first.name, "b": second.name},
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    assert compact_bf16_hf_checkpoint(str(tmp_path)) == 1
    after = json.loads(index_path.read_text())
    assert after["metadata"] == {"total_size": 24, "extra": "preserved"}
    assert after["weight_map"] == index["weight_map"]
    before = [(p.stat().st_mtime_ns, _digest(p)) for p in (first, second, index_path)]
    assert compact_bf16_hf_checkpoint(str(tmp_path)) == 0
    assert before == [(p.stat().st_mtime_ns, _digest(p)) for p in (first, second, index_path)]


def test_failed_conversion_preserves_original_and_removes_temporary_files(tmp_path, monkeypatch):
    _config(tmp_path)
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.randn(8)}, str(path))
    before = _digest(path)

    def fail(tensors, filename, **kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr("visharness.rl_trainer.checkpoint_export.save_file", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        compact_bf16_hf_checkpoint(str(tmp_path))
    assert _digest(path) == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_archiving_compacts_only_hf_export_and_preserves_exact_resume_state(tmp_path):
    role = tmp_path / "global_step_10" / "actor"
    source = role / "huggingface"
    _config(source)
    original = torch.randn(8)
    save_file({"weight": original}, str(source / "model.safetensors"))
    resume = role / "model_world_size_1_rank_0.pt"
    optimizer = role / "optim_world_size_1_rank_0.pt"
    torch.save({"weight": original}, resume)
    torch.save({"exp_avg": original.clone()}, optimizer)
    before = {_file: _digest(_file) for _file in (resume, optimizer)}
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 10
    trainer.config = OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path)}})

    trainer._archive_current_hf_checkpoint()

    archive = tmp_path / "archived" / "global_step_10" / "model.safetensors"
    assert torch.equal(load_file(str(archive))["weight"], original.bfloat16())
    assert load_file(str(source / "model.safetensors"))["weight"].dtype == torch.bfloat16
    for path, digest in before.items():
        assert _digest(path) == digest
    assert torch.equal(torch.load(resume, weights_only=True)["weight"], original)
    assert torch.load(resume, weights_only=True)["weight"].dtype == torch.float32


def test_transformers_bf16_load_has_identical_logits_before_and_after_export(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).bfloat16()
    # Reproduce verl's BF16 save model plus explicit FP32 FSDP master state.
    master = {
        key: value.float() + torch.full_like(value.float(), 0.00001)
        for key, value in model.state_dict().items()
    }
    model.save_pretrained(tmp_path, state_dict=master)
    inputs = torch.tensor([[1, 2, 3, 4]])
    before = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    with torch.no_grad():
        expected = before(inputs).logits.clone()
    assert compact_bf16_hf_checkpoint(str(tmp_path)) > 0
    after = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    with torch.no_grad():
        actual = after(inputs).logits
    assert torch.equal(actual, expected)
    for key, value in before.state_dict().items():
        assert torch.equal(after.state_dict()[key], value)
    assert all(tensor.dtype == torch.float32 for tensor in master.values())
