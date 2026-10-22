"""Compact BF16 inference exports without changing exact-resume checkpoints."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def compact_bf16_hf_checkpoint(checkpoint_dir: str) -> int:
    """Cast FP32 tensors in a BF16-configured HF export, never training shards.

    FSDP keeps FP32 master weights and verl passes those tensors explicitly to
    save_pretrained, so a BF16 config alone does not make its export BF16.
    Inference already loads this model as BF16. Casting the detached, on-disk
    export is equivalent to that load-time cast; optimizer/model .pt files,
    live parameters, RNG, and sampler state are not read or modified.

    Replace one safetensors shard atomically at a time. An interrupted export
    remains loadable (FP32/BF16 shards may coexist) and can be compacted again.
    Float64/FP16 and integer/bool tensors are deliberately left unchanged.
    """
    root = Path(checkpoint_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in {"bfloat16", "torch.bfloat16"}:
        return 0

    converted = 0
    total_size = 0
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            needs_cast = any(source.get_slice(key).get_dtype() == "F32" for key in keys)
            if not needs_cast:
                for key in keys:
                    tensor = source.get_tensor(key)
                    total_size += tensor.numel() * tensor.element_size()
                continue
            metadata = source.metadata()
            tensors = {}
            for key in keys:
                tensor = source.get_tensor(key)
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(dtype=torch.bfloat16)
                    converted += 1
                tensors[key] = tensor
                total_size += tensor.numel() * tensor.element_size()

            temporary = path.with_name(f".{path.name}.bf16-{uuid.uuid4().hex}.tmp")
            try:
                save_file(tensors, str(temporary), metadata=metadata)
                # Keep permissions and leave the original intact on write failure.
                os.chmod(temporary, path.stat().st_mode & 0o777)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
                del tensors

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = index.setdefault("metadata", {})
        if metadata.get("total_size") != total_size:
            metadata["total_size"] = total_size
            temporary = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
                os.chmod(temporary, index_path.stat().st_mode & 0o777)
                os.replace(temporary, index_path)
            finally:
                temporary.unlink(missing_ok=True)

    return converted
