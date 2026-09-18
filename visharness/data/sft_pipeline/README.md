# VisHarness SFT data pipeline

This package owns the offline path from trajectory-runner outputs to an
ms-swift training file:

1. `filter.py` scores the latest checkpoint for each trajectory and retains
   the latest matching SFT snapshots. A trajectory is eligible for task
   scoring only after the model explicitly submits its final answer; merely
   exhausting the turn budget with a valid visual result is rejected.
2. `postprocess.py` normalizes message content and enforces the target-turn
   schema without rejecting harmless errors in historical assistant turns.
3. `merge.py` verifies that filtered model sources have disjoint trajectory
   IDs, then combines them without changing sample IDs or image paths.
4. `swift.py` converts messages, applies loss only to the explicit target
   assistant turn, and excludes samples whose aggregate pre-merge Qwen3-VL
   image-patch count exceeds the configured training budget.
5. `pipeline.py` composes postprocessing, merging, and Swift conversion from
   explicitly filtered model sources. It never scores raw checkpoints again.

The filter stage writes semantically named artifacts:

- `accepted_checkpoints.jsonl` and `rejected_checkpoints.jsonl` retain the
  complete checkpoint records for inspection and visualization.
- `accepted_sft_snapshots.jsonl` contains the SFT snapshots selected for
  postprocessing.
- `accepted_ids.jsonl`, `filter_decisions.jsonl`, and `filter_report.json`
  provide compact cascade-generation and audit metadata.

Every stage takes explicit input/output paths, validates its input, and writes
JSON/JSONL outputs atomically. `SubmitFinalAnswer` is a generation-only tool:
valid calls have already become `<answer>...</answer>` in trajectory
serialization, and any remaining reference is sanitized or rejected before
the Swift file is written.

Run an individual stage with:

```bash
python -m visharness.data.sft_pipeline filter --help
python -m visharness.data.sft_pipeline postprocess --help
python -m visharness.data.sft_pipeline merge --help
python -m visharness.data.sft_pipeline swift --help
python -m visharness.data.sft_pipeline build --help
```

Run `filter` once after each model finishes. In a cascaded Kimi-to-Qwen job,
Qwen uses Kimi's compact `accepted_ids.jsonl` as one of its
`resume_from_ckpt` inputs. After Qwen finishes and is filtered, both models'
`accepted_sft_snapshots.jsonl` files become the inputs to `build`.

Pass every filtered source and its image root directly to `build`. Source and
image-root aliases must match exactly:

```bash
python -m visharness.data.sft_pipeline build \
  --source kimi=/path/to/kimi/accepted_sft_snapshots.jsonl \
  --image-root kimi=/path/to/kimi \
  --source qwen=/path/to/qwen/accepted_sft_snapshots.jsonl \
  --image-root qwen=/path/to/qwen \
  --output-dir /path/to/merged-run \
  --merged-name merged_sft_data.jsonl \
  --swift-name merged_sft_data_swift.jsonl \
  --image-max-token-num 2048 \
  --max-total-raw-image-patches 56000
```

`IMAGE_MAX_TOKEN_NUM` is a per-image limit, so it does not protect training
from a sample containing many individually valid images. The build command
uses the same Qwen3-VL smart-resize geometry as training and sums the raw ViT
patches across every image occurrence. By default, a sample is written to the
Swift training file only when that sum is at most 56,000. The merged snapshot
file remains complete, and `visual_budget_decisions.jsonl` records the image
count, raw patch count, merged visual-token count, and decision for every
snapshot.

Per-source normalized snapshots are written under
`<output-dir>/postprocessed/<alias>/sft_postprocessed.jsonl` before merging.

The four scripts under `key_scripts/` remain as thin compatibility entry
points, but new jobs should invoke this package directly.
