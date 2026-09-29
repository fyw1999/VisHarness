# VisHarness

This package contains the project-owned implementation of VisHarness.

Core responsibilities:

- `data/`: dataset loading and conversion
- `agent_loop/`: asynchronous multi-turn visual-agent rollout
- `prompts/`: canonical system prompts, tool guidance, and OpenAI tool schemas
- `tools/`: adapters for the existing visual-tool controller
- `rewards/`: trajectory reward plus replayed step-potential and delayed-credit functions
- `trainer/`: trajectory filtering and per-turn training transforms

Experiment configurations and launch scripts live in `recipe/visharness/`.
The upstream verl checkout lives in `verl/` and should remain as close to
upstream as possible.

## Prepare verl data

Generate a small smoke-test dataset directly from the raw VisionAgent-4K,
REC-8K, GRES, and ReasonSeg annotations:

```bash
python visharness/data/prepare_verl_data.py \
  --output-dir /tmp/visharness_verl_smoke \
  --max-samples 60 \
  --overwrite true
```

Generate the complete train/validation parquet files:

```bash
python visharness/data/prepare_verl_data.py --overwrite true
```

The complete files are written to
`training_data/GRPO/verl_visharness/{train,val}.parquet` by default.
By default, parquet rows contain only the user message. `VisHarnessDataset`
injects the current canonical system prompt before prompt-length filtering and
rollout. Pass `--embed-system-prompt` only when a self-contained parquet file
is required.

Build the fixed 300-sample checkpoint-selection set from the official
ReasonSeg, GRES, and REC-8K validation splits:

```bash
python -m visharness.data.prepare_official_validation --overwrite true
```

This writes `training_data/GRPO/verl_visharness_official_val/val.parquet`,
three frozen 100-sample manifests, resized images, and
`selection_report.json`. The sampler prioritizes distinct images, excludes
train-image overlap, preserves task-specific difficulty strata, and applies
the REC-8K contiguous-frame filter. The training launcher uses this parquet by
default and saves per-checkpoint validation summaries under
`<checkpoint_dir>/validation/global_step_<N>/`.

The trajectory runner, verl tool adapters, and data preparation code all read
their prompt/tool definitions from `visharness/prompts/visual_agent.py`.
Changing the canonical RL system prompt takes effect when a new training or
validation process loads the parquet; the parquet does not need to be rebuilt.
