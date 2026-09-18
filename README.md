# VisHarness

This is the official code repository for
[*Train the Agent, Not the Expert: Learning to Harness Heterogeneous Experts
for Multi-Turn Visual Reasoning*](https://arxiv.org/abs/2605.29894).

The current release includes the code for inference, SFT data generation, and
SFT training. The reinforcement learning (RL) code is not included in this
release and will be made available in a future update.

Model weights, datasets, generated trajectories, and training checkpoints are
not included.

## Installation

Clone and install the repository:

```bash
git clone https://github.com/fyw1999/VisHarness.git
cd VisHarness
python -m pip install -e '.[evaluation,tools,dev]'
```

## Visual-tool services

VisHarness uses a controller plus independently deployable tool workers. The
published production workers are `PhraseToPoint`, `PhraseToBoxMask`,
`PointToBoxMask`, `SplitImageIntoPatches`, `SuperResolution`, and
`MergeBoxMask`.

| Tool | Model used by the published worker | Official download |
| --- | --- | --- |
| `PhraseToPoint` | Molmo2-4B | [allenai/Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) |
| `PhraseToBoxMask` | SAM 3 image model (`sam3.pt`) | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| `PointToBoxMask` | SAM 3 image model (`sam3.pt`) | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| `SplitImageIntoPatches` | No learned model; deterministic image tiling | Not applicable |
| `SuperResolution` | Real-ESRGAN (`realesr-general-x4v3.pth` and `realesr-general-wdn-x4v3.pth`) with GFPGAN (`GFPGANv1.3.pth`) | [Real-ESRGAN x4v3](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth), [Real-ESRGAN WDN x4v3](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth), and [GFPGAN v1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) |
| `MergeBoxMask` | No learned model; deterministic coordinate and mask merging | Not applicable |

SAM 3 requires accepting the model's access conditions on Hugging Face before
downloading `sam3.pt`; its official implementation and setup instructions are
available in the [SAM 3 repository](https://github.com/facebookresearch/sam3).
With the published `SuperResolution` defaults, place all three listed
checkpoints in the same directory and retain their original filenames. The
worker derives the WDN and GFPGAN paths from the configured
`realesr-general-x4v3.pth` path.

Edit the paths, environments, GPU assignments, controller address, and model
locations in these templates:

- `tool_server/tool_workers/scripts/launch_scripts/config/local_tools_controller.yaml`
- `tool_server/tool_workers/scripts/launch_scripts/config/remote_controller.yaml`

Start the controller and CPU tools first:

```bash
python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
  --config tool_server/tool_workers/scripts/launch_scripts/config/local_tools_controller.yaml
```

Start GPU workers on the same or another machine:

```bash
python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
  --config tool_server/tool_workers/scripts/launch_scripts/config/remote_controller.yaml
```

## Inference

Start an OpenAI-compatible Qwen3-VL endpoint. `MODEL_PATH` may point to either
the base model or a VisHarness SFT checkpoint:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Thinking \
bash recipe/visharness/scripts/trajectory_runner/start_vllm_qwen3vl_8b.sh
```

Copy and edit the inference template, then run:

```bash
cp recipe/visharness/configs/trajectory_runner/inference_qwen3vl_8b.example.yaml \
  local_inference.yaml

python -m visharness.trajectory_runner --config local_inference.yaml
```

## SFT workflow

The current data-generation recipe uses a cascaded teacher setup. Kimi-K2.5
first processes the complete training set. Qwen3.5-397B-A17B-FP8 then processes
only the samples that were not accepted from the Kimi run.

```text
Training image-text pairs
        |
        v
Kimi-K2.5 trajectory generation
        |
        v
Filter accepted Kimi trajectories ------------------+
        |                                             |
        +--> accepted_ids.jsonl                       |
                     |                                |
                     v                                |
        Qwen3.5 generation on remaining samples       |
                     |                                |
                     v                                |
        Filter accepted Qwen trajectories             |
                     |                                |
                     +--------------------+-----------+
                                          |
                                          v
                         postprocess -> merge -> Swift
                                          |
                                          v
                                     SFT training
```

Run all commands below from the repository root:

```bash
cd /path/to/VisHarness
```

## Prerequisites

Before generating trajectories:

1. Install the project and its runtime dependencies in the intended
   environment.
2. Start the visual-tool controller and all required tool workers.
3. Make the training manifests and benchmark annotations available locally.
4. Update model and dataset paths in the trajectory-runner YAML files for the
   current machine.
5. Start an OpenAI-compatible model endpoint. The provided generation configs
   use `http://localhost:8000/v1` by default.

The model-server scripts require `MODEL_PATH`; optional environment variables
such as `TP_SIZE`, `SERVED_MODEL_NAME`, and `GPU_MEMORY_UTILIZATION` override
the published defaults.

## 1. Generate trajectories with Kimi-K2.5

Review the Kimi configuration first:

[`recipe/visharness/configs/trajectory_runner/data_generation_kimi_k2_5.example.yaml`](recipe/visharness/configs/trajectory_runner/data_generation_kimi_k2_5.example.yaml)

At minimum, verify:

- `model_args.base_url` and `model_args.model_name`
- `dataset_args.dataset_path`
- `dataset_args.save_path`
- `batch_size`, `max_rounds`, and `generation_args`

Start the Kimi vLLM server:

```bash
MODEL_PATH=/path/to/Kimi-K2.5 \
bash recipe/visharness/scripts/trajectory_runner/start_vllm_kimi_k2_5.sh
```

Run trajectory generation in another terminal:

```bash
python -m visharness.trajectory_runner \
  --config recipe/visharness/configs/trajectory_runner/data_generation_kimi_k2_5.example.yaml
```

With the repository defaults, outputs are written to:

```text
outputs/data_generation/Kimi-K2.5-VisionAgent-4K/
```

The trajectory runner writes the following primary artifacts:

| Artifact | Description |
| --- | --- |
| `Kimi-K2.5-VisionAgent-4K_ckpt.jsonl` | One final checkpoint record per processed trajectory, including status, termination reason, final visual result, and runtime metrics. |
| `Kimi-K2.5-VisionAgent-4K_trajectory.jsonl` | Per-turn conversation snapshots eligible for downstream SFT filtering. |
| `images/` | Original and tool-produced visualization images referenced by the snapshots. |
| `benchmark_run.json` | Summary for the latest generation run. |
| `benchmark_runs.jsonl` | Append-only history of generation-run summaries. |
| `resource_trace.jsonl` | Optional GPU/resource samples when resource tracing is enabled. |

An invalid, truncated, or parameter-invalid model turn is not saved as an SFT
target. If the model corrects itself after environment feedback, a later valid
turn may still become an SFT snapshot. Historical assistant errors may remain
in the context, but only the target assistant turn receives loss after Swift
conversion.

### Resume an interrupted Kimi run

Add the existing Kimi checkpoint to `dataset_args.resume_from_ckpt`:

```yaml
dataset_args:
  resume_from_ckpt:
    - outputs/data_generation/Kimi-K2.5-VisionAgent-4K/Kimi-K2.5-VisionAgent-4K_ckpt.jsonl
```

The resumed run skips IDs already present in that checkpoint.

## 2. Filter Kimi trajectories

Set the dataset locations for the current machine:

```bash
export DATASET_ROOT=/path/to/datasets
export KIMI_RUN_DIR=outputs/data_generation/Kimi-K2.5-VisionAgent-4K
```

Run the filter:

```bash
python -m visharness.data.sft_pipeline filter \
  --checkpoint "${KIMI_RUN_DIR}/Kimi-K2.5-VisionAgent-4K_ckpt.jsonl" \
  --trajectory "${KIMI_RUN_DIR}/Kimi-K2.5-VisionAgent-4K_trajectory.jsonl" \
  --output-dir "${KIMI_RUN_DIR}" \
  --rec8k-annotations "${DATASET_ROOT}/REC-8K/annotations.json" \
  --gres-dataset-root "${DATASET_ROOT}/GRES" \
  --reasonseg-dataset-root "${DATASET_ROOT}/ReasonSeg/train"
```

Default acceptance thresholds are:

| Dataset | Metric | Threshold |
| --- | --- | ---: |
| GRES | Mask IoU | 0.70 |
| ReasonSeg | Mask IoU | 0.70 |
| REC-8K | Relative count error | 0.30 |
| All applicable tasks | Aspect-ratio tolerance | 0.05 |

A trajectory is accepted only if the model submitted a final answer and its
task metric passed the configured threshold. A valid final answer submitted on
the last allowed turn is accepted. A trajectory that produced a valid visual
result but never submitted a final answer is rejected.

The filter writes:

| Artifact | Description |
| --- | --- |
| `accepted_checkpoints.jsonl` | Complete checkpoint records for accepted trajectories. |
| `rejected_checkpoints.jsonl` | Complete checkpoint records for rejected trajectories. |
| `accepted_sft_snapshots.jsonl` | Accepted SFT snapshots consumed by `build`. |
| `accepted_ids.jsonl` | Compact accepted-ID list used by cascaded generation. |
| `filter_decisions.jsonl` | Per-trajectory decision, metric, and rejection reason. |
| `filter_report.json` | Aggregate filter counts, thresholds, and dataset metadata. |

## 3. Generate remaining trajectories with Qwen3.5

Stop the Kimi model server before starting Qwen if both use the same GPUs and
port. Review the Qwen configuration:

[`recipe/visharness/configs/trajectory_runner/data_generation_qwen3_5_397b.example.yaml`](recipe/visharness/configs/trajectory_runner/data_generation_qwen3_5_397b.example.yaml)

Its `dataset_args.resume_from_ckpt` should include the Kimi
`accepted_ids.jsonl`. This causes Qwen to process only the samples that were not
accepted from the Kimi run.

Start Qwen3.5:

```bash
MODEL_PATH=/path/to/Qwen3.5-397B-A17B-FP8 \
bash recipe/visharness/scripts/trajectory_runner/start_vllm_qwen3_5_397b.sh
```

Run trajectory generation:

```bash
python -m visharness.trajectory_runner \
  --config recipe/visharness/configs/trajectory_runner/data_generation_qwen3_5_397b.example.yaml
```

The default output directory is:

```text
outputs/data_generation/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/
```

### Resume an interrupted Qwen run

Keep the Kimi accepted-ID file and add Qwen's own checkpoint:

```yaml
dataset_args:
  resume_from_ckpt:
    - outputs/data_generation/Kimi-K2.5-VisionAgent-4K/accepted_ids.jsonl
    - outputs/data_generation/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_ckpt.jsonl
```

The first file skips samples already solved by Kimi. The second skips samples
already processed by the interrupted Qwen run.

## 4. Filter Qwen trajectories

```bash
export QWEN_RUN_DIR=outputs/data_generation/Qwen3.5-397B-A17B-FP8-VisionAgent-4K

python -m visharness.data.sft_pipeline filter \
  --checkpoint "${QWEN_RUN_DIR}/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_ckpt.jsonl" \
  --trajectory "${QWEN_RUN_DIR}/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_trajectory.jsonl" \
  --output-dir "${QWEN_RUN_DIR}" \
  --rec8k-annotations "${DATASET_ROOT}/REC-8K/annotations.json" \
  --gres-dataset-root "${DATASET_ROOT}/GRES" \
  --reasonseg-dataset-root "${DATASET_ROOT}/ReasonSeg/train"
```

After filtering, both run directories must contain an
`accepted_sft_snapshots.jsonl` file.

## 5. Build the final SFT dataset

`build` consumes only explicitly filtered `accepted_sft_snapshots.jsonl`
files. It does not read raw checkpoints or score trajectories again.

Use a new or empty output directory for each build:

```bash
export BUILD_DIR=outputs/sft_data/Kimi-Qwen-Merged-$(date +%Y%m%d)

python -m visharness.data.sft_pipeline build \
  --source "Kimi-K2.5=${KIMI_RUN_DIR}/accepted_sft_snapshots.jsonl" \
  --image-root "Kimi-K2.5=${KIMI_RUN_DIR}" \
  --source "Qwen3.5-397B-A17B-FP8=${QWEN_RUN_DIR}/accepted_sft_snapshots.jsonl" \
  --image-root "Qwen3.5-397B-A17B-FP8=${QWEN_RUN_DIR}" \
  --output-dir "${BUILD_DIR}" \
  --merged-name merged_sft_data.jsonl \
  --swift-name merged_sft_data_swift_cmd.jsonl \
  --image-max-token-num 2048 \
  --max-total-raw-image-patches 56000
```

The build consists of three stages:

1. **Postprocess** normalizes the system prompt and message content, removes
   generation-only tool-specific guidance while retaining the shared visual
   verification prompt, removes `SubmitFinalAnswer` artifacts, and validates
   every SFT target.
2. **Merge** requires disjoint trajectory IDs across sources, preserves sample
   identity, adds source metadata, and copies every referenced image.
3. **Swift conversion** emits `ms-swift` messages, assigns loss only to the
   explicit target assistant output, and excludes samples exceeding 56,000
   aggregate pre-merge Qwen3-VL image patches while retaining the merged
   source snapshot for audit.

The output layout is:

```text
<BUILD_DIR>/
├── postprocessed/
│   ├── Kimi-K2.5/
│   │   └── sft_postprocessed.jsonl
│   └── Qwen3.5-397B-A17B-FP8/
│       └── sft_postprocessed.jsonl
├── images/
├── merged_sft_data.jsonl
├── merged_sft_data_swift_cmd.jsonl
├── visual_budget_decisions.jsonl
├── merge_report.json
└── sft_pipeline_report.json
```

The final training dataset consists of
`merged_sft_data_swift_cmd.jsonl` **and** `images/`. The Swift file contains
paths to images under the build directory and is not a standalone text-only
dataset.

Keep `merged_sft_data.jsonl`, `merge_report.json`, and
`sft_pipeline_report.json` for auditing and reproducibility. The
`postprocessed/` directory contains per-source intermediate snapshots and is
useful when diagnosing source-specific failures.

## 6. Launch SFT training

The provided full-parameter training recipe is:

[`recipe/visharness/scripts/sft/train_qwen3vl_8b_thinking_full.sh`](recipe/visharness/scripts/sft/train_qwen3vl_8b_thinking_full.sh)

The dataset must point to the Swift file produced by the current build:

```text
<BUILD_DIR>/merged_sft_data_swift_cmd.jsonl
```

Start training:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Thinking \
DATASET_PATH="${BUILD_DIR}/merged_sft_data_swift_cmd.jsonl" \
OUTPUT_DIR=outputs/sft/Qwen3-VL-8B-Thinking \
bash recipe/visharness/scripts/sft/train_qwen3vl_8b_thinking_full.sh
```

The current recipe uses:

- eight GPUs;
- full-parameter tuning of Qwen3-VL-8B-Thinking;
- BF16 and DeepSpeed ZeRO-3;
- FlashAttention 2, gradient checkpointing, and Liger Kernel;
- `IMAGE_MAX_TOKEN_NUM=2048` and `max_length=25848`;
- sequence parallel size 2;
- per-device batch size 1 with 32 gradient-accumulation steps; and
- an optional logging backend selected with `REPORT_TO`.

By default, checkpoints are written to:

```text
outputs/sft/Qwen3-VL-8B-Thinking/
```

## Inspect a run

Inspect filter summaries:

```bash
python -m json.tool "${KIMI_RUN_DIR}/filter_report.json"
python -m json.tool "${QWEN_RUN_DIR}/filter_report.json"
```

Inspect the build summary:

```bash
python -m json.tool "${BUILD_DIR}/sft_pipeline_report.json"
```

Before rebuilding, remove only the corresponding build output directory or
choose a new one. Do not remove the original model-run checkpoints,
trajectories, filtered snapshots, or image directories; they are the inputs
required to reproduce the build.

For implementation details and individual pipeline commands, see
[`visharness/data/sft_pipeline/README.md`](visharness/data/sft_pipeline/README.md).
