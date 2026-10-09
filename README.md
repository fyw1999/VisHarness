> **Project status:** Our paper has been accepted to NeurIPS 2026. We are
> currently organizing the release. All source code is now publicly available;
> the model weights and datasets are the remaining artifacts to be released. We
> expect to complete the full open-source release in the near future.

# Environment Setup

## Step 1: clone the repository and initialize verl

```bash
git clone --recurse-submodules https://github.com/fyw1999/VisHarness.git
cd VisHarness
```

If the repository was cloned without submodules, initialize verl before
creating the environment:

```bash
git submodule update --init --recursive
```

## Step 2: create the main environment

Run the following commands from the repository root. This is required because
the environment installs the checked-out verl submodule in editable mode.

```bash
conda env create -f environment/VisHarness.yml
conda activate VisHarness
python -m pip check
```

`VisHarness` is the main environment for data generation, evaluation, SFT, RL,
the tool controller, and the two local CPU tools. The GPU-backed visual experts
use isolated environments because their model stacks have different Python and
runtime requirements.

## Step 3: create the visual expert environments

Create all three expert environments with:

```bash
bash environment/create_expert_envs.sh
```

The script creates `Molmo`, `SAM`, and `SuperResolution` from the versioned
files under `environment/experts/`. Existing environments are left unchanged.
Model checkpoints are not downloaded by these environment files and must be
configured separately as described below.

## Step 4: verify the environments

```bash
conda run -n VisHarness python -c "import torch, vllm, verl, visharness"
conda run -n Molmo python -c "import torch, transformers, vllm"
conda run -n SAM python -c "import torch, sam3"
conda run -n SuperResolution python -c "import basicsr, gfpgan, realesrgan"
```

# Visual Expert Server Setup

VisHarness exposes six visual tools. The default model or processing backend
used by each tool is summarized below:

| Experts/Tools | Model or backend | Purpose |
| --- | --- | --- |
| `PhraseToPoint` | [Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) | Returns the center points of all objects matching a complex text description, including attributes, relative positions, and other constraints. |
| `PhraseToBoxMask` | [SAM 3](https://huggingface.co/facebook/sam3) (`sam3.pt`) | Returns bounding boxes and masks for all objects matching a simple noun-phrase description. |
| `PointToBoxMask` | [SAM 3](https://huggingface.co/facebook/sam3) (`sam3.pt`) | Converts supplied center points into object bounding boxes and masks. |
| `SuperResolution` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (`realesr-general-x4v3`) with [GFPGAN](https://github.com/TencentARC/GFPGAN) v1.3 face enhancement | Produces 4x super-resolved images and restores detected faces. |
| `SplitImageIntoPatches` | Deterministic CPU processing; no learned model | Splits an image into overlapping patches so that small targets occupy a larger proportion of each relevant patch, and produces an overview. |
| `MergeBoxMask` | Rule-based CPU geometry and mask processing; no learned model | Merges results from different patches and maps the final bounding boxes and masks back to their locations in the original image. |

After downloading the model checkpoints, set the corresponding `model-path`
values in `tool_server/tool_workers/scripts/launch_scripts/config/remote_tools.yaml`.
The provided `Molmo`, `SAM`, and `SuperResolution` environments match the
default `conda_env` values in that file. If custom environment names are used,
update the corresponding values before launching the workers.
`SplitImageIntoPatches` and `MergeBoxMask` run as local CPU tools and therefore
do not require model checkpoints.

First, configure the controller and local visual tools in
`tool_server/tool_workers/scripts/launch_scripts/config/controller_local_tools.yaml`.
Open a dedicated terminal session and run the following command from the project
root to start the controller and local tools:

```bash
python tool_server/tool_workers/scripts/launch_scripts/start_controller_local_tools.py
```

Next, configure the visual expert model settings in
`tool_server/tool_workers/scripts/launch_scripts/config/remote_tools.yaml`.
Open another dedicated terminal session and start the visual expert workers:

```bash
python tool_server/tool_workers/scripts/launch_scripts/start_remote_tools.py
```

If a single server does not have enough GPU resources, the visual expert models
can run on a second server, while the controller, local visual tools, and the
agent training or inference process run on the first server. To set up the
second server, copy the complete `tool_server/` directory from the project root
to that server, then configure `remote_tools.yaml` and run
`start_remote_tools.py` as described above. Configure network routing or port
forwarding in advance so that the visual expert server can reach the controller
through the controller address specified in `remote_tools.yaml`. For example,
the two servers can be connected and the required ports forwarded through
[Tailscale](https://tailscale.com/).

Keep the controller, local tools, and remote visual expert workers running
throughout SFT data generation, inference, and RL training, because all three
workflows call this service. Standalone SFT training consumes the prepared
dataset and does not require the visual expert service.

# SFT Data Generation

The SFT data generation pipeline starts with a seed pool of 4,499 image-text
pairs drawn from three tasks: GRES, ReasonSeg, and referring expression
counting. Kimi-K2.5 first generates one tool-augmented trajectory for every
image-text pair. During generation, the model invokes the visual tools and uses
their outputs to decide the next step. Each completed trajectory is compared
with the ground truth, and successful trajectories are selected using a hard
threshold. Failed examples are then regenerated with Qwen-397B and filtered to
retain only correct trajectories. Finally, the correct trajectories produced
by the two models are merged and converted into the required SFT training
format.

## Step 1: generation with Kimi-K2.5

In `recipe/visharness/scripts/trajectory_runner/start_vLLM_Kimi_K2.5.bash`, set `MODEL_PATH` to the path of your downloaded Kimi-K2.5 model:

```bash
MODEL_PATH="/path/to/downloaded/Kimi-K2.5"
```

Then run the launch script from the project root:

```bash
bash recipe/visharness/scripts/trajectory_runner/start_vLLM_Kimi_K2.5.bash
```

After the vLLM server is running, configure the data generation settings in
`recipe/visharness/configs/trajectory_runner/Kimi_2.5_online_config.yaml`.
Under `dataset_args`, set `dataset_path` to the path of the seed data pool and
set `save_path` to the path where the generated results should be saved:

```yaml
dataset_args:
  dataset_path: /path/to/seed/data
  save_path: training_data/SFT/Kimi-K2.5-VisionAgent-4K
  resume_from_ckpt:
```

A relative `save_path` is resolved from the project root. An absolute
`save_path` can also be provided and is used unchanged.

If data generation is interrupted, set `resume_from_ckpt` to the path of the
generated `*_ckpt.jsonl` checkpoint file to resume from the previous progress:

```yaml
dataset_args:
  resume_from_ckpt: training_data/SFT/Kimi-K2.5-VisionAgent-4K/Kimi-K2.5-VisionAgent-4K_ckpt.jsonl
```

Like `save_path`, a relative `resume_from_ckpt` path is resolved from the
project root, while an absolute path is used unchanged. The trajectory runner
raises an error if a configured checkpoint file does not exist.

After completing the configuration, run the trajectory runner from the project
root with the Kimi-K2.5 configuration:

```bash
python -m visharness.trajectory_runner \
  --config recipe/visharness/configs/trajectory_runner/Kimi_2.5_online_config.yaml
```

Generating the full dataset can take a long time—approximately 10 hours on our
hardware. The actual runtime depends on the available hardware.

## Step 2: filter Kimi-K2.5 trajectories

After Kimi-K2.5 finishes generating trajectories, compare the generated
results with the ground-truth annotations and retain the successful
trajectories:

```bash
python -m visharness.data.sft_pipeline filter \
  --checkpoint training_data/SFT/Kimi-K2.5-VisionAgent-4K/Kimi-K2.5-VisionAgent-4K_ckpt.jsonl \
  --trajectory training_data/SFT/Kimi-K2.5-VisionAgent-4K/Kimi-K2.5-VisionAgent-4K_trajectory.jsonl \
  --output-dir training_data/SFT/Kimi-K2.5-VisionAgent-4K \
  --rec8k-annotations /path/to/REC-8K/annotations.json \
  --gres-dataset-root /path/to/GRES \
  --reasonseg-dataset-root /path/to/ReasonSeg/train
```

Set `--rec8k-annotations` to the REC8K annotation JSON file,
`--gres-dataset-root` to the GRES dataset directory, and
`--reasonseg-dataset-root` to the ReasonSeg training-data directory.

The command writes the following files to `--output-dir`:

- `accepted_sft_snapshots.jsonl`: SFT snapshots for the trajectories that
  Kimi-K2.5 completed successfully. These snapshots are used by the later
  `build` step.
- `accepted_checkpoints.jsonl`: Checkpoints for successful samples.
- `rejected_checkpoints.jsonl`: Checkpoints for failed samples.
- `accepted_ids.jsonl`: IDs of samples already completed successfully by
  Kimi-K2.5. A later Qwen data-generation run can use these IDs to skip those
  samples.
- `filter_decisions.jsonl`: The filtering decision and reason for each
  trajectory.
- `filter_report.json`: Aggregate filtering statistics. For example, one of
  our runs retained 2,573 successful trajectories and rejected 1,926 failed
  trajectories.

## Step 3: generation with Qwen3.5-397B

In
`recipe/visharness/scripts/trajectory_runner/start_vLLM_qwen_3.5_397B-A17B-FP8.bash`,
set `MODEL_PATH` to the path of your downloaded Qwen3.5-397B-A17B-FP8 model:

```bash
MODEL_PATH="/path/to/downloaded/Qwen3.5-397B-A17B-FP8"
```

Then run the launch script from the project root to start Qwen3.5-397B:

```bash
bash recipe/visharness/scripts/trajectory_runner/start_vLLM_qwen_3.5_397B-A17B-FP8.bash
```

After the vLLM server is running, configure
`recipe/visharness/configs/trajectory_runner/Qwen3.5-397B-A17B-FP8_online_config.yaml`.
For the initial Qwen3.5-397B generation run, set `resume_from_ckpt` to the
`accepted_ids.jsonl` file produced by the Kimi-K2.5 filtering step. This makes
Qwen3.5-397B skip the samples that Kimi-K2.5 has already completed
successfully:

```yaml
dataset_args:
  resume_from_ckpt:
    - training_data/SFT/Kimi-K2.5-VisionAgent-4K/accepted_ids.jsonl
```

If Qwen3.5-397B generation is interrupted, restart it with both the Kimi-K2.5
accepted IDs and the Qwen3.5-397B checkpoint:

```yaml
dataset_args:
  resume_from_ckpt:
    - training_data/SFT/Kimi-K2.5-VisionAgent-4K/accepted_ids.jsonl
    - training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_ckpt.jsonl
```

After completing the configuration, run the trajectory runner from the
project root:

```bash
python -m visharness.trajectory_runner \
  --config recipe/visharness/configs/trajectory_runner/Qwen3.5-397B-A17B-FP8_online_config.yaml
```

The Qwen3.5-397B generation process takes approximately 4 hours on our
hardware. The actual runtime depends on the available hardware.

## Step 4: filter Qwen3.5-397B trajectories

After Qwen3.5-397B finishes generating trajectories, filter the generated
results against the ground-truth annotations:

```bash
python -m visharness.data.sft_pipeline filter \
  --checkpoint training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_ckpt.jsonl \
  --trajectory training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/Qwen3.5-397B-A17B-FP8-VisionAgent-4K_trajectory.jsonl \
  --output-dir training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K \
  --rec8k-annotations /path/to/REC-8K/annotations.json \
  --gres-dataset-root /path/to/GRES \
  --reasonseg-dataset-root /path/to/ReasonSeg/train
```

Set `--rec8k-annotations` to the REC8K annotation JSON file,
`--gres-dataset-root` to the GRES dataset directory, and
`--reasonseg-dataset-root` to the ReasonSeg training-data directory.

## Step 5: build the SFT dataset

After filtering the Qwen3.5-397B trajectories, build the final SFT dataset.
This step performs the following operations:

1. Post-process each data source.
2. Merge the two datasets and their images.
3. Filter samples that may cause training OOM errors based on the visual patch
   limit.
4. Convert the merged data to the Swift SFT format.

Run the following command from the project root:

```bash
python -m visharness.data.sft_pipeline build \
  --source Kimi-K2.5=training_data/SFT/Kimi-K2.5-VisionAgent-4K/accepted_sft_snapshots.jsonl \
  --image-root Kimi-K2.5=training_data/SFT/Kimi-K2.5-VisionAgent-4K \
  --source Qwen3.5-397B-A17B-FP8=training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K/accepted_sft_snapshots.jsonl \
  --image-root Qwen3.5-397B-A17B-FP8=training_data/SFT/Qwen3.5-397B-A17B-FP8-VisionAgent-4K \
  --output-dir training_data/SFT/KimiK2.5-Qwen3.5_397B_FP8-Merged-VisionAgent-4K-20260927-patch56k \
  --merged-name merged_sft_data.jsonl \
  --swift-name merged_sft_data_swift_cmd.jsonl \
  --image-max-token-num 2048 \
  --max-total-raw-image-patches 56000
```

The two visual-budget arguments control image preprocessing and memory usage:

- `--image-max-token-num` sets the maximum number of visual tokens produced
  for each individual image after Qwen3-VL smart resizing. Images are resized
  as needed to remain within this per-image limit.
- `--max-total-raw-image-patches` sets the maximum total number of raw vision
  patches across all image occurrences in one training sample. Samples that
  exceed this limit are excluded from the Swift training file to reduce the
  risk of training OOM errors.

The generated SFT dataset is stored in
`training_data/SFT/KimiK2.5-Qwen3.5_397B_FP8-Merged-VisionAgent-4K-20260927-patch56k`.
Within this directory, `merged_sft_data_swift_cmd.jsonl` is the final training
dataset in the format required by the Swift training framework. Use this file
as the dataset input for SFT training.

# SFT Training

In `recipe/visharness/scripts/sft/train_qwen3vl_8b_thinking_full.sh`, set
`MODEL_PATH` to the path of the downloaded model to fine-tune, such as a
Qwen3-VL model:

```bash
MODEL_PATH="/path/to/downloaded/Qwen3-VL-8B-Thinking"
```

Then run the training script from the project root:

```bash
bash recipe/visharness/scripts/sft/train_qwen3vl_8b_thinking_full.sh
```

With the configuration above, training for one epoch on our 13,323-sample SFT
dataset takes approximately 2 hours and 45 minutes (roughly 3 hours) on eight
NVIDIA A100-SXM4-80GB GPUs. The actual runtime depends on the hardware, dataset
size, and training configuration.

To inspect the data that will be passed to the model during SFT, use the
`debug_swift_sft.py` helper included in the checkout after setting its
`model_id_or_path` and `dataset_path` values.

The script prints one encoded SFT training sample to the console, including the
complete model input and the supervised portion that is used to compute the
training loss.

# Inference after SFT

## Step 1: deploy the SFT model with vLLM

After SFT training finishes, edit
`recipe/visharness/scripts/trajectory_runner/start_vLLM_VisHarness.bash` and set
`MODEL` to your SFT checkpoint. By default, use the last checkpoint saved by
training, such as `checkpoint-105` for the run described above:

```bash
MODEL="${MODEL:-checkpoints/sft/Qwen3-VL-8B-Thinking/<sft-run>/checkpoint-105}"
```

Replace `<sft-run>` with your training run directory. The script automatically
locates the project root and resolves relative `MODEL` paths against it,
regardless of the current working directory. Absolute paths are also accepted.
You can override the checkpoint through the `MODEL` environment variable
without editing the script. The checkpoint must exist at the selected path.

Also adjust `CUDA_VISIBLE_DEVICES`, `--tensor-parallel-size`, and
`--data-parallel-size` to match your GPUs. The example in the script uses four
GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`) with:

```bash
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
```

This configuration runs four independent model replicas, one per GPU, to
maximize concurrent request throughput when the model fits on a single GPU.
Each replica can process multiple requests; data parallel size is not the
maximum number of concurrent trajectories. For the 8B model on four A100 80GB
GPUs, this is a throughput-oriented starting point, not a guarantee of the
highest throughput for every workload. With fewer GPUs or less memory, reduce
the number of replicas or increase tensor parallel size. The product of the
tensor parallel and data parallel sizes must match the number of GPUs used.

Start the server from the project root:

```bash
bash recipe/visharness/scripts/trajectory_runner/start_vLLM_VisHarness.bash
```

## Step 2: run inference

In `recipe/visharness/configs/trajectory_runner/VisHarness.yaml`, keep
`mode: inference`, set `model_args.tokenizer_path` to the same SFT checkpoint,
and use `model_args.model_name: VisHarness` to match the server's
`--served-model-name`. Set `model_args.base_url` to the vLLM API address
(`http://localhost:8000/v1` when running locally).

Configure `dataset_args.task_names`, `split`, `dataset_path`, and `save_path`
for the dataset to evaluate. `batch_size` controls the maximum number of
concurrent trajectories; adjust it to make use of the deployed replicas.
Keep the visual expert services running, then launch inference from the
project root in another terminal:

```bash
python -m visharness.trajectory_runner \
  --config recipe/visharness/configs/trajectory_runner/VisHarness.yaml
```

# RL Training

## Step 1: prepare the training and validation parquet files

RL training uses all 4,499 image-text pairs from the VisionAgent-4K training
manifests. Generate the training parquet from the project root:

```bash
python visharness/data/prepare_verl_data.py \
  --source-root /path/to/VisionAgent-4K \
  --rec8k-anno-path /path/to/REC-8K/annotations.json \
  --gres-data-root /path/to/GRES \
  --reasonseg-data-root /path/to/ReasonSeg/train \
  --output-dir training_data/GRPO/verl_visharness \
  --val-ratio 0 \
  --shuffle true \
  --seed 42 \
  --overwrite true
```

`--val-ratio 0` keeps all 4,499 examples in the training parquet because
checkpoint evaluation uses the separate fixed validation set built below.
With the complete source manifests, the training data contains 1,600 GRES,
2,660 REC-8K, and 239 ReasonSeg samples. The generated files are:

```text
training_data/GRPO/verl_visharness/
├── train.parquet
└── images/
```

Next, build the fixed 300-sample validation set from the official validation
splits:

```bash
python -m visharness.data.prepare_official_validation \
  --rec8k-data-root /path/to/REC-8K \
  --gres-data-root /path/to/GRES \
  --reasonseg-data-root /path/to/ReasonSeg \
  --train-manifest-root /path/to/VisionAgent-4K \
  --output-dir training_data/GRPO/verl_visharness_official_val \
  --samples-per-task 100 \
  --seed 42 \
  --overwrite true
```

The validation builder selects 100 examples from each of ReasonSeg, GRES, and
REC-8K. `--train-manifest-root` is used only to exclude and audit training IDs
and images; samples from the training manifests are never added to the
validation set. The generated files are:

```text
training_data/GRPO/verl_visharness_official_val/
├── val.parquet
├── selection_report.json
├── manifests/
└── images/
```

By default, neither parquet embeds the system prompt. During training and
validation, `VisHarnessDataset` injects the current
`TRAIN_TEST_SYSTEM_PROMPT` before multimodal prompt-length filtering, while the
tool schemas are loaded dynamically from the tool configuration. Pass
`--embed-system-prompt` to either data-preparation command only when a
self-contained parquet is required.

Both builders are deterministic when the source data, code version,
dependencies, command-line arguments, and seed are unchanged. The parquet
stores absolute image paths, so the datasets should normally be generated on
the machine and under the project checkout used for training. The RL launcher
expects the two parquet files at the default output locations shown above.

## Step 2: launch RL training

Before starting RL training, configure
`recipe/visharness/scripts/rl/train_visharness.sh`. In particular, set
`MODEL_PATH` to the SFT checkpoint that will initialize the RL policy and set
`EXPERIMENT_NAME` to a unique name for the run. For example:

```bash
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/sft/Qwen3-VL-8B-Thinking/<sft-run>/checkpoint-<step>}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-my-visharness-grpo-run}"
```

The launcher derives the checkpoint and validation-output directories from
`EXPERIMENT_NAME`. After the visual expert services are running and these
parameters have been configured, start RL training from the project root:

```bash
bash recipe/visharness/scripts/rl/train_visharness.sh
```
