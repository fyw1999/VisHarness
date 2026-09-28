> **Project status:** Our paper has been accepted to NeurIPS 2026. We are
> currently organizing the release. All source code is now publicly available;
> the model weights and datasets are the remaining artifacts to be released. We
> expect to complete the full open-source release in the near future.

# Visual Expert Server Setup

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
