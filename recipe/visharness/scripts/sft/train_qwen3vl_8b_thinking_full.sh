#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3-VL-8B-Thinking/}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/training_data/SFT/KimiK2.5-Qwen3.5_397B_FP8-Merged-VisionAgent-4K-20260927-patch56k}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
IMAGE_MAX_TOKEN_NUM=2048 \
CELOSS_PARALLEL_SIZE=2048 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ROOT_IMAGE_DIR="$DATASET_DIR" \
swift sft \
    --model "$MODEL_PATH" \
    --tuner_type full \
    --dataset "$DATASET_DIR/merged_sft_data_swift_cmd.jsonl" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --output_dir checkpoints/sft/Qwen3-VL-8B-Thinking \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 25848 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero3_offload \
    --gradient_checkpointing true \
    --attn_impl flash_attention_2 \
    --sequence_parallel_size 2 \
    --padding_free true \
    --use_logits_to_keep false \
    --use_liger_kernel true \
    --report_to swanlab
