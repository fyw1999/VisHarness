#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-VL-8B-Thinking model directory}"
: "${DATASET_PATH:?Set DATASET_PATH to merged_sft_data_swift_cmd.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/Qwen3-VL-8B-Thinking}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
CELOSS_PARALLEL_SIZE="${CELOSS_PARALLEL_SIZE:-2048}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES IMAGE_MAX_TOKEN_NUM CELOSS_PARALLEL_SIZE

swift sft \
    --model "$MODEL_PATH" \
    --tuner_type full \
    --dataset "$DATASET_PATH" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --output_dir "$OUTPUT_DIR" \
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
    --report_to "${REPORT_TO:-none}"
