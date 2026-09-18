#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-397B-A17B-FP8 model directory}"
TP_SIZE="${TP_SIZE:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
unset LD_PRELOAD || true

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "${MAX_MODEL_LEN:-78000}" \
    --enforce-eager \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --chat-template-content-format openai
