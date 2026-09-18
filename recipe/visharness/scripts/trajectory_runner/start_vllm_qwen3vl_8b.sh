#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a base or fine-tuned Qwen3-VL model directory}"
TP_SIZE="${TP_SIZE:-2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
unset LD_PRELOAD || true

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-125000}" \
    --chat-template-content-format openai
