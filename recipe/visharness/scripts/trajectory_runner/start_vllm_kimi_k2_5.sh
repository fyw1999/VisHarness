#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Kimi-K2.5 model directory}"
TP_SIZE="${TP_SIZE:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K2.5}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
unset LD_PRELOAD || true

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
    --max-model-len "${MAX_MODEL_LEN:-49152}" \
    --enforce-eager \
    --chat-template-content-format openai
