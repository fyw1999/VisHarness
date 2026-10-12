#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

# MODEL may override the local weights; relative paths use this checkout.
MODEL="${MODEL:-/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3-VL-8B-Thinking}"
if [[ "$MODEL" != /* ]]; then
    MODEL="$PROJECT_ROOT/$MODEL"
fi
if [[ ! -f "$MODEL/config.json" ]]; then
    echo ">>> Model config.json not found: $MODEL/config.json" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset LD_PRELOAD
# Match the current VisHarness deployment: one full replica per GPU.
# Explicit KV dtype also lets resource monitoring calculate active KV memory.
echo ">>> Starting Qwen3-VL-8B (TP=1, DP=4): $MODEL"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "$MODEL" \
    --served-model-name Qwen3-VL-8B \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --async-scheduling \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 125000 \
    --chat-template-content-format openai

echo ">>> The vLLM server has stopped."
