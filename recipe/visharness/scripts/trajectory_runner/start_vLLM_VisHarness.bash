#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

# Resolve relative checkpoint paths against this checkout, not the working directory.
MODEL="${MODEL:-checkpoints/sft/Qwen3-VL-8B-Thinking/v0-20260928-211731/checkpoint-105}"
if [[ "$MODEL" != /* ]]; then
    MODEL="$PROJECT_ROOT/$MODEL"
fi

# unset LD_LIBRARY_PATH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD
echo ">>> Starting the vLLM server..."
# The wrapper preserves vLLM arguments/output and records live startup KV
# capacities automatically for the trajectory runner's memory benchmark.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python "$PROJECT_ROOT/visharness/trajectory_runner/vllm_launch.py" serve "$MODEL" \
    --served-model-name VisHarness \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
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
