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
# Keep the normal vLLM CLI; retain its output for automatic KV-capacity lookup.
# exec preserves this PID, so the monitor can reject logs from replaced servers.
mkdir -p "$PROJECT_ROOT/outputs/vllm"
PROC_STAT="$(<"/proc/$$/stat")"
read -r -a PROC_FIELDS <<< "${PROC_STAT##*) }"
read -r BOOT_ID < /proc/sys/kernel/random/boot_id
exec > >(tee "$PROJECT_ROOT/outputs/vllm/server.log") 2>&1
printf 'VISHARNESS_VLLM_STARTUP pid=%s start_ticks=%s boot_id=%s\n' \
    "$$" "${PROC_FIELDS[19]}" "$BOOT_ID"
echo ">>> Starting the vLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
exec vllm serve "$MODEL" \
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
