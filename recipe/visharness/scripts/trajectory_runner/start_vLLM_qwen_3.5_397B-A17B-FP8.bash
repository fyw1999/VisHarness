#!/bin/bash

MODEL_PATH="/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3.5-397B-A17B-FP8/"

# ==============================================================================
# vLLM server launcher for Qwen3.5.
# ==============================================================================
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD
echo ">>> Starting the vLLM server..."
vllm serve "$MODEL_PATH" \
    --served-model-name Qwen3.5-397B-A17B-FP8 \
    -tp 8 \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu_memory_utilization 0.95 \
    --max_model_len 78000 \
    --enforce-eager \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --chat-template-content-format openai

echo ">>> The vLLM server has stopped."
