#!/bin/bash

MODEL_PATH="/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Kimi-K2.5/moonshotai/Kimi-K2.5/"

echo ">>> [1/2] Configuring environment variables (CUDA 12.9 compatibility)..."
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD

echo ">>> [2/2] Starting the vLLM server..."
vllm serve "$MODEL_PATH" \
    --served-model-name moonshotai/Kimi-K2.5 \
    -tp 8 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu_memory_utilization 0.97 \
    --max_model_len 49152 \
    --enforce-eager \
    --chat-template-content-format openai

echo ">>> The vLLM server has stopped."
