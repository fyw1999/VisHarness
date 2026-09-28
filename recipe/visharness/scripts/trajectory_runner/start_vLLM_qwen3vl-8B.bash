#!/bin/bash
# unset LD_LIBRARY_PATH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD
echo ">>> Starting the vLLM server..."
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3-VL-8B-Thinking/ \
    --served-model-name Qwen3-VL-8B \
    -tp 2 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu_memory_utilization 0.85 \
    --max_model_len 125000 \
    --chat-template-content-format openai

echo ">>> The vLLM server has stopped."
