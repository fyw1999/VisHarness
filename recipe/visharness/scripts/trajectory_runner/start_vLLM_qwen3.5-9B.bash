#!/bin/bash
unset LD_LIBRARY_PATH

echo ">>> Starting the vLLM server..."
vllm serve  /vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/models/LLM/Qwen3.5-9B/ \
    --served-model-name Qwen3.5-9B \
    -tp 4 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu_memory_utilization 0.95 \
    --max_model_len 125000 \

echo ">>> The vLLM server has stopped."
