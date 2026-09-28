#!/bin/bash
unset LD_LIBRARY_PATH

echo ">>> Starting the vLLM server..."
vllm serve  /vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/code/VisionAgent/output/Qwen3.5-9B/v0-20260407-160733/checkpoint-382 \
    --served-model-name VisionAgent \
    -tp 1 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu_memory_utilization 0.45 \
    --max_model_len 80000 \

echo ">>> The vLLM server has stopped."
