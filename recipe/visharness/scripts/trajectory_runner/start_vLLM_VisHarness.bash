#!/bin/bash
# unset LD_LIBRARY_PATH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
unset LD_PRELOAD
MODEL="/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/code/VisHarness-public/checkpoints/VisHarness/VisHarness-grpo-sft105-p8-filterstd02-trajeq-valsample-v1/archived/global_step_550"
echo ">>> Starting the vLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "$MODEL" \
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
