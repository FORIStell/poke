#!/usr/bin/env bash
# Start the teacher/judge models from configs/gintaras-9b.yaml with vLLM
# (OpenAI-compatible API). Adjust GPU ids / tensor parallelism to your machine.
#   pip install vllm
#   bash scripts/serve_teachers.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size 8 --max-model-len 16384 --port 8000 &

# Second teacher on another node/GPU set (or comment out and keep one teacher).
# CUDA_VISIBLE_DEVICES=0,1 vllm serve utter-project/EuroLLM-22B-Instruct-2512 \
#   --tensor-parallel-size 2 --max-model-len 8192 --port 8001 &

wait
