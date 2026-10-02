#!/usr/bin/env bash
# One-command setup + training on a rented 2-GPU machine (RunPod / Vast.ai / Lambda).
# GPU 0: teacher + judge (Qwen3-30B-A3B FP8 via vLLM). GPU 1: Gintaras student.
# Runs the whole pipeline, then the self-improvement loop, which sits the exam
# ladder (NMPP 8 -> PUPP 10 -> VBE) after every round until VBE is passed 3x in a row.
# Safe to re-run: finished stages are kept, the ladder state is resumed.
#   bash scripts/gpu_bootstrap.sh            # inside a tmux session, it runs for days
set -euo pipefail
cd "$(dirname "$0")/.."
C=${CONFIG:-configs/gintaras-9b-budget.yaml}
mkdir -p runs/logs

apt-get update -qq && apt-get install -y -qq poppler-utils tmux >/dev/null || true
pip install -q -e ".[gpu,dev,web]" vllm

# Smaller cards (40 GB, e.g. A100 40GB): smaller training batches, tighter teacher memory.
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
SETS=()
TEACHER_UTIL=0.92
if [ "${GPU_MB:-0}" -lt 46000 ]; then
  echo "GPU has ${GPU_MB} MB: using 40 GB settings"
  SETS=(--set cpt.batch_size=2 --set cpt.grad_accum=16 --set sft.batch_size=2 --set sft.grad_accum=16
        --set dpo.batch_size=1 --set dpo.grad_accum=32 --set eval.batch_size=8)
  TEACHER_UTIL=0.95
fi

if ! curl -sf localhost:8000/v1/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --max-model-len 32768 --gpu-memory-utilization $TEACHER_UTIL --port 8000 > runs/logs/vllm.log 2>&1 &
  echo "waiting for the teacher server..."
  until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done
fi

export CUDA_VISIBLE_DEVICES=1
OUT=$(python -c "from gintaras.config import load_config; print(load_config('$C').output_dir)")
run() { echo "== $*"; python -m gintaras "$@" -c "$C" "${SETS[@]}" 2>&1 | tee -a runs/logs/pipeline.log; }
have() { [ -e "$OUT/$1" ]; }   # skip stages that already finished

run exams-fetch
run exams-convert                                   # skips exams already converted
have data/contexts.jsonl            || run prepare
have data/selfsup_sft.jsonl         || run selfsup
have data/synth_sft.jsonl           || run synth
have checkpoints/cpt/config.json    || run cpt
have checkpoints/sft/config.json    || run sft
have checkpoints/dpo/config.json    || run dpo || echo "no DPO pairs yet; the loop will create them"
run improve           # loops: train -> exams -> train ... until the ladder is done
run report
echo "Done. Front page: python -m gintaras serve -c $C --host 0.0.0.0 --port 8080"

# Stop paying as soon as training is finished (Vast.ai): STOP_WHEN_DONE=1 bash scripts/gpu_bootstrap.sh
if [ "${STOP_WHEN_DONE:-0}" = "1" ] && [ -n "${CONTAINER_ID:-}" ]; then
  pip install -q vastai && vastai stop instance "$CONTAINER_ID" --api-key "${CONTAINER_API_KEY:-$VAST_API_KEY}" || true
fi
