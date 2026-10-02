#!/usr/bin/env bash
# One Kaggle session (free 2x T4): restore previous progress, train up to ~8 h,
# evaluate, sit 3 exams, write a progress chart. Re-run in a new session to continue.
set -uo pipefail
cd "$(dirname "$0")/.."
C=configs/gintaras-9b-kaggle.yaml
OUT=/kaggle/working/gintaras-9b-kaggle
mkdir -p /kaggle/working/logs

# 1) bring back the previous session's output (attach it as an input in the notebook)
PREV=$(ls -d /kaggle/input/*/gintaras-9b-kaggle 2>/dev/null | head -1)
if [ -n "$PREV" ] && [ ! -d "$OUT" ]; then echo "Restoring progress from $PREV"; cp -r "$PREV" "$OUT"; fi
# exams converted on the first GPU run (upload gintaras-progress.tgz as a Kaggle dataset)
TGZ=$(ls /kaggle/input/*/gintaras-progress.tgz 2>/dev/null | head -1)
[ -n "$TGZ" ] && tar xzf "$TGZ" data/exams 2>/dev/null && echo "Exams restored from $TGZ"

# 2) install
apt-get install -y -qq poppler-utils >/dev/null 2>&1 || true
pip install -q -e ".[gpu]" vllm 2>&1 | tail -2

# 3) judge/teacher on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen3-14B-AWQ --dtype half --max-model-len 16384 \
  --gpu-memory-utilization 0.92 --port 8000 > /kaggle/working/logs/vllm.log 2>&1 &
echo "waiting for the judge model..."
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done

# 4) student on GPU 1
export CUDA_VISIBLE_DEVICES=1 GINTARAS_4BIT=1
run() { echo "== $*"; python -m gintaras "$@" -c "$C" 2>&1 | tee -a /kaggle/working/logs/pipeline.log; }
ls data/exams/nmpp8/*.json >/dev/null 2>&1 || { run exams-fetch; run exams-convert; }
[ -f $OUT/data/contexts.jsonl ] || run prepare
[ -f $OUT/data/selfsup_sft.jsonl ] || run selfsup
[ -f $OUT/data/synth_sft.jsonl ] || zcat data/distilled/synth_sft.*.jsonl.gz > $OUT/data/synth_sft.jsonl
run sft
run eval --model $OUT/checkpoints/sft_adapter
run exam --model $OUT/checkpoints/sft_adapter --exams 3
run report
cp $OUT/progress.png /kaggle/working/ 2>/dev/null
echo "SESSION DONE. Save Version -> the output (gintaras-9b-kaggle folder) is your progress."
