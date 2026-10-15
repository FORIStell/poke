#!/bin/bash
# Endless training: 30k-step cycles, each with a short warm restart and a learning-rate wind-down.
# After every cycle the EMA weights are exported to models/tinyface.pt (fp16).
cd "$(dirname "$0")"
PY=/root/.venvs/gintaras/bin/python
while true; do
  S=$($PY -c "import torch;print(torch.load('runs/tiny/last.pt',weights_only=False)['step'])")
  START=$(( 75000 + ((S - 75000) / 30000) * 30000 )); [ $START -lt 75000 ] && START=75000
  END=$(( START + 30000 ))
  echo "CYCLE $START -> $END (now $S)"
  $PY -u tinyface.py train --data data --out runs/tiny --steps $END --decay-from $START --lr 5e-5 \
      --base 96 --batch 64 --sample-every 2500 --save-every 250 || { sleep 30; continue; }
  $PY - <<'PYEOF'
import torch
ck = torch.load("runs/tiny/ema.pt", weights_only=False)
ck["ema"] = {k: v.half() for k, v in ck["ema"].items()}
torch.save(ck, "models/tinyface.pt")
print("CYCLE_DONE exported step", ck["step"])
PYEOF
done
