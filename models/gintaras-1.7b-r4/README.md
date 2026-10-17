# Gintaras 1.7B, round 4 (LoRA adapter, stacks on round 1)

Round 1's data mix (QA/instructions + diacritics/error exercises), 150 new CPU steps at a lower
learning rate, continued from round 1. Same test questions as round 1:
QA F1 0.599 (r1 0.598), diacritics 0.730 (r1 0.721), bpc 0.886 (r1 0.884). A small step up.

Use: `python run_gintaras.py --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r4"`
