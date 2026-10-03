# Gintaras 1.7B, round 2 (LoRA adapter, stacks on round 1)

Trained on CPU for another 150 steps (1,200 new examples) on top of round 1 (merged).
Loss 1.55 -> ~1.17 on the new examples.

Use both rounds together:
`python run_gintaras.py --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r2"`
Exams: `python run_gintaras.py --exam --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r2"`
