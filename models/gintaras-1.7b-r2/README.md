# Gintaras 1.7B, round 2 (LoRA adapter, stacks on round 1)

Trained on CPU for another 150 steps (1,200 new examples) on top of round 1 (merged).
Loss 1.55 -> ~1.17 on the new examples.

**Not recommended:** worse than round 1 (QA F1 0.49 vs 0.60, diacritics 0.59 vs 0.72), because it was trained
on teacher data only and forgot the exercises. Kept for reference; later rounds restart from round 1.

Use both rounds together:
`python run_gintaras.py --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r2"`
Exams: `python run_gintaras.py --exam --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r2"`
