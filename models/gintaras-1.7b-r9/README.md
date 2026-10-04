# Gintaras 1.7B, round 9 (LoRA adapter, stacks on rounds 1, 4 and 5) - current best

300 CPU steps from round 5: exercises + balanced "not in the text" examples + short teacher answers.
QA F1 0.596, diacritics 0.810 (r5: 0.588 / 0.772; base model: 0.377 / 0.150).
Rounds 6-8 were tried and rejected (not better than round 5).

Use: `python run_gintaras.py --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r4+models/gintaras-1.7b-r5+models/gintaras-1.7b-r9"`
