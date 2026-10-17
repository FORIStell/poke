# Gintaras 1.7B, round 5 (LoRA adapter, stacks on rounds 1 and 4)

150 more CPU steps from round 4, with added "not in the text" examples.
QA F1 0.588, diacritics 0.772 (r4: 0.599 / 0.730).

Use: `python run_gintaras.py --model "models/gintaras-1.7b-r1+models/gintaras-1.7b-r4+models/gintaras-1.7b-r5"`
