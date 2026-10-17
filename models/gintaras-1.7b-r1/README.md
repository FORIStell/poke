# Gintaras 1.7B, round 1 (LoRA adapter)

Base: utter-project/EuroLLM-1.7B-Instruct. Trained on CPU for 150 steps (1,200 examples) on
Lithuanian QA/instruction data plus self-made diacritics and error-correction exercises.
Loss 1.30 -> ~1.10.

Use it: `python run_gintaras.py --model models/gintaras-1.7b-r1`
Exams:  `python run_gintaras.py --exam --model models/gintaras-1.7b-r1`
