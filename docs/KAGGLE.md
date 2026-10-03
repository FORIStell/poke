# Training Gintaras for free on Kaggle

Kaggle gives about **30 free GPU hours per week**. Each session can run up to 12 h on **2× Tesla T4 (16 GB)**.
`scripts/kaggle_run.sh` uses one session per run:
- **GPU 0:** a judge model (Qwen3-14B, 4-bit).
- **GPU 1:** trains Gintaras 9B (4-bit QLoRA) for up to ~8 h, then tests it and sits 3 exams.

The next session continues where the last one stopped.

## One-time setup
1. Create an account at kaggle.com, then **Settings → Phone verification** (needed for GPUs and internet).
2. Optional, recommended: **Datasets → New Dataset** → upload `gintaras-progress.tgz`, name it `gintaras-progress`.
   This brings in the exams already converted on the first GPU run.

## Each session
1. **Create → New Notebook.**
2. Right panel → **Session options**: Accelerator **GPU T4 ×2**, Internet **On**.
3. **Add Input** (right panel):
   - your `gintaras-progress` dataset;
   - from the second session on, **also your notebook's previous output**. Use *Add Input → Your Work → this notebook*; it contains the `gintaras-9b-kaggle` folder.
4. Paste into the first cell:
   ```
   !git clone -q -b claude/inspiring-goldberg-4d0m46 https://github.com/FORIStell/poke.git
   %cd poke
   !bash scripts/kaggle_run.sh
   ```
5. Click **Save Version → Save & Run All (Commit)**. This runs in the background, up to 12 h, even if you close the browser.
6. When it finishes, open the version → **Output**:
   - `progress.png` is the chart;
   - `gintaras-9b-kaggle/` is the progress (adapter, exam results).

   Send the chart and `gintaras-9b-kaggle/exam_ladder.json` to Claude.

Repeat each week. Every session adds about 8 h of training on top of the last one.
