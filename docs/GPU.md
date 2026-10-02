# Renting a GPU for Gintaras

You need two GPUs: one to serve the teacher and judge, and one to train Gintaras.
The cheapest route is a rented cloud GPU machine billed by the hour.

## What to rent

| Option | GPUs | Approx. price | Notes |
|---|---|---|---|
| **Recommended** | 2× A100 80GB | ~$2.5–3.5 / h | Enough for the 9B student plus the Qwen3-30B teacher and judge |
| Faster | 2× H100 80GB | ~$4–6 / h | About 2× faster |
| Budget | 2× RTX 4090 / L40S 48GB | ~$1–1.5 / h | Lower `max_seq_len` to 2048. The teacher (about 30 GB) fits on one card |

Disk: **250 GB** is enough with the budget settings. Disk is billed by the hour too: 533 GB costs ~$0.50/h on Vast, 250 GB ~$0.23/h. Use any of these providers:
- **RunPod**: runpod.io → *Pods* → *Deploy* → pick "2× A100 80GB" → template **RunPod PyTorch 2.x** → set container disk to 250 GB.
- **Vast.ai**: vast.ai → *Search* → filter for 2 GPUs, A100 80GB, disk ≥ 250 GB → template **PyTorch**.
- **Lambda**: lambdalabs.com → *Instances* → "2× A100 (80 GB SXM4)".

## Start training (one command)

Open the machine's web terminal (or SSH in), then run:

```bash
git clone https://github.com/FORIStell/poke.git && cd poke
git checkout claude/inspiring-goldberg-4d0m46   # until the PR is merged
tmux new -s gintaras                            # keeps running if you close the browser
STOP_WHEN_DONE=1 bash scripts/gpu_bootstrap.sh   # budget settings; stops the machine when finished
```

By default it uses `configs/gintaras-9b-budget.yaml`, which takes about 3–4× fewer GPU hours than the full config:
- vLLM for all generation (student answers, evaluation, exams), which is about 10–20× faster than plain `generate`;
- bf16 LoRA instead of 4-bit QLoRA (faster on 48 GB cards);
- a short pretraining refresh (EuroLLM already knows Lithuanian);
- a smaller synthetic set and smaller rounds;
- an automatic stop when the exam ladder is done.

For the bigger run, use `CONFIG=configs/gintaras-9b-2gpu.yaml bash scripts/gpu_bootstrap.sh`.

The script installs everything and starts the teacher model on GPU 0. Then it runs these steps:
1. Downloads the past NMPP 8, PUPP 10 and VBE exams and converts them into the exam format.
2. Prepares the Lithuanian data.
3. Generates teacher-free exercises and distills from the teacher.
4. Pretrains, then runs SFT and DPO.
5. Runs the **improvement loop**. After every training round the model sits the exam ladder. It only stops when VBE has been passed 3 times in a row with ≥ 8/10. This takes days.

Re-running the script resumes where it stopped. Detach from tmux with `Ctrl-b d` and re-attach with `tmux attach -t gintaras`.

## Watch progress

```bash
python -m gintaras report -c configs/gintaras-9b-2gpu.yaml       # writes runs/gintaras-9b-2gpu/progress.png
python -m gintaras serve  -c configs/gintaras-9b-2gpu.yaml --host 0.0.0.0 --port 8080
```

To open the front page, expose port 8080 in the provider's dashboard. On RunPod this is under *Edit pod → Expose HTTP ports*.

## Letting Claude drive the GPU machine

Install Claude Code on the machine (`npm i -g @anthropic-ai/claude-code`) and run `claude` in the `poke` folder. Claude then works directly on the GPU box.

## Cost control

- **Stop the machine when you are not training.** The disk is kept on RunPod and Vast "stopped" pods, so the run resumes later.
- With the budget config on 2× 48 GB cards (~$1.2–1.5/h including disk), expect roughly 1.5–3 days, about **$45–110**. This is an estimate; it depends on how fast the exams get passed.
- `STOP_WHEN_DONE=1` stops the instance automatically at the end. Billing for the GPU stops; only the small disk fee remains until you delete the instance.

## If the credit runs out (or the machine stops)

Vast **stops** the instance when your balance hits $0. The disk and all progress are kept, but the storage fee keeps running, and Vast deletes the instance if the balance stays negative.
1. Add credit, then press **▶ Start** on the instance. If the GPUs were taken by someone else in the meantime, you may have to wait until they are free.
2. Open the Jupyter Terminal and run:
   ```bash
   cd /workspace/poke && git pull && STOP_WHEN_DONE=1 bash scripts/gpu_bootstrap.sh
   ```
3. Finished stages are skipped. The interrupted training stage continues from its last checkpoint, saved every ~100 steps. The improvement loop continues from its best model, and the exam ladder keeps its progress.
