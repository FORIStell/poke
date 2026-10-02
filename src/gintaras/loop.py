"""Iterative self-improvement ("train until it's nearly perfect").

Each round:
  1. build fresh Lithuanian prompts (new seed, never the eval prompts);
  2. the current student answers each prompt several times (on-policy samples);
  3. every teacher answers the same prompts;
  4. the judge scores all candidates;
  5. DPO pairs = (best candidate, worst *student* answer) — on-policy negatives
     teach the student exactly which of its own habits to drop; prompts where
     the student failed but a teacher succeeded become extra SFT data;
  6. train SFT (optional) + DPO on top of the current student, evaluate;
  7. keep the new model only if the eval score improved; stop when the target
     score is reached or improvement stalls for `patience` rounds.
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
from pathlib import Path

from gintaras.backends import make_backend
from gintaras.config import Config
from gintaras.data.synth import STYLE_GUIDE, Prompt, build_prompts, score_candidates, teacher_answers
from gintaras.evaluate import evaluate, summary_score
from gintaras.generation import with_system
from gintaras.judge import Score
from gintaras.train import latest_checkpoint, run_dpo, run_sft, to_prompt_completion
from gintaras.utils import append_jsonl

log = logging.getLogger(__name__)


def student_samples(cfg: Config, model_path: str, prompts: list[Prompt]) -> list[list[str]]:
    from gintaras.fastgen import generate

    convs = [with_system(p.messages, cfg.system_prompt) for p in prompts]
    outs = generate(cfg, model_path, convs, cfg.eval.max_new_tokens, temperature=0.8, n=cfg.loop.student_samples)
    gc.collect()
    return [[t for t, _ in row if t] for row in outs]


def round_data(
    prompts: list[Prompt], scored: list[list[tuple[str, str, Score]]], min_score: float, margin: float
) -> tuple[list[dict], list[dict], list[float]]:
    dpo, sft, student_scores = [], [], []
    for p, cands in zip(prompts, scored):
        if not cands:
            continue
        students = [c for c in cands if c[0] == "student"]
        teachers = [c for c in cands if c[0] != "student"]
        student_scores += [c[2].overall for c in students]
        best = max(cands, key=lambda c: c[2].overall)
        if best[2].overall < min_score:
            continue
        if students:
            worst = min(students, key=lambda c: c[2].overall)
            if best[2].overall - worst[2].overall >= margin:
                dpo.append({"id": p.id, "task": p.task, "prompt": p.messages,
                            "chosen": [{"role": "assistant", "content": best[1]}],
                            "rejected": [{"role": "assistant", "content": worst[1]}],
                            "chosen_score": best[2].overall, "rejected_score": worst[2].overall,
                            "chosen_by": best[0]})
        if teachers and students and max(c[2].overall for c in students) < min_score:
            t_best = max(teachers, key=lambda c: c[2].overall)
            if t_best[2].overall >= min_score:
                sft.append({"id": p.id, "task": p.task, "teacher": t_best[0], "score": t_best[2].overall,
                            "messages": [*p.messages, {"role": "assistant", "content": t_best[1]}]})
    return dpo, sft, student_scores


def _remove_checkpoint(cfg: Config, path: str) -> None:
    p = Path(path)
    if p.parent == cfg.ckpt_path("x").parent and p.name.startswith("loop_r"):
        shutil.rmtree(p, ignore_errors=True)


def _resume_state(cfg: Config, hist_path) -> tuple[list[dict], str, float] | None:
    """Continue an interrupted loop (e.g. the rented machine was stopped):
    keep the history and start from the best accepted model so far."""
    if not hist_path.exists():
        return None
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    accepted = [h for h in history if h.get("accepted", h["round"] == 0) and Path(h["model"], "config.json").exists()]
    if not history or not accepted:
        return None
    best = accepted[-1]
    return history, best["model"], summary_score(best["eval"])


def improve(cfg: Config) -> dict:
    if not cfg.teachers or cfg.judge is None:
        raise ValueError("The improvement loop needs `teachers` and a `judge` in the config")
    teachers = [make_backend(t) for t in cfg.teachers]
    judge = make_backend(cfg.judge)
    lc = cfg.loop

    hist_path = cfg.out / "loop_history.json"
    resumed = _resume_state(cfg, hist_path)
    if resumed:
        history, current, best_score = resumed
        log.info("Resuming the loop after round %d from %s (score %.3f)", history[-1]["round"], current, best_score)
    else:
        current = latest_checkpoint(cfg, "dpo", "sft", "cpt")
        history = [{"round": 0, "model": current, "eval": evaluate(cfg, current, name="loop_r0"), "accepted": True}]
        best_score = summary_score(history[0]["eval"])
        log.info("Round 0 (%s): score %.3f", current, best_score)
    first_round = history[-1]["round"] + 1
    stale = 0
    system = f"{cfg.system_prompt}\n\n{STYLE_GUIDE}"

    from gintaras.exams import load_exams, run_ladder

    use_ladder = lc.exam_ladder and any(load_exams(cfg).values())
    ladder_done = False
    if use_ladder:
        ladder_done = run_ladder(cfg, current, max_exams=cfg.exam.streak, judge=judge)["completed"]

    for r in range(first_round, first_round + lc.rounds):
        if ladder_done:
            log.info("Exam ladder completed (VBE passed %d times in a row); stopping", cfg.exam.streak)
            break
        if not use_ladder and best_score >= lc.target_score:
            log.info("Target score %.2f reached; stopping", lc.target_score)
            break
        prompts = build_prompts(cfg, teachers, lc.prompts_per_round, seed=cfg.seed + 10_000 * r, id_prefix=f"r{r}")
        prompts = [p for p in prompts if p.reference is None]
        students = student_samples(cfg, current, prompts)
        teacher_c = teacher_answers(teachers, prompts, system)
        candidates = [[("student", s) for s in st] + tc for st, tc in zip(students, teacher_c)]
        scored = score_candidates(judge, prompts, candidates)
        dpo, sft, s_scores = round_data(prompts, scored, cfg.synth.min_judge_score, cfg.synth.dpo_margin)
        append_jsonl(cfg.data_path("loop_dpo.jsonl"), dpo)
        append_jsonl(cfg.data_path("loop_sft.jsonl"), sft)
        student_mean = sum(s_scores) / len(s_scores) if s_scores else 0.0
        log.info("Round %d: student mean judge score %.2f, %d DPO pairs, %d SFT rows",
                 r, student_mean, len(dpo), len(sft))

        candidate = current
        if lc.sft_on_teacher_wins and sft:
            rows = [to_prompt_completion(x["messages"], cfg.system_prompt) for x in sft]
            candidate = str(run_sft(cfg, base=candidate, rows=rows, name=f"loop_r{r}_sft"))
        if dpo:
            sys_msg = {"role": "system", "content": cfg.system_prompt}
            rows = [{"prompt": [sys_msg, *x["prompt"]], "chosen": x["chosen"], "rejected": x["rejected"]} for x in dpo]
            new = str(run_dpo(cfg, base=candidate, rows=rows, name=f"loop_r{r}"))
            if candidate != current:
                _remove_checkpoint(cfg, candidate)
        else:
            new = candidate
        if new == current:
            log.info("Round %d produced no training data; stopping", r)
            break

        res = evaluate(cfg, new, name=f"loop_r{r}")
        score = summary_score(res)
        accepted = score > best_score
        history.append({"round": r, "model": new, "eval": res, "student_train_score": student_mean,
                        "dpo_pairs": len(dpo), "sft_rows": len(sft), "accepted": accepted})
        if accepted:
            log.info("Round %d improved %.3f -> %.3f", r, best_score, score)
            _remove_checkpoint(cfg, current)
            current, best_score, stale = new, score, 0
            (cfg.out / "BEST_MODEL").write_text(current + "\n", encoding="utf-8")
        else:
            log.info("Round %d did not improve (%.3f <= %.3f); discarding", r, score, best_score)
            _remove_checkpoint(cfg, new)
            stale += 1
            if stale >= lc.patience:
                log.info("No improvement for %d rounds; stopping", stale)
                break
        if use_ladder and accepted:
            ladder_done = run_ladder(cfg, current, max_exams=cfg.exam.streak, judge=judge)["completed"]
        (cfg.out / "loop_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    (cfg.out / "loop_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.out / "BEST_MODEL").write_text(current + "\n", encoding="utf-8")
    log.info("Best model: %s (score %.3f)", current, best_score)
    return {"best_model": current, "best_score": best_score, "rounds": len(history) - 1}
