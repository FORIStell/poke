"""Evaluation suite for Lithuanian ability.

Metrics (higher is better unless noted):
  bpc                 bits per character on held-out Lithuanian text (lower is better;
                      tokenizer-independent, so comparable across base models)
  qa_f1               grounded QA token F1 (inflection-tolerant stems)
  qa_unanswerable_acc says "not in the text" when the answer is absent
  diacritics_word_acc restores ą č ę ė į š ų ū ž correctly
  lt_consistency      share of open-ended answers that are actually in Lithuanian
  judge_*             LLM-judge scores (1-10) on a fixed prompt set, if a judge is configured
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import torch

from gintaras.config import Config
from gintaras.data.synth import is_refusal
from gintaras.data.tasks import UNANSWERABLE, _sentences, context_qa_prompt
from gintaras.generation import load_for_inference, with_system
from gintaras.lt import LT_LETTERS_ALL, answer_tokens, lt_score, strip_diacritics
from gintaras.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_QA = "data/eval/gold_context_qa.jsonl"
JUDGE_PROMPTS = "data/eval/judge_prompts.jsonl"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.exists() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)


def diacritics_word_accuracy(prediction: str, original: str) -> float:
    """Share of words that need Lithuanian letters which the model restored exactly."""
    orig_w, pred_w = original.split(), prediction.split()
    targets = [i for i, w in enumerate(orig_w) if any(c in LT_LETTERS_ALL for c in w)]
    if not targets:
        return 1.0
    matcher = difflib.SequenceMatcher(a=orig_w, b=pred_w, autojunk=False)
    matched = set()
    for block in matcher.get_matching_blocks():
        matched.update(range(block.a, block.a + block.size))
    return sum(1 for i in targets if i in matched) / len(targets)


@torch.no_grad()
def bits_per_char(model, tok, docs: list[str], max_len: int = 1024) -> float:
    device = next(model.parameters()).device
    total_nll, total_chars = 0.0, 0
    for doc in docs:
        ids = tok(doc, return_tensors="pt", add_special_tokens=False)["input_ids"][0][:max_len]
        if len(ids) < 2:
            continue
        text_used = tok.decode(ids, skip_special_tokens=True)
        ids = ids.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids)
        total_nll += out.loss.item() * (ids.shape[1] - 1)
        total_chars += max(1, len(text_used))
    return total_nll / max(1, total_chars) / math.log(2)


# ---------------------------------------------------------------------------
# Eval sets
# ---------------------------------------------------------------------------


def qa_items(cfg: Config) -> list[dict]:
    held_out = cfg.data_path("instructions_eval.jsonl")
    rows = read_jsonl(held_out) if held_out.exists() else []
    items = [
        {"messages": r["messages"][:-1], "answer": r["messages"][-1]["content"], "source": "held_out"}
        for r in rows
        if r.get("task") == "context_qa"
    ]
    random.Random(cfg.seed).shuffle(items)
    items = items[: cfg.eval.num_qa]
    gold = resolve(GOLD_QA)
    if gold.exists():
        for r in read_jsonl(gold):
            items.append(
                {"messages": [{"role": "user", "content": context_qa_prompt(r["context"], r["question"])}],
                 "answer": r["answer"], "source": "gold"}
            )
    return items


def diacritic_items(cfg: Config) -> list[dict]:
    path = cfg.data_path("corpus_eval.jsonl")
    if not path.exists():
        return []
    rng = random.Random(cfg.seed)
    items = []
    for r in read_jsonl(path):
        sents = [s for s in _sentences(r["text"]) if 40 <= len(s) <= 300 and strip_diacritics(s) != s]
        if sents:
            original = rng.choice(sents)
            user = f"Atkurk lietuviškas raides (ą, č, ę, ė, į, š, ų, ū, ž) šiame tekste:\n\n{strip_diacritics(original)}"
            items.append({"messages": [{"role": "user", "content": user}], "original": original})
        if len(items) >= cfg.eval.num_diacritics:
            break
    return items


def judge_prompt_items(cfg: Config) -> list[dict]:
    path = resolve(JUDGE_PROMPTS)
    if not path.exists():
        return []
    rows = list(read_jsonl(path))[: cfg.eval.num_judge_prompts]
    return [{"id": r["id"], "category": r.get("category"), "messages": [{"role": "user", "content": r["prompt"]}]} for r in rows]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate(cfg: Config, model_path: str, name: str | None = None, use_judge: bool = True) -> dict:
    from gintaras.fastgen import engine, generate

    name = name or Path(model_path).name
    ev = cfg.eval
    hf = load_for_inference(model_path, cfg.model.bf16) if engine(cfg) == "hf" else None
    results: dict = {"model": model_path}
    samples: list[dict] = []

    qa, dia, jp = qa_items(cfg), diacritic_items(cfg), judge_prompt_items(cfg)
    convs = [with_system(x["messages"], cfg.system_prompt) for x in qa + dia + jp]
    outs = [c[0][0] for c in generate(cfg, model_path, convs, ev.max_new_tokens, 0.0, 1, hf)]
    qa_preds, dia_preds, jp_preds = outs[: len(qa)], outs[len(qa) : len(qa) + len(dia)], outs[len(qa) + len(dia) :]

    if qa:
        answerable = [(p, q) for p, q in zip(qa_preds, qa) if q["answer"] != UNANSWERABLE]
        unanswerable = [(p, q) for p, q in zip(qa_preds, qa) if q["answer"] == UNANSWERABLE]
        if answerable:
            results["qa_f1"] = round(sum(token_f1(p, q["answer"]) for p, q in answerable) / len(answerable), 4)
        if unanswerable:
            results["qa_unanswerable_acc"] = round(sum(is_refusal(p) for p, _ in unanswerable) / len(unanswerable), 4)
        samples += [{"kind": "qa", "prompt": q["messages"][-1]["content"], "reference": q["answer"], "output": p}
                    for p, q in zip(qa_preds, qa)]

    if dia:
        results["diacritics_word_acc"] = round(
            sum(diacritics_word_accuracy(p, d["original"]) for p, d in zip(dia_preds, dia)) / len(dia), 4
        )
        samples += [{"kind": "diacritics", "reference": d["original"], "output": p} for p, d in zip(dia_preds, dia)]

    if jp:
        results["lt_consistency"] = round(sum(lt_score(p) >= 0.55 for p in jp_preds) / len(jp_preds), 4)
        samples += [{"kind": "open", "prompt": j["messages"][-1]["content"], "output": p} for p, j in zip(jp_preds, jp)]
        if use_judge and cfg.judge is not None:
            from gintaras.backends import make_backend
            from gintaras.judge import CRITERIA, judge_many

            scores = judge_many(make_backend(cfg.judge), [(j["messages"], p, None) for p, j in zip(jp_preds, jp)])
            valid = [s for s in scores if s is not None]
            if valid:
                for c in CRITERIA:
                    results[f"judge_{c}"] = round(sum(getattr(s, c) for s in valid) / len(valid), 3)

    ppl_path = cfg.data_path("corpus_eval.jsonl")
    docs = [r["text"] for r in read_jsonl(ppl_path)][: ev.num_ppl_docs] if ppl_path.exists() else []
    if docs:
        model, tok = hf or load_for_inference(model_path, cfg.model.bf16)
        results["bpc"] = round(bits_per_char(model, tok, docs, cfg.model.max_seq_len), 4)
        del model
    hf = None

    out_dir = cfg.out / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / f"{name}_samples.jsonl", samples)
    log.info("Eval %s: %s", name, results)
    return results


def summary_score(results: dict) -> float:
    """Single number used by the improvement loop to pick the best checkpoint:
    the judge's overall score when available, otherwise an average of the
    automatic metrics mapped onto the same 0-10 scale."""
    if "judge_overall" in results:
        return results["judge_overall"]
    parts = [results[k] * 10 for k in ("qa_f1", "qa_unanswerable_acc", "diacritics_word_acc", "lt_consistency") if k in results]
    return sum(parts) / len(parts) if parts else 0.0
