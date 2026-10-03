"""Download, clean, filter and deduplicate Lithuanian data.

Outputs (under <output_dir>/data/):
  corpus.jsonl              {"text", "source"}           continued pretraining
  corpus_eval.jsonl         held-out docs                perplexity / bits-per-char
  instructions.jsonl        {"id", "task", "messages"}   existing human/translated SFT data
  instructions_eval.jsonl   held-out instruction rows    QA evaluation
  contexts.jsonl            {"context"}                  grounding passages for distillation
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Callable, Iterator

from gintaras.config import Config, SourceConfig
from gintaras.data.tasks import context_qa_prompt, split_passages
from gintaras.lt import clean_text, dedup_key, diacritic_density, lt_score
from gintaras.utils import read_jsonl, stable_fraction, write_jsonl

log = logging.getLogger(__name__)

CORPUS_EVAL_FRACTION = 0.005
MIN_DIACRITIC_DENSITY = 0.01  # real Lithuanian prose is ~3-5%; ASCII-only text is low quality


def iter_source(src: SourceConfig) -> Iterator[dict]:
    if src.path:
        rows: Iterator[dict] = read_jsonl(src.path)
    elif src.name:
        from datasets import load_dataset

        rows = iter(load_dataset(src.name, src.config, split=src.split, streaming=True))
    else:
        raise ValueError("A data source needs either `name` (HF dataset) or `path` (local .jsonl)")
    for i, row in enumerate(rows):
        if src.max_docs is not None and i >= src.max_docs:
            break
        yield row


def _source_label(src: SourceConfig) -> str:
    return src.path or f"{src.name}:{src.config or ''}"


def keep_document(text: str, cfg: Config) -> bool:
    d = cfg.data
    return (
        d.min_chars <= len(text) <= d.max_chars
        and diacritic_density(text) >= MIN_DIACRITIC_DENSITY
        and lt_score(text) >= d.min_lt_score
    )


def prepare_corpus(cfg: Config) -> dict:
    """Stream every corpus source through cleaning, filtering and dedup,
    writing documents as we go so multi-GB corpora never sit in memory."""
    seen: set[str] = set()
    stats = {"read": 0, "kept": 0, "filtered": 0, "duplicates": 0, "eval": 0}
    train_path, eval_path = cfg.data_path("corpus.jsonl"), cfg.data_path("corpus_eval.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f_train, eval_path.open("w", encoding="utf-8") as f_eval:
        for src in cfg.data.corpus:
            label = _source_label(src)
            log.info("Corpus source: %s", label)
            for row in iter_source(src):
                stats["read"] += 1
                text = clean_text(str(row.get(src.text_field) or ""))
                if not keep_document(text, cfg):
                    stats["filtered"] += 1
                    continue
                if cfg.data.dedup:
                    key = dedup_key(text)
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                line = json.dumps({"text": text, "source": label}, ensure_ascii=False) + "\n"
                if stable_fraction(text[:2000]) < CORPUS_EVAL_FRACTION:
                    f_eval.write(line)
                    stats["eval"] += 1
                else:
                    f_train.write(line)
                    stats["kept"] += 1
                if stats["read"] % 100_000 == 0:
                    log.info("  %s", stats)
    log.info("Corpus: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Instruction datasets
# ---------------------------------------------------------------------------


def _clean_field(value) -> str:
    text = clean_text(str(value or ""))
    return "" if text.lower() in {"nan", "none", "null"} else text


def _qa(row: dict) -> list[dict] | None:
    q, a = _clean_field(row.get("question")), _clean_field(row.get("answer"))
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}] if q and a else None


def _context_qa(row: dict) -> list[dict] | None:
    c, q, a = (_clean_field(row.get(k)) for k in ("context", "question", "answer"))
    if not (c and q and a):
        return None
    return [{"role": "user", "content": context_qa_prompt(c, q)}, {"role": "assistant", "content": a}]


def _alpaca(row: dict) -> list[dict] | None:
    ins, inp, out = (_clean_field(row.get(k)) for k in ("instruction", "input", "output"))
    if not (ins and out):
        return None
    user = f"{ins}\n\n{inp}" if inp else ins
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


def _messages(row: dict) -> list[dict] | None:
    msgs = row.get("messages") or row.get("conversations")
    if not msgs:
        return None
    out = []
    for m in msgs:
        role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from", ""), "")
        content = _clean_field(m.get("content") or m.get("value"))
        if role in ("system", "user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out if out and out[-1]["role"] == "assistant" else None


ADAPTERS: dict[str, Callable[[dict], list[dict] | None]] = {
    "qa": _qa,
    "context_qa": _context_qa,
    "alpaca": _alpaca,
    "messages": _messages,
}

# Machine-translated English datasets are full of "As an AI ..." boilerplate.
_BAD_PHRASES = re.compile(r"\bkaip (?:ai\b|dirbtinis intelektas|dirbtinio intelekto|kalbos model)", re.IGNORECASE)


def keep_conversation(messages: list[dict], cfg: Config) -> bool:
    answer = messages[-1]["content"]
    if _BAD_PHRASES.search(answer):
        return False
    if len(answer) > 120 and lt_score(answer) < cfg.data.min_lt_score:
        return False
    return True


def prepare_instructions(cfg: Config) -> dict:
    seen: set[str] = set()
    stats = {"read": 0, "kept": 0, "filtered": 0, "eval": 0}
    train, held_out = [], []
    for src in cfg.data.instructions:
        if src.kind not in ADAPTERS:
            raise ValueError(f"Instruction source {_source_label(src)}: unknown kind '{src.kind}' ({list(ADAPTERS)})")
        adapter = ADAPTERS[src.kind]
        label = _source_label(src)
        for i, row in enumerate(iter_source(src)):
            stats["read"] += 1
            msgs = adapter(row)
            if not msgs or not keep_conversation(msgs, cfg):
                stats["filtered"] += 1
                continue
            key = dedup_key(" ".join(m["content"] for m in msgs))
            if key in seen:
                stats["filtered"] += 1
                continue
            seen.add(key)
            rec = {"id": f"{label}#{i}", "task": src.kind, "source": label, "messages": msgs}
            if stable_fraction(key) < src.eval_fraction:
                held_out.append(rec)
                stats["eval"] += 1
            else:
                train.append(rec)
                stats["kept"] += 1
    write_jsonl(cfg.data_path("instructions.jsonl"), train)
    write_jsonl(cfg.data_path("instructions_eval.jsonl"), held_out)
    log.info("Instructions: %s", stats)
    return stats


def prepare_contexts(cfg: Config) -> int:
    """Uniformly sample grounding passages from the (train) corpus for
    distillation (reservoir sampling: one pass, bounded memory)."""
    rng = random.Random(cfg.seed)
    k = cfg.synth.num_contexts
    reservoir: list[str] = []
    seen = 0
    for row in read_jsonl(cfg.data_path("corpus.jsonl")):
        for p in split_passages(row["text"]):
            seen += 1
            if len(reservoir) < k:
                reservoir.append(p)
            else:
                j = rng.randrange(seen)
                if j < k:
                    reservoir[j] = p
    rng.shuffle(reservoir)
    n = write_jsonl(cfg.data_path("contexts.jsonl"), ({"context": p} for p in reservoir))
    log.info("Contexts: %d passages", n)
    return n


def prepare_all(cfg: Config) -> None:
    prepare_corpus(cfg)
    prepare_instructions(cfg)
    prepare_contexts(cfg)
