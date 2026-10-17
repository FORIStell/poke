"""Teacher-free reading-comprehension exercises in the national-exam format.

Built from real Lithuanian passages, with answers that are checkable by construction:
  - choice: which word was in the text (a-d, one letter)
  - multi:  mark every statement that is true according to the text (letters)
  - order:  put the text's sentences back in order (letters)
Prompts mirror gintaras.exams.question_prompt and its answer hints, so the model
practises exactly what the exam ladder (grade 2 upwards) asks.

    gintaras reading -c configs/gintaras-1.7b-cpu.yaml   ->  <output_dir>/data/reading_sft.jsonl
"""

from __future__ import annotations

import logging
import random
import re

from gintaras.config import Config
from gintaras.data.tasks import _sentences
from gintaras.lt import lt_score
from gintaras.utils import append_jsonl

log = logging.getLogger(__name__)

EXAM_SYSTEM = (  # same as gintaras.exams.EXAM_SYSTEM (not imported to keep this module light)
    "Tu laikai lietuvių kalbos ir literatūros egzaminą. Atsakyk taisyklinga lietuvių kalba, "
    "tiksliai pagal užduoties nurodymus. Rašyk tik atsakymą, be paaiškinimų apie save."
)
HINTS = {
    "choice": "Atsakyk tik teisingo atsakymo raide.",
    "multi": "Parašyk visų teisingų atsakymų raides, atskirtas kableliais (pvz.: a, c).",
    "order": "Parašyk raides teisinga eilės tvarka, atskirtas kableliais (pvz.: c, a, b).",
}
LETTERS = "abcdefgh"
_WORD = re.compile(r"^[A-PR-VYZĄČĘĖĮŠŲŪŽa-pr-vyząčęėįšųūž]+$")  # Lithuanian letters only (no q, w, x)
_GOOD_SENT = re.compile(r"^[A-ZĄČĘĖĮŠŲŪŽ„–-][^()\[\]{}=|/]*[.!?…“]$")


def _good(s: str) -> bool:
    """A clean, complete prose sentence: no lists, brackets, codes, dates soup or foreign words."""
    n = len(s.split())
    digits = sum(c.isdigit() for c in s)
    return 5 <= n <= 30 and digits <= 4 and bool(_GOOD_SENT.match(s)) and lt_score(s) >= 0.6


def _content_words(sentence: str) -> list[str]:
    toks = sentence.split()
    return [w for w in toks[1:] if _WORD.match(w.strip(",.;:!?„“\"()")) and len(w.strip(",.;:!?„“\"()")) >= 5]


def _clean(w: str) -> str:
    return w.strip(",.;:!?„“\"()")


def _prompt(text: str, task: str, kind: str) -> str:
    return f"Tekstas:\n{text}\n\n{task}\n\n{HINTS[kind]}"


def _passage(rng: random.Random, ctx: str, lo: int = 4, hi: int = 8) -> list[str] | None:
    sents = _sentences(ctx)
    ok = [_good(s) for s in sents]
    # longest run of consecutive good sentences, so the passage reads as one text
    best, cur = (0, 0), 0
    for i, g in enumerate(ok):
        cur = cur + 1 if g else 0
        if cur > best[1] - best[0]:
            best = (i - cur + 1, i + 1)
    run = sents[best[0] : best[1]]
    if len(run) < lo:
        return None
    k = min(len(run), rng.randint(lo, hi))
    start = rng.randint(0, len(run) - k)
    return run[start : start + k]


def build_word_choice(rng: random.Random, ctx: str) -> tuple[str, str] | None:
    sents = _passage(rng, ctx)
    if not sents:
        return None
    i = rng.randrange(len(sents))
    words = _content_words(sents[i])
    pool = {_clean(w) for s in sents for w in _content_words(s)}
    if not words:
        return None
    target = _clean(rng.choice(words))
    # distractors: words from the same passage with the same ending (same case/form), not in this sentence
    same_end = [w for w in pool if w != target and w[-2:] == target[-2:] and w not in sents[i]]
    others = [w for w in pool if w != target and w not in sents[i]]
    distractors = rng.sample(same_end, 3) if len(same_end) >= 3 else (rng.sample(others, 3) if len(others) >= 3 else None)
    if not distractors:
        return None
    opts = distractors + [target]
    rng.shuffle(opts)
    gap = sents[i].replace(target, "______", 1)
    task = f"Tekste parašyta: „{gap}“ Kuris žodis praleistas?\n" + "\n".join(f"{LETTERS[j]} {o}" for j, o in enumerate(opts))
    return _prompt(" ".join(sents), task, "choice"), LETTERS[opts.index(target)]


def _falsify(rng: random.Random, sentence: str, pool: list[str]) -> str | None:
    words = _content_words(sentence)
    rng.shuffle(words)
    for w in words:
        cw = _clean(w)
        subs = [p for p in pool if p != cw and p[-2:] == cw[-2:] and p.lower() != cw.lower()]
        if subs:
            return sentence.replace(cw, rng.choice(subs), 1)
    return None


def build_true_statements(rng: random.Random, ctx: str) -> tuple[str, str] | None:
    sents = _passage(rng, ctx, 5, 9)
    if not sents:
        return None
    pool = sorted({_clean(w) for s in sents for w in _content_words(s)})
    picked = rng.sample(range(len(sents)), min(5, len(sents)))
    n_false = rng.randint(1, max(1, len(picked) - 2))
    opts, true_flags = [], []
    for j, idx in enumerate(picked):
        if j < n_false:
            f = _falsify(rng, sents[idx], pool)
            if f is None:
                continue
            opts.append(f), true_flags.append(False)
        else:
            opts.append(sents[idx]), true_flags.append(True)
    if not any(true_flags) or all(true_flags):
        return None
    order = list(range(len(opts)))
    rng.shuffle(order)
    opts, true_flags = [opts[o] for o in order], [true_flags[o] for o in order]
    task = "Pažymėk visus sakinius, kurie teisingi pagal tekstą.\n" + "\n".join(f"{LETTERS[j]} {o}" for j, o in enumerate(opts))
    answer = ", ".join(LETTERS[j] for j, t in enumerate(true_flags) if t)
    return _prompt(" ".join(sents), task, "multi"), answer


def build_order(rng: random.Random, ctx: str) -> tuple[str, str] | None:
    sents = _passage(rng, ctx, 4, 6)
    if not sents:
        return None
    k = min(len(sents), rng.randint(3, 5))
    chosen = sents[:k]
    shuffled = list(range(k))
    while shuffled == sorted(shuffled):
        rng.shuffle(shuffled)
    task = "Sudėliok šiuos teksto sakinius iš eilės, kaip jie eina tekste.\n" + "\n".join(
        f"{LETTERS[j]} {chosen[i]}" for j, i in enumerate(shuffled))
    answer = ", ".join(LETTERS[shuffled.index(i)] for i in range(k))
    return _prompt(" ".join(sents), task, "order"), answer


BUILDERS = {"reading_choice": build_word_choice, "reading_true": build_true_statements, "reading_order": build_order}


def make_reading(cfg: Config, n: int = 3000) -> dict:
    from gintaras.data.synth import load_contexts

    from gintaras.utils import read_jsonl

    contexts = load_contexts(cfg)
    corpus = cfg.data_path("corpus.jsonl")
    if corpus.exists():  # general web/encyclopedia prose: more narrative variety than QA contexts
        contexts += [r["text"] for r in read_jsonl(corpus)]
    if not contexts:
        raise ValueError("No contexts found; run `gintaras prepare` first")
    rng = random.Random(cfg.seed + 11)
    names = list(BUILDERS)
    rows, tries = [], 0
    while len(rows) < n and tries < 5 * n:
        tries += 1
        task = names[len(rows) % len(names)]
        made = BUILDERS[task](rng, rng.choice(contexts))
        if made:
            user, answer = made
            rows.append({"id": f"reading-{len(rows)}", "task": task, "teacher": "reference",
                         "messages": [{"role": "system", "content": EXAM_SYSTEM},
                                      {"role": "user", "content": user},
                                      {"role": "assistant", "content": answer}]})
    path = cfg.data_path("reading_sft.jsonl")
    path.unlink(missing_ok=True)
    stats = {"reading": append_jsonl(path, rows)}
    log.info("Reading exercises: %s", stats)
    return stats
