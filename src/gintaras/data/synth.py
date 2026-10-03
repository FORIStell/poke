"""Multi-teacher distillation: build Lithuanian prompts, let several teacher
models answer, keep the judge's best answer for SFT and (best, worst) pairs
for DPO.

Teachers receive a detailed Lithuanian style guide in their system prompt;
the student is trained with only the short production system prompt, so it
learns to behave as if the guide were always there ("context distillation").
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from gintaras.backends import Backend, make_backend, parse_json_object
from gintaras.config import Config
from gintaras.data.tasks import TASKS, UNANSWERABLE, Draft, _sentences, context_qa_prompt, sample_task
from gintaras.lt import answer_tokens
from gintaras.judge import Score, judge_many
from gintaras.utils import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

STYLE_GUIDE = """Rašymo gairės:
- Rašyk taisyklinga bendrine lietuvių kalba, su visomis lietuviškomis raidėmis (ą, č, ę, ė, į, š, ų, ū, ž).
- Venk anglicizmų, vertinių ir barbarizmų; rinkis lietuviškus terminus.
- Laikykis visų vartotojo nurodymų: apimties, formos ir stiliaus.
- Jei pateiktas tekstas, remkis tik juo ir nieko neišsigalvok. Jei atsakymo tekste nėra, aiškiai tai pasakyk.
- Rašinio struktūra (kaip vertinama per PUPP ir VBE):
  1) Įžanga: parodyk, kad raktinis žodis svarbus visais laikais, trumpai pateik kontekstą
     (istorinį, biblinį, antikos mitą ar kūrinį) ir užbaik probleminiu klausimu („Taigi kyla klausimas, ...“).
  2) Dėstymas: teiginys → samprotavimas (paaiškink teiginį) → kultūrinis kontekstas → konkretus kūrinys
     ir jo autorius (įvykiai, veikėjo jausmai, charakterio savybės) → samprotavimas → dalinė išvada.
     Antroje pastraipoje gali pateikti savo patirties ar gyvenimo pavyzdį („Mano nuomone, ...“), rašyk „mes“, ne „tu“.
  3) Apibendrinimas (2–3 sakiniai): atsakyk į temos klausimą („Apibendrinant galima teigti, kad ...“),
     be naujų idėjų.
- Ypač saugok šias taisykles: kablelius prie įterpinių (deja, beje, žinoma, matyt, rodos, mano nuomone,
  pasak / anot ko), prieš prijungiamuosius jungtukus (kad, jog, nes, kai, kuris) ir prieš „o“, „bet“, „tačiau“;
  nosines ą, ę, į, ų pagal kaitą (skęsti – skendo, gęsta – užgeso, lįsti – lindo);
  dalyvių, padalyvių ir pusdalyvių formas (bėgantis, bėgęs, bėgdamas, bėgant);
  „ne veltui“ rašomas atskirai, dalelytė „gi“ – kartu tik su nekaitomais žodžiais (kurgi, nejaugi, kadangi).
- Matematikos ir logikos uždavinius spręsk žingsnis po žingsnio ir aiškiai nurodyk galutinį atsakymą.
- Atsakyk tiesiai, be nereikalingų įžangų, pasikartojimų ir atsiprašinėjimų."""

_REFUSAL_MARKERS = (
    "nėra", "nepateikta", "neminima", "nenurodyta", "neužsimenama", "nepaminėta",
    "negalima nustatyti", "neįmanoma nustatyti", "nekalbama",
)

CHUNK = 256


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


@dataclass
class Prompt:
    id: str
    task: str
    user: str
    reference: str | None = None

    @property
    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}]


def load_contexts(cfg: Config) -> list[str]:
    path = cfg.data_path("contexts.jsonl")
    if not path.exists():
        return []
    return [r["context"] for r in read_jsonl(path)]


def make_drafts(cfg: Config, n: int, rng: random.Random, contexts: list[str]) -> list[Draft]:
    drafts = []
    weights = cfg.synth.task_weights or None
    for _ in range(n):
        task = sample_task(rng, weights)
        spec = TASKS[task]
        if spec.needs_context:
            if not contexts:
                continue
            drafts.append(spec.builder(rng, rng.choice(contexts)))
        else:
            drafts.append(spec.builder(rng))
    return drafts


def resolve_prompts(drafts: list[Draft], teachers: list[Backend], id_prefix: str) -> list[Prompt]:
    """Have teachers write the user turns that a template can't provide.
    Requests are spread round-robin across teachers for diversity."""
    pending = [(i, d) for i, d in enumerate(drafts) if d.user is None and d.instruction_request]
    for t_idx, teacher in enumerate(teachers):
        mine = pending[t_idx :: len(teachers)]
        if not mine:
            continue
        convs = [[{"role": "user", "content": d.instruction_request}] for _, d in mine]
        replies = teacher.chat_many(convs, temperature=0.9, max_tokens=1024)
        for (_, d), reply in zip(mine, replies):
            obj = parse_json_object(reply)
            text = str(obj.get("instruction", "")).strip() if obj else ""
            if text and d.wrap:
                d.user = d.wrap(text)
    return [
        Prompt(id=f"{id_prefix}-{i}", task=d.task, user=d.user, reference=d.reference)
        for i, d in enumerate(drafts)
        if d.user
    ]


def build_prompts(cfg: Config, teachers: list[Backend], n: int, seed: int, id_prefix: str) -> list[Prompt]:
    rng = random.Random(seed)
    return resolve_prompts(make_drafts(cfg, n, rng, load_contexts(cfg)), teachers, id_prefix)


def teacher_answers(teachers: list[Backend], prompts: list[Prompt], system: str) -> list[list[tuple[str, str]]]:
    """Every teacher answers every prompt -> per prompt: [(teacher_name, answer), ...]."""
    convs = [[{"role": "system", "content": system}, *p.messages] for p in prompts]
    per_prompt: list[list[tuple[str, str]]] = [[] for _ in prompts]
    for teacher in teachers:
        for i, ans in enumerate(teacher.chat_many(convs)):
            if ans:
                per_prompt[i].append((teacher.name, ans))
    return per_prompt


def score_candidates(
    judge: Backend, prompts: list[Prompt], candidates: list[list[tuple[str, str]]]
) -> list[list[tuple[str, str, Score]]]:
    flat = [(pi, name, ans) for pi, cands in enumerate(candidates) for name, ans in cands]
    scores = judge_many(judge, [(prompts[pi].messages, ans, prompts[pi].reference) for pi, _, ans in flat])
    out: list[list[tuple[str, str, Score]]] = [[] for _ in prompts]
    for (pi, name, ans), s in zip(flat, scores):
        if s is not None:
            out[pi].append((name, ans, s))
    return out


def select(
    prompts: list[Prompt], scored: list[list[tuple[str, str, Score]]], min_score: float, margin: float
) -> tuple[list[dict], list[dict]]:
    sft, dpo = [], []
    for p, cands in zip(prompts, scored):
        if not cands:
            continue
        ranked = sorted(cands, key=lambda c: c[2].overall, reverse=True)
        best_name, best, best_s = ranked[0]
        if best_s.overall >= min_score:
            sft.append(
                {"id": p.id, "task": p.task, "teacher": best_name, "score": best_s.overall,
                 "messages": [*p.messages, {"role": "assistant", "content": best}]}
            )
        worst_name, worst, worst_s = ranked[-1]
        if len(ranked) > 1 and best_s.overall - worst_s.overall >= margin and best_s.overall >= min_score:
            dpo.append(
                {"id": p.id, "task": p.task, "prompt": p.messages,
                 "chosen": [{"role": "assistant", "content": best}],
                 "rejected": [{"role": "assistant", "content": worst}],
                 "chosen_score": best_s.overall, "rejected_score": worst_s.overall,
                 "chosen_by": best_name, "rejected_by": worst_name}
            )
    return sft, dpo


def reference_rows(prompts: list[Prompt], candidates: list[list[tuple[str, str]]]) -> list[dict]:
    """Rows whose target is a gold reference. Unanswerable-question rows are
    kept only if most teachers also judged the question unanswerable, which
    filters out teacher-written questions that were actually answerable."""
    rows = []
    for p, cands in zip(prompts, candidates):
        if p.reference is None:
            continue
        if p.reference == UNANSWERABLE:
            refusals = sum(is_refusal(a) for _, a in cands)
            if not cands or refusals * 2 <= len(cands):
                continue
        rows.append(
            {"id": p.id, "task": p.task, "teacher": "reference",
             "messages": [*p.messages, {"role": "assistant", "content": p.reference}]}
        )
    return rows


def synthesize(cfg: Config) -> dict:
    if not cfg.teachers:
        raise ValueError("Config has no `teachers`; distillation needs at least one teacher endpoint")
    if cfg.judge is None:
        raise ValueError("Config has no `judge` endpoint")
    teachers = [make_backend(t) for t in cfg.teachers]
    judge = make_backend(cfg.judge)
    system = f"{cfg.system_prompt}\n\n{STYLE_GUIDE}"
    sft_path, dpo_path = cfg.data_path("synth_sft.jsonl"), cfg.data_path("synth_dpo.jsonl")
    for p in (sft_path, dpo_path):
        p.unlink(missing_ok=True)

    stats = {"prompts": 0, "sft": 0, "dpo": 0, "reference": 0}
    total = cfg.synth.num_prompts
    for start in range(0, total, CHUNK):
        n = min(CHUNK, total - start)
        prompts = build_prompts(cfg, teachers, n, seed=cfg.seed + start, id_prefix=f"synth{start}")
        needs_answers = [p for p in prompts if p.reference is None or p.reference == UNANSWERABLE]
        candidates = teacher_answers(teachers, needs_answers, system)
        ref_rows = reference_rows(needs_answers, candidates)
        ref_rows += [
            {"id": p.id, "task": p.task, "teacher": "reference",
             "messages": [*p.messages, {"role": "assistant", "content": p.reference}]}
            for p in prompts
            if p.reference is not None and p.reference != UNANSWERABLE
        ]
        open_idx = [i for i, p in enumerate(needs_answers) if p.reference is None]
        scored = score_candidates(judge, [needs_answers[i] for i in open_idx], [candidates[i] for i in open_idx])
        sft, dpo = select(
            [needs_answers[i] for i in open_idx], scored, cfg.synth.min_judge_score, cfg.synth.dpo_margin
        )
        stats["prompts"] += len(prompts)
        stats["sft"] += append_jsonl(sft_path, sft + ref_rows)
        stats["reference"] += len(ref_rows)
        stats["dpo"] += append_jsonl(dpo_path, dpo)
        log.info("Synth progress %d/%d: %s", start + n, total, stats)
    return stats


SELF_SUPERVISED_TASKS = ("diacritics", "grammar_fix")


def synthesize_selfsup(cfg: Config, n: int | None = None) -> dict:
    """Teacher-free training material: exercises whose gold answer is real
    Lithuanian text (restore diacritics, fix injected spelling/punctuation
    errors). Works without any GPU or teacher server."""
    contexts = load_contexts(cfg)
    if not contexts:
        raise ValueError("No contexts found; run `gintaras prepare` first")
    rng = random.Random(cfg.seed + 7)
    n = n if n is not None else min(cfg.synth.num_prompts, 2 * len(contexts))
    rows = []
    for i in range(n):
        task = SELF_SUPERVISED_TASKS[i % len(SELF_SUPERVISED_TASKS)]
        d = TASKS[task].builder(rng, rng.choice(contexts))
        if d.user and d.reference and d.user != d.reference:
            rows.append({"id": f"selfsup-{i}", "task": task, "teacher": "reference",
                         "messages": [{"role": "user", "content": d.user},
                                      {"role": "assistant", "content": d.reference}]})
    rows += unanswerable_rows(cfg, contexts, rng, k=len(rows) // 10)
    rows += answer_removed_rows(cfg, rng, k=len(rows) // 6)
    path = cfg.data_path("selfsup_sft.jsonl")
    path.unlink(missing_ok=True)
    stats = {"selfsup": append_jsonl(path, rows)}
    log.info("Self-supervised material: %s", stats)
    return stats


def unanswerable_rows(cfg: Config, contexts: list[str], rng: random.Random, k: int) -> list[dict]:
    """Teacher-free "not in the text" examples: a real question (from the instruction
    data) paired with an unrelated passage, so the gold answer is UNANSWERABLE."""
    path = cfg.data_path("instructions.jsonl")
    questions = [r["messages"][0]["content"] for r in read_jsonl(path) if r.get("task") == "qa"] if path.exists() else []
    questions = [q for q in questions if len(q) <= 300 and q.rstrip().endswith("?")]
    if not questions or not contexts:
        return []
    rows = []
    for i in range(k):
        q, ctx = rng.choice(questions), rng.choice(contexts)
        keywords = {w.lower().strip("?,.") for w in q.split() if len(w) > 5}
        if any(w in ctx.lower() for w in keywords):  # might be answerable after all
            continue
        rows.append({"id": f"selfsup-unans-{i}", "task": "context_qa_unanswerable", "teacher": "reference",
                     "messages": [{"role": "user", "content": context_qa_prompt(ctx, q)},
                                  {"role": "assistant", "content": UNANSWERABLE}]})
    return rows


def _split_context_qa(prompt: str) -> tuple[str, str] | None:
    head, sep, question = prompt.rpartition("\n\nKlausimas: ")
    _, sep2, context = head.partition("Tekstas:\n")
    return (context, question) if sep and sep2 else None


def remove_answer(context: str, question: str, answer: str) -> str | None:
    """Drop every sentence that carries the answer; None if that guts the passage."""
    key = {t for t in answer_tokens(answer) if len(t) >= 4 or t.isdigit()} - set(answer_tokens(question))
    if not key:
        return None
    sents = _sentences(context)
    kept = [x for x in sents if not key & set(answer_tokens(x))]
    if len(kept) == len(sents) or sum(map(len, kept)) < max(200, len(context) // 2):
        return None
    return " ".join(kept)


def answer_removed_rows(cfg: Config, rng: random.Random, k: int) -> list[dict]:
    """Hard, same-topic "not in the text" examples: a real grounded question whose
    context has had the answer-bearing sentences removed."""
    sources = [cfg.data_path("instructions.jsonl"), *sorted((Path(__file__).resolve().parents[3] / "data/distilled").glob("synth_sft.*.jsonl.gz"))]
    pool = []
    for path in sources:
        if path.exists():
            pool += [r for r in read_jsonl(path) if r.get("task") == "context_qa"]
    rng.shuffle(pool)
    rows = []
    for r in pool:
        if len(rows) >= k:
            break
        parts = _split_context_qa(r["messages"][-2]["content"])
        if parts is None:
            continue
        context = remove_answer(*parts, r["messages"][-1]["content"])
        if context:
            rows.append({"id": f"selfsup-unans-hard-{len(rows)}", "task": "context_qa_unanswerable", "teacher": "reference",
                         "messages": [{"role": "user", "content": context_qa_prompt(context, parts[1])},
                                      {"role": "assistant", "content": UNANSWERABLE}]})
    return rows
