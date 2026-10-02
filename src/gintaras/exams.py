"""Lithuanian school exam ladder (NMPP 8 → PUPP 10 → VBE 12).

Exams are real past papers from the National Agency for Education (NŠA,
nsa.smsm.lt), converted into JSON (see `data/exams/README.md` for the format):

    {"id", "level", "year", "title", "sources", "texts": {name: text},
     "questions": [{"id", "type", "prompt", "texts", "points", ...}],
     "grade_table": [[min_points, grade], ...]   # optional official table}

Question types and how they are graded:
  choice      exact letter match against "answer"
  forms       list of short answers; each matched against alternatives ("a / b")
  fill_text   model rewrites a text with gaps filled; word errors vs "reference",
              converted to points with the official "error_table"
  punctuation model rewrites a text with punctuation; punctuation errors vs
              "reference", converted with "error_table"
  open        short open answer graded against the official key + rubric
  essay       long text graded against the official rubric
`open`/`essay` need an LLM judge (`judge:` in the config). Without one they
are scored approximately by overlap with the key and the result is marked
`approximate`.

Grades are on the 10-point scale. If the exam has no official points→grade
table, grade = round(10 × share of points) (so 8 ⇔ at least 75 %).
VBE results are also reported on the 0–100 scale.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from gintaras.config import Config
from gintaras.evaluate import resolve, token_f1
from gintaras.lt import words

log = logging.getLogger(__name__)

EXAM_SYSTEM = (
    "Tu laikai lietuvių kalbos ir literatūros egzaminą. Atsakyk taisyklinga lietuvių kalba, "
    "tiksliai pagal užduoties nurodymus. Rašyk tik atsakymą, be paaiškinimų apie save."
)

LEVEL_NAMES = {
    "nmpp8": "8 klasės NMPP (nacionalinis mokinių pasiekimų patikrinimas)",
    "pupp10": "10 klasės PUPP (pagrindinio ugdymo pasiekimų patikrinimas)",
    "vbe12": "12 klasės VBE (valstybinis brandos egzaminas)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_exams(cfg: Config) -> dict[str, list[dict]]:
    """level -> exams sorted by year (newest first, so 'different years' come first)."""
    root = resolve(cfg.exam.exams_dir)
    by_level: dict[str, list[dict]] = {lvl: [] for lvl in cfg.exam.levels}
    if not root.exists():
        return by_level
    for path in sorted(root.glob("**/*.json")):
        try:
            exam = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("Skipping invalid exam %s: %s", path, e)
            continue
        if exam.get("level") in by_level and exam.get("questions"):
            exam["_path"] = str(path)
            by_level[exam["level"]].append(exam)
    for lvl in by_level:
        by_level[lvl].sort(key=lambda e: (-int(e.get("year", 0)), e["id"]))
    return by_level


def max_points(exam: dict) -> float:
    return float(exam.get("max_points") or sum(q["points"] for q in exam["questions"]))


def question_prompt(exam: dict, q: dict) -> str:
    parts = []
    for name in q.get("texts", []):
        parts.append(f"{name}:\n{exam['texts'][name]}")
    parts.append(q["prompt"])
    if q["type"] == "forms":
        parts.append("Atsakymus pateik sunumeruotus, po vieną eilutėje (1. ..., 2. ...).")
    elif q["type"] in ("fill_text", "punctuation"):
        parts.append("Perrašyk visą tekstą su pataisymais. Nieko daugiau nerašyk.")
    elif q["type"] == "choice":
        parts.append("Atsakyk tik teisingo atsakymo raide.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Objective graders
# ---------------------------------------------------------------------------

_PUNCT = ",.;:–—-!?„“\"()"


def _norm_word(w: str) -> str:
    return w.strip(_PUNCT + "…'").lower()


def table_points(errors: int, table: list[float]) -> float:
    return table[min(errors, len(table) - 1)]


def word_errors(answer: str, reference: str) -> int:
    """Reference words the answer does not reproduce exactly (spelling counts,
    punctuation and case don't)."""
    ref = [_norm_word(w) for w in reference.split() if _norm_word(w)]
    ans = [_norm_word(w) for w in answer.split() if _norm_word(w)]
    sm = difflib.SequenceMatcher(a=ref, b=ans, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return len(ref) - matched


def _punct_slots(text: str) -> tuple[list[str], list[str]]:
    """Words and the punctuation that follows each word."""
    toks = re.findall(r"[^\s,.;:!?–—„“\"()]+|[,.;:!?–—„“\"()]", text.replace(" - ", " – "))
    ws, after = [], []
    for t in toks:
        if re.fullmatch(r"[,.;:!?–—„“\"()]", t):
            if after:
                after[-1] += t.replace("—", "–")
        else:
            ws.append(t.lower())
            after.append("")
    return ws, after


def punctuation_errors(answer: str, reference: str) -> int:
    rw, rp = _punct_slots(reference)
    aw, ap = _punct_slots(answer)
    sm = difflib.SequenceMatcher(a=rw, b=aw, autojunk=False)
    errors = 0
    aligned = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            aligned[blk.a + k] = blk.b + k
    for i, p in enumerate(rp):
        j = aligned.get(i)
        got = ap[j] if j is not None else None
        if got is None:
            errors += 1 if p else 0
        elif sorted(got.replace("„", "").replace("“", "")) != sorted(p.replace("„", "").replace("“", "")):
            errors += 1
    return errors


def parse_numbered(answer: str, n: int) -> list[str]:
    found: dict[int, str] = {}
    for line in answer.splitlines():
        m = re.match(r"\s*\(?(\d+)[.)]?\s*[-–:]?\s*(.+)", line)
        if m and 1 <= int(m.group(1)) <= n:
            found.setdefault(int(m.group(1)), m.group(2).strip())
    if not found:  # comma separated fallback
        parts = [p.strip() for p in re.split(r"[,;\n]", answer) if p.strip()]
        found = {i + 1: p for i, p in enumerate(parts[:n])}
    return [found.get(i + 1, "") for i in range(n)]


def grade_objective(q: dict, answer: str) -> tuple[float, str]:
    t = q["type"]
    if t == "choice":
        m = re.search(r"\b([A-HА-Я])\b", answer.strip().upper())
        ok = bool(m) and m.group(1) == q["answer"].upper()
        return (q["points"] if ok else 0.0), f"answer {m.group(1) if m else '?'} vs {q['answer']}"
    if t == "forms":
        got = parse_numbered(answer, len(q["answers"]))
        each = q.get("points_each", q["points"] / len(q["answers"]))
        correct = 0
        for g, alts in zip(got, q["answers"]):
            g = _norm_word(g.split()[0]) if g.split() else ""
            if g in {_norm_word(a) for a in alts}:
                correct += 1
        if "error_table" in q:
            return table_points(len(q["answers"]) - correct, q["error_table"]), f"{correct}/{len(q['answers'])} forms"
        return correct * each, f"{correct}/{len(q['answers'])} forms"
    if t == "fill_text":
        e = word_errors(answer, q["reference"])
        return table_points(e, q["error_table"]), f"{e} spelling errors"
    if t == "punctuation":
        e = punctuation_errors(answer, q["reference"])
        return table_points(e, q["error_table"]), f"{e} punctuation errors"
    raise ValueError(f"not an objective question type: {t}")


# ---------------------------------------------------------------------------
# Judge-graded questions
# ---------------------------------------------------------------------------

EXAM_JUDGE = """You are an official grader of the Lithuanian national exam ({level}). \
Grade the STUDENT ANSWER strictly by the official marking key below, exactly as a Lithuanian \
exam commission would. Answers may use different words than the key; give the points the key \
assigns to an equivalent answer. Penalize language errors only where the key says so.

### TASK
{prompt}

### OFFICIAL MARKING KEY (max {points} points)
{key}

### STUDENT ANSWER
{answer}

Return ONLY JSON: {{"points": <number between 0 and {points}>, "reason": "<short, English>"}}"""


def judge_prompt(exam: dict, q: dict, answer: str) -> str:
    key = q.get("key", "")
    if q.get("rubric"):
        key = f"{key}\n{q['rubric']}".strip()
    return EXAM_JUDGE.format(level=LEVEL_NAMES.get(exam["level"], exam["level"]),
                             prompt=question_prompt(exam, q), points=q["points"], key=key, answer=answer)


def approx_points(q: dict, answer: str) -> float:
    """No-judge fallback: overlap with the key, scaled. Crude; flagged as approximate."""
    if q["type"] == "essay":
        n = len(words(answer))
        target = q.get("min_words", 250)
        return round(q["points"] * min(1.0, n / target) * 0.5, 2)
    f1 = token_f1(answer, q.get("key", ""))
    return round(q["points"] * min(1.0, f1 * 2), 2)


# ---------------------------------------------------------------------------
# Taking an exam
# ---------------------------------------------------------------------------


@dataclass
class ExamResult:
    exam_id: str
    level: str
    year: int
    points: float
    max_points: float
    grade: float
    passed: bool
    approximate: bool
    details: list[dict] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return 100 * self.points / self.max_points if self.max_points else 0.0

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "details"}
        d["percent"] = round(self.percent, 1)
        return d


def to_grade(exam: dict, points: float) -> float:
    table = exam.get("grade_table")
    if table:
        grade = 1.0
        for min_pts, g in sorted(table):
            if points >= min_pts:
                grade = float(g)
        return grade
    return float(max(1, min(10, round(10 * points / max_points(exam)))))


def take_exam(cfg: Config, exam: dict, model_path: str, hf=None, judge=None) -> ExamResult:
    from gintaras.backends import parse_json_object
    from gintaras.fastgen import generate_best

    questions = exam["questions"]
    convs = [[{"role": "system", "content": EXAM_SYSTEM}, {"role": "user", "content": question_prompt(exam, q)}]
             for q in questions]
    answers: list[str] = [""] * len(questions)
    for is_essay in (False, True):  # one batched call per answer length
        idx = [i for i, q in enumerate(questions) if (q["type"] == "essay") == is_essay]
        if idx:
            mnt = cfg.exam.essay_max_new_tokens if is_essay else cfg.exam.max_new_tokens
            outs = generate_best(cfg, model_path, [convs[i] for i in idx], cfg.exam.strength, mnt, hf)
            for i, a in zip(idx, outs):
                answers[i] = a

    details, approximate = [], False
    judged = [i for i, q in enumerate(questions) if q["type"] in ("open", "essay")]
    judge_points: dict[int, tuple[float, str]] = {}
    if judged and judge is not None:
        replies = judge.chat_many(
            [[{"role": "user", "content": judge_prompt(exam, questions[i], answers[i])}] for i in judged],
            temperature=0.0, max_tokens=400,
        )
        for i, r in zip(judged, replies):
            obj = parse_json_object(r)
            try:
                pts = max(0.0, min(float(questions[i]["points"]), float(obj["points"])))
                judge_points[i] = (pts, str(obj.get("reason", ""))[:300])
            except (TypeError, KeyError, ValueError):
                pass
    total = 0.0
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q["type"] in ("open", "essay"):
            if i in judge_points:
                pts, why = judge_points[i]
            else:
                pts, why, approximate = approx_points(q, a), "approximate (no judge)", True
        else:
            pts, why = grade_objective(q, a)
        total += pts
        details.append({"id": q["id"], "type": q["type"], "points": pts, "max": q["points"], "why": why, "answer": a})
    grade = to_grade(exam, total)
    return ExamResult(exam["id"], exam["level"], int(exam.get("year", 0)), round(total, 2), max_points(exam),
                      grade, grade >= cfg.exam.pass_grade, approximate, details)


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def ladder_path(cfg: Config) -> Path:
    return cfg.out / "exam_ladder.json"


def load_ladder(cfg: Config) -> dict:
    p = ladder_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"level_index": 0, "streak": 0, "history": [], "completed": False}


def save_ladder(cfg: Config, state: dict) -> None:
    ladder_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    ladder_path(cfg).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def next_exam(state: dict, exams: list[dict], model_path: str) -> dict | None:
    """Prefer an exam this model hasn't taken yet (a different year); otherwise
    the one it took longest ago."""
    taken = [h["exam_id"] for h in state["history"] if h.get("model") == model_path]
    for e in exams:
        if e["id"] not in taken:
            return e
    if not exams:
        return None
    order = {eid: i for i, eid in enumerate(taken)}
    return min(exams, key=lambda e: order.get(e["id"], -1))


def run_ladder(cfg: Config, model_path: str, max_exams: int = 3, judge=None) -> dict:
    """Sit up to `max_exams` exams at the current level; advance a level after
    `streak` consecutive passes. State persists across training rounds."""
    from gintaras.fastgen import engine
    from gintaras.generation import load_for_inference

    state = load_ladder(cfg)
    all_exams = load_exams(cfg)
    if judge is None and cfg.judge is not None:
        from gintaras.backends import make_backend

        judge = make_backend(cfg.judge)
    hf = load_for_inference(model_path, cfg.model.bf16) if engine(cfg) == "hf" else None
    sat = 0
    while sat < max_exams and not state["completed"]:
        level = cfg.exam.levels[state["level_index"]]
        exam = next_exam(state, all_exams.get(level, []), model_path)
        if exam is None:
            log.warning("No exams available for level %s (put them in %s)", level, cfg.exam.exams_dir)
            break
        res = take_exam(cfg, exam, model_path, hf, judge)
        sat += 1
        state["streak"] = state["streak"] + 1 if res.passed else 0
        entry = {**res.summary(), "model": model_path, "streak_after": state["streak"]}
        state["history"].append(entry)
        log.info("Exam %s: %.1f/%.1f points (%.0f%%) → grade %.0f %s%s", res.exam_id, res.points, res.max_points,
                 res.percent, res.grade, "PASS" if res.passed else "fail", " [approximate]" if res.approximate else "")
        out = cfg.out / "exams" / f"{Path(model_path).name}_{res.exam_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**res.summary(), "details": res.details}, ensure_ascii=False, indent=2), encoding="utf-8")
        if state["streak"] >= cfg.exam.streak:
            log.info("Passed %d %s exams in a row!", state["streak"], level)
            state["streak"] = 0
            if state["level_index"] + 1 < len(cfg.exam.levels):
                state["level_index"] += 1
            else:
                state["completed"] = True
        save_ladder(cfg, state)
    hf = None
    state["current_level"] = cfg.exam.levels[state["level_index"]]
    return state


# ---------------------------------------------------------------------------
# Converting official PDFs into exam JSON (needs a strong LLM; run on the GPU box)
# ---------------------------------------------------------------------------

CONVERT_PROMPT = """Below are two documents from a Lithuanian national exam: the EXAM PAPER and \
its OFFICIAL MARKING KEY (vertinimo instrukcija), extracted from PDF (layout may be messy).

Convert them into ONE JSON object with this schema:
{{"id": "{exam_id}", "level": "{level}", "year": {year}, "title": "<Lithuanian title>",
 "texts": {{"<text name>": "<full reading text, cleaned>"}},
 "questions": [
   {{"id": "<number>", "type": "open|essay|choice|forms|fill_text|punctuation",
     "texts": ["<names of texts this question refers to>"],
     "prompt": "<the task exactly as written, with any task text included>",
     "points": <max points>,
     "key": "<official answers / criteria for open & essay>",
     "rubric": "<point breakdown for open & essay>",
     "answer": "<letter, for choice>",
     "answers": [["<alternative forms>"]],          // forms
     "reference": "<fully correct text>",           // fill_text, punctuation
     "error_table": [<points for 0,1,2,... errors>] // fill_text, punctuation, optional for forms
   }}]}}
Rules: if the student must CHOOSE one of several tasks (e.g. essay topics), include only the \
first option and use its points; keep Lithuanian text exactly (fix only PDF line-break hyphenation); skip tasks that require \
drawing symbols on paper or cannot be done in plain text; omit fields that do not apply.
Return ONLY the JSON.

### EXAM PAPER
{paper}

### OFFICIAL MARKING KEY
{key}"""


def convert_exam(backend, paper_text: str, key_text: str, exam_id: str, level: str, year: int) -> dict | None:
    from gintaras.backends import parse_json_object

    prompt = CONVERT_PROMPT.format(exam_id=exam_id, level=level, year=year, paper=paper_text, key=key_text)
    reply = backend.chat_many([[{"role": "user", "content": prompt}]], temperature=0.0, max_tokens=16000)[0]
    exam = parse_json_object(reply)
    if not exam or not exam.get("questions"):
        return None
    exam.update({"id": exam_id, "level": level, "year": year})
    return exam


def load_catalog(cfg: Config) -> dict[str, list[dict]]:
    import yaml

    path = resolve(Path(cfg.exam.exams_dir) / "catalog.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pdf_text(pdf: Path) -> str:
    import subprocess

    out = pdf.with_suffix(".txt")
    if not out.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=True)
    return out.read_text(encoding="utf-8", errors="replace")


def fetch_exams(cfg: Config) -> int:
    """Download every catalogued paper + key and extract their text (needs `pdftotext`)."""
    import urllib.request

    raw = resolve(cfg.exam.exams_dir) / "raw"
    n = 0
    for level, entries in load_catalog(cfg).items():
        for e in entries:
            for kind in ("paper", "key"):
                dest = raw / level / f"{e['id']}_{kind}.pdf"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    log.info("Downloading %s", e[kind])
                    req = urllib.request.Request(e[kind], headers={"User-Agent": "Mozilla/5.0 gintaras"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        dest.write_bytes(r.read())
                _pdf_text(dest)
                n += 1
    log.info("Fetched %d documents into %s", n, raw)
    return n


def convert_all(cfg: Config, overwrite: bool = False) -> int:
    """Turn fetched papers + keys into exam JSON using the judge (or first teacher) LLM."""
    from gintaras.backends import make_backend

    endpoint = cfg.judge or (cfg.teachers[0] if cfg.teachers else None)
    if endpoint is None:
        raise ValueError("exams-convert needs a strong LLM: configure `judge` or `teachers`")
    backend = make_backend(endpoint)
    root = resolve(cfg.exam.exams_dir)
    done = 0
    for level, entries in load_catalog(cfg).items():
        for e in entries:
            out = root / level / f"{e['id']}.json"
            if out.exists() and not overwrite:
                continue
            raw = root / "raw" / level
            paper = (raw / f"{e['id']}_paper.txt").read_text(encoding="utf-8", errors="replace")
            key = (raw / f"{e['id']}_key.txt").read_text(encoding="utf-8", errors="replace")
            exam = convert_exam(backend, paper, key, e["id"], level, int(e["year"]))
            if exam is None:
                log.warning("Conversion failed for %s", e["id"])
                continue
            exam["sources"] = [e["paper"], e["key"]]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
            log.info("Converted %s (%d questions)", e["id"], len(exam["questions"]))
    return done
