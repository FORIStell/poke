"""Task catalogue for Lithuanian instruction data.

Each task builds a `Draft`: the user turn the student will see, plus how to
obtain the answer:
  * self-supervised tasks carry a gold `reference` taken from real Lithuanian
    text (diacritic restoration, error correction, back-translation target);
  * all other tasks are answered by the teacher models and filtered by the
    judge (see synth.py).

Some user turns are themselves written by a teacher (e.g. a question about a
passage); those drafts carry an `instruction_request` meta-prompt and a
`wrap` function that turns the teacher's text into the final user turn.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from gintaras.lt import strip_diacritics

UNANSWERABLE = "Pateiktame tekste atsakymo į šį klausimą nėra."


def context_qa_prompt(context: str, question: str) -> str:
    """The canonical grounded-QA user turn. Shared by training, eval and inference."""
    return (
        "Atsakyk į klausimą remdamasis tik pateiktu tekstu. "
        "Jei tekste atsakymo nėra, taip ir parašyk.\n\n"
        f"Tekstas:\n{context.strip()}\n\nKlausimas: {question.strip()}"
    )


# ---------------------------------------------------------------------------
# Seed material
# ---------------------------------------------------------------------------

ESSAY_TOPICS = [
    "Ar žmogui reikia praeities?",
    "Kas suteikia gyvenimui prasmę?",
    "Ar laisvė visada džiugina?",
    "Žmogus ir gamta: darna ar konfliktas?",
    "Ar įmanoma išlikti savimi visuomenėje?",
    "Kodėl verta skaityti klasikinę literatūrą?",
    "Ką reiškia būti atsakingam?",
    "Technologijos: nauda ar grėsmė žmogui?",
    "Ar svarbu puoselėti gimtąją kalbą?",
    "Draugystės vertė šiuolaikiniame pasaulyje",
    "Ar kiekvienas gali tapti lyderiu?",
    "Tėvynės meilė lietuvių literatūroje",
    "Kokią reikšmę žmogui turi namai?",
    "Ar sėkmė priklauso tik nuo žmogaus pastangų?",
    "Ar vienas žmogus gali pakeisti pasaulį?",
    "Kaltė ir atsakomybė",
    "Menas kaip būdas pažinti save",
    "Ar būtina laikytis tradicijų?",
    "Gėris ir blogis žmoguje",
    "Vienatvė: bausmė ar dovana?",
    "Ar mokslas gali atsakyti į visus klausimus?",
    "Kodėl žmonės kuria?",
    "Ar svarbu atleisti?",
    "Žmogaus pasirinkimai ir jų pasekmės",
]

KNOWLEDGE_TOPICS = [
    # istorija
    "karalius Mindaugas", "Vytautas Didysis", "Žalgirio mūšis", "Liublino unija",
    "Abiejų Tautų Respublika", "knygnešystė ir spaudos draudimas", "Vasario 16-osios aktas",
    "Kovo 11-osios aktas", "Baltijos kelias", "Sausio 13-osios įvykiai", "Lietuvos krikštas",
    "Lietuvos partizanų kova", "Lietuvos įstojimas į Europos Sąjungą ir NATO",
    # kultūra ir kalba
    "Kūčios ir Kalėdų tradicijos", "Joninės (Rasos)", "Užgavėnės", "sutartinės", "Dainų šventė",
    "lietuvių kalbos linksniai", "kirčiavimas lietuvių kalboje", "dalyviai ir padalyviai",
    "skyrybos taisyklės lietuvių kalboje", "lietuvių kalbos vieta indoeuropiečių kalbų šeimoje",
    # literatūra ir menas
    "Kristijono Donelaičio poema „Metai“", "Maironio poezija", "Žemaitės apsakymai",
    "Jono Biliūno kūryba", "Vaižganto „Dėdės ir dėdienės“", "Salomėjos Nėries poezija",
    "Balio Sruogos „Dievų miškas“", "Justino Marcinkevičiaus dramų trilogija",
    "Mikalojaus Konstantino Čiurlionio kūryba",
    # geografija
    "Kuršių nerija", "Nemunas", "Aukštaitija", "Žemaitija", "Dzūkija", "Suvalkija",
    "Mažoji Lietuva", "Lietuvos ežerai", "Vilniaus senamiestis",
    # mokslas ir technologijos
    "fotosintezė", "gravitacija", "DNR ir paveldimumas", "klimato kaita", "elektros srovė",
    "dirbtinis intelektas", "kompiuterių tinklai ir internetas", "vakcinos ir imunitetas",
    "Saulės sistema", "atsinaujinantys energijos šaltiniai", "programavimo pagrindai",
    # visuomenė ir kasdienybė
    "infliacija", "palūkanos ir paskolos", "asmeninio biudžeto planavimas", "sveika mityba",
    "miego svarba", "laiko planavimas", "pasiruošimas darbo pokalbiui", "demokratija",
    "žmogaus teisės", "kibernetinis saugumas kasdienybėje",
]

QUESTION_STYLES = [
    "a factual question",
    "a 'why' question that needs an explanation",
    "a 'how' question asking for practical steps",
    "a comparison question",
    "a question a high-school student might ask while studying",
    "a question asking for an explanation suitable for a 10-year-old",
    "a question asking for advice",
]

CREATIVE_SUBJECTS = [
    "rudens lietų", "senąjį ąžuolą", "jūrą Palangoje", "močiutės sodą", "žiemos vakarą",
    "gandrus, grįžtančius pavasarį", "draugystę", "miestą naktį", "pirmąją meilę",
    "išdykusį katiną", "Kūčių vakarą", "Neries pakrantę", "vaikystės prisiminimus",
]

FORMAL_TASKS = [
    "Parašyk oficialų prašymą savivaldybei dėl gatvės apšvietimo įrengimo.",
    "Parašyk el. laišką darbdaviui, kuriame prašai leisti dirbti nuotoliniu būdu du kartus per savaitę.",
    "Parašyk motyvacinį laišką jaunesniojo programuotojo pareigoms užimti.",
    "Parašyk skundą parduotuvei dėl sugedusios prekės ir paprašyk grąžinti pinigus.",
    "Parašyk padėkos laišką mokytojai mokslo metų pabaigos proga.",
    "Parašyk kvietimą į įmonės vasaros šventę.",
    "Parašyk trumpą pranešimą spaudai apie naujos bibliotekos atidarymą.",
    "Parašyk prašymą universitetui dėl akademinių atostogų.",
    "Parašyk el. laišką klientui, kuriame atsiprašai dėl vėluojančio užsakymo.",
    "Parašyk gyvenimo aprašymo (CV) santrauką žmogui, turinčiam penkerių metų pardavimų patirtį.",
]

REASONING_KINDS = [
    "a primary-school arithmetic word problem",
    "a middle-school percentage or ratio word problem",
    "a high-school algebra word problem",
    "a logic puzzle with a unique answer",
    "a probability question",
    "an everyday planning problem that needs step-by-step reasoning (time, money, distances)",
]


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    task: str
    user: str | None = None
    reference: str | None = None
    instruction_request: str | None = None
    wrap: Callable[[str], str] | None = None
    meta: dict = field(default_factory=dict)


def _teacher_json_hint(key: str) -> str:
    return f'Return ONLY a JSON object of the form {{"{key}": "..."}} with no other text.'


def build_context_qa(rng: random.Random, ctx: str) -> Draft:
    style = rng.choice(QUESTION_STYLES[:5])
    req = (
        "Below is a passage in Lithuanian.\n\n"
        f"PASSAGE:\n{ctx}\n\n"
        f"Write {style} in Lithuanian whose answer is clearly contained in the passage. "
        "Do not copy a full sentence from the passage. Use natural, grammatical Lithuanian.\n"
        + _teacher_json_hint("instruction")
    )
    return Draft("context_qa", instruction_request=req, wrap=lambda q: context_qa_prompt(ctx, q), meta={"context": ctx})


def build_context_qa_unanswerable(rng: random.Random, ctx: str) -> Draft:
    req = (
        "Below is a passage in Lithuanian.\n\n"
        f"PASSAGE:\n{ctx}\n\n"
        "Write one question in Lithuanian that is on the same topic as the passage and sounds plausible, "
        "but whose answer is NOT contained in the passage.\n" + _teacher_json_hint("instruction")
    )
    return Draft(
        "context_qa_unanswerable",
        instruction_request=req,
        wrap=lambda q: context_qa_prompt(ctx, q),
        reference=UNANSWERABLE,
        meta={"context": ctx},
    )


def build_summarize(rng: random.Random, ctx: str) -> Draft:
    n = rng.choice([2, 3, 4])
    tmpl = rng.choice(
        [
            f"Apibendrink šį tekstą {n} sakiniais:\n\n{{c}}",
            "Parašyk trumpą šio teksto santrauką:\n\n{c}",
            "Išvardyk pagrindines šio teksto mintis punktais:\n\n{c}",
            "Kokia yra pagrindinė šio teksto mintis? Paaiškink trumpai.\n\n{c}",
        ]
    )
    return Draft("summarize", user=tmpl.format(c=ctx))


def build_rewrite(rng: random.Random, ctx: str) -> Draft:
    tmpl = rng.choice(
        [
            "Perrašyk šį tekstą paprastesne kalba, kad suprastų dvylikametis:\n\n{c}",
            "Perrašyk šį tekstą oficialiu, dalykiniu stiliumi:\n\n{c}",
            "Sutrumpink šį tekstą maždaug perpus, išsaugodamas svarbiausią informaciją:\n\n{c}",
            "Perrašyk šį tekstą kaip trumpą įrašą socialiniams tinklams:\n\n{c}",
        ]
    )
    return Draft("rewrite", user=tmpl.format(c=ctx))


def build_diacritics(rng: random.Random, ctx: str) -> Draft:
    sentences = _sentences(ctx)
    k = min(len(sentences), rng.randint(2, 4))
    start = rng.randint(0, len(sentences) - k)
    original = " ".join(sentences[start : start + k])
    stripped = strip_diacritics(original)
    if stripped == original:
        original, stripped = ctx, strip_diacritics(ctx)
    user = rng.choice(
        [
            "Atkurk lietuviškas raides (ą, č, ę, ė, į, š, ų, ū, ž) šiame tekste:\n\n{t}",
            "Šis tekstas parašytas be lietuviškų raidžių. Perrašyk jį taisyklingai:\n\n{t}",
        ]
    ).format(t=stripped)
    return Draft("diacritics", user=user, reference=original)


def build_grammar_fix(rng: random.Random, ctx: str) -> Draft:
    sentences = _sentences(ctx)
    k = min(len(sentences), rng.randint(2, 4))
    start = rng.randint(0, len(sentences) - k)
    original = " ".join(sentences[start : start + k])
    corrupted = corrupt(original, rng)
    if corrupted == original:
        corrupted = strip_diacritics(original)
    user = rng.choice(
        [
            "Ištaisyk rašybos ir skyrybos klaidas šiame tekste. Pateik tik ištaisytą tekstą.\n\n{t}",
            "Šiame tekste yra klaidų. Parašyk jį taisyklingai:\n\n{t}",
        ]
    ).format(t=corrupted)
    return Draft("grammar_fix", user=user, reference=original)


def build_translate(rng: random.Random, ctx: str) -> Draft:
    """Back-translation: a teacher translates authentic Lithuanian into English;
    the training target is the original human-written Lithuanian."""
    req = (
        "Translate the following Lithuanian text into fluent English. "
        "Preserve meaning, names and numbers exactly.\n\n"
        f"TEXT:\n{ctx}\n\n" + _teacher_json_hint("instruction")
    )
    prefix = rng.choice(["Išversk į lietuvių kalbą:\n\n", "Išversk šį tekstą į lietuvių kalbą:\n\n"])
    return Draft("translate_en_lt", instruction_request=req, wrap=lambda en: prefix + en, reference=ctx)


def build_essay(rng: random.Random, ctx: str | None = None) -> Draft:
    topic = rng.choice(ESSAY_TOPICS)
    words_n = rng.choice([300, 400, 500, 600])
    tmpl = rng.choice(
        [
            "Parašyk argumentuotą rašinį tema „{t}“. Rašinyje turi būti įžanga, bent du argumentai su pavyzdžiais "
            "ir apibendrinanti pabaiga. Apimtis – apie {n} žodžių.",
            "Parašyk samprotaujamąjį rašinį tema „{t}“. Remkis literatūros, istorijos ar gyvenimo pavyzdžiais. "
            "Apimtis – apie {n} žodžių.",
            "Sukurk rašinį tema „{t}“, tinkamą lietuvių kalbos brandos egzaminui. Aiškiai suformuluok tezę ir ją "
            "pagrįsk. Apimtis – apie {n} žodžių.",
        ]
    )
    return Draft("essay", user=tmpl.format(t=topic, n=words_n), meta={"topic": topic})


def build_open_qa(rng: random.Random, ctx: str | None = None) -> Draft:
    topic = rng.choice(KNOWLEDGE_TOPICS)
    style = rng.choice(QUESTION_STYLES)
    req = (
        f"Write {style} that a Lithuanian speaker might ask an AI assistant about the topic: {topic}. "
        "Write it in natural, grammatical Lithuanian, as a real user would type it.\n" + _teacher_json_hint("instruction")
    )
    return Draft("open_qa", instruction_request=req, wrap=lambda q: q, meta={"topic": topic})


def build_reasoning(rng: random.Random, ctx: str | None = None) -> Draft:
    kind = rng.choice(REASONING_KINDS)
    req = (
        f"Write {kind} in Lithuanian, set in a realistic Lithuanian everyday context (names, places, euros). "
        "It must have exactly one correct answer. Write only the problem, not the solution.\n"
        + _teacher_json_hint("instruction")
    )
    suffix = rng.choice(["", "\n\nSpręsk žingsnis po žingsnio.", "\n\nPaaiškink sprendimą."])
    return Draft("reasoning", instruction_request=req, wrap=lambda q: q + suffix, meta={"kind": kind})


def build_creative(rng: random.Random, ctx: str | None = None) -> Draft:
    subject = rng.choice(CREATIVE_SUBJECTS)
    tmpl = rng.choice(
        [
            "Parašyk trumpą eilėraštį apie {s}.",
            "Sukurk trumpą pasaką vaikams apie {s}.",
            "Parašyk trumpą apsakymą, kuriame svarbų vaidmenį atlieka {s}.",
            "Parašyk nuotaikingą sveikinimą, kuriame būtų paminėtas {s}.",
        ]
    )
    return Draft("creative", user=tmpl.format(s=subject))


def build_formal(rng: random.Random, ctx: str | None = None) -> Draft:
    return Draft("formal_writing", user=rng.choice(FORMAL_TASKS))


@dataclass(frozen=True)
class TaskSpec:
    builder: Callable[..., Draft]
    needs_context: bool


TASKS: dict[str, TaskSpec] = {
    "context_qa": TaskSpec(build_context_qa, True),
    "context_qa_unanswerable": TaskSpec(build_context_qa_unanswerable, True),
    "summarize": TaskSpec(build_summarize, True),
    "rewrite": TaskSpec(build_rewrite, True),
    "diacritics": TaskSpec(build_diacritics, True),
    "grammar_fix": TaskSpec(build_grammar_fix, True),
    "translate_en_lt": TaskSpec(build_translate, True),
    "essay": TaskSpec(build_essay, False),
    "open_qa": TaskSpec(build_open_qa, False),
    "reasoning": TaskSpec(build_reasoning, False),
    "creative": TaskSpec(build_creative, False),
    "formal_writing": TaskSpec(build_formal, False),
}

DEFAULT_TASK_WEIGHTS = {
    "context_qa": 0.22,
    "context_qa_unanswerable": 0.05,
    "summarize": 0.08,
    "rewrite": 0.05,
    "diacritics": 0.04,
    "grammar_fix": 0.06,
    "translate_en_lt": 0.06,
    "essay": 0.12,
    "open_qa": 0.16,
    "reasoning": 0.07,
    "creative": 0.05,
    "formal_writing": 0.04,
}


def sample_task(rng: random.Random, weights: dict[str, float] | None) -> str:
    weights = weights or DEFAULT_TASK_WEIGHTS
    unknown = set(weights) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s) in task_weights: {sorted(unknown)}")
    names = list(weights)
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄČĘĖĮŠŲŪŽ„\"(])")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text.replace("\n", " ")) if s.strip()]
    return parts or [text.strip()]


def corrupt(text: str, rng: random.Random, rate: float = 0.08) -> str:
    """Inject realistic typos: lost diacritics, dropped commas, swapped letters,
    wrong capitalization. Used to synthesize error-correction pairs whose
    target is the original (correct) text."""
    out: list[str] = []
    words_ = text.split(" ")
    for i, w in enumerate(words_):
        r = rng.random()
        if r < rate:
            w = strip_diacritics(w)
        elif r < rate * 1.6 and w.endswith(","):
            w = w[:-1]
        elif r < rate * 2.1 and len(w) > 4:
            j = rng.randint(1, len(w) - 3)
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        elif r < rate * 2.4 and i > 0 and w[:1].isupper():
            w = w[:1].lower() + w[1:]
        out.append(w)
    return " ".join(out)


def split_passages(doc: str, min_words: int = 60, max_words: int = 300) -> list[str]:
    """Split a document into paragraph-aligned passages for grounding."""
    passages, cur, n = [], [], 0
    for para in (p.strip() for p in doc.split("\n")):
        if not para:
            continue
        w = len(para.split())
        if n + w > max_words and n >= min_words:
            passages.append("\n".join(cur))
            cur, n = [], 0
        if w > max_words:  # very long paragraph: cut by sentences
            sent_buf, sn = [], 0
            for s in _sentences(para):
                sw = len(s.split())
                if sn + sw > max_words and sn >= min_words:
                    passages.append(" ".join(sent_buf))
                    sent_buf, sn = [], 0
                sent_buf.append(s)
                sn += sw
            if sent_buf:
                cur.append(" ".join(sent_buf))
                n += sn
            continue
        cur.append(para)
        n += w
    if cur and n >= min_words:
        passages.append("\n".join(cur))
    return passages
