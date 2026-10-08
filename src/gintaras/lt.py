"""Lithuanian text utilities: language scoring, cleaning, diacritics, normalization.

These are dependency-free heuristics. They are used to
  * filter pretraining text and teacher outputs (is this really Lithuanian?),
  * build self-supervised tasks (diacritic restoration),
  * normalize answers for QA metrics (Lithuanian is highly inflected).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

LT_LETTERS = "ąčęėįšųūž"
LT_LETTERS_ALL = LT_LETTERS + LT_LETTERS.upper()
_STRIP_MAP = str.maketrans("ąčęėįšųūžĄČĘĖĮŠŲŪŽ", "aceeisuuzACEEISUUZ")

# Letters that are common in neighbouring languages (Latvian, Polish, German,
# Estonian, ...) but never appear in standard Lithuanian orthography.
_FOREIGN_LETTERS = set("āēīōģķļņŗõäöüßłśżźćńřůěýťďňáéíóúàèìòùâêîôûñçøåæœ")

# High-frequency Lithuanian function words. A real Lithuanian text has a
# substantial share of its tokens in this set. Words that are also common in
# English ("to", "be", "per", "net", "ten", "o") are deliberately left out.
LT_STOPWORDS = frozenset(
    """
    ir yra kad su į iš bet tai kaip ar jo jos jų buvo nuo taip dėl apie
    kuris kuri kurie kurios kurį kurią kurio kuriuo kuriame kur kai kas ką
    kam kuo jau dar tik arba nes jei jeigu todėl tačiau bei prie po prieš
    tarp už ant pagal iki link šis ši šie šios šį šią šio tas tie tos tą
    jis ji jie jam jai jiems joms mes jūs aš tu mūsų jūsų mano tavo
    savo save sau ne nėra būti bus būtų gali galima reikia labai daug mažai
    pat pats pati patys vienas viena vieno metu metais metų kitas kita
    kiti kitos kiek kada čia visi visos visas visa viską kiekvienas
    """.split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_WS_RE = re.compile(r"[ \t ]+")
_MANY_NL_RE = re.compile(r"\n{3,}")
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​﻿]")


def normalize_unicode(text: str) -> str:
    """NFC-normalize so that e.g. 'e' + combining dot == 'ė'."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Light cleaning that keeps document structure (paragraphs) intact."""
    text = normalize_unicode(text)
    text = _CTRL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NL_RE.sub("\n\n", text)
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def lt_score(text: str) -> float:
    """Return a 0..1 score for how likely `text` is standard Lithuanian.

    Combines (a) share of Lithuanian function words, (b) density of
    Lithuanian-specific letters, (c) Latin script share, minus a penalty for
    letters that Lithuanian never uses. Calibrated so that ordinary Lithuanian
    prose scores > 0.7 and English/Latvian/Polish/Russian score < 0.4.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return 0.0
    n = len(letters)
    lower = [c.lower() for c in letters]
    latin = sum(1 for c in lower if "a" <= c <= "z" or c in LT_LETTERS or c in _FOREIGN_LETTERS)
    lt_chars = sum(1 for c in lower if c in LT_LETTERS)
    foreign = sum(1 for c in lower if c in _FOREIGN_LETTERS)

    toks = words(text)
    if not toks:
        return 0.0
    stop_ratio = sum(1 for t in toks if t in LT_STOPWORDS) / len(toks)

    stop_part = min(1.0, stop_ratio / 0.15)
    diac_part = min(1.0, (lt_chars / n) / 0.02)
    latin_part = latin / n
    penalty = min(1.0, (foreign / n) / 0.01)

    score = 0.5 * stop_part + 0.3 * diac_part + 0.2 * latin_part - 0.6 * penalty
    if latin_part < 0.8:  # mostly non-Latin script (e.g. Cyrillic)
        score *= latin_part
    return max(0.0, min(1.0, score))


def strip_diacritics(text: str) -> str:
    """'Ąžuolas žaliuoja' -> 'Azuolas zaliuoja' (only Lithuanian letters)."""
    return text.translate(_STRIP_MAP)


def diacritic_density(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in LT_LETTERS_ALL) / len(letters)


def dedup_key(text: str) -> str:
    """Hash of an aggressively normalized form, for exact/near-exact dedup."""
    norm = " ".join(words(strip_diacritics(text)))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def stem(word: str, keep: int = 5) -> str:
    """Crude Lithuanian 'stemmer': diacritic-free prefix of the word.

    Lithuanian inflects nouns into 7 cases x 2 numbers, so exact token match
    badly under-counts correct answers ("Vilnius" vs "Vilniuje"). Comparing
    fixed-length prefixes is a cheap, language-agnostic approximation of
    lemma matching for answer F1.
    """
    w = strip_diacritics(word.lower())
    return w[:keep] if len(w) > keep else w


def answer_tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text)]
