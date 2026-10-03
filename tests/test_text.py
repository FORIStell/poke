import random

from gintaras.data.tasks import (
    TASKS,
    UNANSWERABLE,
    context_qa_prompt,
    corrupt,
    sample_task,
    split_passages,
)
from gintaras.lt import clean_text, dedup_key, lt_score, stem, strip_diacritics

LT = (
    "Lietuva yra valstybė Baltijos jūros rytinėje pakrantėje. Jos sostinė yra Vilnius, "
    "o didžiausi miestai – Kaunas ir Klaipėda. Šalis atkūrė nepriklausomybę 1990 metais."
)


def test_lt_score_separates_languages():
    assert lt_score(LT) > 0.8
    assert lt_score("Lithuania is a country in the Baltic region. Its capital is Vilnius and it is in the EU.") < 0.4
    assert lt_score("Latvija ir valsts Baltijas jūras austrumu krastā. Tās galvaspilsēta ir Rīga.") < 0.4
    assert lt_score("Litwa jest państwem położonym nad Morzem Bałtyckim. Jej stolicą jest Wilno.") < 0.4
    assert lt_score("Литва — государство на восточном побережье Балтийского моря.") < 0.2
    assert lt_score("per trumpas") == 0.0


def test_diacritics_and_cleaning():
    assert strip_diacritics("Ąžuolas žaliuoja, ėjo į šilą") == "Azuolas zaliuoja, ejo i sila"
    assert clean_text("  Labas​   rytas \r\n\n\n\nVilniau ") == "Labas rytas\n\nVilniau"
    # NFC: 'e' + combining dot above == 'ė'
    assert clean_text("ė") == "ė"


def test_dedup_key_ignores_case_whitespace_and_diacritics():
    assert dedup_key("Labas   rytas, Lietuva!") == dedup_key("labas rytas lietuva")
    assert dedup_key("Žąsis") == dedup_key("zasis")
    assert dedup_key("Labas rytas") != dedup_key("Labas vakaras")


def test_stem_matches_inflected_forms():
    assert stem("Vilnius") == stem("Vilniuje") == stem("Vilniaus")


def test_split_passages_respects_bounds():
    para = " ".join(["žodis"] * 50)
    doc = "\n".join([para] * 10)
    passages = split_passages(doc, min_words=60, max_words=120)
    assert passages
    assert all(60 <= len(p.split()) <= 120 for p in passages)


def test_corrupt_changes_text_but_keeps_length_scale():
    rng = random.Random(1)
    text = LT * 3
    bad = corrupt(text, rng, rate=0.3)
    assert bad != text
    assert abs(len(bad) - len(text)) < len(text) * 0.1


def test_every_task_builds_a_draft():
    rng = random.Random(0)
    ctx = LT + " " + LT
    for name, spec in TASKS.items():
        draft = spec.builder(rng, ctx) if spec.needs_context else spec.builder(rng)
        assert draft.task == name
        assert draft.user or draft.instruction_request
        if draft.instruction_request:
            assert draft.wrap is not None


def test_self_supervised_tasks_have_gold_references():
    rng = random.Random(0)
    d = TASKS["diacritics"].builder(rng, LT)
    assert strip_diacritics(d.reference) in d.user and d.reference not in d.user
    g = TASKS["grammar_fix"].builder(rng, LT)
    assert g.reference and g.reference in LT
    u = TASKS["context_qa_unanswerable"].builder(rng, LT)
    assert u.reference == UNANSWERABLE
    assert u.wrap("Kiek?") == context_qa_prompt(LT, "Kiek?")


def test_sample_task_respects_weights():
    rng = random.Random(0)
    picks = [sample_task(rng, {"essay": 1.0, "open_qa": 0.0}) for _ in range(50)]
    assert set(picks) == {"essay"}
