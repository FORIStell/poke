import json
from pathlib import Path

import pytest

from gintaras.backends import DummyBackend, parse_json_object
from gintaras.config import EndpointConfig, load_config
from gintaras.data.prepare import ADAPTERS, prepare_contexts, prepare_corpus, prepare_instructions
from gintaras.data.synth import Prompt, is_refusal, reference_rows, select, synthesize
from gintaras.data.tasks import UNANSWERABLE
from gintaras.evaluate import diacritics_word_accuracy, summary_score, token_f1
from gintaras.judge import Score, judge_many, parse_score
from gintaras.loop import round_data
from gintaras.train import to_prompt_completion
from gintaras.utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    return load_config(ROOT / "configs/smoke.yaml", {"output_dir": str(tmp_path / "run")})


def test_configs_load():
    full = load_config(ROOT / "configs/gintaras-9b.yaml")
    assert full.model.base == "utter-project/EuroLLM-9B-Instruct-2512"
    assert len(full.teachers) == 2 and full.judge is not None
    assert abs(sum(full.synth.task_weights.values()) - 1.0) < 1e-6


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  bse: x\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(p)


def test_config_overrides(tmp_path):
    cfg = load_config(ROOT / "configs/smoke.yaml", {"sft.epochs": 3, "model.base": "foo"})
    assert cfg.sft.epochs == 3.0 and cfg.model.base == "foo"


def test_parse_json_object_is_tolerant():
    assert parse_json_object('Sure!\n```json\n{"a": "b {x}"}\n```') == {"a": "b {x}"}
    assert parse_json_object('<think>{"no": 1}</think>{"yes": 2}') == {"yes": 2}
    assert parse_json_object("no json here") is None


def test_parse_score_clamps_and_validates():
    s = parse_score('{"faithfulness": 12, "language": 8, "helpfulness": "7", "overall": 0, "issues": "x"}')
    assert (s.faithfulness, s.language, s.helpfulness, s.overall) == (10.0, 8.0, 7.0, 1.0)
    assert parse_score('{"overall": 5}') is None


def test_judge_caps_non_lithuanian_answers():
    judge = DummyBackend(EndpointConfig(name="j", backend="dummy"))
    english = "This answer is written entirely in English and therefore should be rejected by the judge. " * 2
    lithuanian = "Lietuva yra valstybė Baltijos jūros rytinėje pakrantėje. Jos sostinė yra Vilnius, o šalis yra ES narė."
    msgs = [{"role": "user", "content": "Papasakok apie Lietuvą."}]
    en, lt = judge_many(judge, [(msgs, english, None), (msgs, lithuanian, None)])
    assert en.overall <= 3.0 and en.language == 1.0
    assert lt.overall >= 8.0


def test_ai_boilerplate_filter(cfg):
    from gintaras.data.prepare import keep_conversation

    convo = lambda a: [{"role": "user", "content": "?"}, {"role": "assistant", "content": a}]  # noqa: E731
    assert not keep_conversation(convo("Kaip AI, aš negaliu to padaryti."), cfg)
    assert keep_conversation(convo("Paaiškinsiu kaip aiškiai parašyti rašinį."), cfg)


def test_adapters():
    assert ADAPTERS["alpaca"]({"instruction": " Išvardyk ", "input": "nan", "output": "Gerai."})[0]["content"] == "Išvardyk"
    msgs = ADAPTERS["context_qa"]({"context": "Tekstas.", "question": "Kas?", "answer": "Tai."})
    assert "Tekstas:" in msgs[0]["content"] and msgs[1]["content"] == "Tai."
    assert ADAPTERS["messages"]({"conversations": [{"from": "human", "value": "Labas"}, {"from": "gpt", "value": "Sveiki"}]})


def test_prepare_on_fixtures(cfg):
    stats = prepare_corpus(cfg)
    assert stats["kept"] + stats["eval"] > 10
    rows = list(read_jsonl(cfg.data_path("corpus.jsonl")))
    assert all(r["text"] and r["source"] for r in rows)
    istats = prepare_instructions(cfg)
    assert istats["kept"] > 0 and istats["eval"] > 0
    assert prepare_contexts(cfg) > 0


def test_synthesize_with_dummy_teachers(cfg):
    prepare_corpus(cfg)
    prepare_contexts(cfg)
    stats = synthesize(cfg)
    assert stats["prompts"] > 0 and stats["sft"] > 0
    rows = list(read_jsonl(cfg.data_path("synth_sft.jsonl")))
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)


def test_select_builds_sft_and_dpo():
    p = Prompt(id="1", task="essay", user="Parašyk rašinį.")
    good, bad = Score(9, 9, 9, 9), Score(4, 4, 4, 4)
    sft, dpo = select([p], [[("a", "geras", good), ("b", "blogas", bad)]], min_score=8, margin=2)
    assert sft[0]["messages"][-1]["content"] == "geras"
    assert dpo[0]["chosen"][0]["content"] == "geras" and dpo[0]["rejected"][0]["content"] == "blogas"
    sft, dpo = select([p], [[("a", "vidutinis", Score(6, 6, 6, 6))]], min_score=8, margin=2)
    assert not sft and not dpo


def test_unanswerable_rows_need_teacher_agreement():
    p = Prompt(id="1", task="context_qa_unanswerable", user="...", reference=UNANSWERABLE)
    agree = [("a", "Tekste šios informacijos nėra."), ("b", "Apie tai tekste nekalbama.")]
    disagree = [("a", "Atsakymas yra 42."), ("b", "Tekste šios informacijos nėra.")]
    assert len(reference_rows([p], [agree])) == 1
    assert reference_rows([p], [disagree]) == []
    assert is_refusal(UNANSWERABLE)


def test_loop_round_data_uses_student_negatives():
    p = Prompt(id="1", task="open_qa", user="Kas?")
    scored = [[("student", "silpnas", Score(3, 3, 3, 3)), ("teacher", "puikus", Score(9, 9, 9, 9))]]
    dpo, sft, student_scores = round_data([p], scored, min_score=8, margin=2)
    assert dpo[0]["rejected"][0]["content"] == "silpnas" and dpo[0]["chosen_by"] == "teacher"
    assert sft[0]["messages"][-1]["content"] == "puikus"
    assert student_scores == [3]


def test_prompt_completion_format():
    row = to_prompt_completion(
        [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}], system="S"
    )
    assert row["prompt"][0] == {"role": "system", "content": "S"}
    assert row["completion"] == [{"role": "assistant", "content": "B"}]


def test_metrics():
    assert token_f1("Vilniuje", "Vilnius") == 1.0
    assert token_f1("Kaune", "Vilnius") == 0.0
    original = "Ąžuolas žaliuoja prie upės."
    assert diacritics_word_accuracy(original, original) == 1.0
    assert diacritics_word_accuracy("Azuolas zaliuoja prie upes.", original) == 0.0
    assert summary_score({"judge_overall": 8.5, "qa_f1": 0.1}) == 8.5
    assert summary_score({"qa_f1": 0.5, "lt_consistency": 1.0}) == 7.5


def test_eval_files_are_valid():
    gold = list(read_jsonl(ROOT / "data/eval/gold_context_qa.jsonl"))
    assert gold and all(r["question"] and r["answer"] and r["context"] for r in gold)
    prompts = list(read_jsonl(ROOT / "data/eval/judge_prompts.jsonl"))
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert json.dumps(prompts, ensure_ascii=False)


def test_loop_resumes_from_best_accepted_round(cfg, tmp_path):
    import json

    from gintaras.loop import _resume_state

    good = tmp_path / "loop_r2"
    good.mkdir()
    (good / "config.json").write_text("{}")
    hist = [
        {"round": 0, "model": str(tmp_path / "gone"), "eval": {"qa_f1": 0.1}},
        {"round": 1, "model": str(tmp_path / "loop_r1"), "eval": {"qa_f1": 0.9}, "accepted": False},
        {"round": 2, "model": str(good), "eval": {"judge_overall": 7.5}, "accepted": True},
    ]
    path = tmp_path / "loop_history.json"
    path.write_text(json.dumps(hist))
    history, current, score = _resume_state(cfg, path)
    assert current == str(good) and score == 7.5 and history[-1]["round"] == 2
    assert _resume_state(cfg, tmp_path / "missing.json") is None


def test_unanswerable_rows_pair_questions_with_unrelated_text(cfg):
    import random

    from gintaras.data.synth import unanswerable_rows
    from gintaras.utils import append_jsonl

    append_jsonl(cfg.data_path("instructions.jsonl"), [
        {"task": "qa", "messages": [{"role": "user", "content": "Kada įkurtas Vilniaus universitetas?"},
                                    {"role": "assistant", "content": "1579 metais."}]}])
    contexts = ["Vilniaus universitetas – seniausias Lietuvos universitetas.", "Nemunas – ilgiausia Lietuvos upė."]
    rows = unanswerable_rows(cfg, contexts, random.Random(0), k=20)
    assert rows and all(r["messages"][1]["content"] == UNANSWERABLE for r in rows)
    assert all("Nemunas" in r["messages"][0]["content"] for r in rows)  # the related passage is never used
    assert is_refusal(UNANSWERABLE)


def test_remove_answer_drops_the_answer_sentence():
    from gintaras.data.synth import remove_answer

    ctx = ("Nida yra kurortinė gyvenvietė Kuršių nerijoje. Joje gyvena apie 1650 gyventojų. "
           "Nidoje stovi Thomo Manno vasarnamis, kuriame dabar veikia memorialinis muziejus. "
           "Kopos aplink gyvenvietę yra saugomos, o vasarą čia atvyksta daugybė poilsiautojų iš visos Europos.")
    out = remove_answer(ctx, "Kiek gyventojų gyvena Nidoje?", "Apie 1650 gyventojų.")
    assert out and "1650" not in out and "Thomo Manno" in out
    assert remove_answer(ctx, "Kas tai?", "Visai kas kita") is None  # nothing removed → no example


def test_reading_exercises_have_checkable_answers():
    import random

    from gintaras.data.reading import build_order, build_true_statements, build_word_choice

    ctx = ("Vilnius yra Lietuvos sostinė ir didžiausias šalies miestas. Mieste gyvena daugiau kaip "
           "pusė milijono žmonių. Senamiestis įtrauktas į pasaulio paveldo sąrašą. Per miestą teka "
           "Neris ir Vilnelė. Vilniaus universitetas įkurtas šešioliktame amžiuje. Gedimino pilies "
           "bokštas stovi ant aukštos kalvos. Kiekvieną pavasarį mieste vyksta Kaziuko mugė.")
    rng = random.Random(0)
    user, ans = build_word_choice(rng, ctx)
    assert "______" in user and ans in "abcd" and len(ans) == 1
    user, ans = build_order(rng, ctx)
    letters = [a.strip() for a in ans.split(",")]
    assert sorted(letters) == sorted(set(letters)) and len(letters) >= 3
    made = build_true_statements(random.Random(3), ctx)
    assert made is None or all(a.strip() in "abcde" for a in made[1].split(","))


def test_own_system_prompt_is_kept():
    rows = to_prompt_completion([{"role": "system", "content": "EGZAMINAS"}, {"role": "user", "content": "k"},
                                 {"role": "assistant", "content": "a"}], "numatytasis")
    assert rows["prompt"][0]["content"] == "EGZAMINAS"
