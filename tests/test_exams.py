from pathlib import Path

import pytest

from gintaras.config import load_config
from gintaras.exams import (
    grade_objective,
    next_exam,
    parse_numbered,
    punctuation_errors,
    to_grade,
    word_errors,
)

ROOT = Path(__file__).resolve().parents[1]

REF = "Beveik per visą neriją driekiasi smėlio kopų grandinės. Ne veltui Kuršių nerija įtraukta į sąrašą."


def test_word_errors_counts_spelling_mistakes():
    assert word_errors(REF, REF) == 0
    bad = REF.replace("driekiasi", "driekesi").replace("grandinės", "grandynės")
    assert word_errors(bad, REF) == 2


def test_punctuation_errors():
    ref = "Pasak prodiuserio, ši juosta yra bendras darbas, o režisierius teigė, kad viskas gerai."
    assert punctuation_errors(ref, ref) == 0
    missing_two = "Pasak prodiuserio ši juosta yra bendras darbas, o režisierius teigė kad viskas gerai."
    assert punctuation_errors(missing_two, ref) == 2
    extra = "Pasak prodiuserio, ši juosta, yra bendras darbas, o režisierius teigė, kad viskas gerai."
    assert punctuation_errors(extra, ref) == 1


def test_objective_graders():
    choice = {"type": "choice", "answer": "B", "points": 1}
    assert grade_objective(choice, "B")[0] == 1
    assert grade_objective(choice, "Atsakymas: C")[0] == 0
    forms = {"type": "forms", "points": 2.5, "answers": [["garsiąją"], ["pirmiesiems"], ["klausydamiesi", "besiklausydami"]]}
    pts, _ = grade_objective(forms, "1. garsiąją\n2. pirmiems\n3. besiklausydami")
    assert pts == pytest.approx(2.5 * 2 / 3)
    table = {"type": "forms", "points": 2, "error_table": [2, 1, 1, 0, 0], "answers": [["a"], ["b"], ["c"], ["d"]]}
    assert grade_objective(table, "1. a\n2. b\n3. x\n4. d")[0] == 1
    fill = {"type": "fill_text", "points": 3, "reference": REF, "error_table": [3, 2.5, 2, 1.5, 1, 0.5, 0]}
    assert grade_objective(fill, REF.replace("visą", "visa"))[0] == 2.5


def test_parse_numbered_fallback():
    assert parse_numbered("garsiąją, pirmiesiems", 2) == ["garsiąją", "pirmiesiems"]


def test_grades():
    exam = {"questions": [{"points": 20}]}
    assert to_grade(exam, 15) == 8  # 75 % -> 8
    assert to_grade(exam, 14) == 7
    assert to_grade({"questions": [], "max_points": 30, "grade_table": [[0, 1], [10, 4], [24, 8]]}, 25) == 8


def test_next_exam_prefers_untaken_years():
    exams = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    state = {"history": [{"exam_id": "a", "model": "m"}, {"exam_id": "b", "model": "m"}]}
    assert next_exam(state, exams, "m")["id"] == "c"
    state["history"].append({"exam_id": "c", "model": "m"})
    assert next_exam(state, exams, "m")["id"] == "a"


def test_catalog_lists_all_levels():
    from gintaras.exams import load_catalog

    cat = load_catalog(load_config(ROOT / "configs/gintaras-1.7b-cpu.yaml"))
    assert set(cat) == {"nmpp2", "nmpp4", "nmpp6", "nmpp8", "pupp10", "vbe12"}
    assert all(len({e["year"] for e in v}) >= 3 for v in cat.values())  # 3 different years per level


def test_vllm_worker_protocol(tmp_path, monkeypatch):
    """The vLLM subprocess reads requests and writes (text, mean logprob) rows."""
    import json
    import sys
    import types
    from argparse import Namespace

    from gintaras import fastgen

    class Out:
        def __init__(self, text):
            self.text, self.token_ids, self.cumulative_logprob = f" {text} ", [1, 2], -1.0

    class LLM:
        def __init__(self, **kw):
            self.kw = kw

        def chat(self, convs, params, use_tqdm=False):
            return [types.SimpleNamespace(outputs=[Out(c[-1]["content"])] * params.n) for c in convs]

    fake = types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "vllm", fake)
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"convs": [[{"role": "user", "content": "Labas"}]], "max_tokens": 8,
                               "temperature": 0.7, "n": 2}))
    fastgen._worker(Namespace(model="m", inp=str(inp), out=str(out), max_model_len=64, gpu_util=0.8))
    assert json.loads(out.read_text()) == [[["Labas", -0.5], ["Labas", -0.5]]]


def test_engine_falls_back_to_hf_without_gpu():
    from gintaras.fastgen import engine

    cfg = load_config(ROOT / "configs/smoke.yaml")
    assert engine(cfg) == "hf"


def test_fill_text_without_error_table_uses_default():
    from gintaras.exams import grade_objective

    q = {"type": "fill_text", "points": 3, "reference": "Vilnius yra Lietuvos sostinė"}
    assert grade_objective(q, "Vilnius yra Lietuvos sostinė")[0] == 3
    assert grade_objective(q, "Vilnius yra Lietuvos sostine")[0] == 2


def test_objective_question_without_key_is_not_gradable():
    from gintaras.exams import has_answer_key

    assert not has_answer_key({"type": "choice", "points": 1})
    assert has_answer_key({"type": "choice", "points": 1, "answer": "B"})
    assert has_answer_key({"type": "open", "points": 1})


def test_multi_select_needs_exact_set():
    from gintaras.exams import grade_objective

    q = {"type": "multi", "points": 1, "answer": "a, c, d"}
    assert grade_objective(q, "a, c, d")[0] == 1
    assert grade_objective(q, "Teisingi: c, a ir d")[0] == 1
    assert grade_objective(q, "a, c")[0] == 0
    assert grade_objective(q, "a, b, c, d")[0] == 0


def test_order_needs_exact_sequence():
    from gintaras.exams import grade_objective

    q = {"type": "order", "points": 1, "answer": "d, e, a, c, b"}
    assert grade_objective(q, "d, e, a, c, b")[0] == 1
    assert grade_objective(q, "d, e, a, b, c")[0] == 0
