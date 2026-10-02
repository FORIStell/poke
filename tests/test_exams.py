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
    assert set(cat) == {"nmpp8", "pupp10", "vbe12"}
    assert all(len({e["year"] for e in v}) >= 3 for v in cat.values())  # 3 different years per level
