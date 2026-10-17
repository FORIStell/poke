"""End-to-end CPU smoke run: every stage with a tiny model and dummy teachers.

Downloads a ~5 MB test model from the Hugging Face Hub. Run with:
    pytest -m slow
"""

import json
from pathlib import Path

import pytest

from gintaras.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    main(["all", "-c", "configs/smoke.yaml", "--set", f"output_dir={out}"])

    for name in ("cpt", "sft"):
        assert (out / "checkpoints" / name / "config.json").exists(), name
    base_eval = json.loads((out / "eval" / "base.json").read_text())
    assert "bpc" in base_eval and "qa_f1" in base_eval
    best = (out / "BEST_MODEL").read_text().strip()
    assert Path(best, "config.json").exists()
    history = json.loads((out / "loop_history.json").read_text())
    assert history[0]["round"] == 0
