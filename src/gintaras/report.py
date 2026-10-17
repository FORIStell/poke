"""Progress report image: training loss, eval metrics (base vs latest), exam grades.

    gintaras report -c configs/gintaras-1.7b-cpu.yaml   ->  <output_dir>/progress.png
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from gintaras.config import Config
from gintaras.exams import load_ladder

log = logging.getLogger(__name__)

BASE_COLOR, MODEL_COLOR = "#2F6DB5", "#C46A0A"  # validated pair (CVD-safe, >=3:1 on white)
INK, MUTED, GRID = "#2b2118", "#7a6a58", "#eadfce"
METRICS = {
    "qa_f1": "Answers from text (F1)",
    "qa_unanswerable_acc": "Says 'not in text'",
    "diacritics_word_acc": "Restores ą č ę ė į š ų ū ž",
    "lt_consistency": "Answers in Lithuanian",
}


def _loss_history(cfg: Config) -> list[tuple[int, float]]:
    best: list[tuple[int, float]] = []
    for state in cfg.out.glob("checkpoints/**/trainer_state.json"):
        hist = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
        pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        if len(pts) > len(best):
            best = pts
    return best


def _evals(cfg: Config) -> list[tuple[str, dict]]:
    d = cfg.out / "eval"
    if not d.exists():
        return []
    files = sorted((p for p in d.glob("*.json")), key=lambda p: p.stat().st_mtime)
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in files]


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_report(cfg: Config, out: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or cfg.out / "progress.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)
    fig.patch.set_facecolor("white")

    # 1) training loss
    ax = axes[0]
    _style(ax, "Training loss (lower is better)")
    loss = _loss_history(cfg)
    if loss:
        xs, ys = zip(*loss)
        ax.plot(xs, ys, color=MODEL_COLOR, linewidth=2)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 4), textcoords="offset points", color=INK, fontsize=9)
        ax.set_xlabel("step", color=MUTED, fontsize=9)
    else:
        ax.text(0.5, 0.5, "no training yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)

    # 2) eval metrics: first eval (base) vs latest
    ax = axes[1]
    _style(ax, "Lithuanian skills (higher is better)")
    evals = _evals(cfg)
    if evals:
        base_name, base = evals[0]
        last_name, last = evals[-1]
        keys = [k for k in METRICS if k in base or k in last]
        y = range(len(keys))
        h = 0.36
        ax.barh([i + h / 2 for i in y], [100 * base.get(k, 0) for k in keys], height=h, color=BASE_COLOR,
                label=f"before ({base_name})")
        if last_name != base_name:
            ax.barh([i - h / 2 for i in y], [100 * last.get(k, 0) for k in keys], height=h, color=MODEL_COLOR,
                    label=f"now ({last_name})")
        for i, k in enumerate(keys):
            ax.text(100 * base.get(k, 0) + 1, i + h / 2, f"{100 * base.get(k, 0):.0f}%", va="center", fontsize=8, color=INK)
            if last_name != base_name:
                ax.text(100 * last.get(k, 0) + 1, i - h / 2, f"{100 * last.get(k, 0):.0f}%", va="center", fontsize=8, color=INK)
        ax.set_yticks(list(y), [METRICS[k] for k in keys], color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 110)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no evaluation yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)

    # 3) exam grades
    ax = axes[2]
    _style(ax, "Exam grades (pass = 8/10)")
    hist = [h for h in load_ladder(cfg)["history"] if not h.get("skipped")]
    if hist:
        labels = [h["exam_id"].replace("-skaitymas", "\nreading").replace("-testas", "\ntest") for h in hist[-8:]]
        grades = [h["grade"] for h in hist[-8:]]
        bars = ax.bar(range(len(grades)), grades, color=MODEL_COLOR, width=0.6)
        for b, g, hh in zip(bars, grades, hist[-8:]):
            ax.text(b.get_x() + b.get_width() / 2, g + 0.15, f"{g:.0f}{' ≈' if hh.get('approximate') else ''}",
                    ha="center", fontsize=9, color=INK)
        ax.axhline(cfg.exam.pass_grade, color=INK, linewidth=1, linestyle="--")
        ax.text(len(grades) - 0.5, cfg.exam.pass_grade + 0.15, "pass", ha="right", fontsize=8, color=INK)
        ax.set_xticks(range(len(grades)), labels, fontsize=8, color=INK)
        ax.set_ylim(0, 10.5)
    else:
        ax.text(0.5, 0.5, "no exams yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_ylim(0, 10.5)

    fig.suptitle(f"Gintaras progress · {cfg.model.base} · {datetime.now():%Y-%m-%d %H:%M}",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    log.info("Report written to %s", out)
    return out
