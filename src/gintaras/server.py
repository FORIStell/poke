"""Tiny web server for the Gintaras front page (stdlib only).

    gintaras serve -c configs/gintaras-1.7b-cpu.yaml [--model PATH] [--port 8080]

GET  /            front page (web/index.html)
GET  /api/status  model name, exam-ladder progress, latest training/eval numbers
POST /api/chat    {"messages": [...], "strength": "low|medium|max"} -> {"reply", "seconds", "strength"}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gintaras.config import Config
from gintaras.evaluate import REPO_ROOT

log = logging.getLogger(__name__)
WEB = REPO_ROOT / "web"
TYPES = {".html": "text/html; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".css": "text/css"}


def training_summary(cfg: Config) -> dict:
    """Latest numbers for the progress card."""
    out: dict = {}
    eval_dir = cfg.out / "eval"
    evals = sorted(eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime) if eval_dir.exists() else []
    evals = [p for p in evals if not p.name.endswith("_samples.jsonl")]
    if evals:
        res = json.loads(evals[-1].read_text(encoding="utf-8"))
        out["Paskutinis įvertinimas"] = evals[-1].stem
        names = {"qa_f1": "Atsakymai iš teksto (F1)", "diacritics_word_acc": "Lietuviškos raidės",
                 "lt_consistency": "Atsako lietuviškai", "bpc": "Kalbos modelis (bpc ↓)",
                 "qa_unanswerable_acc": "Atpažįsta, kai atsakymo nėra", "judge_overall": "Teisėjo balas"}
        for k, label in names.items():
            if k in res:
                v = res[k]
                out[label] = f"{v * 100:.0f}%" if k not in ("bpc", "judge_overall") else f"{v:.2f}"
    for state in sorted(cfg.out.glob("checkpoints/*_adapter/trainer_state.json")) + sorted(
            cfg.out.glob("checkpoints/*_adapter/checkpoint-*/trainer_state.json")):
        st = json.loads(state.read_text(encoding="utf-8"))
        losses = [h["loss"] for h in st.get("log_history", []) if "loss" in h]
        if losses:
            out["Mokymo žingsniai"] = f"{st.get('global_step', 0)} / {st.get('max_steps', '?')}"
            out["Klaida (loss)"] = f"{losses[0]:.2f} → {losses[-1]:.2f}"
    return out


class App:
    def __init__(self, cfg: Config, model_path: str):
        self.cfg, self.model_path = cfg, model_path
        self.model = self.tok = None
        self.lock = threading.Lock()

    def ensure_model(self):
        if self.model is None:
            from gintaras.generation import load_for_inference

            log.info("Loading %s", self.model_path)
            self.model, self.tok = load_for_inference(self.model_path, self.cfg.model.bf16)

    def chat(self, messages: list[dict], strength: str) -> dict:
        from gintaras.generation import generate_with_strength, with_system

        msgs = [{"role": m["role"], "content": str(m["content"])} for m in messages
                if m.get("role") in ("user", "assistant") and m.get("content")]
        if not msgs or msgs[-1]["role"] != "user":
            raise ValueError("last message must be from the user")
        with self.lock:
            self.ensure_model()
            t0 = time.time()
            reply = generate_with_strength(self.model, self.tok, [with_system(msgs, self.cfg.system_prompt)], strength)[0]
        return {"reply": reply, "seconds": time.time() - t0, "strength": strength}

    def status(self) -> dict:
        from gintaras.exams import load_ladder

        return {"model": self.model_path, "model_name": f"Gintaras · {Path(self.model_path).name}",
                "levels": self.cfg.exam.levels, "ladder": load_ladder(self.cfg),
                "training": training_summary(self.cfg)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/status":
                return self._json(200, app.status())
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (WEB / name).resolve()
            if WEB.resolve() in f.parents and f.is_file():
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/chat":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._json(200, app.chat(body.get("messages", []), body.get("strength", "medium")))
            except ValueError as e:
                self._json(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def serve(cfg: Config, model_path: str, host: str = "127.0.0.1", port: int = 8080, preload: bool = False) -> None:
    app = App(cfg, model_path)
    if preload:
        app.ensure_model()
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    log.info("Gintaras front page on http://%s:%d", host, port)
    srv.serve_forever()
