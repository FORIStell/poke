"""Generation engine switch: vLLM (fast, GPU) or Hugging Face `generate` (CPU/fallback).

vLLM runs in a short-lived subprocess so its GPU memory is fully released
before training resumes in the main process.

    generate(cfg, model_path, convs, max_new_tokens, temperature, n) -> [[(text, mean_logprob), ...], ...]
    generate_best(cfg, model_path, convs, strength, max_new_tokens)   -> [text, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gintaras.config import Config

log = logging.getLogger(__name__)
Message = dict[str, str]
Candidate = tuple[str, float]


def vllm_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def engine(cfg: Config) -> str:
    choice = cfg.generation_engine
    if choice == "auto":
        return "vllm" if vllm_available() else "hf"
    if choice not in ("hf", "vllm"):
        raise ValueError(f"generation_engine must be auto, hf or vllm (got {choice})")
    return choice


def _hf_generate(hf, convs, max_new_tokens, temperature, n, batch_size, score) -> list[list[Candidate]]:
    from gintaras.generation import generate_chat, mean_logprob

    model, tok = hf
    out: list[list[Candidate]] = [[] for _ in convs]
    for _ in range(n):
        texts = generate_chat(model, tok, convs, max_new_tokens=max_new_tokens, temperature=temperature,
                              batch_size=batch_size)
        for i, t in enumerate(texts):
            out[i].append((t, mean_logprob(model, tok, convs[i], t) if score and t else 0.0))
    return out


def _vllm_generate(cfg: Config, model_path: str, convs, max_new_tokens, temperature, n) -> list[list[Candidate]]:
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / "in.json", Path(d) / "out.json"
        inp.write_text(json.dumps({"convs": convs, "max_tokens": max_new_tokens, "temperature": temperature, "n": n},
                                  ensure_ascii=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "gintaras.fastgen", "--model", model_path, "--in", str(inp), "--out", str(outp),
               "--max-model-len", str(cfg.model.max_seq_len + max_new_tokens)]
        subprocess.run(cmd, check=True, env=os.environ.copy())
        return [[(t, s) for t, s in row] for row in json.loads(outp.read_text(encoding="utf-8"))]


def generate(cfg: Config, model_path: str, convs: list[list[Message]], max_new_tokens: int,
             temperature: float = 0.0, n: int = 1, hf=None, score: bool = False) -> list[list[Candidate]]:
    """`score=True` fills in mean token log-probabilities (always present with vLLM)."""
    if not convs:
        return []
    if engine(cfg) == "vllm":
        return _vllm_generate(cfg, model_path, convs, max_new_tokens, temperature, n)
    if hf is None:
        from gintaras.generation import load_for_inference

        hf = load_for_inference(model_path, cfg.model.bf16)
    return _hf_generate(hf, convs, max_new_tokens, temperature, n, cfg.eval.batch_size, score)


def generate_best(cfg: Config, model_path: str, convs: list[list[Message]], strength: str,
                  max_new_tokens: int | None = None, hf=None) -> list[str]:
    """Strength levels: low/medium = one answer; max = greedy + sampled
    candidates, keep the one with the highest mean token log-probability."""
    from gintaras.generation import STRENGTHS

    s = STRENGTHS[strength]
    mnt = max_new_tokens or s["max_new_tokens"]
    if s["samples"] == 1:
        return [c[0][0] for c in generate(cfg, model_path, convs, mnt, s["temperature"], 1, hf)]
    if hf is None and engine(cfg) == "hf":
        from gintaras.generation import load_for_inference

        hf = load_for_inference(model_path, cfg.model.bf16)
    greedy = generate(cfg, model_path, convs, mnt, 0.0, 1, hf, score=True)
    sampled = generate(cfg, model_path, convs, mnt, s["temperature"], s["samples"] - 1, hf, score=True)
    best = []
    for g, smp in zip(greedy, sampled):
        opts = [c for c in g + smp if c[0]]
        best.append(max(opts, key=lambda c: c[1])[0] if opts else "")
    return best


# ---------------------------------------------------------------------------
# vLLM worker (runs in its own process)
# ---------------------------------------------------------------------------


def _worker(args) -> None:
    from vllm import LLM, SamplingParams

    req = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True)
    t = req["temperature"]
    params = SamplingParams(n=req["n"], temperature=t, top_p=0.95 if t > 0 else 1.0,
                            max_tokens=req["max_tokens"], logprobs=1)
    outs = llm.chat(req["convs"], params, use_tqdm=False)
    rows = []
    for o in outs:
        rows.append([(c.text.strip(), (c.cumulative_logprob or 0.0) / max(1, len(c.token_ids))) for c in o.outputs])
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-util", type=float, default=0.80)
    _worker(p.parse_args())
