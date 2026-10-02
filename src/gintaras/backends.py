"""Teacher / judge model backends.

* `openai`: any OpenAI-compatible chat server. Serve open-weight teachers with
  e.g. `vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --tensor-parallel-size 8`.
* `hf`: a local Hugging Face model (handy for small teachers or the student).
* `dummy`: deterministic offline stand-in, ONLY for testing the pipeline.

Licensing note: only distill from models whose license permits training other
models on their outputs (e.g. Apache-2.0 Qwen3, EuroLLM, Mistral). Commercial
APIs such as Claude or GPT forbid using outputs to build competing models.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from gintaras.config import EndpointConfig
from gintaras.lt import lt_score

log = logging.getLogger(__name__)
Message = dict[str, str]


class Backend(Protocol):
    name: str

    def chat_many(
        self, conversations: list[list[Message]], temperature: float | None = None, max_tokens: int | None = None
    ) -> list[str | None]: ...


class OpenAIBackend:
    def __init__(self, cfg: EndpointConfig):
        from openai import OpenAI

        self.cfg = cfg
        self.name = cfg.name
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env, "EMPTY"),
            timeout=cfg.timeout,
            max_retries=5,
        )

    def _one(self, messages: list[Message], temperature: float, max_tokens: int) -> str | None:
        import openai

        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            log.warning("[%s] request failed: %s", self.name, e)
            return None
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            return None  # truncated answers make bad training targets
        return (choice.message.content or "").strip() or None

    def chat_many(self, conversations, temperature=None, max_tokens=None):
        t = self.cfg.temperature if temperature is None else temperature
        m = self.cfg.max_tokens if max_tokens is None else max_tokens
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(lambda c: self._one(c, t, m), conversations))


class HFBackend:
    def __init__(self, cfg: EndpointConfig, model=None, tokenizer=None):
        from gintaras.generation import load_for_inference

        self.cfg = cfg
        self.name = cfg.name
        if model is None:
            model, tokenizer = load_for_inference(cfg.model)
        self.model, self.tok = model, tokenizer

    def chat_many(self, conversations, temperature=None, max_tokens=None):
        from gintaras.generation import generate_chat

        outs = generate_chat(
            self.model,
            self.tok,
            conversations,
            max_new_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
            temperature=self.cfg.temperature if temperature is None else temperature,
            batch_size=self.cfg.concurrency,
        )
        return [o or None for o in outs]


class DummyBackend:
    """Offline stand-in that lets the whole pipeline run without a GPU server.

    It answers with fixed Lithuanian text and, when asked to judge, scores the
    candidate with the `lt_score` heuristic. Never use it for real training.
    """

    ANSWER = (
        "Lietuva yra valstybė Baltijos jūros rytinėje pakrantėje. Jos sostinė yra Vilnius. "
        "Šis atsakymas sugeneruotas bandomuoju režimu, todėl jis visada vienodas."
    )

    def __init__(self, cfg: EndpointConfig):
        self.cfg = cfg
        self.name = cfg.name

    def _one(self, messages: list[Message]) -> str:
        last = messages[-1]["content"]
        if "### CANDIDATE RESPONSE" in last:
            cand = last.split("### CANDIDATE RESPONSE", 1)[1].split("Return ONLY a JSON object", 1)[0]
            s = round(3 + 7 * lt_score(cand), 1)
            return json.dumps(
                {"faithfulness": s, "language": s, "helpfulness": s, "overall": s, "issues": "dummy"},
                ensure_ascii=False,
            )
        m = re.search(r'\{"(\w+)": "\.\.\."\}', last)
        if m:
            return json.dumps({m.group(1): "Kokia yra pagrindinė šio teksto mintis?"}, ensure_ascii=False)
        return self.ANSWER

    def chat_many(self, conversations, temperature=None, max_tokens=None):
        return [self._one(c) for c in conversations]


def make_backend(cfg: EndpointConfig) -> Backend:
    if cfg.backend == "openai":
        return OpenAIBackend(cfg)
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "dummy":
        return DummyBackend(cfg)
    raise ValueError(f"Unknown backend '{cfg.backend}' for endpoint '{cfg.name}'")


def parse_json_object(text: str | None) -> dict | None:
    """Extract the first JSON object from a model reply (tolerates code fences,
    leading chatter and reasoning tags)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
