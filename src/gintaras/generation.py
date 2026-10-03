"""Local Hugging Face model loading and batched chat generation."""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Message = dict[str, str]


def pick_dtype(bf16: bool = True) -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_for_inference(path: str, bf16: bool = True):
    """Load a full model, or a small LoRA adapter folder (adapter_config.json):
    adapters are applied on top of their base model and merged. Several
    adapters can be stacked with "+": "base_or_adapter1+adapter2"."""
    import json
    from pathlib import Path

    parts = path.split("+")
    first = Path(parts[0])
    adapters = parts[1:]
    if (first / "adapter_config.json").exists():
        base = json.loads((first / "adapter_config.json").read_text())["base_model_name_or_path"]
        adapters = [parts[0], *adapters]
    else:
        base = parts[0]
    tok = load_tokenizer(adapters[-1] if adapters and (Path(adapters[-1]) / "tokenizer_config.json").exists() else base)
    import os

    kw: dict = {}
    four_bit = os.environ.get("GINTARAS_4BIT") == "1" and torch.cuda.is_available()
    if four_bit:  # small GPUs (e.g. Kaggle T4 16 GB): keep the 9B model in 4-bit
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=pick_dtype(bf16))
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=pick_dtype(bf16),
        device_map="auto" if torch.cuda.is_available() else None,
        **kw,
    )
    if adapters:
        from peft import PeftModel

        for a in adapters:
            model = PeftModel.from_pretrained(model, a)
            if not four_bit:
                model = model.merge_and_unload()
    model.eval()
    return model, tok


def with_system(messages: list[Message], system: str | None) -> list[Message]:
    if not system or (messages and messages[0]["role"] == "system"):
        return list(messages)
    return [{"role": "system", "content": system}, *messages]


def render_chat(tok, messages: list[Message], add_generation_prompt: bool = True) -> str:
    """Apply the chat template; fold the system turn into the first user turn
    for templates that reject a system role."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        if not messages or messages[0]["role"] != "system":
            raise
        sys_msg, rest = messages[0]["content"], list(messages[1:])
        if rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": f"{sys_msg}\n\n{rest[0]['content']}"}
        return tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=add_generation_prompt)


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@torch.no_grad()
def generate_chat(
    model,
    tok,
    conversations: list[list[Message]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch_size: int = 8,
) -> list[str]:
    """Generate one assistant reply per conversation (left-padded batches)."""
    tok.padding_side = "left"
    outputs: list[str] = []
    device = next(model.parameters()).device
    for batch in _batches(conversations, batch_size):
        prompts = [render_chat(tok, conv) for conv in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        outputs.extend(t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


# ---------------------------------------------------------------------------
# Strength levels (used by the chat UI and the exam ladder)
# ---------------------------------------------------------------------------

STRENGTHS = {
    # fastest: one greedy answer, short
    "low": {"samples": 1, "temperature": 0.0, "max_new_tokens": 256},
    # balanced: one sampled answer, normal length
    "medium": {"samples": 1, "temperature": 0.6, "max_new_tokens": 768},
    # slowest/best: several candidates (greedy + sampled), keep the one the
    # model itself is most confident in (highest mean token log-probability)
    "max": {"samples": 4, "temperature": 0.7, "max_new_tokens": 1536},
}


@torch.no_grad()
def mean_logprob(model, tok, conversation: list[Message], answer: str) -> float:
    prompt = render_chat(tok, conversation)
    device = next(model.parameters()).device
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if a_ids.shape[1] == 0:
        return float("-inf")
    ids = torch.cat([p_ids, a_ids], dim=1).to(device)
    logits = model(input_ids=ids).logits[0, p_ids.shape[1] - 1 : -1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, a_ids[0].to(device).unsqueeze(1))
    return logp.mean().item()


def generate_with_strength(
    model, tok, conversations: list[list[Message]], strength: str = "medium",
    max_new_tokens: int | None = None, batch_size: int = 4,
) -> list[str]:
    if strength not in STRENGTHS:
        raise ValueError(f"Unknown strength '{strength}', choose from {list(STRENGTHS)}")
    s = STRENGTHS[strength]
    mnt = max_new_tokens or s["max_new_tokens"]
    if s["samples"] == 1:
        return generate_chat(model, tok, conversations, max_new_tokens=mnt,
                             temperature=s["temperature"], batch_size=batch_size)
    candidates = [generate_chat(model, tok, conversations, max_new_tokens=mnt, temperature=0.0, batch_size=batch_size)]
    for _ in range(s["samples"] - 1):
        candidates.append(generate_chat(model, tok, conversations, max_new_tokens=mnt,
                                        temperature=s["temperature"], batch_size=batch_size))
    best = []
    for i, conv in enumerate(conversations):
        opts = [c[i] for c in candidates if c[i]]
        if not opts:
            best.append("")
            continue
        best.append(max(opts, key=lambda a: mean_logprob(model, tok, conv, a)))
    return best
