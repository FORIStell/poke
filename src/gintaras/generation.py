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
    tok = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=pick_dtype(bf16),
        device_map="auto" if torch.cuda.is_available() else None,
    )
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
