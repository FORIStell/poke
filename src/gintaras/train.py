"""Training stages: continued pretraining (CPT), SFT and DPO, all as (Q)LoRA
adapters that are merged back into full checkpoints between stages."""

from __future__ import annotations

import gc
import logging
import random
import shutil
from pathlib import Path

import torch

from gintaras.config import Config, TrainStageConfig
from gintaras.generation import pick_dtype
from gintaras.utils import read_jsonl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / adapter plumbing
# ---------------------------------------------------------------------------


def peft_config(cfg: Config):
    from peft import LoraConfig

    lc = cfg.model.lora
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=lc.target_modules,
        modules_to_save=["embed_tokens", "lm_head"] if lc.train_embeddings else None,
        task_type="CAUSAL_LM",
    )


def quantization_config(cfg: Config):
    if not (cfg.model.load_in_4bit and torch.cuda.is_available()):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=pick_dtype(cfg.model.bf16),
    )


def model_init_kwargs(cfg: Config) -> dict:
    kw: dict = {"dtype": pick_dtype(cfg.model.bf16)}
    if cfg.model.attn_implementation and torch.cuda.is_available():
        kw["attn_implementation"] = cfg.model.attn_implementation
    return kw


def common_args(cfg: Config, stage: TrainStageConfig, out_dir: Path) -> dict:
    cuda = torch.cuda.is_available()
    return dict(
        output_dir=str(out_dir),
        num_train_epochs=stage.epochs,
        max_steps=stage.max_steps,
        learning_rate=stage.learning_rate,
        per_device_train_batch_size=stage.batch_size,
        gradient_accumulation_steps=stage.grad_accum,
        warmup_steps=stage.warmup_ratio,
        lr_scheduler_type=stage.lr_scheduler,
        weight_decay=stage.weight_decay,
        logging_steps=stage.logging_steps,
        save_steps=stage.save_steps,
        save_total_limit=2,
        bf16=cuda and cfg.model.bf16 and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        max_length=cfg.model.max_seq_len,
        model_init_kwargs=model_init_kwargs(cfg),
        report_to="none",
        seed=cfg.seed,
    )


def merge_adapter(base: str, adapter_dir: Path, out_dir: Path, bf16: bool = True) -> Path:
    """Merge a LoRA adapter into full-precision base weights and save a
    standalone checkpoint (with tokenizer) that the next stage starts from."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Merging %s into %s -> %s", adapter_dir, base, out_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=pick_dtype(bf16), device_map="auto" if torch.cuda.is_available() else None
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(out_dir))
    del model
    gc.collect()
    return out_dir


def _finish(trainer, cfg: Config, base: str, name: str) -> Path:
    adapter_dir = cfg.ckpt_path(f"{name}_adapter")
    trainer.save_model(str(adapter_dir))
    trainer.processing_class.save_pretrained(str(adapter_dir))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = merge_adapter(base, adapter_dir, cfg.ckpt_path(name), cfg.model.bf16)
    for ckpt in adapter_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    return out


def _tokenizer(path: str):
    from gintaras.generation import load_tokenizer

    return load_tokenizer(path)


# ---------------------------------------------------------------------------
# Stage inputs
# ---------------------------------------------------------------------------


def latest_checkpoint(cfg: Config, *names: str) -> str:
    """First existing merged checkpoint among `names`, else the base model."""
    for name in names:
        p = cfg.ckpt_path(name)
        if (p / "config.json").exists():
            return str(p)
    return cfg.model.base


def to_prompt_completion(messages: list[dict], system: str) -> dict:
    """[user, assistant, ..., assistant] -> TRL conversational prompt/completion
    (loss is computed on the final assistant turn only)."""
    msgs = [m for m in messages if m["role"] != "system"]
    assert msgs and msgs[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    return {
        "prompt": [{"role": "system", "content": system}, *msgs[:-1]],
        "completion": [msgs[-1]],
    }


def sft_rows(cfg: Config) -> list[dict]:
    rows = []
    for name in ("instructions.jsonl", "selfsup_sft.jsonl", "synth_sft.jsonl", "loop_sft.jsonl"):
        path = cfg.data_path(name)
        if path.exists():
            n0 = len(rows)
            rows.extend(to_prompt_completion(r["messages"], cfg.system_prompt) for r in read_jsonl(path))
            log.info("SFT data: %d rows from %s", len(rows) - n0, name)
    random.Random(cfg.seed).shuffle(rows)
    return rows


def dpo_rows(cfg: Config, files: tuple[str, ...] = ("synth_dpo.jsonl",)) -> list[dict]:
    rows = []
    sys_msg = {"role": "system", "content": cfg.system_prompt}
    for name in files:
        path = cfg.data_path(name)
        if path.exists():
            for r in read_jsonl(path):
                rows.append({"prompt": [sys_msg, *r["prompt"]], "chosen": r["chosen"], "rejected": r["rejected"]})
    random.Random(cfg.seed).shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_cpt(cfg: Config) -> Path | None:
    """Continued pretraining on raw Lithuanian text (next-token prediction)."""
    if not cfg.cpt.enabled:
        log.info("CPT disabled; skipping")
        return None
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    base = cfg.model.base
    ds = load_dataset("json", data_files=str(cfg.data_path("corpus.jsonl")), split="train")
    ds = ds.select_columns(["text"])
    args = SFTConfig(
        **common_args(cfg, cfg.cpt, cfg.ckpt_path("cpt_adapter")),
        dataset_text_field="text",
        packing=cfg.cpt.packing,
    )
    trainer = SFTTrainer(
        model=base,
        args=args,
        train_dataset=ds,
        processing_class=_tokenizer(base),
        peft_config=peft_config(cfg),
        quantization_config=quantization_config(cfg),
    )
    trainer.train()
    return _finish(trainer, cfg, base, "cpt")


def run_sft(cfg: Config, base: str | None = None, rows: list[dict] | None = None, name: str = "sft") -> Path:
    """Supervised fine-tuning on human + distilled instruction data."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    base = base or latest_checkpoint(cfg, "cpt")
    rows = rows if rows is not None else sft_rows(cfg)
    if not rows:
        raise ValueError("No SFT data found; run `prepare` and/or `synth` first")
    args = SFTConfig(**common_args(cfg, cfg.sft, cfg.ckpt_path(f"{name}_adapter")), packing=cfg.sft.packing)
    trainer = SFTTrainer(
        model=base,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=_tokenizer(base),
        peft_config=peft_config(cfg),
        quantization_config=quantization_config(cfg),
    )
    trainer.train()
    return _finish(trainer, cfg, base, name)


def run_dpo(cfg: Config, base: str | None = None, rows: list[dict] | None = None, name: str = "dpo") -> Path:
    """Direct preference optimization on judge-ranked (chosen, rejected) pairs.
    With LoRA, the frozen base (adapter disabled) acts as the reference model."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    base = base or latest_checkpoint(cfg, "sft", "cpt")
    rows = rows if rows is not None else dpo_rows(cfg)
    if not rows:
        raise ValueError("No preference pairs found; run `synth` (or the improvement loop) first")
    args = DPOConfig(**common_args(cfg, cfg.dpo, cfg.ckpt_path(f"{name}_adapter")), beta=cfg.dpo.beta)
    trainer = DPOTrainer(
        model=base,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=_tokenizer(base),
        peft_config=peft_config(cfg),
        quantization_config=quantization_config(cfg),
    )
    trainer.train()
    return _finish(trainer, cfg, base, name)
