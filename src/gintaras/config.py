"""Typed configuration loaded from YAML.

Every stage of the pipeline reads the same config file, so a single YAML fully
describes a training run (see configs/).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class LoraConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: Any = "all-linear"
    # Training the embeddings + LM head helps a lot when the base model's
    # tokenizer/embeddings saw little Lithuanian. Costs extra memory.
    train_embeddings: bool = False


@dataclass
class ModelConfig:
    base: str = "utter-project/EuroLLM-9B-Instruct-2512"
    load_in_4bit: bool = True
    bf16: bool = True
    max_seq_len: int = 4096
    attn_implementation: str | None = "sdpa"
    gradient_checkpointing: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class SourceConfig:
    """A dataset source: a Hugging Face dataset id or a local .jsonl path."""

    name: str | None = None
    path: str | None = None
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    max_docs: int | None = None
    # Adapter used to turn a row into chat messages (instruction sources only).
    kind: str | None = None
    # Fraction of rows held out for evaluation (instruction sources only).
    eval_fraction: float = 0.0


@dataclass
class DataConfig:
    corpus: list[SourceConfig] = field(default_factory=list)
    instructions: list[SourceConfig] = field(default_factory=list)
    min_chars: int = 200
    max_chars: int = 100_000
    min_lt_score: float = 0.55
    dedup: bool = True


@dataclass
class EndpointConfig:
    """A teacher/judge model behind any OpenAI-compatible server (vLLM, TGI,
    llama.cpp, Ollama, ...), a local HF model, or the offline dummy backend."""

    name: str = "teacher"
    backend: str = "openai"  # openai | hf | dummy
    model: str = ""
    base_url: str | None = "http://localhost:8000/v1"
    api_key_env: str = "TEACHER_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    concurrency: int = 16
    timeout: float = 600.0


@dataclass
class SynthConfig:
    num_prompts: int = 20_000
    # Mixture weights over task types (see gintaras/data/tasks.py).
    task_weights: dict[str, float] = field(default_factory=dict)
    min_judge_score: float = 8.0
    # Minimum score gap between chosen and rejected to form a DPO pair.
    dpo_margin: float = 2.0
    # Number of Wikipedia passages to sample as grounding contexts.
    num_contexts: int = 50_000


@dataclass
class TrainStageConfig:
    enabled: bool = True
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 500
    packing: bool = False
    # DPO only.
    beta: float = 0.1


@dataclass
class LoopConfig:
    rounds: int = 5
    prompts_per_round: int = 2_000
    student_samples: int = 2
    target_score: float = 9.0
    patience: int = 2
    sft_on_teacher_wins: bool = True
    # After every round, sit the exam ladder; the loop then only stops when
    # the top level (VBE) has been passed `exam.streak` times in a row.
    exam_ladder: bool = True


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    num_judge_prompts: int = 100
    num_qa: int = 300
    num_diacritics: int = 200
    num_ppl_docs: int = 200
    batch_size: int = 8


@dataclass
class ExamConfig:
    """The exam ladder: the model must pass `streak` exams in a row (different
    years) with at least `pass_grade` on a 10-point scale before moving up."""

    exams_dir: str = "data/exams"
    levels: list[str] = field(default_factory=lambda: ["nmpp8", "pupp10", "vbe12"])
    pass_grade: float = 8.0
    streak: int = 3
    strength: str = "max"
    max_new_tokens: int = 1024
    essay_max_new_tokens: int = 2048


@dataclass
class Config:
    name: str = "gintaras"
    output_dir: str = "runs/gintaras"
    seed: int = 42
    # auto: vLLM when a GPU + vllm are available (10-20x faster), else Hugging Face generate
    generation_engine: str = "auto"
    system_prompt: str = (
        "Tu esi Gintaras – išmanus, mandagus ir tikslus dirbtinio intelekto asistentas. "
        "Visada atsakyk taisyklinga lietuvių kalba, nebent vartotojas aiškiai paprašo kitos kalbos. "
        "Jei atsakymui pateiktas kontekstas, remkis tik juo; jei kontekste informacijos nėra, taip ir pasakyk."
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teachers: list[EndpointConfig] = field(default_factory=list)
    judge: EndpointConfig | None = None
    synth: SynthConfig = field(default_factory=SynthConfig)
    cpt: TrainStageConfig = field(default_factory=lambda: TrainStageConfig(learning_rate=1e-4, packing=True))
    sft: TrainStageConfig = field(default_factory=lambda: TrainStageConfig(epochs=2))
    dpo: TrainStageConfig = field(default_factory=lambda: TrainStageConfig(learning_rate=5e-6, batch_size=2))
    loop: LoopConfig = field(default_factory=LoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    exam: ExamConfig = field(default_factory=ExamConfig)

    # ---- paths -----------------------------------------------------------------
    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    def data_path(self, name: str) -> Path:
        return self.out / "data" / name

    def ckpt_path(self, name: str) -> Path:
        return self.out / "checkpoints" / name


def _build(cls: type, raw: Any, where: str) -> Any:
    """Recursively build dataclass `cls` from plain YAML data, rejecting unknown keys."""
    if raw is None:
        return None
    if not dataclasses.is_dataclass(cls):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, value in raw.items():
        kwargs[key] = _coerce(hints[key], value, f"{where}.{key}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, where: str) -> Any:
    origin = get_origin(hint)
    args = [a for a in get_args(hint) if a is not type(None)]
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    if origin is list and args and dataclasses.is_dataclass(args[0]):
        return [_build(args[0], v, f"{where}[{i}]") for i, v in enumerate(value or [])]
    # Optional[Dataclass] / Dataclass | None
    if args and len(args) == 1 and dataclasses.is_dataclass(args[0]):
        return _build(args[0], value, where)
    if hint is float and isinstance(value, (int, str)):
        return float(value)
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return _build(Config, raw, "config")
