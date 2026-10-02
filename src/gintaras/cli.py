"""Command line interface.

    gintaras prepare  -c configs/gintaras-9b.yaml     # data: corpus, instructions, contexts
    gintaras selfsup  -c ...                          # teacher-free exercises (diacritics, error fixing)
    gintaras synth    -c ...                          # multi-teacher distillation
    gintaras cpt|sft|dpo -c ...                       # training stages
    gintaras improve  -c ...                          # self-improvement loop
    gintaras eval     -c ... [--model PATH]
    gintaras all      -c ...                          # everything, in order
    gintaras exams-fetch|exams-convert -c ...         # download + convert past NŠA exams
    gintaras exam     -c ... [--model PATH]           # sit the exam ladder (NMPP 8 → PUPP 10 → VBE)
    gintaras serve    -c ... [--port 8080]            # front page with chat
    gintaras report   -c ...                          # progress chart PNG
    gintaras chat     -c ... [--model PATH]
    gintaras ask      -c ... --context FILE --question "..."
    gintaras essay    -c ... --topic "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from gintaras.config import Config, load_config
from gintaras.utils import setup_logging

log = logging.getLogger("gintaras")


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got '{pair}'")
        out[key] = yaml.safe_load(value)
    return out


def _default_model(cfg: Config) -> str:
    best = cfg.out / "BEST_MODEL"
    if best.exists():
        return best.read_text(encoding="utf-8").strip()
    from gintaras.train import latest_checkpoint

    return latest_checkpoint(cfg, "dpo", "sft", "cpt")


def cmd_prepare(cfg: Config, args) -> None:
    from gintaras.data import prepare

    steps = {"corpus": prepare.prepare_corpus, "instructions": prepare.prepare_instructions,
             "contexts": prepare.prepare_contexts}
    for name in [args.only] if args.only else steps:
        steps[name](cfg)


def cmd_synth(cfg: Config, args) -> None:
    from gintaras.data.synth import synthesize

    print(json.dumps(synthesize(cfg), indent=2))


def cmd_selfsup(cfg: Config, args) -> None:
    from gintaras.data.synth import synthesize_selfsup

    print(json.dumps(synthesize_selfsup(cfg), indent=2))


def cmd_cpt(cfg: Config, args) -> None:
    from gintaras.train import run_cpt

    run_cpt(cfg)


def cmd_sft(cfg: Config, args) -> None:
    from gintaras.train import run_sft

    run_sft(cfg, base=args.model)


def cmd_dpo(cfg: Config, args) -> None:
    from gintaras.train import run_dpo

    run_dpo(cfg, base=args.model)


def cmd_improve(cfg: Config, args) -> None:
    from gintaras.loop import improve

    print(json.dumps(improve(cfg), indent=2, ensure_ascii=False))


def cmd_eval(cfg: Config, args) -> None:
    from gintaras.evaluate import evaluate

    model = args.model or _default_model(cfg)
    print(json.dumps(evaluate(cfg, model, use_judge=not args.no_judge), indent=2, ensure_ascii=False))


def cmd_exams_fetch(cfg: Config, args) -> None:
    from gintaras.exams import fetch_exams

    fetch_exams(cfg)


def cmd_exams_convert(cfg: Config, args) -> None:
    from gintaras.exams import convert_all

    print(f"converted {convert_all(cfg)} exams")


def cmd_exam(cfg: Config, args) -> None:
    from gintaras.exams import run_ladder

    state = run_ladder(cfg, args.model or _default_model(cfg), max_exams=args.exams)
    print(json.dumps({k: v for k, v in state.items() if k != "history"} | {"last": state["history"][-args.exams:]},
                     indent=2, ensure_ascii=False))


def cmd_report(cfg: Config, args) -> None:
    from gintaras.report import make_report

    print(make_report(cfg))


def cmd_serve(cfg: Config, args) -> None:
    from gintaras.server import serve

    serve(cfg, args.model or _default_model(cfg), host=args.host, port=args.port)


def cmd_all(cfg: Config, args) -> None:
    from gintaras.data.prepare import prepare_all
    from gintaras.data.synth import synthesize, synthesize_selfsup
    from gintaras.evaluate import evaluate
    from gintaras.loop import improve
    from gintaras.train import dpo_rows, run_cpt, run_dpo, run_sft

    prepare_all(cfg)
    synthesize_selfsup(cfg)
    if cfg.teachers and cfg.judge:
        synthesize(cfg)
    else:
        log.warning("No teachers/judge configured: skipping distillation, DPO and the improvement loop")
    evaluate(cfg, cfg.model.base, name="base")
    run_cpt(cfg)
    sft = run_sft(cfg)
    final = sft
    if cfg.dpo.enabled and dpo_rows(cfg):
        final = run_dpo(cfg)
    evaluate(cfg, str(final))
    if cfg.teachers and cfg.judge and cfg.loop.rounds > 0:
        improve(cfg)


def _chat_model(cfg: Config, args):
    from gintaras.generation import load_for_inference

    path = args.model or _default_model(cfg)
    log.info("Loading %s", path)
    return load_for_inference(path, cfg.model.bf16)


def _reply(cfg: Config, model, tok, messages: list[dict], max_new_tokens: int = 1024) -> str:
    from gintaras.generation import generate_chat, with_system

    return generate_chat(model, tok, [with_system(messages, cfg.system_prompt)],
                         max_new_tokens=max_new_tokens, temperature=0.3)[0]


def cmd_chat(cfg: Config, args) -> None:
    model, tok = _chat_model(cfg, args)
    history: list[dict] = []
    print("Gintaras. Rašykite žinutę (/naujas – pradėti iš naujo, /iseiti – baigti).")
    while True:
        try:
            user = input("\nJūs: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in {"/iseiti", "/exit", "/quit"}:
            break
        if user in {"/naujas", "/reset"}:
            history = []
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        answer = _reply(cfg, model, tok, history)
        history.append({"role": "assistant", "content": answer})
        print(f"\nGintaras: {answer}")


def cmd_ask(cfg: Config, args) -> None:
    from gintaras.data.tasks import context_qa_prompt

    context = Path(args.context).read_text(encoding="utf-8") if args.context else ""
    model, tok = _chat_model(cfg, args)
    user = context_qa_prompt(context, args.question) if context else args.question
    print(_reply(cfg, model, tok, [{"role": "user", "content": user}]))


def cmd_essay(cfg: Config, args) -> None:
    model, tok = _chat_model(cfg, args)
    user = (f"Parašyk argumentuotą rašinį tema „{args.topic}“. Rašinyje turi būti įžanga, bent du argumentai "
            f"su pavyzdžiais ir apibendrinanti pabaiga. Apimtis – apie {args.words} žodžių.")
    print(_reply(cfg, model, tok, [{"role": "user", "content": user}], max_new_tokens=2048))


COMMANDS = {
    "prepare": cmd_prepare, "selfsup": cmd_selfsup, "synth": cmd_synth, "cpt": cmd_cpt, "sft": cmd_sft, "dpo": cmd_dpo,
    "improve": cmd_improve, "eval": cmd_eval, "all": cmd_all, "chat": cmd_chat, "ask": cmd_ask,
    "essay": cmd_essay, "exams-fetch": cmd_exams_fetch, "exams-convert": cmd_exams_convert, "exam": cmd_exam, "serve": cmd_serve, "report": cmd_report,
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gintaras", description="Lithuanian LLM training pipeline")
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("-c", "--config", default="configs/gintaras-9b.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set sft.epochs=1")
    p.add_argument("--model", help="model path/id (default: best trained checkpoint)")
    p.add_argument("--only", choices=["corpus", "instructions", "contexts"], help="prepare: run one step")
    p.add_argument("--no-judge", action="store_true", help="eval: skip LLM-judge scoring")
    p.add_argument("--context", help="ask: file with the context text")
    p.add_argument("--question", help="ask: the question")
    p.add_argument("--topic", help="essay: topic")
    p.add_argument("--words", type=int, default=500, help="essay: target length in words")
    p.add_argument("--host", default="127.0.0.1", help="serve: bind address")
    p.add_argument("--port", type=int, default=8080, help="serve: port")
    p.add_argument("--exams", type=int, default=3, help="exam: how many exams to sit this time")
    args = p.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config, _overrides(args.set))
    if args.command == "ask" and not args.question:
        p.error("ask needs --question")
    if args.command == "essay" and not args.topic:
        p.error("essay needs --topic")
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
