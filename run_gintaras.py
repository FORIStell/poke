"""Run Gintaras on your own PC and open it in the browser.

    Double-click this file, or run:
    python run_gintaras.py                      # best trained Gintaras (1.7B), works on CPU or a 4 GB GPU
    python run_gintaras.py --model PATH_OR_ID   # e.g. a trained checkpoint folder
    python run_gintaras.py --lan                # also reachable from your phone on the same Wi-Fi
    python run_gintaras.py --exam               # sit the converted exams (needs gintaras-progress.tgz
                                                # in this folder or in Downloads) and save results

First start installs the needed packages and downloads the model (~3.5 GB).
"""

import argparse
import importlib.util
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEEDED = {"torch": "torch", "transformers": "transformers", "yaml": "pyyaml", "accelerate": "accelerate", "peft": "peft"}


VENV = ROOT / ".venv"
# best trained model so far: base EuroLLM-1.7B + the accepted CPU rounds (LoRA adapters, stacked)
BEST = "+".join(f"models/gintaras-1.7b-{r}" for r in ("r1", "r4", "r5", "r9", "r12", "r17"))


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_venv() -> None:
    """Run inside a private environment next to this file, so a broken system
    Python (e.g. "Cannot uninstall idna ... no RECORD file") can't interfere."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    if not venv_python().exists():
        print("Creating a private Python environment in", VENV)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    sys.exit(subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]))


def ensure_packages() -> None:
    missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing (first start only):", " ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Host the Gintaras chat page on this PC")
    p.add_argument("--model", default=None, help="model folder or Hugging Face id (default: best trained Gintaras)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--lan", action="store_true", help="allow other devices on your network to connect")
    p.add_argument("--exam", action="store_true", help="sit the exams instead of starting the chat page")
    p.add_argument("--level", default="nmpp8", help="exam level for --exam: nmpp8, pupp10 or vbe12")
    args = p.parse_args()

    ensure_venv()
    if args.model is None:
        have = all((ROOT / m).exists() for m in BEST.split("+"))
        args.model = "+".join(str(ROOT / m) for m in BEST.split("+")) if have else "utter-project/EuroLLM-1.7B-Instruct"
    if args.exam:
        return run_exams(args)
    ensure_packages()
    sys.path.insert(0, str(ROOT / "src"))
    from gintaras.config import load_config
    from gintaras.server import serve
    from gintaras.utils import setup_logging

    setup_logging()
    cfg = load_config(ROOT / "configs/gintaras-1.7b-cpu.yaml")
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nGintaras: open {url}")
    if args.lan:
        print(f"From phone/other PC on the same Wi-Fi: http://{lan_ip()}:{args.port}/")
    print("The model loads on your first message (can take a minute). Stop with Ctrl+C.\n")
    threading.Timer(2, lambda: webbrowser.open(url)).start()
    serve(cfg, args.model, host=host, port=args.port)


def find_progress_archive() -> Path | None:
    for d in (ROOT, Path.home() / "Downloads", Path.home() / "Desktop"):
        hits = sorted(d.glob("*gintaras-progress*.tgz"))
        if hits:
            return hits[-1]
    return None


def run_exams(args) -> None:
    """Sit every converted exam of one level on this PC (CPU) and save the grades."""
    import json
    import tarfile

    if importlib.util.find_spec("peft") is None:
        print("Installing the full toolkit (first --exam run only)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
    exams_dir = ROOT / "data" / "exams" / args.level
    if not any(exams_dir.glob("*.json")):
        archive = find_progress_archive()
        if archive is None:
            raise SystemExit("Put gintaras-progress.tgz in this folder (or Downloads) first.")
        print("Unpacking exams from", archive)
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if m.name.startswith("data/exams/")]
            t.extractall(ROOT, members=members, filter="data")
    sys.path.insert(0, str(ROOT / "src"))
    from gintaras.config import load_config
    from gintaras.exams import load_exams, run_ladder
    from gintaras.utils import setup_logging

    setup_logging()
    if sys.platform == "win32":  # keep Windows awake while exams run (sleep freezes them)
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    cfg = load_config(ROOT / "configs/gintaras-1.7b-cpu.yaml", {
        "output_dir": str(ROOT / "runs" / "pc"), "generation_engine": "hf",
        "exam.levels": [args.level], "exam.strength": "low", "exam.max_new_tokens": 200,
    })
    n = len(load_exams(cfg)[args.level])
    print(f"\nSitting {n} {args.level} exams with {args.model}. This takes a while on a CPU (~15-40 min each).\n")
    state = run_ladder(cfg, args.model, max_exams=n)
    out = ROOT / "pc_exam_results.json"
    out.write_text(json.dumps(state["history"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResults:")
    for h in state["history"]:
        if h.get("skipped"):
            print(f"  {h['exam_id']:28s} not graded (essay needs a judge model)")
        else:
            print(f"  {h['exam_id']:28s} {h['grade']:.0f}/10  ({h['percent']}%){'  approx.' if h['approximate'] else ''}")
    print(f"\nSaved to {out} - send this file to Claude.")
    input("Press Enter to close...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # keep the window open when started by double-click
        print(f"\nError: {e}")
        input("Press Enter to close...")
