"""Run Gintaras on your own PC and open it in the browser.

    Double-click this file, or run:
    python run_gintaras.py                      # small model (EuroLLM-1.7B), works on CPU or a 4 GB GPU
    python run_gintaras.py --model PATH_OR_ID   # e.g. a trained checkpoint folder
    python run_gintaras.py --lan                # also reachable from your phone on the same Wi-Fi

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
NEEDED = {"torch": "torch", "transformers": "transformers", "yaml": "pyyaml", "accelerate": "accelerate"}


def ensure_packages() -> None:
    missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing:", " ".join(missing))
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
    p.add_argument("--model", default="utter-project/EuroLLM-1.7B-Instruct", help="model folder or Hugging Face id")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--lan", action="store_true", help="allow other devices on your network to connect")
    args = p.parse_args()

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # keep the window open when started by double-click
        print(f"\nError: {e}")
        input("Press Enter to close...")
