"""Run the people image generator on your own PC and open it in the browser.

    Double-click this file, or run:
    python run_imagegen.py          # http://127.0.0.1:8090/
    python run_imagegen.py --lan    # also reachable from your phone on the same Wi-Fi
    python run_imagegen.py --cpu    # don't use the graphics card

First start creates a private Python environment (.venv-img), installs PyTorch with
NVIDIA GPU support (~2.5 GB) and downloads the ready-made model on first use (~2.5 GB).
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
VENV = ROOT / ".venv-img"
NEEDED = {"diffusers": "diffusers", "transformers": "transformers", "accelerate": "accelerate",
          "cv2": "opencv-python-headless<5", "PIL": "pillow", "safetensors": "safetensors"}
TINY = ROOT / "imagegen" / "models" / "tinyface.pt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_venv() -> None:
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    if not venv_python().exists():
        print("Creating a private Python environment in", VENV)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    sys.exit(subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]))


def ensure_packages(cpu: bool) -> None:
    pip = [sys.executable, "-m", "pip", "install"]
    if importlib.util.find_spec("torch") is None:
        print("Installing PyTorch (first start only, a few GB)...")
        subprocess.check_call(pip + ["--upgrade", "pip"])
        if cpu or sys.platform == "darwin":
            subprocess.check_call(pip + ["torch"])
        else:  # CUDA 12.6 build still supports older cards like the GTX 1050 Ti
            subprocess.check_call(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu126"])
    missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.check_call(pip + missing)


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
    p = argparse.ArgumentParser(description="Host the people image generator on this PC")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--lan", action="store_true", help="allow other devices on your network to connect")
    p.add_argument("--cpu", action="store_true", help="don't use the graphics card")
    args = p.parse_args()

    ensure_venv()
    ensure_packages(args.cpu)
    sys.path.insert(0, str(ROOT / "imagegen"))
    from app import serve

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nImage generator: open {url}")
    if args.lan:
        print(f"From phone/other PC on the same Wi-Fi: http://{lan_ip()}:{args.port}/")
    print("Models load on the first picture (the first time also downloads them). Stop with Ctrl+C.\n")
    threading.Timer(2, lambda: webbrowser.open(url)).start()
    serve(TINY if TINY.exists() else None, host=host, port=args.port, device="cpu" if args.cpu else None)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # keep the window open when started by double-click
        print(f"\nError: {e}")
        input("Press Enter to close...")
