"""Screenshot the front page: python scripts/screenshot.py [url] [out_dir] [--chat "question"]"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img")
chat = sys.argv[sys.argv.index("--chat") + 1] if "--chat" in sys.argv else None
strength = sys.argv[sys.argv.index("--strength") + 1] if "--strength" in sys.argv else "medium"
out.mkdir(parents=True, exist_ok=True)
exe = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"), None)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=str(exe) if exe else None)
    for name, opts in {"front-desktop": {"viewport": {"width": 1280, "height": 860}},
                       "front-dark": {"viewport": {"width": 1280, "height": 860}, "color_scheme": "dark"},
                       "front-mobile": {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 2}}.items():
        if chat and name != "front-desktop":
            continue
        page = browser.new_page(**opts)
        page.goto(url)
        page.wait_for_timeout(1500)
        if chat:
            page.check(f"input[value={strength}]", force=True)
            page.fill("#q", chat)
            page.click("#send")
            page.wait_for_selector(".msg.bot .meta", timeout=1_800_000)
            name = "front-chat"
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
        print("saved", out / f"{name}.png")
    browser.close()
