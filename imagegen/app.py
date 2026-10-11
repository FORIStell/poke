"""Image generator web app (stdlib server + diffusers).

Two engines:
  - "turbo": ready-made SD-Turbo (~2.5 GB, 512px, 4 steps; fast on a 4 GB GPU, slow but OK on CPU)
  - "tiny":  our own tiny face model trained from scratch (imagegen/tinyface.py, 128px faces)

Modes: "face" (portrait) or "body" (full body, optional face blur).
Safety: no nudity/sexual content, adults only. Blocked words in the prompt are refused, and every
picture is checked by an image classifier (Falconsai/nsfw_image_detection); flagged pictures are not shown.

GET  /                page (imagegen/web/index.html)
GET  /api/options     option lists for the page
POST /api/generate    {"engine", "mode", "options": {...}, "prompt", "count", "seed", "blur_face"} -> {"images": [data-URLs], ...}
"""

from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
sys.path.insert(0, str(HERE))

OPTIONS = {
    "gender": ["any", "man", "woman"],
    "age": ["any", "young adult", "adult", "middle-aged", "elderly"],
    "expression": ["any", "neutral", "smiling", "laughing", "serious", "surprised", "angry", "sad", "thoughtful"],
    "hair_color": ["any", "black", "brown", "blond", "red", "gray", "white", "bald"],
    "hair_style": ["any", "short", "long", "curly", "wavy", "straight", "ponytail", "buzz cut"],
    "facial_hair": ["any", "clean-shaven", "stubble", "beard", "mustache"],
    "skin_tone": ["any", "light", "medium", "dark"],
    "clothing": ["any", "casual t-shirt", "hoodie", "sweater", "business suit", "shirt and tie", "dress",
                 "leather jacket", "denim jacket", "sportswear", "winter coat", "uniform", "swimwear"],
    "accessories": ["none", "glasses", "sunglasses", "hat", "cap", "earrings", "necklace", "headphones"],
    "body_type": ["any", "slim", "average", "athletic", "muscular", "heavy"],
    "pose": ["any", "standing", "walking", "sitting", "arms crossed", "running"],
    "background": ["any", "studio", "plain white", "city street", "park", "beach", "office", "forest", "room"],
    "style": ["photo", "film photo", "painting", "anime", "3D render", "pencil sketch"],
}
TINY_OPTIONS = ["gender", "age", "expression", "hair_color", "facial_hair", "skin_tone", "accessories", "clothing"]

BLOCKED = re.compile(
    r"\b(nude|naked|nsfw|sex\w*|porn\w*|topless|bottomless|nipples?|genitals?|lingerie|undress\w*|erotic\w*|"
    r"hentai|explicit|fetish\w*|boobs?|breasts?|penis|vagina|butt|buttocks|strip(ping|tease|per)s?|child\w*|kids?|minors?|"
    r"teen\w*|underage|loli\w*|shota\w*|schoolgirls?|schoolboys?|baby|babies|toddlers?|little (girl|boy))\b",
    re.I)
NEGATIVE = "nude, naked, nsfw, child, deformed, extra limbs, bad anatomy, blurry, watermark, text"


def build_prompt(mode: str, o: dict, extra: str) -> str:
    def pick(k):
        v = o.get(k, "any")
        return None if v in (None, "", "any", "none") else v

    who = {"man": "man", "woman": "woman"}.get(pick("gender") or "", "person")
    parts = []
    if pick("age"):
        parts.append(pick("age"))
    if pick("skin_tone"):
        parts.append(f"{pick('skin_tone')} skin")
    desc = " ".join(parts + [who]) if parts else who
    bits = [desc if pick("age") else f"adult {desc}"]
    if pick("expression"):
        bits.append(f"{pick('expression')} expression")
    hair = " ".join(x for x in (pick("hair_style"), pick("hair_color")) if x)
    if pick("hair_color") == "bald":
        bits.append("bald")
    elif hair:
        bits.append(f"{hair} hair")
    if pick("facial_hair"):
        bits.append(pick("facial_hair"))
    if pick("clothing"):
        bits.append(f"wearing {pick('clothing')}")
    if pick("accessories"):
        bits.append(f"with {pick('accessories')}")
    if mode == "body":
        if pick("body_type"):
            bits.append(f"{pick('body_type')} build")
        bits.append(pick("pose") or "standing")
        bits.append("facing the camera, full body shot, head to toe, whole person visible")
    else:
        bits.append("head and shoulders portrait, face in focus")
    if pick("background"):
        bits.append(f"{pick('background')} background")
    style = o.get("style") or "photo"
    quality = {"photo": "realistic photo, natural light, sharp focus, detailed",
               "film photo": "35mm film photo, natural grain",
               "painting": "oil painting", "anime": "anime illustration", "3D render": "3D render, soft lighting",
               "pencil sketch": "pencil sketch"}.get(style, style)
    prompt = ", ".join(bits + [quality])
    if extra.strip():
        prompt = f"{extra.strip()}, {prompt}"
    return prompt


def tiny_attrs(o: dict) -> dict:
    """Map page options onto the tiny model's face options (+1 yes / -1 no)."""
    a: dict = {}
    g = o.get("gender")
    if g in ("man", "woman"):
        a["Male"] = 1 if g == "man" else -1
    age = o.get("age")
    if age in ("young adult", "adult"):
        a["Young"] = 1
    elif age in ("middle-aged", "elderly"):
        a["Young"] = -1
    e = o.get("expression")
    if e in ("smiling", "laughing"):
        a["Smiling"] = 1
        if e == "laughing":
            a["Mouth_Slightly_Open"] = 1
    elif e in ("neutral", "serious", "angry", "sad", "thoughtful"):
        a["Smiling"] = -1
    hc = {"black": "Black_Hair", "brown": "Brown_Hair", "blond": "Blond_Hair", "gray": "Gray_Hair",
          "white": "Gray_Hair", "bald": "Bald"}.get(o.get("hair_color") or "")
    if hc:
        a[hc] = 1
    fh = o.get("facial_hair")
    if fh == "clean-shaven":
        a["No_Beard"], a["Mustache"], a["Goatee"] = 1, -1, -1
    elif fh in ("beard", "stubble"):
        a["No_Beard"] = -1
    elif fh == "mustache":
        a["Mustache"] = 1
    if o.get("skin_tone") == "light":
        a["Pale_Skin"] = 1
    acc = {"glasses": "Eyeglasses", "sunglasses": "Eyeglasses", "hat": "Wearing_Hat", "cap": "Wearing_Hat",
           "earrings": "Wearing_Earrings", "necklace": "Wearing_Necklace"}.get(o.get("accessories") or "")
    if acc:
        a[acc] = 1
    if o.get("clothing") in ("business suit", "shirt and tie"):
        a["Wearing_Necktie"] = 1
    return a


def blur_faces(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    W, H = img.size
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(16, 16), maxSize=(int(W * 0.22), int(W * 0.22)))
    faces = [f for f in faces if f[1] < H * 0.6]  # in a full-body picture the head is in the upper part
    out = img.copy()
    for (x, y, w, h) in [tuple(int(v) for v in f) for f in faces]:
        pad = int(0.25 * w)
        box = (max(0, x - pad), max(0, y - pad), min(img.width, x + w + pad), min(img.height, y + h + pad))
        out.paste(out.crop(box).filter(ImageFilter.GaussianBlur(radius=max(6, w // 6))), box)
    return out


class Engines:
    def __init__(self, tiny_ckpt: Path | None, device: str):
        self.device = device
        self.tiny_ckpt = tiny_ckpt
        self._turbo = self._tiny = self._nsfw = None
        self.lock = threading.Lock()

    def turbo(self):
        if self._turbo is None:
            from diffusers import AutoPipelineForText2Image

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            p = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype,
                                                          variant="fp16")  # fp16 files (half the download), cast on load
            p = p.to(self.device)
            p.enable_attention_slicing()
            p.set_progress_bar_config(disable=True)
            self._turbo = p
        return self._turbo

    def tiny(self):
        if self._tiny is None:
            from tinyface import load

            self._tiny = load(str(self.tiny_ckpt))
        return self._tiny

    def is_nsfw(self, img: Image.Image) -> bool:
        if self._nsfw is None:
            from transformers import pipeline

            self._nsfw = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=-1)
        scores = {r["label"]: r["score"] for r in self._nsfw(img)}
        return scores.get("nsfw", 0.0) > 0.35

    def generate(self, req: dict) -> dict:
        engine = req.get("engine", "turbo")
        mode = req.get("mode", "face")
        opts = req.get("options") or {}
        extra = (req.get("prompt") or "")[:400]
        count = max(1, min(int(req.get("count", 2)), 8))
        seed = int(req.get("seed") or random.randint(0, 2**31 - 1))
        if BLOCKED.search(extra) or any(BLOCKED.search(str(v)) for v in opts.values()):
            return {"error": "This request isn't allowed: no nudity or sexual content, and only adults."}
        t0 = time.time()
        with self.lock:
            if engine == "tiny":
                if not self.tiny_ckpt or not Path(self.tiny_ckpt).exists():
                    return {"error": "The tiny model isn't downloaded yet (imagegen/models/tinyface.pt)."}
                from tinyface import ATTRS, decode, sample

                model, scale = self.tiny()
                a = torch.zeros(count, len(ATTRS))
                for k, v in tiny_attrs(opts).items():
                    a[:, ATTRS.index(k)] = v
                x = decode(sample(model, a, guidance=float(req.get("guidance", 2.5)), seed=seed), scale)
                x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
                imgs = [Image.fromarray(i).resize((256, 256), Image.LANCZOS) for i in x]
                prompt = "options: " + ", ".join(f"{k}={'yes' if v > 0 else 'no'}" for k, v in tiny_attrs(opts).items())
            else:
                prompt = build_prompt(mode, opts, extra)
                w, h = (512, 768) if mode == "body" else (512, 512)
                g = torch.Generator(self.device).manual_seed(seed)
                imgs = self.turbo()(prompt=[prompt] * count, num_inference_steps=int(req.get("steps", 4)),
                                    guidance_scale=0.0, width=w, height=h, generator=g).images
                if mode == "body" and req.get("blur_face"):
                    imgs = [blur_faces(i) for i in imgs]
            shown, hidden = [], 0
            for im in imgs:
                if self.is_nsfw(im):
                    hidden += 1
                    continue
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                shown.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return {"images": shown, "hidden": hidden, "prompt": prompt, "seed": seed,
                "seconds": round(time.time() - t0, 1)}


def serve(tiny_ckpt: Path | None, host: str = "127.0.0.1", port: int = 8090, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engines = Engines(tiny_ckpt, device)
    print(f"Device: {device}" + ("" if device == "cuda" else " (no GPU found: the ready-made model will be slow)"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/options":
                have_tiny = bool(tiny_ckpt and Path(tiny_ckpt).exists())
                self._send(200, json.dumps({"options": OPTIONS, "tiny_options": TINY_OPTIONS,
                                            "tiny_available": have_tiny, "device": device}).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/generate":
                return self._send(404, b"{}")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                res = engines.generate(req)
            except Exception as e:  # show the problem on the page instead of a dead request
                res = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(res).encode())

    ThreadingHTTPServer((host, port), H).serve_forever()
