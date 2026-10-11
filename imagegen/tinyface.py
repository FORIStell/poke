"""Tiny face generator trained from scratch: a small diffusion U-Net conditioned on face options.

It draws 128x128 faces in TAESD's compressed latent space (4x16x16); the tiny pretrained TAESD
decoder (~5 MB) turns that into the picture. Only the U-Net is trained here.

Each option is +1 (yes), -1 (no) or 0 (don't care). During training options are randomly hidden,
so the model also learns to draw a face when only some (or none) are given; at sampling time this
gives classifier-free guidance towards the chosen options.

    python tinyface.py train --data data --out runs/tiny --steps 60000
    python tinyface.py sample --ckpt runs/tiny/ema.pt --set Male=1 Smiling=1 Blond_Hair=1 -n 16 -o out.png
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image

ATTRS = ["Male", "Young", "Smiling", "Mouth_Slightly_Open", "Eyeglasses", "Wearing_Hat",
         "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald", "Bangs", "Wavy_Hair",
         "Straight_Hair", "No_Beard", "Mustache", "Goatee", "Heavy_Makeup", "Wearing_Lipstick",
         "Wearing_Earrings", "Wearing_Necklace", "Wearing_Necktie", "Pale_Skin", "Chubby"]


class TinyFace(nn.Module):
    def __init__(self, n_attrs: int = len(ATTRS), base: int = 128):
        super().__init__()
        tdim = base * 4
        self.unet = UNet2DModel(
            sample_size=16, in_channels=4, out_channels=4, layers_per_block=2,
            block_out_channels=(base, base * 2, base * 3),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            class_embed_type="identity", attention_head_dim=64,
        )
        self.attr_embed = nn.Sequential(nn.Linear(n_attrs, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

    def forward(self, x, t, attrs):
        return self.unet(x, t, class_labels=self.attr_embed(attrs)).sample


def hide_options(a: torch.Tensor) -> torch.Tensor:
    """Randomly set options to 0 ("don't care"): 30% of each, and all of them for 10% of images."""
    keep = (torch.rand_like(a) > 0.3).float()
    keep *= (torch.rand(a.shape[0], 1) > 0.1).float()
    return a * keep


def save_grid(imgs: torch.Tensor, path: Path, cols: int = 8, scale: int = 2) -> None:
    x = ((imgs.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    n, h, w, _ = x.shape
    rows = math.ceil(n / cols)
    grid = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = x[i]
    Image.fromarray(grid).resize((cols * w * scale, rows * h * scale), Image.LANCZOS).save(path)


@torch.no_grad()
def sample(model: TinyFace, attrs: torch.Tensor, steps: int = 40, guidance: float = 2.5, seed: int = 0) -> torch.Tensor:
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    sched.set_timesteps(steps)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(attrs.shape[0], 4, 16, 16, generator=g)
    blank = torch.zeros_like(attrs)
    for t in sched.timesteps:
        tt = t.repeat(attrs.shape[0])
        eps = model(x, tt, attrs)
        if guidance != 1:
            eps_u = model(x, tt, blank)
            eps = eps_u + guidance * (eps - eps_u)
        x = sched.step(eps, t, x).prev_sample
    return x


_VAE = None


@torch.no_grad()
def decode(latents: torch.Tensor, scale: float) -> torch.Tensor:
    """Latents (N,4,16,16) -> pictures (N,3,128,128) in [-1, 1] with the tiny TAESD decoder."""
    global _VAE
    if _VAE is None:
        from diffusers import AutoencoderTiny
        _VAE = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval()
    return _VAE.decode(latents * scale).sample


def demo_attrs() -> tuple[torch.Tensor, list[str]]:
    """Fixed sample sheet: one row per option set, so rounds can be compared side by side."""
    idx = {a: i for i, a in enumerate(ATTRS)}
    rows = [
        ("any face", {}),
        ("woman, smiling", {"Male": -1, "Smiling": 1}),
        ("man, beard", {"Male": 1, "No_Beard": -1}),
        ("blond woman", {"Male": -1, "Blond_Hair": 1}),
        ("man, glasses", {"Male": 1, "Eyeglasses": 1}),
        ("old man, gray hair", {"Male": 1, "Young": -1, "Gray_Hair": 1}),
        ("black hair, hat", {"Black_Hair": 1, "Wearing_Hat": 1}),
        ("woman, serious, lipstick", {"Male": -1, "Smiling": -1, "Wearing_Lipstick": 1}),
    ]
    out = torch.zeros(len(rows) * 8, len(ATTRS))
    for r, (_, opts) in enumerate(rows):
        for k, v in opts.items():
            out[r * 8:(r + 1) * 8, idx[k]] = v
    return out, [name for name, _ in rows]


def train(args) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    faces = np.load(Path(args.data) / "latents.npy")
    scale = float(faces[:5000].astype(np.float32).std())
    attrs = torch.from_numpy(np.load(Path(args.data) / "attrs.npy").astype(np.float32) * 2 - 1)
    n = len(faces)
    model = TinyFace(base=args.base)
    ema = TinyFace(base=args.base)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]), ema.load_state_dict(ck["ema"]), opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, images: {n}", flush=True)
    rng = np.random.default_rng(step)
    demo, _ = demo_attrs()
    t0, losses = time.time(), []
    while step < args.steps:
        model.train()
        idx = np.sort(rng.choice(n, args.batch, replace=False))
        x = torch.from_numpy(faces[idx].astype(np.float32) / scale)
        if rng.random() < 0.5:
            x = x.flip(3)  # mirror: same person, same options
        a = hide_options(attrs[idx])
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],))
        loss = F.mse_loss(model(sched.add_noise(x, noise, t), t, a), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for gp in opt.param_groups:
            gp["lr"] = args.lr * min(1.0, (step + 1) / 1000)
        opt.step()
        step += 1
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step} loss {np.mean(losses):.4f} {(time.time() - t0) / 100:.2f}s/step", flush=True)
            losses, t0 = [], time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step},
                       out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(out / "last.pt")
            torch.save({"ema": ema.state_dict(), "base": args.base, "attrs": ATTRS, "step": step, "scale": scale},
                       out / "ema.pt")
        if step % args.sample_every == 0 or step == args.steps:
            save_grid(decode(sample(ema, demo), scale), out / "samples" / f"step{step:06d}.png", scale=1)
            print(f"samples saved at step {step}", flush=True)


def load(ckpt: str) -> tuple[TinyFace, float]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = TinyFace(base=ck["base"])
    m.load_state_dict(ck["ema"])
    return m.eval(), ck["scale"]


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="runs/tiny")
    t.add_argument("--steps", type=int, default=60000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--base", type=int, default=128)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--sample-every", type=int, default=2500)
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--set", nargs="*", default=[], help="Option=1/-1, e.g. Male=1 Smiling=1")
    s.add_argument("-n", type=int, default=16)
    s.add_argument("-o", default="out.png")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--guidance", type=float, default=2.5)
    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        a = torch.zeros(args.n, len(ATTRS))
        for kv in args.set:
            k, v = kv.split("=")
            a[:, ATTRS.index(k)] = float(v)
        m, scale = load(args.ckpt)
        save_grid(decode(sample(m, a, guidance=args.guidance, seed=args.seed), scale), Path(args.o), scale=1)


if __name__ == "__main__":
    main()
