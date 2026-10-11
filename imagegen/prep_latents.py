"""CelebA parquet shards -> TAESD latents of 128x128 face crops.

Writes latents.npy (float16 N,4,16,16), attrs.npy (uint8 N,A) and a decoded check image.
TAESD (madebyollin/taesd, ~10 MB) only compresses/decompresses pictures; the generator is ours.
"""
import glob, io, sys
import numpy as np, pyarrow.parquet as pq, torch
from diffusers import AutoencoderTiny
from PIL import Image
from tinyface import ATTRS

src, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval()
lats, attrs = [], []
for f in sorted(glob.glob(f"{src}/*.parquet")):
    part = f.replace(".parquet", ".lat.npz")
    try:
        z = np.load(part)
        lats.append(z["lat"]), attrs.extend(z["attrs"].tolist())
        print(f, "cached", flush=True)
        continue
    except FileNotFoundError:
        pass
    t = pq.read_table(f, columns=["image"] + ATTRS).to_pydict()
    imgs = [np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB").crop((19, 49, 159, 189))
                       .resize((128, 128), Image.BICUBIC)) for im in t["image"]]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1
    with torch.no_grad():
        lat = np.concatenate([vae.encode(x[i:i + 256]).latents.half().numpy() for i in range(0, len(x), 256)])
    att = [[int(t[a][i]) for a in ATTRS] for i in range(len(imgs))]
    np.savez(part, lat=lat, attrs=np.array(att, dtype=np.uint8))
    lats.append(lat), attrs.extend(att)
    print(f, sum(len(l) for l in lats), flush=True)
L = np.concatenate(lats)
np.save(f"{out}/latents.npy", L)
np.save(f"{out}/attrs.npy", np.array(attrs, dtype=np.uint8))
print("latent mean/std", float(L.astype(np.float32).mean()), float(L.astype(np.float32).std()))
with torch.no_grad():
    y = vae.decode(torch.from_numpy(L[:16]).float()).sample
y = ((y.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
Image.fromarray(np.concatenate(list(y), axis=1)).save(f"{out}/latent_check.png")
