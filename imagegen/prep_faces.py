"""CelebA parquet shards -> faces{SIZE}.npy (uint8 N,64,64,3) + attrs.npy (uint8 N,A) + attr_names.txt"""
import io, sys, glob
import numpy as np, pyarrow.parquet as pq
from PIL import Image

ATTRS = ["Male", "Young", "Smiling", "Mouth_Slightly_Open", "Eyeglasses", "Wearing_Hat",
         "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald", "Bangs", "Wavy_Hair",
         "Straight_Hair", "No_Beard", "Mustache", "Goatee", "Heavy_Makeup", "Wearing_Lipstick",
         "Wearing_Earrings", "Wearing_Necklace", "Wearing_Necktie", "Pale_Skin", "Chubby"]

src, out = sys.argv[1], sys.argv[2]
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 64
imgs, attrs = [], []
for f in sorted(glob.glob(f"{src}/*.parquet")):
    t = pq.read_table(f, columns=["image"] + ATTRS).to_pydict()
    for i, im in enumerate(t["image"]):
        x = Image.open(io.BytesIO(im["bytes"])).convert("RGB")  # 178x218 aligned
        x = x.crop((19, 49, 159, 189)).resize((SIZE, SIZE), Image.BICUBIC)  # face-centred square
        imgs.append(np.asarray(x, dtype=np.uint8))
        attrs.append([int(t[a][i]) for a in ATTRS])
    print(f, len(imgs), flush=True)
np.save(f"{out}/faces{SIZE}.npy", np.stack(imgs))
np.save(f"{out}/attrs.npy", np.array(attrs, dtype=np.uint8))
open(f"{out}/attr_names.txt", "w").write("\n".join(ATTRS))
