"""Cut the ghost-guide photos (data/samples/*.jpg) for the phone viewfinder from the official 500-set pack.

The official photos are confidential, so data/samples/ is not in git: every teammate regenerates it locally.
The official ImageType labels are often wrong, so one clean front corner and one clean rear corner are used and
the other side is mirrored; the phone page has a 「鏡像範例」 toggle too. Which sets to use is chosen locally:
IRENT_GHOST_EXT = '<set>' (its 取車_02 / 取車_03 are the front / rear corners), IRENT_GHOST_INT = '<set>'.

    set IRENT_PIC500=C:\\path\\to\\Hims_Pic500        (the extracted 500-set folder)
    set IRENT_GHOST_EXT=<set>  &  set IRENT_GHOST_INT=<set>
    .venv\\Scripts\\python.exe scripts\\make_samples.py
"""
import os
import pathlib

from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIC500 = pathlib.Path(os.environ.get("IRENT_PIC500", ""))
OUT = ROOT / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def load(set_id, name):
    return ImageOps.exif_transpose(Image.open(PIC500 / set_id / name)).convert("RGB")


EXT, INT = os.environ["IRENT_GHOST_EXT"], os.environ["IRENT_GHOST_INT"]
front, rear = load(EXT, "取車_02.png"), load(EXT, "取車_03.png")
picks = {"fr": front, "fl": ImageOps.mirror(front), "rl": rear, "rr": ImageOps.mirror(rear),
         "int_front": load(INT, "還車_10.png"), "int_rear": load(INT, "還車_11.png")}
for k, im in picks.items():
    s = 540 / im.width
    im.resize((540, round(im.height * s))).save(OUT / f"{k}.jpg", quality=82)
    print(k, "->", OUT / f"{k}.jpg")
