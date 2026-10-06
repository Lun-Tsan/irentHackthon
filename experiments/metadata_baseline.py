"""Shortcut check: how well can the 141 借→還 pairs be classified WITHOUT looking at the pixels?

Features per pair come only from file metadata (size in px, aspect, bytes, bytes/pixel, extension, EXIF orientation),
for both photos and their differences. Same plate-grouped 5-fold CV and labels as v1. If this AUC is high, an image
model's AUC is not evidence that it sees damage.

Run: python experiments/metadata_baseline.py   (needs IRENT_DATA)
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finetune_classifier import OUT, SEED, auc, auc_ci, folds, index_photos, lr_oof, pairs_from, recall_at  # noqa: E402


def meta(path):
    from PIL import Image
    p = pathlib.Path(path)
    im = Image.open(p)
    w, h = im.size
    orient = im.getexif().get(0x0112, 0)
    if orient in (5, 6, 7, 8):
        w, h = h, w
    size = p.stat().st_size
    return [w, h, w / h, size / 1e6, size / (w * h), float(p.suffix.lower() == ".jfif"), float(orient != 0)]


def main():
    rows = index_photos()
    pairs = pairs_from(rows)
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    M = {i: np.array(meta(rows[i]["path"])) for pr in pairs for i in pr[:2]}
    X = np.stack([np.concatenate([M[i], M[j], np.abs(M[j] - M[i])]) for i, j, *_ in pairs])
    out = {}
    for C in (0.01, 0.1, 1.0):
        p = lr_oof(X, y, folds(y, g, SEED), C=C)
        out[f"metadata LR C={C}"] = {"auc": round(auc(p, y), 3), "auc_95ci_by_car": auc_ci(p, y, g),
                                     "recall_at_flag": recall_at(p, y)}
    # single-feature views, no training
    names = ["width", "height", "aspect", "MB", "bytes_per_px", "is_jfif", "has_exif_orient"]
    for k, n in enumerate(names):
        s = X[:, 7 + k]                          # return photo's value
        out[f"return photo {n} alone"] = round(max(auc(s, y), 1 - auc(s, y)), 3)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (OUT / "metadata_baseline.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
