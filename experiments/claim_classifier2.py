"""v2 of the pair experiment: rotate photos upright, crop the car body, then compare.

Steps per photo
  1. 轉向：try 0/90/180/270°, keep the rotation where YOLO11n is most confident there is a vehicle
  2. 框車體：largest vehicle box (+6% margin); no box → whole photo
  3. embed the crop: CLIP (512), DINOv2 CLS+mean (768) and DINOv2 patch tokens (16×16×384)

Pair scores (借 → 還), all training-free unless marked:
  global_cos_full   1 − cos(CLIP) on the full photo            (v1 baseline, AUC 0.84)
  global_cos_crop   1 − cos(CLIP) on the car crop
  patch_nn_mean     mean over 還 patches of (1 − best cosine to any 借 patch)      "how much of 還 is new"
  patch_nn_top10    mean of the 10% least-explained 還 patches                       sensitive to a local scratch
  head_crop         DINOv2 difference features + logistic head (trained, grouped CV)

Run:  .venv/Scripts/python.exe experiments/claim_classifier2.py
"""
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.vision.quality import load_image, resize_width  # noqa: E402
from experiments.claim_classifier import OUT, SEED, auc, contact_sheet, grouped_cv, index_photos, recall_at_flag_rate  # noqa: E402

VEHICLE = [2, 3, 5, 7]  # COCO car, motorcycle, bus, truck


def detect_all(rows):
    """Rotation search + car box with YOLO11n. Cached to out/detections.json."""
    cache = OUT / "detections.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if set(d) == {r["name"] for r in rows}:
            return d
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "yolo11n.pt")) if (ROOT / "yolo11n.pt").exists() else YOLO("yolo11n.pt")
    det, t0 = {}, time.time()
    for k, r in enumerate(rows):
        img = resize_width(load_image(pathlib.Path(r["path"]).read_bytes()), 640)
        best = {"rot": 0, "conf": 0.0, "box": None}
        for rot in (0, 90, 180, 270):
            im = img.rotate(rot, expand=True) if rot else img
            res = model.predict(im, imgsz=640, conf=0.2, classes=VEHICLE, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            areas = (res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])
            ok = areas >= 0.05 * im.width * im.height
            if not bool(ok.any()):
                continue
            confs = res.boxes.conf.clone()
            confs[~ok] = 0
            i = int(confs.argmax())
            # prefer the upright reading: a rotation only wins if it is clearly more confident
            score = float(confs[i]) + (0.08 if rot == 0 else 0.0)
            if score > best["conf"]:
                best = {"rot": rot, "conf": score, "box": [float(v) for v in res.boxes.xyxy[i]], "size": [im.width, im.height]}
        det[r["name"]] = best
        if k % 40 == 0:
            print(f"  detected {k}/{len(rows)} ({time.time() - t0:.0f}s)")
    print(f"detection {time.time() - t0:.0f}s = {1000 * (time.time() - t0) / len(rows):.0f} ms/photo incl. 4 rotations")
    cache.write_text(json.dumps(det), encoding="utf-8")
    return det


def crop_for(r, d):
    img = resize_width(load_image(pathlib.Path(r["path"]).read_bytes()), 640)
    if d["rot"]:
        img = img.rotate(d["rot"], expand=True)
    if not d.get("box"):
        return img, False
    x0, y0, x1, y1 = d["box"]
    mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    box = (max(0, x0 - mx), max(0, y0 - my), min(img.width, x1 + mx), min(img.height, y1 + my))
    return img.crop(tuple(int(v) for v in box)), True


def embed_crops(rows, det):
    cache = OUT / "embeddings_crop.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == [r["name"] for r in rows]:
            return z["clip"], z["dino"], z["patches"]
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dm = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip, dino, patches, t0 = [], [], [], time.time()
    with torch.no_grad():
        for k, r in enumerate(rows):
            crop, _ = crop_for(r, det[r["name"]])
            o = dm(**dp(images=crop, return_tensors="pt"))
            h = o.last_hidden_state[0]
            dino.append(torch.cat([h[0], h[1:].mean(0)]).numpy())
            pt = h[1:]
            pt = pt / pt.norm(dim=-1, keepdim=True)
            patches.append(pt.numpy().astype(np.float16))
            f = cm.get_image_features(**cp(images=crop, return_tensors="pt"))
            f = (f if torch.is_tensor(f) else f.pooler_output)[0]
            clip.append((f / f.norm()).numpy())
            if k % 40 == 0:
                print(f"  embedded {k}/{len(rows)} ({time.time() - t0:.0f}s)")
    clip, dino, patches = np.stack(clip), np.stack(dino), np.stack(patches)
    np.savez(cache, clip=clip, dino=dino, patches=patches, names=np.array([r["name"] for r in rows]))
    print(f"crop embedding {time.time() - t0:.0f}s = {1000 * (time.time() - t0) / len(rows):.0f} ms/photo")
    return clip, dino, patches


def patch_scores(pa, pb):
    """pa = 借 patches, pb = 還 patches (both L2-normalised). How much of 還 has no counterpart in 借?"""
    s = pb.astype(np.float32) @ pa.astype(np.float32).T   # (nB, nA)
    nn = s.max(1)
    unexplained = 1 - nn
    top = np.sort(unexplained)[-max(1, len(unexplained) // 10):]
    return float(unexplained.mean()), float(top.mean())


def pair_sheet(rows, det, pairs, scores, path, title, k=8):
    from PIL import ImageDraw, ImageFont, ImageOps
    order = np.argsort(-scores)[:k]
    tw, th = 150, 200
    sheet = Image.new("RGB", (tw * 2 * k + 10 * (k - 1), th + 50), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 14)
    d.text((6, 4), title, fill=(255, 220, 120), font=font)
    for n, idx in enumerate(order):
        i, j, pl, c = pairs[idx]
        x = n * (tw * 2 + 10)
        for m, ii in enumerate((i, j)):
            crop, _ = crop_for(rows[ii], det[rows[ii]["name"]])
            sheet.paste(ImageOps.fit(crop, (tw, th)), (x + m * tw, 50))
        d.text((x + 4, 28), f"{pl} {rows[j]['view']} 借|還  s={scores[idx]:.2f} {'索賠' if c else '正常'}", fill=(120, 255, 160) if c else (255, 140, 120), font=font)
    sheet.save(path, quality=85)


def main():
    rows = index_photos()
    det = detect_all(rows)
    rot = sum(1 for d in det.values() if d["rot"])
    nobox = sum(1 for d in det.values() if not d.get("box"))
    print(f"photos {len(rows)}: rotated upright {rot}, no vehicle box {nobox}")

    clip_c, dino_c, patches = embed_crops(rows, det)
    z = np.load(OUT / "embeddings.npz", allow_pickle=True)
    clip_full = z["clip"]

    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["plate"], r["view"]), {})[r["phase"]] = i
    pairs = []
    for (plate, view), d in by.items():
        if "借" in d and ("還" in d or "還2" in d):
            j = d.get("還", d.get("還2"))
            pairs.append((d["借"], j, plate, rows[j]["claim"]))
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    print(f"pairs {len(pairs)}, claim pairs {int(y.sum())}")

    scores = {
        "global_cos_full": np.array([1 - float(clip_full[i] @ clip_full[j]) for i, j, _, _ in pairs]),
        "global_cos_crop": np.array([1 - float(clip_c[i] @ clip_c[j]) for i, j, _, _ in pairs]),
    }
    pm, pt = zip(*[patch_scores(patches[i], patches[j]) for i, j, _, _ in pairs])
    scores["patch_nn_mean"] = np.array(pm)
    scores["patch_nn_top10"] = np.array(pt)
    Xp = np.stack([np.concatenate([dino_c[j] - dino_c[i], np.abs(dino_c[j] - dino_c[i]), [scores["global_cos_crop"][k], pm[k], pt[k]]]) for k, (i, j, _, _) in enumerate(pairs)])
    scores["head_crop (trained)"] = grouped_cv(Xp, y, g, "pairs-crop", calibrate=False, C=0.01)

    results = {"n_pairs": len(pairs), "claim_pairs": int(y.sum()), "rotated_upright": rot, "no_vehicle_box": nobox, "scores": {}}
    print(f"\n{'score':24s} {'AUC':>6s}  {'top10%':>7s} {'top20%':>7s} {'top30%':>7s} {'top50%':>7s}   (recall of claim pairs when that share is flagged)")
    for name, s in scores.items():
        r = recall_at_flag_rate(s, y)
        results["scores"][name] = {"auc": auc(s, y), "recall_at_flag_rate": r}
        print(f"{name:24s} {auc(s, y):6.3f}  " + "  ".join(f"{r[f'flag_top_{p}%']['recall']:7.2f}" for p in (10, 20, 30, 50)))
    (OUT / "results2.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    best = max(scores, key=lambda n: results["scores"][n]["auc"])
    pair_sheet(rows, det, pairs, scores[best], OUT / "q2v2_top_pairs.jpg", f"v2 車體裁切後 · {best} 排名前 8 對（左借右還）")
    pair_sheet(rows, det, pairs, scores["patch_nn_top10"], OUT / "q2v2_top_pairs_patch.jpg", "v2 patch_nn_top10 排名前 8 對（左借右還）")
    print("wrote", OUT / "results2.json")


if __name__ == "__main__":
    main()
