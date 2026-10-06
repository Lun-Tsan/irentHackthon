"""Jev-style fast image classifier on the official iRent photos.

Frozen encoder (DINOv2-small / CLIP ViT-B/32, both off the shelf) → tiny decision head (logistic regression)
→ typed answer + calibrated probability → confidence-band routing (pass / review / human).

Two questions, evaluated with plate-grouped, shuffled 5-fold cross-validation (a car never appears in both
train and test folds):
  Q1 single photo : "does this photo show a damaged car?"      claim 還/還2/車況回報 = 1, no-claim photos = 0
  Q2 pickup/return: "did this rental add new damage?"           claim 借→還 pairs = 1, no-claim 借→還 pairs = 0
Plus a zero-training CLIP baseline for Q1 (prompts only).

Run:  .venv/Scripts/python.exe experiments/claim_classifier.py
Data: set IRENT_DATA to the official folder that holds 進行索賠資料/ and 沒有進行索賠/ (confidential, not in git).
"""
import json
import os
import pathlib
import random
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.vision.quality import load_image, resize_width  # noqa: E402

DATA = pathlib.Path(os.environ.get("IRENT_DATA", ""))
OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261003
NAME_RE = re.compile(r"^([A-Z]{3}-?\d{4})\s*(左前|右前|左後|右後)?\s*(借|還2|還|車況回報)")


def index_photos():
    rows = []
    for folder, claim in (("進行索賠資料", 1), ("沒有進行索賠", 0)):
        for p in sorted((DATA / folder).iterdir()):
            m = NAME_RE.match(p.stem)
            if not m:
                print("skip (name not parsed):", p.name)
                continue
            plate, view, phase = m.group(1).replace("-", ""), m.group(2) or "", m.group(3)
            rows.append({"path": str(p), "name": p.name, "plate": plate, "view": view, "phase": phase, "claim": claim})
    return rows


# ---------- embeddings (cached) ----------

def _feat(out):
    """transformers 5 returns a ModelOutput whose pooler_output is the projected CLIP embedding; v4 returned the tensor."""
    import torch
    if torch.is_tensor(out):
        return out
    return out.pooler_output


def embed_all(rows):
    cache = OUT / "embeddings.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == [r["name"] for r in rows]:
            return z["dino"], z["clip"]
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dm = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    dino, clip = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            img = resize_width(load_image(pathlib.Path(r["path"]).read_bytes()), 640)
            o = dm(**dp(images=img, return_tensors="pt"))
            h = o.last_hidden_state[0]
            dino.append(torch.cat([h[0], h[1:].mean(0)]).numpy())          # CLS + mean patch = 768 dims
            f = _feat(cm.get_image_features(**cp(images=img, return_tensors="pt")))[0]
            clip.append((f / f.norm()).numpy())
            if i % 25 == 0:
                print(f"  embedded {i}/{len(rows)}  ({time.time() - t0:.0f}s)")
    dino, clip = np.stack(dino), np.stack(clip)
    np.savez(cache, dino=dino, clip=clip, names=np.array([r["name"] for r in rows]))
    print(f"embedding time {time.time() - t0:.0f}s for {len(rows)} photos "
          f"= {1000 * (time.time() - t0) / len(rows):.0f} ms/photo (both encoders, CPU)")
    return dino, clip


def clip_zero_shot(rows, clip_feats):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    prompts = {
        1: ["a photo of a damaged car with scratches", "a photo of a car with a dent", "a photo of a scraped car bumper", "a car with body damage"],
        0: ["a photo of an undamaged car", "a photo of a clean car in good condition", "a photo of a car with no damage", "a parked car"],
    }
    with torch.no_grad():
        txt = {k: _feat(cm.get_text_features(**cp(text=v, return_tensors="pt", padding=True))) for k, v in prompts.items()}
        txt = {k: (t / t.norm(dim=-1, keepdim=True)).numpy() for k, t in txt.items()}
    logits = {k: (clip_feats @ t.T).max(1) * 100 for k, t in txt.items()}
    p = 1 / (1 + np.exp(-(logits[1] - logits[0])))
    return p


# ---------- evaluation helpers ----------

def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def grouped_cv(X, y, groups, name, calibrate=True, seed=SEED, C=0.05):
    """Shuffled, plate-grouped 5-fold CV. Returns out-of-fold calibrated probabilities."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups):
        base = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        if calibrate and y[tr].sum() >= 6:
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            clf = base
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def recall_at_flag_rate(p, y, rates=(0.1, 0.2, 0.3, 0.5)):
    """Threshold-free view of routing: if the top X% most suspicious go to review/human, what share of true positives is caught?"""
    out = {}
    order = np.argsort(-p)
    for r in rates:
        k = max(1, int(round(r * len(p))))
        flagged = np.zeros(len(p), bool)
        flagged[order[:k]] = True
        out[f"flag_top_{int(r * 100)}%"] = {"recall": float(y[flagged].sum() / max(1, y.sum())), "precision": float(y[flagged].mean())}
    return out


def sweep(X, y, groups, label):
    """Try a few head settings; report AUC and recall when the top 30% is flagged. Picks nothing — just shows sensitivity."""
    rows = []
    for C in (0.01, 0.1, 1.0):
        for cal in (False, True):
            p = grouped_cv(X, y, groups, label, calibrate=cal, C=C)
            r = recall_at_flag_rate(p, y)
            rows.append({"C": C, "calibrated": cal, "auc": round(auc(p, y), 3), "recall@top30%": round(r["flag_top_30%"]["recall"], 3), "ece": round(ece(p, y), 3)})
    return rows


def routing(p, y, t1, t2):
    bands = np.where(p < t1, "pass", np.where(p < t2, "review", "human"))
    out = {}
    for b in ("pass", "review", "human"):
        m = bands == b
        out[b] = {"share": float(m.mean()), "n": int(m.sum()), "positives": int(y[m].sum())}
    out["recall_not_auto_passed"] = float(1 - y[bands == "pass"].sum() / max(1, y.sum()))
    return out


def report_block(title, p, y, thr=0.5):
    from sklearn.metrics import confusion_matrix
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "title": title, "n": int(len(y)), "positives": int(y.sum()), "auc": auc(p, y),
        "accuracy@0.5": float((pred == y).mean()), "ece": ece(p, y),
        "confusion[[tn,fp],[fn,tp]]": cm.tolist(),
        "routing_t1_0.25_t2_0.6": routing(p, y, 0.25, 0.6),
        "recall_at_flag_rate": recall_at_flag_rate(p, y),
    }


def contact_sheet(rows, p, y, path, title, k=16):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    order = np.argsort(-p)[:k]
    tw, th = 160, 285
    sheet = Image.new("RGB", (tw * 8, (th + 24) * 2 + 30), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 15)  # Microsoft JhengHei: has the CJK glyphs
    d.text((6, 6), title, fill=(255, 220, 120), font=font)
    for i, j in enumerate(order):
        im = ImageOps.exif_transpose(Image.open(rows[j]["path"])).convert("RGB")
        im = ImageOps.fit(im, (tw, th))
        x, yy = (i % 8) * tw, 30 + (i // 8) * (th + 24)
        sheet.paste(im, (x, yy + 24))
        d.text((x + 4, yy + 4), f"p={p[j]:.2f} {'索賠' if y[j] else '正常'}", fill=(120, 255, 160) if y[j] else (255, 140, 120), font=font)
    sheet.save(path, quality=85)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rows = index_photos()
    print(f"photos: {len(rows)}  claim folder: {sum(r['claim'] for r in rows)}  no-claim: {sum(1 - r['claim'] for r in rows)}")
    dino, clip = embed_all(rows)

    # ----- Q1 single photo -----
    y1 = np.array([1 if (r["claim"] and r["phase"] in ("還", "還2", "車況回報")) else 0 for r in rows])
    g1 = np.array([r["plate"] for r in rows])
    results = {"seed": SEED, "n_photos": len(rows)}
    pz = clip_zero_shot(rows, clip)
    results["Q1_clip_zero_shot"] = report_block("Q1 CLIP zero-shot（完全不訓練）", pz, y1)
    p_d = grouped_cv(dino, y1, g1, "dino")
    results["Q1_dino_probe"] = report_block("Q1 DINOv2 + 決策頭（分組交叉驗證）", p_d, y1)
    p_c = grouped_cv(clip, y1, g1, "clip")
    results["Q1_clip_probe"] = report_block("Q1 CLIP 特徵 + 決策頭（分組交叉驗證）", p_c, y1)
    contact_sheet(rows, p_d, y1, OUT / "q1_top_damaged.jpg", "Q1 DINOv2 決策頭：判定最像有損傷的 16 張（綠＝索賠資料夾，紅＝正常）")
    results["Q1_sweep_dino"] = sweep(dino, y1, g1, "dino")
    results["Q1_sweep_clip"] = sweep(clip, y1, g1, "clip")

    # shuffled spot check the user asked for: 20 random photos, prediction vs truth
    idx = list(range(len(rows)))
    random.shuffle(idx)
    results["Q1_shuffled_spot_check"] = [
        {"photo": rows[i]["name"], "truth": "索賠" if y1[i] else "正常", "p_damage": round(float(p_d[i]), 3),
         "band": "pass" if p_d[i] < 0.25 else ("review" if p_d[i] < 0.6 else "human")} for i in idx[:20]]

    # ----- Q2 pickup→return pairs -----
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["plate"], r["view"]), {})[r["phase"]] = i
    pairs = []
    for (plate, view), d in by.items():
        if "借" in d and ("還" in d or "還2" in d):
            j = d.get("還", d.get("還2"))
            pairs.append((d["借"], j, plate, rows[j]["claim"]))
    if pairs:
        Xp = np.stack([np.concatenate([dino[j] - dino[i], np.abs(dino[j] - dino[i]), [float(clip[i] @ clip[j])]]) for i, j, _, _ in pairs])
        y2 = np.array([c for *_, c in pairs])
        g2 = np.array([pl for _, _, pl, _ in pairs])
        p2 = grouped_cv(Xp, y2, g2, "pairs")
        results["Q2_pair_new_damage"] = report_block("Q2 借→還 成對（新車損）DINOv2 差異 + 決策頭", p2, y2)
        results["Q2_pairs_n"] = {"pairs": len(pairs), "claim_pairs": int(y2.sum())}
        results["Q2_sweep"] = sweep(Xp, y2, g2, "pairs")
        # the simplest possible pair signal: how different are the two photos? (no training at all)
        cos = np.array([float(clip[i] @ clip[j]) for i, j, _, _ in pairs])
        results["Q2_cosine_only_no_training"] = {"auc": auc(1 - cos, y2), "recall_at_flag_rate": recall_at_flag_rate(1 - cos, y2)}
        contact_sheet([rows[j] for _, j, _, _ in pairs], p2, y2, OUT / "q2_top_new_damage.jpg", "Q2 成對比對：判定最像有新車損的 16 筆還車照（綠＝索賠，紅＝正常）")
        results["Q2_shuffled_spot_check"] = [
            {"pair": f"{pl} {rows[i]['view']} 借→還", "truth": "索賠" if c else "正常", "p_new_damage": round(float(p2[k]), 3)}
            for k, (i, j, pl, c) in enumerate(pairs) if k in set(random.sample(range(len(pairs)), min(16, len(pairs))))]

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k.startswith("Q") and not k.endswith("spot_check")}, ensure_ascii=False, indent=1))
    print("\n--- Q1 shuffled spot check ---")
    for s in results["Q1_shuffled_spot_check"]:
        print(f"  {s['truth']}  p={s['p_damage']:.2f}  {s['band']:6s}  {s['photo']}")
    if pairs:
        print("\n--- Q2 shuffled spot check ---")
        for s in results["Q2_shuffled_spot_check"]:
            print(f"  {s['truth']}  p={s['p_new_damage']:.2f}  {s['pair']}")
    print("\nwrote", OUT / "results.json", "and", OUT / "q1_top_damaged.jpg")


if __name__ == "__main__":
    main()
