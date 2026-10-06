"""Fine-tuned ImageNet classifiers (an ordinary image classifier) on the official iRent photos.

Same data, labels and plate-grouped shuffled 5-fold CV as claim_classifier.py (v1), so the numbers are comparable:
  q1  single photo : claim photo (進行索賠資料 還/還2/車況回報) = 1, everything else = 0      (32 vs 261)
  q2  借→還 pair   : pair from 進行索賠資料 = 1, pair from 沒有進行索賠 = 0, siamese            (21 vs 120)
  v   view angle   : 500-set filename ImageType 01–04 / 10 / 11, grouped by set               (labels are noisy per README)

Every backbone runs twice: frozen ImageNet features + logistic regression (like v1), and a full fine-tune.
Controls:
  * label permutation (labels shuffled across cars) -> what a "result" looks like when there is nothing to learn
  * q1 split by subset: inside the claim folder only (same car, same corner, 借 vs 還) the folder/camera/batch
    cues are gone, so what is left is the damage itself (plus lighting).

Run:  set IRENT_DATA / IRENT_PIC500, then
      python experiments/finetune_classifier.py --tasks q1,q2,v
      python experiments/finetune_classifier.py --quick          (1 fold, 1 epoch, smoke test)
Images are resized to a fixed square (whole frame kept, no centre crop) and cached as uint8 arrays outside the repo.
"""
import argparse
import json
import math
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "out"
DATA = pathlib.Path(os.environ.get("IRENT_DATA", ""))
PIC500 = pathlib.Path(os.environ.get("IRENT_PIC500", ""))
CACHE = pathlib.Path(os.environ.get("IRENT_CACHE", str(pathlib.Path.home() / "irent_cache")))
SEED = 20261003                     # same CV seed as v1
NAME_RE = re.compile(r"^([A-Z]{3}-?\d{4})\s*(左前|右前|左後|右後)?\s*(借|還2|還|車況回報)")   # identical to v1
NAME500 = re.compile(r"^(取車|還車)_(\d{2})$")
VIEW_TYPES = [1, 2, 3, 4, 10, 11]
VIEW_NAMES = ["左前", "右前", "左後", "右後", "前車內", "後車內"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data

def index_photos():
    """Same rows, same order as claim_classifier.index_photos (v1)."""
    rows = []
    for folder, claim in (("進行索賠資料", 1), ("沒有進行索賠", 0)):
        for p in sorted((DATA / folder).iterdir()):
            m = NAME_RE.match(p.stem)
            if not m:
                print("skip (name not parsed):", p.name)
                continue
            rows.append({"path": str(p), "name": p.name, "plate": m.group(1).replace("-", ""),
                         "view": m.group(2) or "", "phase": m.group(3), "claim": claim})
    return rows


def pairs_from(rows):
    """Same pairing rule as v1: (plate, view) with 借 and 還 (else 還2)."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["plate"], r["view"]), {})[r["phase"]] = i
    pairs = []
    for (plate, view), d in by.items():
        if "借" in d and ("還" in d or "還2" in d):
            j = d.get("還", d.get("還2"))
            pairs.append((d["借"], j, plate, rows[j]["claim"]))
    return pairs


def index_pic500():
    rows = []
    for d in sorted(PIC500.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            m = NAME500.match(p.stem)
            if m and int(m.group(2)) in VIEW_TYPES:
                rows.append({"path": str(p), "name": f"{d.name}/{p.name}", "set": d.name, "phase": m.group(1),
                             "label": VIEW_TYPES.index(int(m.group(2)))})
    return rows


def load_square(path, size):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im.draft("RGB", (size * 2, size * 2))           # JPEG: decode at reduced scale (all files here are JPEG, even *.png)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im.resize((size, size), Image.BICUBIC))     # squash, keep the whole frame


def load_all(rows, size, tag):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tag}_{size}.npz"
    names = [r["name"] for r in rows]
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if list(z["names"]) == names:
            return z["x"]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        x = np.stack(list(ex.map(lambda r: load_square(r["path"], size), rows)))
    np.savez(f, x=x, names=np.array(names))
    print(f"  loaded {len(rows)} images at {size}px in {time.time() - t0:.0f}s -> {f}")
    return x


# ---------------------------------------------------------------- models

def make_backbone(arch):
    import torch.nn as nn
    import torchvision.models as tvm
    if arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        dim, m.fc = m.fc.in_features, nn.Identity()
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim, m.classifier = m.classifier[1].in_features, nn.Identity()
    elif arch == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    else:
        raise ValueError(arch)
    return m, dim


def to_gpu(xb):
    import torch
    x = xb.cuda(non_blocking=True).float().div_(255)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def make_aug(size):
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),   # day/night, basement lighting
    ])


def chw(x_uint8):
    import torch
    return torch.from_numpy(x_uint8).permute(2, 0, 1)


def batches(idx, bs, drop_last_single=True):
    out = [idx[k:k + bs] for k in range(0, len(idx), bs)]
    if drop_last_single and len(out) > 1 and len(out[-1]) < 2:      # BatchNorm cannot train on a batch of 1
        out[-2] = np.concatenate([out[-2], out[-1]])
        out = out[:-1]
    return out


def make_opt(backbone, head, steps, lr_bb=1e-4, lr_head=1e-3):
    import torch
    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": lr_bb},
                             {"params": head.parameters(), "lr": lr_head}], weight_decay=0.05)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    return opt, sched


# ---------------------------------------------------------------- fine-tune one fold

def ft_single(X, y, tr, te, arch, size, epochs, bs, seed):
    """Binary fine-tune on single photos. Returns P(y=1) for te (mean of plain + h-flip)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bb, dim = make_backbone(arch)
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 1))
    model = nn.Sequential(bb, head).cuda().to(memory_format=torch.channels_last)
    aug = make_aug(size)
    steps = epochs * len(batches(tr, bs))
    opt, sched = make_opt(bb, head, steps)
    pw = torch.tensor(float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum())), device="cuda")
    model.train()
    for _ in range(epochs):
        for b in batches(rng.permutation(tr), bs):
            xs = []
            for i in b:
                t = aug(chw(X[i]))
                xs.append(t.flip(-1) if rng.random() < 0.5 else t)
            xb = to_gpu(torch.stack(xs))
            yb = torch.tensor(y[b], dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return predict_single(model, X, te)


def predict_single(model, X, idx):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(np.asarray(idx), 32, drop_last_single=False):
            xb = to_gpu(torch.stack([chw(X[i]) for i in b]))
            p = (torch.sigmoid(model(xb).float()) + torch.sigmoid(model(xb.flip(-1)).float())) / 2
            out.append(p.squeeze(1).cpu().numpy())
    return np.concatenate(out)


class Siamese:
    """Shared backbone; head sees |f_return - f_pickup| and f_return * f_pickup."""

    def __init__(self, arch):
        import torch
        import torch.nn as nn
        self.bb, dim = make_backbone(arch)
        self.head = nn.Sequential(nn.Linear(2 * dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.bb = self.bb.cuda().to(memory_format=torch.channels_last)
        self.head = self.head.cuda()

    def __call__(self, xa, xb):
        import torch
        f = self.bb(torch.cat([xa, xb]))
        fa, fb = f[: len(xa)], f[len(xa):]
        return self.head(torch.cat([(fb - fa).abs(), fa * fb], 1)).squeeze(1)

    def train(self, on=True):
        self.bb.train(on)
        self.head.train(on)


def ft_pairs(X, pairs, y, tr, te, arch, size, epochs, bs, seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Siamese(arch)
    aug = make_aug(size)
    steps = epochs * len(batches(tr, bs))
    opt, sched = make_opt(model.bb, model.head, steps)
    pw = torch.tensor(float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum())), device="cuda")
    model.train()
    for _ in range(epochs):
        for b in batches(rng.permutation(tr), bs):
            xa, xb = [], []
            for k in b:
                i, j = pairs[k][0], pairs[k][1]
                a, c = aug(chw(X[i])), aug(chw(X[j]))           # independent crop / lighting per photo
                if rng.random() < 0.5:                          # same flip for both, or the corners stop matching
                    a, c = a.flip(-1), c.flip(-1)
                xa.append(a)
                xb.append(c)
            yb = torch.tensor(y[b], dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(to_gpu(torch.stack(xa)), to_gpu(torch.stack(xb)))
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.train(False)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(np.asarray(te), 16, drop_last_single=False):
            xa = to_gpu(torch.stack([chw(X[pairs[k][0]]) for k in b]))
            xb = to_gpu(torch.stack([chw(X[pairs[k][1]]) for k in b]))
            p = (torch.sigmoid(model(xa, xb).float()) + torch.sigmoid(model(xa.flip(-1), xb.flip(-1)).float())) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def ft_view(X, y, tr, te, arch, size, epochs, bs, seed):
    """6-class view fine-tune. No horizontal flip: it would swap 左 and 右."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bb, dim = make_backbone(arch)
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, len(VIEW_TYPES)))
    model = nn.Sequential(bb, head).cuda().to(memory_format=torch.channels_last)
    aug = make_aug(size)
    steps = epochs * len(batches(tr, bs))
    opt, sched = make_opt(bb, head, steps)
    model.train()
    for _ in range(epochs):
        for b in batches(rng.permutation(tr), bs):
            xb = to_gpu(torch.stack([aug(chw(X[i])) for i in b]))
            yb = torch.tensor(y[b], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb)
            loss = F.cross_entropy(logit.float(), yb, label_smoothing=0.1)      # labels are known to be noisy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(np.asarray(te), 64, drop_last_single=False):
            out.append(torch.softmax(model(to_gpu(torch.stack([chw(X[i]) for i in b]))).float(), 1).cpu().numpy())
    return np.concatenate(out)


def frozen_features(X, arch, bs=64):
    import torch
    bb, _ = make_backbone(arch)
    bb = bb.cuda().eval().to(memory_format=torch.channels_last)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(X), bs):
            out.append(bb(to_gpu(torch.stack([chw(x) for x in X[k:k + bs]]))).float().cpu().numpy())
    del bb
    torch.cuda.empty_cache()
    return np.concatenate(out)


# ---------------------------------------------------------------- evaluation

def folds(y, groups, seed, n=5):
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def auc_ci(p, y, groups, n_boot=2000, seed=0):
    """95% interval by resampling cars (groups), since photos of one car are not independent."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[idx].tolist())) > 1:
            vals.append(auc(p[idx], y[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def null_auc_sd(y):
    """SD of AUC when scores carry no information (Mann-Whitney null)."""
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return math.sqrt((n1 + n0 + 1) / (12 * n1 * n0))


def recall_at(p, y, rates=(0.1, 0.2, 0.3, 0.5)):
    # ties broken at random (fixed seed): claim rows come first in the data, so a stable sort would favour them
    order = np.lexsort((np.random.default_rng(0).random(len(p)), -np.asarray(p, float)))
    out = {}
    for r in rates:
        k = max(1, int(round(r * len(p))))
        out[f"top{int(r * 100)}%"] = round(float(y[order[:k]].sum() / max(1, y.sum())), 3)
    return out


def summary(p, y, groups):
    return {"auc": round(auc(p, y), 3), "auc_95ci_by_car": auc_ci(p, y, groups), "recall_at_flag": recall_at(p, y)}


def lr_oof(F, y, fl, C=0.05):
    """v1-style head: standardise + balanced logistic regression, out-of-fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    multi = len(set(y.tolist())) > 2
    oof = np.zeros((len(y), len(set(y.tolist())))) if multi else np.zeros(len(y))
    for tr, te in fl:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        clf.fit(F[tr], y[tr])
        pr = clf.predict_proba(F[te])
        oof[te] = pr if multi else pr[:, 1]
    return oof


def permuted_by_group(y, groups, seed):
    """Shuffle labels across cars, keeping each car's own label pattern intact."""
    rng = np.random.default_rng(seed)
    ug = list(dict.fromkeys(groups.tolist()))
    pattern = {g: y[groups == g] for g in ug}
    donors = list(rng.permutation(len(ug)))
    yp = y.copy()
    # cars have different photo counts, so hand each car the label of a random donor car (majority label)
    for g, d in zip(ug, donors):
        yp[groups == g] = int(round(pattern[ug[d]].mean()))
    return yp


# ---------------------------------------------------------------- tasks

def save(results):
    (OUT / "ft_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def run_q1(args, results):
    rows = index_photos()
    X = load_all(rows, args.size, "q1")
    y = np.array([1 if (r["claim"] and r["phase"] in ("還", "還2", "車況回報")) else 0 for r in rows])
    g = np.array([r["plate"] for r in rows])
    in_claim = np.array([r["claim"] == 1 and r["phase"] != "車況回報" for r in rows])     # 借 vs 還/還2, same cars
    no_report = np.array([r["phase"] != "車況回報" for r in rows])
    res = results.setdefault("q1", {"n": len(y), "positives": int(y.sum()), "null_auc_sd": round(null_auc_sd(y), 3),
                                    "runs": {}})
    print(f"q1: {len(y)} photos, {y.sum()} positive; chance AUC = 0.5 ± {null_auc_sd(y):.3f} (1 SD)")

    def report(tag, p, extra=None):
        d = summary(p, y, g)
        d["auc_without_車況回報"] = round(auc(p[no_report], y[no_report]), 3)
        d["auc_inside_claim_folder_借_vs_還"] = round(auc(p[in_claim], y[in_claim]), 3)
        d["n_inside_claim_folder"] = [int((y[in_claim] == 0).sum()), int((y[in_claim] == 1).sum())]
        if extra:
            d.update(extra)
        res["runs"][tag] = d
        res.setdefault("oof", {})[tag] = [round(float(v), 4) for v in p]
        print(f"  {tag:38s} AUC {d['auc']:.3f} {d['auc_95ci_by_car']}  no-車況回報 {d['auc_without_車況回報']:.3f}  "
              f"inside-claim 借vs還 {d['auc_inside_claim_folder_借_vs_還']:.3f}  recall@30% {d['recall_at_flag']['top30%']:.2f}")
        save(results)

    res["names"] = [r["name"] for r in rows]
    fl = folds(y, g, SEED, args.folds)
    for arch in args.archs:
        F = frozen_features(X, arch)
        report(f"{arch}/frozen+LR", lr_oof(F, y, fl))
        for s in args.seeds:
            fls = folds(y, g, s, args.folds)
            t0 = time.time()
            p = np.zeros(len(y))
            for k, (tr, te) in enumerate(fls[: args.max_folds]):
                p[te] = ft_single(X, y, tr, te, arch, args.size, args.epochs_q, args.bs, seed=s + k)
            report(f"{arch}/finetune/cv{s}", p, {"seconds": round(time.time() - t0)})
    for s in range(args.perms):
        yp = permuted_by_group(y, g, 1000 + s)
        fl_p = folds(yp, g, SEED, args.folds)
        p = np.zeros(len(y))
        for k, (tr, te) in enumerate(fl_p[: args.max_folds]):
            p[te] = ft_single(X, yp, tr, te, args.archs[-1], args.size, args.epochs_q, args.bs, seed=SEED + k)
        res["runs"][f"{args.archs[-1]}/finetune/PERMUTED{s}"] = {"auc_vs_permuted_labels": round(auc(p, yp), 3),
                                                                 "positives": int(yp.sum())}
        print(f"  permuted labels #{s}: AUC {auc(p, yp):.3f} (should sit near 0.5)")
        save(results)


def run_q2(args, results):
    rows = index_photos()
    X = load_all(rows, args.size, "q1")
    pairs = pairs_from(rows)
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    res = results.setdefault("q2", {"n": len(y), "positives": int(y.sum()), "null_auc_sd": round(null_auc_sd(y), 3),
                                    "runs": {}})
    res["pairs"] = [f"{rows[i]['name']} -> {rows[j]['name']}" for i, j, _, _ in pairs]
    print(f"q2: {len(y)} pairs, {y.sum()} claim pairs; chance AUC = 0.5 ± {null_auc_sd(y):.3f} (1 SD)")

    def report(tag, p, extra=None):
        d = summary(p, y, g)
        if extra:
            d.update(extra)
        res["runs"][tag] = d
        res.setdefault("oof", {})[tag] = [round(float(v), 4) for v in p]
        print(f"  {tag:38s} AUC {d['auc']:.3f} {d['auc_95ci_by_car']}  recall@30% {d['recall_at_flag']['top30%']:.2f}  "
              f"@50% {d['recall_at_flag']['top50%']:.2f}")
        save(results)

    fl = folds(y, g, SEED, args.folds)
    for arch in args.archs:
        F = frozen_features(X, arch)
        Fa, Fb = F[[i for i, *_ in pairs]], F[[j for _, j, *_ in pairs]]
        na, nb = Fa / np.linalg.norm(Fa, axis=1, keepdims=True), Fb / np.linalg.norm(Fb, axis=1, keepdims=True)
        report(f"{arch}/frozen 1-cos (no training)", 1 - (na * nb).sum(1))
        report(f"{arch}/frozen+LR", lr_oof(np.concatenate([np.abs(Fb - Fa), Fa * Fb], 1), y, fl))
        for s in args.seeds:
            fls = folds(y, g, s, args.folds)
            t0 = time.time()
            p = np.zeros(len(y))
            for k, (tr, te) in enumerate(fls[: args.max_folds]):
                p[te] = ft_pairs(X, pairs, y, tr, te, arch, args.size, args.epochs_p, max(4, args.bs // 2), seed=s + k)
            report(f"{arch}/finetune-siamese/cv{s}", p, {"seconds": round(time.time() - t0)})
    for s in range(args.perms):
        yp = permuted_by_group(y, g, 2000 + s)
        fl_p = folds(yp, g, SEED, args.folds)
        p = np.zeros(len(y))
        for k, (tr, te) in enumerate(fl_p[: args.max_folds]):
            p[te] = ft_pairs(X, pairs, yp, tr, te, args.archs[-1], args.size, args.epochs_p, max(4, args.bs // 2),
                             seed=SEED + k)
        res["runs"][f"{args.archs[-1]}/finetune-siamese/PERMUTED{s}"] = {"auc_vs_permuted_labels": round(auc(p, yp), 3),
                                                                         "positives": int(yp.sum())}
        print(f"  permuted labels #{s}: AUC {auc(p, yp):.3f} (should sit near 0.5)")
        save(results)


def run_view(args, results):
    rows = index_pic500()
    X = load_all(rows, args.size_v, "pic500")
    y = np.array([r["label"] for r in rows])
    g = np.array([r["set"] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    res = results.setdefault("view", {"n": len(y), "classes": VIEW_NAMES,
                                      "class_counts": np.bincount(y, minlength=6).tolist(), "runs": {}})
    print(f"view: {len(y)} photos from {len(set(g.tolist()))} sets, classes {dict(zip(VIEW_NAMES, np.bincount(y).tolist()))}")
    coarse = np.array([0, 0, 1, 1, 2, 3])            # 車頭 / 車尾 / 前車內 / 後車內 (左右不分)
    side = np.array([0, 1, 0, 1, -1, -1])            # 左 / 右 for the four corners

    def report(tag, P):
        pred = P.argmax(1)
        ext = y < 4
        cm = np.zeros((6, 6), int)
        np.add.at(cm, (y, pred), 1)
        d = {"acc_vs_filename_label": round(float((pred == y).mean()), 3),
             "acc_coarse_front_rear_interior": round(float((coarse[pred] == coarse[y]).mean()), 3),
             "acc_left_right_on_corner_photos": round(float((side[pred[ext]] == side[y[ext]]).mean()), 3),
             "acc_取車": round(float((pred[phase == "取車"] == y[phase == "取車"]).mean()), 3),
             "acc_還車": round(float((pred[phase == "還車"] == y[phase == "還車"]).mean()), 3),
             "recall_per_class": {VIEW_NAMES[c]: round(float(cm[c, c] / max(1, cm[c].sum())), 3) for c in range(6)},
             "confusion_rows_label_cols_pred": cm.tolist()}
        conf = P.max(1)
        dis = np.flatnonzero((pred != y) & (conf >= 0.9))
        dis = dis[np.argsort(-conf[dis])]
        d["confident_disagreements"] = int(len(dis))
        d["examples"] = [{"photo": rows[i]["name"], "label": VIEW_NAMES[y[i]], "pred": VIEW_NAMES[pred[i]],
                          "p": round(float(conf[i]), 3)} for i in dis[:60]]
        res["runs"][tag] = d
        print(f"  {tag:30s} acc {d['acc_vs_filename_label']:.3f}  coarse {d['acc_coarse_front_rear_interior']:.3f}  "
              f"left/right {d['acc_left_right_on_corner_photos']:.3f}  confident disagreements {len(dis)}")
        save(results)
        return dis, pred, conf

    fl = folds(y, g, SEED, args.folds)
    arch = args.archs[-1]
    F = frozen_features(X, arch)
    report(f"{arch}/frozen+LR", lr_oof(F, y, fl, C=0.1))
    t0 = time.time()
    P = np.zeros((len(y), 6))
    for k, (tr, te) in enumerate(fl[: args.max_folds]):
        P[te] = ft_view(X, y, tr, te, arch, args.size_v, args.epochs_v, args.bs * 2, seed=SEED + k)
    dis, pred, conf = report(f"{arch}/finetune", P)
    res["runs"][f"{arch}/finetune"]["seconds"] = round(time.time() - t0)
    save(results)
    contact_sheet_view(rows, dis[:32], y, pred, conf)


def contact_sheet_view(rows, idx, y, pred, conf):
    """For the team to eyeball: the photos where the model most confidently disagrees with the upload label."""
    if len(idx) == 0:
        return
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    tw, th, cols = 150, 267, 8
    nrow = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (tw * cols, (th + 22) * nrow + 30), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 14)
    d.text((6, 6), "視角分類：模型高信心但和上傳分類不同（標籤→模型）", fill=(255, 220, 120), font=font)
    for n, i in enumerate(idx):
        im = ImageOps.fit(ImageOps.exif_transpose(Image.open(rows[i]["path"])).convert("RGB"), (tw, th))
        x, yy = (n % cols) * tw, 30 + (n // cols) * (th + 22)
        sheet.paste(im, (x, yy + 22))
        d.text((x + 3, yy + 3), f"{VIEW_NAMES[y[i]]}→{VIEW_NAMES[pred[i]]} {conf[i]:.2f}", fill=(255, 160, 120), font=font)
    sheet.save(OUT / "ft_view_disagree.jpg", quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="q1,q2,v")
    ap.add_argument("--archs", default="resnet50,efficientnet_b0,convnext_tiny")
    ap.add_argument("--seeds", default=str(SEED), help="CV seeds for the fine-tune runs (comma separated)")
    ap.add_argument("--perms", type=int, default=2, help="label-permutation control runs per task (last arch)")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--size-v", type=int, default=320)
    ap.add_argument("--epochs-q", type=int, default=12)
    ap.add_argument("--epochs-p", type=int, default=15)
    ap.add_argument("--epochs-v", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=99)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 fold, 1 epoch, 1 arch, no permutations")
    args = ap.parse_args()
    args.archs = args.archs.split(",")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        args.archs, args.max_folds, args.perms = args.archs[-1:], 1, 0
        args.epochs_q = args.epochs_p = args.epochs_v = 1
    import torch
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    assert (DATA / "進行索賠資料").is_dir(), f"set IRENT_DATA (now {DATA!s})"
    OUT.mkdir(parents=True, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "args": vars(args)}
    t0 = time.time()
    tasks = args.tasks.split(",")
    if "q1" in tasks:
        run_q1(args, results)
    if "q2" in tasks:
        run_q2(args, results)
    if "v" in tasks:
        assert PIC500.is_dir(), f"set IRENT_PIC500 (now {PIC500!s})"
        run_view(args, results)
    results["total_seconds"] = round(time.time() - t0)
    save(results)
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'ft_results.json'}")
    os.kill(os.getpid(), 15)        # handover: Windows Python can hang in ExitProcess after torch/PIL threads


if __name__ == "__main__":
    main()
