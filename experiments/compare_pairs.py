"""One table for the 141 借→還 pairs: prompted VLMs vs trained classifiers vs shortcut baselines.

The no-claim folder mixes .jfif and .jpg exports and two resolutions, while every claim pair is .jpg, so file format
alone reaches AUC ~0.83 (metadata_baseline.py). Every method is therefore scored on three sets:
  all     141 pairs (21 claim)
  jpg     both photos .jpg                           (21 vs 40)
  same    both photos .jpg, 1280x720, EXIF-rotated  (18 vs 28)  <- format cannot help here
Trained models are scored out-of-fold as trained (on all pairs); "retrained-on-same" rows refit the v1 heads inside
the format-matched set only. AUC differences vs the VLM ensemble use a paired bootstrap over cars.

Run: python experiments/compare_pairs.py   (needs IRENT_DATA; reads experiments/out/*)
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finetune_classifier import OUT, SEED, folds, index_photos, lr_oof, pairs_from, recall_at  # noqa: E402


def auc(s, y):
    """Mann-Whitney AUC with average ranks for ties (same value as sklearn's roc_auc_score, ~50x faster)."""
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
from metadata_baseline import meta  # noqa: E402
from vlm_option_probs import PRIMARY  # noqa: E402


def boot_idx(g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    mem = {k: np.flatnonzero(g == k) for k in ug}
    return [np.concatenate([mem[k] for k in rng.choice(ug, len(ug))]) for _ in range(n)]


def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    rows = index_photos()
    pairs = pairs_from(rows)
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    names = [f"{rows[i]['name']}->{rows[j]['name']}" for i, j, *_ in pairs]
    M = {k: meta(rows[k]["path"]) for pr in pairs for k in pr[:2]}
    jpg = np.array([M[i][5] == 0 and M[j][5] == 0 for i, j, *_ in pairs])
    same = np.array([bool(jpg[n] and M[i][:2] == [720, 1280] and M[j][:2] == [720, 1280] and M[i][6] and M[j][6])
                     for n, (i, j, *_) in enumerate(pairs)])
    sets = {"all": np.ones(len(y), bool), "jpg": jpg, "same": same}

    S = {}                                   # method -> score per pair (higher = more likely new damage)
    # prompted VLMs
    vl = {}
    for line in (OUT / "vlm_option_probs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            m, q, pid, mode, size, name = r["key"].split("|")
            vl.setdefault((m, q, mode, size), {}).setdefault(pid, {})[name] = r["logit"]
    for (m, q, mode, size), d in vl.items():
        tag = f"VLM {m.split('/')[-1]}{'' if q == 'none' else ' ' + q}{'' if mode == 'direct' else ' ' + mode} @{size}"
        full = {pid: np.array([v[n] for n in names]) for pid, v in d.items() if all(n in v for n in names)}
        if all(p in full for p in PRIMARY):
            S[f"{tag} | ensemble P1-P3 (primary)"] = np.mean([full[p] for p in PRIMARY], 0)
        for pid in ("P1", "P4", "P1S", "P6", "P1T2"):
            if pid in full:
                S[f"{tag} | {pid}"] = full[pid]
    # trained classifiers (out-of-fold, trained on all 141)
    ft = OUT / "ft_results.json"
    if ft.exists():
        q2 = json.loads(ft.read_text(encoding="utf-8")).get("q2", {})
        for tag, s in q2.get("oof", {}).items():
            if len(s) == len(y):
                S[f"trained {tag}"] = np.array(s)
    # v1 encoders (cached CPU embeddings) and shortcut baselines
    z = np.load(OUT / "embeddings.npz", allow_pickle=True)
    pos = {n: k for k, n in enumerate(z["names"])}
    clip, dino = z["clip"], z["dino"]
    ia = [pos[rows[i]["name"]] for i, *_ in pairs]
    ib = [pos[rows[j]["name"]] for _, j, *_ in pairs]
    S["v1 CLIP 1-cos (no training)"] = 1 - (clip[ia] * clip[ib]).sum(1)
    Xd = np.concatenate([dino[ib] - dino[ia], np.abs(dino[ib] - dino[ia]), (clip[ia] * clip[ib]).sum(1, keepdims=True)], 1)
    S["v1 DINOv2-diff + LR"] = lr_oof(Xd, y, folds(y, g, SEED))
    Xm = np.stack([np.concatenate([np.array(M[i]), np.array(M[j]), np.abs(np.array(M[j]) - np.array(M[i]))])
                   for i, j, *_ in pairs])
    S["SHORTCUT metadata only + LR"] = lr_oof(Xm, y, folds(y, g, SEED), C=0.1)
    S["SHORTCUT is .jpg (1 = jpg)"] = jpg.astype(float)
    # v1 head refit inside the format-matched set only
    k = np.flatnonzero(same)
    s_re = np.full(len(y), np.nan)
    s_re[k] = lr_oof(Xd[k], y[k], folds(y[k], g[k], SEED))
    S["v1 DINOv2-diff + LR retrained-on-same"] = s_re

    prim = [m for m in S if m.endswith("(primary)")]
    ref = S[prim[0]] if prim else None
    boots = {sn: boot_idx(g[m]) for sn, m in sets.items()}
    table = {}
    for meth, s in S.items():
        row = {}
        for sn, m in sets.items():
            ok = m & ~np.isnan(s)
            if ok.sum() < len(y[m]) or len(set(y[ok].tolist())) < 2:
                continue
            ss, yy = s[m], y[m]
            b = boots[sn]
            a = auc(ss, yy)
            row[sn] = {"n": f"{int(yy.sum())}v{int((1 - yy).sum())}", "auc": round(a, 3),
                       "ci": ci([auc(ss[i], yy[i]) for i in b if len(set(yy[i].tolist())) > 1]),
                       "recall@30%": recall_at(ss, yy)["top30%"]}
            if ref is not None and meth != prim[0]:
                rr = ref[m]
                d = [auc(rr[i], yy[i]) - auc(ss[i], yy[i]) for i in b if len(set(yy[i].tolist())) > 1]
                row[sn]["vlm_minus_this"] = round(auc(rr, yy) - a, 3)
                row[sn]["diff_ci"] = ci(d)
        table[meth] = row
    print(f"{'method':62s} {'all (21v120)':>22s} {'jpg (21v40)':>22s} {'same (18v28)':>22s}")
    for meth, row in table.items():
        cells = []
        for sn in ("all", "jpg", "same"):
            c = row.get(sn)
            cells.append(f"{c['auc']:.3f} [{c['ci'][0]:.2f},{c['ci'][1]:.2f}]" if c else "-")
        print(f"{meth[:62]:62s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")
    if ref is not None:
        print(f"\nVLM primary ({prim[0]}) minus method, AUC difference [95% CI, paired bootstrap over cars], set 'same':")
        for meth, row in table.items():
            c = row.get("same", {})
            if "vlm_minus_this" in c:
                print(f"  {meth[:62]:62s} {c['vlm_minus_this']:+.3f} [{c['diff_ci'][0]:+.2f},{c['diff_ci'][1]:+.2f}]")
    (OUT / "compare_pairs.json").write_text(json.dumps({"sets": {k: int(v.sum()) for k, v in sets.items()},
                                                        "table": table}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
