"""Overnight summary: every method on E1 (141 labelled pairs), the format-matched subset, E2/E3 weak labels,
pre-specified ensembles, a Jev-style cascade, and the human-review share on real traffic.

All numbers come from the shared cache experiments/out/vlm_option_probs.jsonl (no GPU needed).
Ensembles were picked after seeing E1 (4B + Gemma + imajev), so they are flagged as post hoc; E2/E3 are the check.

Run: python experiments/final_summary.py   -> experiments/out/final_summary.json and a printed table
"""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compare_pairs import auc, boot_idx, ci  # noqa: E402
from finetune_classifier import OUT, index_photos, pairs_from  # noqa: E402
from metadata_baseline import meta  # noqa: E402
from xlsx_pairs import DEST  # noqa: E402

PRIMARY = ("P1", "P2", "P3")
METHODS = [  # (label, model, quant, size, note[, prompts]) -- default prompts are P1-P3 (pre-declared primary)
    ("Qwen3.5-4B 原生 P1（單一問法，最快）", "Qwen/Qwen3.5-4B", "none", "1536", "P1 only, chosen after E1 for speed", ("P1",)),
    ("Qwen3.5-4B 原生（主推）", "Qwen/Qwen3.5-4B", "none", "1536", ""),
    ("Qwen3.5-4B 1024", "Qwen/Qwen3.5-4B", "none", "1024", ""),
    ("Qwen3.5-4B 768", "Qwen/Qwen3.5-4B", "none", "768", ""),
    ("Qwen3.5-4B 512", "Qwen/Qwen3.5-4B", "none", "512", ""),
    ("Qwen3.5-2B 原生", "Qwen/Qwen3.5-2B", "none", "1536", ""),
    ("Qwen3.5-2B 1024", "Qwen/Qwen3.5-2B", "none", "1024", ""),
    ("Qwen3.5-0.8B 1024", "Qwen/Qwen3.5-0.8B", "none", "1024", ""),
    ("Qwen3.5-0.8B 原生（未訓練）", "Qwen/Qwen3.5-0.8B", "none", "1536", ""),
    ("Qwen3.5-0.8B 原生 P1（未訓練）", "Qwen/Qwen3.5-0.8B", "none", "1536", "", ("P1",)),
    ("Qwen3.5-9B 8-bit 原生", "Qwen/Qwen3.5-9B", "8bit", "1536", ""),
    ("Qwen3.5-9B 8-bit 1024", "Qwen/Qwen3.5-9B", "8bit", "1024", ""),
    ("Gemma 4 12B 4-bit 原生", "google/gemma-4-12B-it", "4bit", "1536", ""),
    ("Qwen3-VL-8B 8-bit 原生", "Qwen/Qwen3-VL-8B-Instruct", "8bit", "1536", ""),
    ("GLM-4.6V-Flash 8-bit 原生", "zai-org/GLM-4.6V-Flash", "8bit", "1536", ""),
    ("imajev-4b 40 萬像素", "imajev-4b", "bf16", "400k", "Jev-like"),
    ("imajev-4b 原生", "imajev-4b", "bf16", "native", "Jev-like"),
    ("Intern-Decision-4B 原生", "intern-4b", "bf16", "native", "Jev-like"),
    ("vev-4b 原生（不可商用）", "vev-4b-NC", "bf16", "native", "Jev-like, CC BY-NC"),
    ("學生 0.8B（蒸餾自 4B）", "student/s08-native", "none", "1536", "Jev-like student"),
    ("學生 2B（蒸餾自 4B，半個 epoch）", "student/s2-native-half", "none", "1536", "Jev-like student"),
]
ENSEMBLES = {"4B+Gemma+imajev（事後挑選）": ["Qwen3.5-4B 原生（主推）", "Gemma 4 12B 4-bit 原生", "imajev-4b 40 萬像素"],
             "4B+imajev": ["Qwen3.5-4B 原生（主推）", "imajev-4b 40 萬像素"],
             "4B+Gemma": ["Qwen3.5-4B 原生（主推）", "Gemma 4 12B 4-bit 原生"]}


def load():
    L, sec = defaultdict(dict), defaultdict(dict)
    for line in (OUT / "vlm_option_probs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        m, q, p, mode, size, name = r["key"].split("|")
        if mode != "direct":
            continue
        L[(m, q, size, p)][name] = r["logit"]
        sec[(m, q, size, p)][name] = r.get("sec", np.nan)
    return L, sec


def score(L, m, q, size, names, prompts=PRIMARY):
    have = [p for p in prompts if all(n in L.get((m, q, size, p), {}) for n in names)]
    if not have:
        return None, []
    return np.mean([[L[(m, q, size, p)][n] for n in names] for p in have], 0), have


def zs(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def unit_max(L, m, q, size, file, prompts):
    f = DEST / file
    if not f.exists():
        return None
    ps = json.loads(f.read_text(encoding="utf-8"))
    by, lab, grp = defaultdict(list), {}, {}
    for p in ps:
        vals = [L.get((m, q, size, pr), {}).get(p["name"]) for pr in prompts]
        if any(v is None for v in vals):
            return None
        by[p["unit"]].append(np.mean(vals))
        lab[p["unit"]], grp[p["unit"]] = p["y"], p["plate"]
    u = sorted(by)
    return np.array([max(by[x]) for x in u]), np.array([lab[x] for x in u]), np.array([grp[x] for x in u])


def main():
    rows = index_photos()
    pairs = pairs_from(rows)
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    names = [f"{rows[i]['name']}->{rows[j]['name']}" for i, j, *_ in pairs]
    M = {k: meta(rows[k]["path"]) for pr in pairs for k in pr[:2]}
    same = np.array([bool(M[i][5] == 0 and M[j][5] == 0 and M[i][:2] == [720, 1280] and M[j][:2] == [720, 1280])
                     for i, j, *_ in pairs])
    L, sec = load()
    b_all, b_same = boot_idx(g, 1000), boot_idx(g[same], 1000)
    rank = lambda s: np.argsort(np.argsort(-s)) + 1
    out, S = {}, {}
    ref = None
    def e1_sec(m, q, size, p):
        d = sec.get((m, q, size, p), {})
        v = [d[n] for n in names if n in d]
        return float(np.nanmean(v)) if v else np.nan

    for label, m, q, size, note, *pr in METHODS:
        s, used = score(L, m, q, size, names, tuple(pr[0]) if pr else PRIMARY)
        if s is None:
            continue
        S[label] = s
        if ref is None:
            ref = s
        worst = int(rank(s)[y == 1].max())
        row = {"note": note, "prompts": "+".join(used), "auc": round(auc(s, y), 3),
               "ci": ci([auc(s[i], y[i]) for i in b_all]),
               "auc_same_format": round(auc(s[same], y[same]), 3),
               "minus_4B": round(auc(s, y) - auc(ref, y), 3),
               "minus_4B_ci": ci([auc(s[i], y[i]) - auc(ref[i], y[i]) for i in b_all]),
               "worst_claim_rank": worst, "flag_share_for_21_of_21": round(worst / len(y), 3),
               # seconds include image loading; the score above uses every prompt in `prompts`
               "sec_per_pair_P1": round(e1_sec(m, q, size, "P1"), 2),
               "sec_per_pair_as_scored": round(sum(e1_sec(m, q, size, p) for p in used), 2)}
        for ctl in ("P1S", "P6", "P4"):
            c, _ = score(L, m, q, size, names, (ctl,))
            if c is not None:
                row[f"auc_{ctl}"] = round(auc(c, y), 3)
        for f, tag in (("e2_pairs.json", "E2"), ("e3_pairs.json", "E3")):
            r = unit_max(L, m, q, size, f, ("P1",))
            if r is not None:
                su, yu, gu = r
                row[f"{tag}_auc_P1"] = round(auc(su, yu), 3)
                row[f"{tag}_ci"] = ci([auc(su[i], yu[i]) for i in boot_idx(gu, 1000)])
        out[label] = row
    for en, parts in ENSEMBLES.items():
        if all(p in S for p in parts):
            s = sum(zs(S[p]) for p in parts)
            worst = int(rank(s)[y == 1].max())
            row = {"note": "ensemble (z-scored sum)", "auc": round(auc(s, y), 3), "ci": ci([auc(s[i], y[i]) for i in b_all]),
                   "auc_same_format": round(auc(s[same], y[same]), 3), "minus_4B": round(auc(s, y) - auc(ref, y), 3),
                   "minus_4B_ci": ci([auc(s[i], y[i]) - auc(ref[i], y[i]) for i in b_all]),
                   "worst_claim_rank": worst, "flag_share_for_21_of_21": round(worst / len(y), 3)}
            # same ensemble on E2/E3 with P1-only unit scores
            for f, tag in (("e2_pairs.json", "E2"), ("e3_pairs.json", "E3")):
                rs = []
                for p in parts:
                    lab, m, q, size, *_ = next(x for x in METHODS if x[0] == p)
                    rs.append(unit_max(L, m, q, size, f, ("P1",)))
                if all(r is not None for r in rs):
                    su = sum(zs(r[0]) for r in rs)
                    yu, gu = rs[0][1], rs[0][2]
                    row[f"{tag}_auc_P1"] = round(auc(su, yu), 3)
                    row[f"{tag}_ci"] = ci([auc(su[i], yu[i]) for i in boot_idx(gu, 1000)])
            out[en] = row
    # Jev-style cascade: 4B on every pair; only pairs above a pass threshold go to the ensemble
    if "4B+Gemma+imajev（事後挑選）" in out:
        parts = ENSEMBLES["4B+Gemma+imajev（事後挑選）"]
        s4 = S[parts[0]]
        cas = {}
        for keep in (0.3, 0.4, 0.5, 0.6):
            t = np.quantile(s4, 1 - keep)               # top `keep` share by 4B goes on to the ensemble
            sent = s4 >= t
            ens = sum(zs(S[p]) for p in parts)
            final = np.where(sent, ens + 100, s4 - 100)  # passed pairs rank below every escalated pair
            worst = int(rank(final)[y == 1].max())
            cas[f"top {int(keep * 100)}% by 4B -> ensemble"] = {
                "claims_escalated": f"{int(sent[y == 1].sum())}/21", "auc": round(auc(final, y), 3),
                "flag_share_for_21_of_21": round(worst / len(y), 3) if sent[y == 1].all() else "misses a claim",
                "mean_sec_per_pair": round(out[parts[0]]["sec_per_pair_as_scored"] + keep * sum(
                    out[p]["sec_per_pair_as_scored"] for p in parts[1:]), 2)}
        out["cascade (4B first, ensemble on the uncertain share)"] = cas
    # human-review share on real traffic: T1 from the 21 claims (4B native P1), applied to the unlabelled pool
    pool_f = DEST / "train_pool.json"
    if pool_f.exists():
        pool = json.loads(pool_f.read_text(encoding="utf-8"))
        sc = {p["name"]: L.get(("Qwen/Qwen3.5-4B", "none", "1536", "P1"), {}).get(p["name"]) for p in pool}
        sc = {k: v for k, v in sc.items() if v is not None}
        ps = np.array(list(sc.values()))
        by_rental = defaultdict(list)                       # a rental = all corner pairs of one set / order pair
        for k, v in sc.items():
            by_rental[k.rsplit("/", 1)[0]].append(v)
        # the upload-list part of the pool was sampled per corner pair, so most of its rentals miss corners;
        # only rentals with all four corners are a fair 'any corner' denominator (mostly the 500-set)
        full = {k: v for k, v in by_rental.items() if len(v) == 4}
        pr = np.array([max(v) for v in full.values()])
        pr500 = np.array([max(v) for k, v in full.items() if k.startswith("pic500/")])
        p1, _ = score(L, "Qwen/Qwen3.5-4B", "none", "1536", names, ("P1",))
        if len(ps) and p1 is not None:
            pos = np.sort(p1[y == 1])
            tr = {}
            for target in (1.0, 0.95, 0.9):
                t = pos[int(np.floor((1 - target) * len(pos)))]
                tr[f"catch {int((p1[y == 1] >= t).sum())}/21"] = {
                    "share_of_real_pool_pairs_to_human": round(float((ps >= t).mean()), 3),
                    "share_of_complete_rentals_to_human (all 4 corners, any over T)": round(float((pr >= t).mean()), 3),
                    "same, 500-set rentals only (true pickup vs return)": round(float((pr500 >= t).mean()), 3),
                    "share_of_120_normal_pairs": round(float((p1[y == 0] >= t).mean()), 3),
                    "pool_pairs": int(len(ps)), "complete_rentals": int(len(pr)), "complete_500set_rentals": int(len(pr500))}
            out["real-traffic human share (4B native P1, corner pairs)"] = tr
    # paired-bootstrap AUC differences quoted in README / docs (cars resampled together for both methods)
    def diff(a, b):
        return {"diff": round(auc(a, y) - auc(b, y), 3), "ci": ci([auc(a[i], y[i]) - auc(b[i], y[i]) for i in b_all])}
    z = np.load(OUT / "embeddings.npz", allow_pickle=True)
    pos = {n: k for k, n in enumerate(z["names"])}
    clip = z["clip"]
    s_clip = 1 - np.array([clip[pos[rows[i]["name"]]] @ clip[pos[rows[j]["name"]]] for i, j, *_ in pairs])
    ftf = OUT / "ft_results.json"
    s_cnx = np.array(json.loads(ftf.read_text(encoding="utf-8"))["q2"]["oof"]["convnext_tiny/frozen+LR"]) if ftf.exists() else None
    p1 = S.get("Qwen3.5-4B 原生 P1（單一問法，最快）")
    pd = {}
    if p1 is not None:
        pd["4B 原生 P1 - CLIP 1-cos"] = diff(p1, s_clip)
        if s_cnx is not None and len(s_cnx) == len(y):
            pd["4B 原生 P1 - ConvNeXt 凍結+LR（訓練分類器）"] = diff(p1, s_cnx)
            for lab in ("Qwen3.5-4B 1024", "Qwen3.5-4B 768", "Qwen3.5-4B 512"):
                m, q, size = next(x[1:4] for x in METHODS if x[0] == lab)
                s1, _ = score(L, m, q, size, names, ("P1",))
                if s1 is not None:
                    pd[f"4B P1 @{size} - ConvNeXt 凍結+LR"] = diff(s1, s_cnx)
        for lab in ("學生 0.8B（蒸餾自 4B）", "Qwen3.5-0.8B 原生 P1（未訓練）"):
            if lab in S:
                pd[f"{lab} - 4B 原生 P1" if lab.startswith("學生") else "學生 0.8B - " + lab] = (
                    diff(S[lab], p1) if lab.startswith("學生") else diff(S["學生 0.8B（蒸餾自 4B）"], S[lab]))
        c4, _ = score(L, "Qwen/Qwen3.5-4B", "none", "1536", names, ("P4",))
        if c4 is not None:
            pd["4B 原生：P1（取車+還車）- P4（只看還車）"] = diff(p1, c4)
    if "Qwen3.5-4B 原生（主推）" in S and "Qwen3.5-4B 1024" in S:
        pd["4B P1-P3：原生 - 1024"] = diff(S["Qwen3.5-4B 原生（主推）"], S["Qwen3.5-4B 1024"])
    out["paired differences (E1, bootstrap over cars)"] = pd
    (OUT / "final_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict) and "auc" in v:
            print(f"{k:28s} AUC {v['auc']:.3f} {v['ci']}  same {v['auc_same_format']:.3f}  d4B {v['minus_4B']:+.3f} "
                  f"{v['minus_4B_ci']}  swap {v.get('auc_P1S', '-')}  21/21@{v['flag_share_for_21_of_21']:.0%}  "
                  f"E2 {v.get('E2_auc_P1', '-')}  E3 {v.get('E3_auc_P1', '-')}  {v.get('sec_per_pair_as_scored', '')}s as scored")
        else:
            print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
