"""Cheap score variants from cached VLM answers (no GPU): does removing order-symmetric or per-photo bias help?

  P1            new damage in B (return) vs A (pickup)
  P1 - P1S      direction contrast: "new in B vs A" minus "new in A vs B" (cancels changes that look the same both
                ways, e.g. lighting); a model that only detects "some difference" scores ~0 here
  P1 - P6       contextual calibration: subtract the model's answer for (B, B), i.e. its bias on this return photo
  ens           mean of P1-P3 (pre-declared primary)
Reported on all 141 pairs and on the format-matched subset (18 v 28), with paired-bootstrap CIs over cars.

Run: python experiments/score_variants.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compare_pairs import boot_idx, ci  # noqa: E402
from finetune_classifier import OUT, auc, index_photos, pairs_from  # noqa: E402
from metadata_baseline import meta  # noqa: E402


def load(model_q):
    d = {}
    for line in (OUT / "vlm_option_probs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            m, q, pid, mode, size, name = r["key"].split("|")
            if f"{m}|{q}|{mode}|{size}" == model_q:
                d.setdefault(pid, {})[name] = r["logit"]
    return d


def main():
    rows = index_photos()
    pairs = pairs_from(rows)
    y = np.array([c for *_, c in pairs])
    g = np.array([pl for _, _, pl, _ in pairs])
    names = [f"{rows[i]['name']}->{rows[j]['name']}" for i, j, *_ in pairs]
    M = {k: meta(rows[k]["path"]) for pr in pairs for k in pr[:2]}
    same = np.array([bool(M[i][5] == 0 and M[j][5] == 0 and M[i][:2] == [720, 1280] and M[j][:2] == [720, 1280])
                     for i, j, *_ in pairs])
    out = {}
    models = sorted({"|".join(json.loads(l)["key"].split("|")[:2] + json.loads(l)["key"].split("|")[3:5])
                     for l in (OUT / "vlm_option_probs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()})
    for mq in models:
        d = load(mq)
        if not all(p in d and all(n in d[p] for n in names) for p in ("P1", "P2", "P3", "P1S", "P6")):
            continue
        L = {p: np.array([d[p][n] for n in names]) for p in d if all(n in d[p] for n in names)}
        S = {"P1": L["P1"], "ens P1-P3": (L["P1"] + L["P2"] + L["P3"]) / 3, "P1 - P1S": L["P1"] - L["P1S"],
             "P1 - P6": L["P1"] - L["P6"], "ens - P1S": (L["P1"] + L["P2"] + L["P3"]) / 3 - L["P1S"],
             "P1 - P1S - P6": L["P1"] - L["P1S"] - L["P6"]}
        if "P4" in L and "P5" in L:
            S["P4 - P5"] = L["P4"] - L["P5"]
        res = {}
        for sn, m in (("all", np.ones(len(y), bool)), ("same", same)):
            b = boot_idx(g[m], n=1000)
            yy = y[m]
            base = S["ens P1-P3"][m]
            for k, s in S.items():
                ss = s[m]
                a = auc(ss, yy)
                dd = [auc(ss[i], yy[i]) - auc(base[i], yy[i]) for i in b if len(set(yy[i].tolist())) > 1]
                res.setdefault(k, {})[sn] = {"auc": round(a, 3), "minus_ens": round(a - auc(base, yy), 3),
                                             "diff_ci": ci(dd)}
        out[mq] = res
        print(mq)
        for k, v in res.items():
            print(f"  {k:16s} all {v['all']['auc']:.3f} ({v['all']['minus_ens']:+.3f} {v['all']['diff_ci']})"
                  f"   same {v['same']['auc']:.3f} ({v['same']['minus_ens']:+.3f} {v['same']['diff_ci']})")
    (OUT / "score_variants.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
