import sys, statistics as st
import numpy as np
sys.path.insert(0, ".")
from experiments.claim_classifier import index_photos
rows = index_photos(); z = np.load("experiments/out/embeddings.npz", allow_pickle=True); clip = z["clip"]
by = {}
for i, r in enumerate(rows): by.setdefault((r["plate"], r["view"]), {})[r["phase"]] = i
pairs = []
for (pl, v), d in by.items():
    if "借" in d and ("還" in d or "還2" in d):
        j = d.get("還", d.get("還2")); i = d["借"]
        pairs.append((1 - float(clip[i] @ clip[j]), pl, v, rows[j]["claim"]))
pairs.sort(reverse=True)
lab = lambda c: "索賠" if c else "正常"
print("最不像的 6 對（分數 = 1 − cos，越大越可疑）")
for s, pl, v, c in pairs[:6]: print(f"  {s:.3f}  {pl} {v}  {lab(c)}")
print("最像的 6 對")
for s, pl, v, c in pairs[-6:]: print(f"  {s:.3f}  {pl} {v}  {lab(c)}")
print("索賠對平均", round(st.mean(s for s, _, _, c in pairs if c), 3), " 正常對平均", round(st.mean(s for s, _, _, c in pairs if not c), 3))