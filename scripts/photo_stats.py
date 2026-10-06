"""Brightness / blur / orientation statistics over all photos of the official 500-set pack,
using the demo's own quality thresholds (config.yaml). Writes experiments/out/pic500_stats.json.

    set IRENT_PIC500=C:\\path\\to\\Hims_Pic500
    .venv\\Scripts\\python.exe scripts\\photo_stats.py
"""
import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import CFG  # noqa: E402
from app.vision import quality  # noqa: E402

PIC500 = pathlib.Path(os.environ.get("IRENT_PIC500", ""))
OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for s in sorted(PIC500.iterdir()):
    for p in sorted(s.iterdir()):
        try:
            img = quality.load_image(p.read_bytes())
        except Exception as e:  # noqa: BLE001
            rows.append({"set": s.name, "file": p.name, "error": str(e)})
            continue
        phase, code = p.stem.split("_")
        kind = "interior" if int(code) in (10, 11) else "exterior"
        q = quality.assess(img, CFG, kind)
        rows.append({"set": s.name, "file": p.name, "phase": phase, "type": int(code), "w": img.width, "h": img.height,
                     "blur": q["blur"], "brightness": q["brightness"], "issues": [i["code"] for i in q["issues"]]})
(OUT / "pic500_stats.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

ok = [r for r in rows if "error" not in r]
th = CFG["thresholds"]


def grp(name, rs):
    n = len(rs)
    if not n:
        return
    dark = sum(r["brightness"] < th["brightness_min"] for r in rs)
    blur = sum("blur" in r["issues"] for r in rs)
    any_issue = sum(bool([c for c in r["issues"] if c not in ("landscape", "portrait")]) for r in rs)
    print(f"{name:14s} n={n:5d}  太暗 {dark / n:5.1%}  太糊 {blur / n:5.1%}  任一問題 {any_issue / n:5.1%}")


grp("全部", ok)
grp("取車 車外1-4", [r for r in ok if r["phase"] == "取車" and r["type"] <= 4])
grp("還車 車外1-4", [r for r in ok if r["phase"] == "還車" and r["type"] <= 4])
grp("還車 車內10", [r for r in ok if r["type"] == 10])
grp("還車 車內11", [r for r in ok if r["type"] == 11])
sets = collections.defaultdict(set)
for r in ok:
    sets[r["set"]].add((r["phase"], r["type"]))
print("sets with extra pickup damage photos:", sum(any(p == "取車" and t >= 5 for p, t in v) for v in sets.values()))
print("sets with extra return damage photos:", sum(any(p == "還車" and 5 <= t <= 9 for p, t in v) for v in sets.values()))
