"""Weak-label evaluation sets E2 (500-set) and E3 (upload list), and scoring any method on them.

Labels come from customer close-ups (ImageType 5-9 / file _05.._09), which README calls "other": a close-up only
counts as damage if Qwen3.5-4B, asked about that single photo, says Yes (key XDMG in the shared cache). The labeller
is the same for every method, so the comparison between methods is fair, but it is the same model family as some
methods; treat E2/E3 as a consistency check, E1 (141 labelled pairs) stays the main number.

  E2 unit = one 500-set rental (4 corner pairs 取車_0k vs 還車_0k)
      1 = damage close-up at return and none at pickup        0 = no damage close-up at either
  E3 unit = two consecutive orders of the same car on the upload day (corner pairs: order n-1 return vs order n return)
      1 = order n has a damage close-up and order n-1 has none 0 = neither has one
A unit's score is the max over its corner pairs. Negative units are subsampled (fixed seed) to cap scoring cost.

Steps:
  python experiments/weak_eval.py xdmg      score the close-ups that are not scored yet (GPU, Qwen3.5-4B)
  python experiments/weak_eval.py build     write e2_pairs.json / e3_pairs.json + unit labels
  (score them: vlm_option_probs.py --pool json --pairs-json <file> --prompts P1 --model ...)
  python experiments/weak_eval.py eval      AUC per method on E2 / E3
"""
import json
import pathlib
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compare_pairs import auc, boot_idx, ci  # noqa: E402
from finetune_classifier import OUT  # noqa: E402
from vlm_option_probs import CACHE_FILE, PIC500, VLM, load_cache, load_img  # noqa: E402
from xlsx_pairs import DEST  # noqa: E402

LABELLER = ("Qwen/Qwen3.5-4B", "none", 1024)
Q_INTRO = "You are checking a photo uploaded by a car-sharing customer."
Q = ("Is this a photo of visible body damage on a car, such as a scratch, scuff, dent, crack, broken light, damaged "
     "wheel rim or mirror? Answer with one word: Yes or No.")
EXTRA = re.compile(r"^(取車|還車)_(0[5-9])$")
THR = 0.5
N_NEG_E2, N_NEG_E3 = 150, 200
N_POOL_XLSX = 2500


def xkey(name):
    m, q, s = LABELLER
    return f"{m}|{q}|XDMG|direct|{s}|{name}"


def closeups():
    out = []
    for d in sorted(p for p in PIC500.iterdir() if p.is_dir()):
        for p in sorted(d.iterdir()):
            m = EXTRA.match(p.stem)
            if m:
                out.append({"name": f"pic500x/{d.name}/{p.stem}", "path": str(p), "unit": d.name, "phase": m.group(1)})
    f = DEST / "xlsx_other_photos.json"
    if f.exists():
        for o in json.loads(f.read_text(encoding="utf-8")):
            out.append({"name": f"xlsxx/{o['order']}/{o['type']}_{o['ts']}", "path": o["path"], "unit": o["order"],
                        "phase": "xlsx"})
    return out


def cmd_xdmg():
    done = load_cache()
    todo = [c for c in closeups() if xkey(c["name"]) not in done]
    print(f"{len(todo)} close-ups to score")
    if not todo:
        return
    vlm = VLM(LABELLER[0], LABELLER[1])
    t0 = time.time()
    with CACHE_FILE.open("a", encoding="utf-8") as fh:
        for n, c in enumerate(todo):
            img = load_img(c["path"], LABELLER[2])
            msgs = [{"role": "user", "content": [{"type": "text", "text": Q_INTRO}, {"type": "image", "image": img},
                                                 {"type": "text", "text": Q}]}]
            r = vlm.score(msgs, [img])
            r.update({"key": xkey(c["name"]), "y": -1, "sec": round(time.time() - t0, 3)})
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                fh.flush()
                print(f"  {n + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def cmd_build():
    done = load_cache()
    dmg = defaultdict(set)                       # unit -> phases with a damage close-up
    for c in closeups():
        r = done.get(xkey(c["name"]))
        if r is None:
            raise SystemExit("run `weak_eval.py xdmg` first")
        if r["p_yes"] >= THR:
            dmg[c["unit"]].add(c["phase"])
    rng = random.Random(20261007)
    # E2
    sets = sorted(p.name for p in PIC500.iterdir() if p.is_dir())
    pos2 = [s for s in sets if dmg.get(s) == {"還車"}]
    neg2 = [s for s in sets if not dmg.get(s)]
    neg2 = sorted(rng.sample(neg2, min(N_NEG_E2, len(neg2))))
    e2 = []
    for s, lab in [(s, 1) for s in pos2] + [(s, 0) for s in neg2]:
        for k in (1, 2, 3, 4):
            a, b = PIC500 / s / f"取車_0{k}.png", PIC500 / s / f"還車_0{k}.png"
            if a.exists() and b.exists():
                e2.append({"a": str(a), "b": str(b), "name": f"pic500/{s}/{k}", "plate": s, "y": lab, "unit": s})
    # E3
    pairs = json.loads((DEST / "xlsx_pairs.json").read_text(encoding="utf-8"))
    units = defaultdict(list)
    for p in pairs:
        car, oo, k = p["name"].split("/")[1:]
        units[(car, oo)].append(p)
    pos3, neg3 = [], []
    for (car, oo), ps in units.items():
        o1, o2 = oo.split("->")
        d1, d2 = bool(dmg.get(o1)), bool(dmg.get(o2))
        if d2 and not d1:
            pos3.append((car, oo))
        elif not d1 and not d2:
            neg3.append((car, oo))
    neg3 = sorted(rng.sample(neg3, min(N_NEG_E3, len(neg3))))
    e3 = []
    for u, lab in [(u, 1) for u in pos3] + [(u, 0) for u in neg3]:
        for p in units[u]:
            e3.append(dict(p, y=lab, unit=f"{u[0]}/{u[1]}"))
    (DEST / "e2_pairs.json").write_text(json.dumps(e2, ensure_ascii=False), encoding="utf-8")
    (DEST / "e3_pairs.json").write_text(json.dumps(e3, ensure_ascii=False), encoding="utf-8")
    # distillation pool: every unlabelled pair whose unit is NOT in E2/E3 (so student scores on E2/E3 stay clean)
    e2u, e3u = {p["unit"] for p in e2}, {p["unit"] for p in e3}
    pool = []
    for s in sets:
        if s in e2u:
            continue
        for k in (1, 2, 3, 4):
            a, b = PIC500 / s / f"取車_0{k}.png", PIC500 / s / f"還車_0{k}.png"
            if a.exists() and b.exists():
                pool.append({"a": str(a), "b": str(b), "name": f"pic500/{s}/{k}", "plate": s, "y": -1})
    xl = [dict(p, y=-1) for u, ps in units.items() if f"{u[0]}/{u[1]}" not in e3u for p in ps]
    rng.shuffle(xl)
    pool += xl[:N_POOL_XLSX]
    # 21 of the 58 cars in the official folders (50 of them in E1 pairs) also show up (other days, other photos) in the upload list: keep them out of training
    from finetune_classifier import index_photos
    e1 = {r["plate"] for r in index_photos()}
    pool = [p for p in pool if p["plate"].replace("-", "") not in e1]
    (DEST / "train_pool.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    info = {"E2": {"pos_units": len(pos2), "neg_units": len(neg2), "pairs": len(e2)},
            "E3": {"pos_units": len(pos3), "neg_units_sampled": len(neg3), "pairs": len(e3)},
            "train_pool_pairs": len(pool),
            "closeups_called_damage": sum(1 for c in closeups() if done[xkey(c["name"])]["p_yes"] >= THR),
            "closeups": len(closeups())}
    print(json.dumps(info, indent=1))
    (OUT / "weak_eval_sets.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def unit_scores(done, file, method):
    """method = (model, quant, mode, size, prompts) -> unit score = max over corners of the mean logit."""
    m, q, mode, size, prompts = method
    ps = json.loads((DEST / file).read_text(encoding="utf-8"))
    by = defaultdict(list)
    lab, grp = {}, {}
    for p in ps:
        ks = [f"{m}|{q}|{pr}|{mode}|{size}|{p['name']}" for pr in prompts]
        if not all(k in done for k in ks):
            return None
        by[p["unit"]].append(np.mean([done[k]["logit"] for k in ks]))
        lab[p["unit"]], grp[p["unit"]] = p["y"], p["plate"]
    u = sorted(by)
    return np.array([max(by[x]) for x in u]), np.array([lab[x] for x in u]), np.array([grp[x] for x in u])


def cmd_eval():
    done = load_cache()
    methods = set()
    for k in done:
        m, q, pr, mode, size, name = k.split("|")
        if pr != "XDMG":
            methods.add((m, q, mode, size))
    out = {}
    for f, tag in (("e2_pairs.json", "E2"), ("e3_pairs.json", "E3")):
        if not (DEST / f).exists():
            continue
        for (m, q, mode, size) in sorted(methods):
            for prompts in (("P1",), ("P1", "P2", "P3")):
                r = unit_scores(done, f, (m, q, mode, size, prompts))
                if r is None:
                    continue
                s, y, g = r
                b = boot_idx(g, n=1000)
                key = f"{m}|{q}|{mode}|{size}|{'+'.join(prompts)}"
                out.setdefault(key, {})[tag] = {"auc": round(auc(s, y), 3), "n": f"{int(y.sum())}v{int((1 - y).sum())}",
                                                "ci": ci([auc(s[i], y[i]) for i in b if len(set(y[i].tolist())) > 1])}
    for k, v in out.items():
        print(f"{k:60s} " + "  ".join(f"{t} {x['auc']:.3f} {x['ci']} ({x['n']})" for t, x in v.items()))
    (OUT / "weak_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"xdmg": cmd_xdmg, "build": cmd_build, "eval": cmd_eval}[sys.argv[1]]()
