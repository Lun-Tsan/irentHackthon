"""Weak labels from the 500 unlabelled sets: the extra photos (_05.._09) customers took at pickup or at return.

README only calls ImageType 5-9 "other", so first ask the VLM whether each extra photo actually shows car body damage
(single photo, P(Yes) from one forward pass). Then group the sets:
  ret_only   damage close-up at return, none at pickup   -> candidate NEW damage in this rental
  pick_only  damage close-up at pickup only               -> damage that was already there (old)
  both       close-ups at both
  none       no damage close-up at all
and compare the corner-pair new-damage scores (vlm_option_probs.py --pool pic500, 4B, P1-P3 mean) across groups,
taking each set's maximum over its four corners. If ret_only sets score higher than none, the pair scorer agrees
with what customers documented; the AUC between the two groups is a weak, independent check on 500 real rentals.

Run: python experiments/pic500_weak_labels.py --model Qwen/Qwen3.5-4B   (needs IRENT_PIC500, GPU)
Outputs experiments/out/pic500_weak_labels.json (set ids only, no images).
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finetune_classifier import OUT, auc  # noqa: E402
from vlm_option_probs import CACHE_FILE, PIC500, PRIMARY, VLM, load_cache, load_img  # noqa: E402

EXTRA = re.compile(r"^(取車|還車)_(0[5-9])$")
Q_INTRO = "You are checking a photo uploaded by a car-sharing customer."
Q = ("Is this a photo of visible body damage on a car, such as a scratch, scuff, dent, crack, broken light, damaged "
     "wheel rim or mirror? Answer with one word: Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--thr", type=float, default=0.5, help="P(Yes) above which an extra photo counts as damage")
    args = ap.parse_args()
    sets = sorted(d for d in PIC500.iterdir() if d.is_dir())
    extras = []
    for d in sets:
        for p in sorted(d.iterdir()):
            m = EXTRA.match(p.stem)
            if m:
                extras.append({"set": d.name, "phase": m.group(1), "path": str(p), "name": f"pic500x/{d.name}/{p.stem}"})
    print(f"{len(extras)} extra photos in {len({e['set'] for e in extras})} sets")

    done = load_cache()
    vlm = None
    t0 = time.time()
    with CACHE_FILE.open("a", encoding="utf-8") as fh:
        for n, e in enumerate(extras):
            key = f"{args.model}|{args.quant}|XDMG|direct|{args.max_side}|{e['name']}"
            if key not in done:
                if vlm is None:
                    vlm = VLM(args.model, args.quant)
                img = load_img(e["path"], args.max_side)
                msgs = [{"role": "user", "content": [{"type": "text", "text": Q_INTRO}, {"type": "image", "image": img},
                                                     {"type": "text", "text": Q}]}]
                r = vlm.score(msgs, [img])
                r.update({"key": key, "y": -1, "sec": round(time.time() - t0, 3)})
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done[key] = r
            e["p_damage"] = done[key]["p_yes"]
            if n % 100 == 0:
                print(f"  {n + 1}/{len(extras)}  {time.time() - t0:.0f}s", flush=True)

    # set-level groups from the extra photos the VLM calls damage
    dmg = {}
    for e in extras:
        if e["p_damage"] >= args.thr:
            dmg.setdefault(e["set"], set()).add(e["phase"])
    group = {}
    for d in sets:
        ph = dmg.get(d.name, set())
        group[d.name] = ("both" if ph == {"取車", "還車"} else "ret_only" if ph == {"還車"} else
                         "pick_only" if ph == {"取車"} else "none")
    # corner-pair new-damage scores, max over the four corners
    score = {}
    for d in sets:
        vals = []
        for k in (1, 2, 3, 4):
            ks = [f"{args.model}|{args.quant}|{p}|direct|{args.max_side}|pic500/{d.name}/{k}" for p in PRIMARY]
            if all(x in done for x in ks):
                vals.append(np.mean([done[x]["logit"] for x in ks]))
        if len(vals) == 4:
            score[d.name] = max(vals)
    out = {"extra_photos": len(extras),
           "extra_photos_called_damage": int(sum(e["p_damage"] >= args.thr for e in extras)),
           "by_phase_called_damage": {ph: f"{sum(e['p_damage'] >= args.thr for e in extras if e['phase'] == ph)}/"
                                           f"{sum(e['phase'] == ph for e in extras)}" for ph in ("取車", "還車")},
           "groups": {g: sum(v == g for v in group.values()) for g in ("ret_only", "pick_only", "both", "none")},
           "sets_with_pair_scores": len(score)}
    s = {g: np.array([score[k] for k in score if group[k] == g]) for g in ("ret_only", "pick_only", "both", "none")}
    out["max_corner_logit_median"] = {g: (round(float(np.median(v)), 2) if len(v) else None) for g, v in s.items()}
    out["share_of_sets_with_any_corner_logit>0"] = {g: (round(float((v > 0).mean()), 3) if len(v) else None)
                                                    for g, v in s.items()}

    def a(pos, neg):
        if len(s[pos]) and len(s[neg]):
            yy = np.r_[np.ones(len(s[pos])), np.zeros(len(s[neg]))]
            return round(auc(np.r_[s[pos], s[neg]], yy), 3)
        return None
    out["AUC ret_only vs none (weak check: new damage documented at return)"] = a("ret_only", "none")
    out["AUC ret_only vs pick_only (new vs old documented damage)"] = a("ret_only", "pick_only")
    out["AUC pick_only vs none (does it fire on old damage? lower is better)"] = a("pick_only", "none")
    out["candidate_new_damage_sets_for_team_review"] = sorted(k for k, g in group.items() if g == "ret_only")
    print(json.dumps({k: v for k, v in out.items() if k != "candidate_new_damage_sets_for_team_review"},
                     ensure_ascii=False, indent=1))
    (OUT / "pic500_weak_labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.kill(os.getpid(), 15)


if __name__ == "__main__":
    main()
