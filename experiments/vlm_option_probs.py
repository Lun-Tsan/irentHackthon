"""Prompted local VLM as a new-damage classifier, scored the Jev way: read P(answer token), do not parse generated text.

For each 借→還 pair (same 141 pairs, same labels as v1 / finetune_classifier.py) the model sees both photos and a
yes/no question; the score is  P(Yes) / (P(Yes) + P(No))  at the answer position, from ONE forward pass.
Optional --think: let the model reason first (Qwen3.5 thinking), then read P(Yes) right after </think>.

Prompts are fixed up front (P1–P3 pair prompts, P4 = return photo only, a control for "is it really comparing?").
Choosing the best prompt on these same 141 pairs would be optimistic, so the pre-declared primary score is the
mean logit of P1–P3; every prompt is still reported.

Run (needs IRENT_DATA; model from the HF cache, nothing is sent anywhere):
  python experiments/vlm_option_probs.py --model Qwen/Qwen3.5-4B --limit 10        smoke test, 5 claim + 5 normal
  python experiments/vlm_option_probs.py --model Qwen/Qwen3.5-9B --quant 8bit
Results append to experiments/out/vlm_option_probs.jsonl (resumable, keyed by model/quant/prompt/mode/size/pair).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from finetune_classifier import OUT, auc, auc_ci, index_photos, null_auc_sd, pairs_from, recall_at  # noqa: E402

CACHE_FILE = OUT / "vlm_option_probs.jsonl"
PIC500 = pathlib.Path(os.environ.get("IRENT_PIC500", ""))
ANSWER_PREFIX = {"zai-org/GLM-4.6V-Flash": "<|begin_of_box|>"}


def pic500_pairs():
    """Unlabelled pool: for each of the 500 sets, 取車_0k vs 還車_0k for the four corners (k = 1..4).
    The ImageType in the file name is what the customer picked, often wrong (README), so some pairs are not the same
    corner; that is what production traffic looks like too."""
    out = []
    for d in sorted(PIC500.iterdir()):
        for k in (1, 2, 3, 4):
            a, b = d / f"取車_0{k}.png", d / f"還車_0{k}.png"
            if a.exists() and b.exists():
                out.append({"a": str(a), "b": str(b), "name": f"pic500/{d.name}/{k}", "plate": d.name, "y": -1})
    return out


def ens_logits(done, model_id, quant, mode, size, names, prompts=None):
    prompts = prompts or PRIMARY
    ks = [[f"{model_id}|{quant}|{p}|{mode}|{size}|{n}" for n in names] for p in prompts]
    if not all(k in done for kk in ks for k in kk):
        return None
    return np.mean([[done[k]["logit"] for k in kk] for kk in ks], 0)


def flag_rates(done, model_id, quant, mode, size, pool_names, off_names, y_off):
    """Pick T1 on the 21 labelled claim pairs for a target recall, then see what share of the unlabelled 500-set pairs
    (real traffic) and of the 120 labelled normal pairs would be sent to a human at that T1."""
    e_off = ens_logits(done, model_id, quant, mode, size, off_names)
    e_pool = ens_logits(done, model_id, quant, mode, size, pool_names)
    if e_off is None or e_pool is None:
        return {}
    pos = np.sort(e_off[y_off == 1])
    out = {"pool_pairs": len(e_pool)}
    for target in (1.0, 0.95, 0.9):
        k = int(math.floor((1 - target) * len(pos)))      # how many claim pairs we allow to slip under T1
        t = float(pos[k])
        out[f"recall>={target:.0%} on the 21 (T1 logit {t:.2f})"] = {
            "caught": f"{int((e_off[y_off == 1] >= t).sum())}/{len(pos)}",
            "share_of_500set_pairs_to_human": round(float((e_pool >= t).mean()), 3),
            "share_of_120_normal_pairs_to_human": round(float((e_off[y_off == 0] >= t).mean()), 3)}
    return out
INTRO_EN = ("You are inspecting a car-sharing rental car. Both photos show the same corner of the same car. "
            "Photo A was taken by the customer at pickup, Photo B by the same customer at return, often at a different "
            "place and time, so lighting, background, angle and distance can differ.")
PROMPTS = {
    "P1": ("en", INTRO_EN, "Did the car get NEW body damage during this rental, that is, a scratch, scuff, dent, crack, "
           "broken light, damaged wheel rim or mirror that is visible in Photo B but not in Photo A? Damage already "
           "visible in Photo A is old and does not count. Differences caused by lighting, reflections, dirt, water "
           "drops, shadows, camera angle or distance do not count. Answer with one word: Yes or No."),
    "P2": ("zh", "你在檢查一台共享汽車。兩張照片拍的是同一台車的同一個角落。照片 A 是顧客取車時拍的，照片 B 是同一位顧客還車時拍的，"
           "地點和時間常常不同，所以光線、背景、角度和遠近都可能不一樣。",
           "這次租車期間，車身有沒有出現「新的」損傷？也就是照片 B 看得到、照片 A 看不到的刮痕、擦痕、凹陷、裂痕、燈具破損、"
           "輪框或後照鏡損傷。照片 A 已經有的是舊傷，不算。光線、反光、髒汙、水滴、陰影、拍攝角度或距離造成的差異也不算。"
           "只用一個英文字回答：Yes 或 No。"),
    "P3": ("en", INTRO_EN, "Compare the two photos part by part: bumper, fender, door, wheel and rim, lights, mirror, "
           "body edges. For each part visible in both photos, check whether Photo B shows a mark or deformation that is "
           "absent in Photo A. Ignore anything already present in Photo A and anything explained by lighting, "
           "reflections, dirt or viewpoint. Is there at least one new damage in Photo B? Answer with one word: Yes or No."),
    "P4": ("en", "You are inspecting a car-sharing rental car. The photo shows one corner of the car, taken at return.",
           "Does this photo show any body damage, such as a scratch, scuff, dent, crack, broken light, damaged wheel "
           "rim or mirror? Ignore dirt, water drops, reflections and shadows. Answer with one word: Yes or No."),
    # P5 = P4 on the pickup photo: does the model see damage that was already there? P4 - P5 is a "difference" score.
    "P5": ("en", "You are inspecting a car-sharing rental car. The photo shows one corner of the car, taken at pickup.",
           "Does this photo show any body damage, such as a scratch, scuff, dent, crack, broken light, damaged wheel "
           "rim or mirror? Ignore dirt, water drops, reflections and shadows. Answer with one word: Yes or No."),
}
PRIMARY = ("P1", "P2", "P3")


def load_img(path, max_side):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    return im


TILES = {"P1T2": [(0.0, 0.0, 1.0, 0.6), (0.0, 0.4, 1.0, 1.0)]}   # top / bottom 60% of the frame, 20% overlap


def tile(img, box, max_side):
    """Crop a fractional box and resize so its long side is max_side (upscaling allowed: this is the zoom)."""
    from PIL import Image
    w, h = img.size
    c = img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
    s = max_side / max(c.size)
    return c.resize((round(c.width * s), round(c.height * s)), Image.BICUBIC)


def build_messages(pid, img_a, img_b):
    # controls that reuse P1's wording: P1S = photos swapped (a model that really compares should now say No for the
    # claim pairs, because the damage sits in the "pickup" photo); P6 = the return photo twice (nothing changed -> No)
    if pid == "P1S":
        pid, img_a, img_b = "P1", img_b, img_a
    elif pid == "P6":
        pid, img_a = "P1", img_b
    lang, intro, q = PROMPTS[pid]
    la, lb = ("照片 A（取車）：", "照片 B（還車）：") if lang == "zh" else ("Photo A (pickup):", "Photo B (return):")
    if pid in ("P4", "P5"):
        img = img_b if pid == "P4" else img_a
        content = [{"type": "text", "text": intro}, {"type": "image", "image": img}, {"type": "text", "text": q}]
        images = [img]
    else:
        content = [{"type": "text", "text": intro}, {"type": "text", "text": la}, {"type": "image", "image": img_a},
                   {"type": "text", "text": lb}, {"type": "image", "image": img_b}, {"type": "text", "text": q}]
        images = [img_a, img_b]
    return [{"role": "user", "content": content}], images


class VLM:
    def __init__(self, model_id, quant):
        import torch
        import transformers.integrations.sdpa_attention as sdpa
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        sdpa.use_gqa_in_sdpa = lambda *a, **k: False      # Windows torch has no flash-attn; GQA flag forces the math kernel
        kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)
        self.proc = AutoProcessor.from_pretrained(model_id)
        ip = getattr(self.proc, "image_processor", None)
        if ip is not None and hasattr(ip, "max_soft_tokens"):
            # Gemma 4 defaults to 280 image tokens; its tech report shows fine detail needs the 1120 budget
            ip.max_soft_tokens = int(os.environ.get("IRENT_IMAGE_TOKENS", 1120))
            if hasattr(self.proc, "image_seq_length"):
                self.proc.image_seq_length = ip.max_soft_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        self._guarded = False
        # GLM-4.6V puts its final answer inside <|begin_of_box|>...; open the box so the next token is the answer
        self.answer_prefix = ANSWER_PREFIX.get(model_id, "")
        tok = self.proc.tokenizer
        self.yes = self._ids(tok, ["Yes", " Yes", "yes", " yes", "YES"])
        self.no = self._ids(tok, ["No", " No", "no", " no", "NO"])
        print(f"loaded {model_id} ({quant}); yes ids {self.yes} no ids {self.no}; "
              f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    @staticmethod
    def _bf16(inputs):
        import torch
        return {k: (v.to(torch.bfloat16) if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
                for k, v in inputs.items()}

    @staticmethod
    def _ids(tok, words):
        ids = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        return ids

    def _template(self, messages, think):
        try:
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=think)
        except TypeError:
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + ("" if think else self.answer_prefix)

    def _answer_probs(self, text, images):
        import torch
        inputs = self.proc(text=[text], images=images, return_tensors="pt").to("cuda")
        if self._guarded:
            inputs = self._bf16(inputs)
        with torch.no_grad():
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except RuntimeError as e:          # Gemma 4 + bnb: fp32 pixel_values meet a bf16 LayerNorm; Qwen never gets here
                if self._guarded or "expected scalar type" not in str(e):
                    raise
                self._guarded = True
                inputs = self._bf16(inputs)
                out = self.model(**inputs, logits_to_keep=1)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        ly = torch.logsumexp(lp[self.yes], 0).item()
        ln = torch.logsumexp(lp[self.no], 0).item()
        top = int(lp.argmax())
        return ly, ln, int(inputs["input_ids"].shape[1]), self.proc.tokenizer.decode([top])

    def score(self, messages, images, think=False, think_tokens=1024):
        """Returns dict with p_yes (normalised over Yes/No), the Yes+No probability mass and the top token."""
        import torch
        text = self._template(messages, think)
        reasoning_len = 0
        if think:
            inputs = self.proc(text=[text], images=images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=think_tokens, do_sample=False)
            new = self.proc.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            reasoning_len = int(gen.shape[1] - inputs["input_ids"].shape[1])
            cut = new.find("</think>")
            new = new[: cut + len("</think>")] if cut >= 0 else new + "\n</think>"
            text = text + new + "\n\n"
        ly, ln, n_tok, top = self._answer_probs(text, images)
        return {"p_yes": 1 / (1 + math.exp(ln - ly)), "logit": ly - ln, "mass": math.exp(ly) + math.exp(ln),
                "top": top, "tokens": n_tok, "reasoning_tokens": reasoning_len}


def load_cache():
    done = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:        # a line cut short by an interrupted writer: recompute it
                    continue
                done[r["key"]] = r
    return done


def evaluate(done, pairs, y, g, model_id, quant, mode, size):
    """AUC per prompt and for the pre-declared ensemble, vs the trained-classifier baselines in ft_results.json."""
    names = [p["name"] for p in pairs]
    res = {}
    logits = {}
    for pid in list(PROMPTS) + ["P1S", "P6"] + list(TILES):
        keys = [f"{model_id}|{quant}|{pid}|{mode}|{size}|{n}" for n in names]
        if all(k in done for k in keys):
            lg = np.array([done[k]["logit"] for k in keys])
            logits[pid] = lg
            p = 1 / (1 + np.exp(-lg))
            res[pid] = {"auc": round(auc(p, y), 3), "auc_95ci_by_car": auc_ci(p, y, g), "recall_at_flag": recall_at(p, y),
                        "mean_yes_no_mass": round(float(np.mean([done[k]["mass"] for k in keys])), 3),
                        "share_answered_yes": round(float((p > 0.5).mean()), 3),
                        "sec_per_pair": round(float(np.mean([done[k]["sec"] for k in keys])), 2)}
    if "P4" in logits and "P5" in logits:
        pdiff = 1 / (1 + np.exp(-(logits["P4"] - logits["P5"])))
        res["P4-P5 (damage at return minus damage at pickup, single photos)"] = {
            "auc": round(auc(pdiff, y), 3), "auc_95ci_by_car": auc_ci(pdiff, y, g), "recall_at_flag": recall_at(pdiff, y)}
        pa = 1 / (1 + np.exp(-logits["P5"]))
        res["P5 on claim pairs: share of pickup photos the model calls damaged"] = round(float((pa[y == 1] > 0.5).mean()), 3)
        res["P5 on normal pairs: share of pickup photos the model calls damaged"] = round(float((pa[y == 0] > 0.5).mean()), 3)
    if all(p in logits for p in PRIMARY):
        ens = np.mean([logits[p] for p in PRIMARY], 0)
        pe = 1 / (1 + np.exp(-ens))
        res["ENSEMBLE_P1-P3 (primary)"] = {"auc": round(auc(pe, y), 3), "auc_95ci_by_car": auc_ci(pe, y, g),
                                           "recall_at_flag": recall_at(pe, y)}
        res.update(compare_to_baselines(pe, y, g))
    return res


def compare_to_baselines(p_vlm, y, g, n_boot=2000):
    """Bootstrap by car: AUC(VLM ensemble) - AUC(baseline), with the baselines' out-of-fold scores from ft_results.json."""
    f = OUT / "ft_results.json"
    if not f.exists():
        return {}
    q2 = json.loads(f.read_text(encoding="utf-8")).get("q2", {})
    out = {}
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    members = {k: np.flatnonzero(g == k) for k in ug}
    boots = [np.concatenate([members[k] for k in rng.choice(ug, len(ug))]) for _ in range(n_boot)]
    for tag, s in q2.get("oof", {}).items():
        if "PERMUTED" in tag or len(s) != len(y):
            continue
        s = np.array(s)
        d = [auc(p_vlm[b], y[b]) - auc(s[b], y[b]) for b in boots if len(set(y[b].tolist())) > 1]
        out[f"vs {tag}"] = {"baseline_auc": round(auc(s, y), 3), "delta_auc": round(auc(p_vlm, y) - auc(s, y), 3),
                            "delta_95ci_by_car": [round(float(np.percentile(d, 2.5)), 3),
                                                  round(float(np.percentile(d, 97.5)), 3)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--prompts", default="P1,P2,P3,P4,P5,P1S,P6")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--think-tokens", type=int, default=1024)
    ap.add_argument("--max-side", type=int, default=1024, help="longest image side in px before the processor")
    ap.add_argument("--limit", type=int, default=0, help="balanced subset: limit/2 claim + limit/2 normal pairs")
    ap.add_argument("--pool", default="official", choices=["official", "pic500", "json"],
                    help="official = the 141 labelled pairs; pic500 = ~2,000 unlabelled corner pairs (needs IRENT_PIC500); "
                         "json = unlabelled pairs from --pairs-json (e.g. xlsx_pairs.py output)")
    ap.add_argument("--pairs-json", default="", help="list of {a, b, name, plate, y} for --pool json")
    args = ap.parse_args()
    import torch
    rows = index_photos()
    pairs_all = pairs_from(rows)
    pairs = [{"a": rows[i]["path"], "b": rows[j]["path"], "name": f"{rows[i]['name']}->{rows[j]['name']}",
              "plate": pl, "y": c} for i, j, pl, c in pairs_all]
    official = pairs
    if args.pool == "pic500":
        pairs = pic500_pairs()
    elif args.pool == "json":
        pairs = json.loads(pathlib.Path(args.pairs_json).read_text(encoding="utf-8"))
    y = np.array([p["y"] for p in pairs])
    g = np.array([p["plate"] for p in pairs])
    order = list(range(len(pairs)))
    if args.limit:
        pos = [k for k in order if y[k] == 1][: args.limit // 2]
        neg = [k for k in order if y[k] == 0][: args.limit - len(pos)]
        order = [k for ab in zip(pos, neg) for k in ab] + pos[len(neg):] + neg[len(pos):]
    mode = f"think{args.think_tokens}" if args.think else "direct"
    done = load_cache()
    vlm = None
    prompts = args.prompts.split(",")
    t_start = time.time()
    with CACHE_FILE.open("a", encoding="utf-8") as fh:
        for n, k in enumerate(order):
            for pid in prompts:
                key = f"{args.model}|{args.quant}|{pid}|{mode}|{args.max_side}|{pairs[k]['name']}"
                if key in done:
                    continue
                if vlm is None:
                    vlm = VLM(args.model, args.quant)
                t0 = time.time()
                if pid in TILES:
                    # zoom: same fractional tile from both photos, P1 wording, keep the most suspicious tile
                    fa, fb = load_img(pairs[k]["a"], 4096), load_img(pairs[k]["b"], 4096)
                    parts = []
                    for box in TILES[pid]:
                        messages, images = build_messages("P1", tile(fa, box, args.max_side), tile(fb, box, args.max_side))
                        parts.append(vlm.score(messages, images))
                    r = dict(max(parts, key=lambda x: x["logit"]))
                    r["tile_logits"] = [round(x["logit"], 4) for x in parts]
                else:
                    ia, ib = load_img(pairs[k]["a"], args.max_side), load_img(pairs[k]["b"], args.max_side)
                    messages, images = build_messages(pid, ia, ib)
                    r = vlm.score(messages, images, think=args.think, think_tokens=args.think_tokens)
                r.update({"key": key, "y": int(y[k]), "sec": round(time.time() - t0, 3)})
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done[key] = r
            if n % 10 == 0:
                print(f"  {n + 1}/{len(order)} pairs  {time.time() - t_start:.0f}s  "
                      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if args.pool in ("pic500", "json"):
        fr = flag_rates(done, args.model, args.quant, mode, args.max_side, [p["name"] for p in pairs],
                        [p["name"] for p in official], np.array([p["y"] for p in official]))
        print(json.dumps(fr, ensure_ascii=False, indent=1))
        summ = OUT / "vlm_flag_rates.json"
        allres = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else {}
        allres[f"{args.pool}|{args.model}|{args.quant}|{mode}|{args.max_side}"] = fr
        summ.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    elif args.limit:
        sub = order
        for pid in prompts:
            ks = [f"{args.model}|{args.quant}|{pid}|{mode}|{args.max_side}|{pairs[k]['name']}" for k in sub]
            p = np.array([done[x]["p_yes"] for x in ks])
            print(f"{pid}: AUC on {len(sub)} pairs {auc(p, y[sub]):.3f}  mass {np.mean([done[x]['mass'] for x in ks]):.3f}  "
                  f"top tokens {sorted(set(done[x]['top'] for x in ks))}  sec/pair {np.mean([done[x]['sec'] for x in ks]):.2f}")
    else:
        res = evaluate(done, pairs, y, g, args.model, args.quant, mode, args.max_side)
        res = {"model": args.model, "quant": args.quant, "mode": mode, "max_side": args.max_side,
               "gpu": torch.cuda.get_device_name(0), "n": len(y), "positives": int(y.sum()),
               "null_auc_sd": round(null_auc_sd(y), 3), "results": res}
        print(json.dumps(res, ensure_ascii=False, indent=1))
        summ = OUT / "vlm_option_probs_summary.json"
        allres = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else {}
        allres[f"{args.model}|{args.quant}|{mode}|{args.max_side}"] = res
        summ.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    os.kill(os.getpid(), 15)


if __name__ == "__main__":
    main()
