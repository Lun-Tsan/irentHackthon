"""Jev-like student: a small Qwen3.5 VLM + LoRA that outputs P(new damage) from ONE forward pass, no text.

Jev's principles, applied to the 借→還 pair:
  * typed answer, no generation: the decision is the Yes/No token at a fixed position,
    p = sigmoid(logit_Yes - logit_No) (log-sum-exp over the Yes / No spellings), read with logits_to_keep=1
  * calibrated: trained with a proper scoring rule (soft-target BCE) against the teacher's probability
  * fast: a 0.8B / 2B model, one forward, nothing generated
Training data: unlabelled pairs only (500-set + consecutive upload-list orders; E2/E3 units are excluded by
weak_eval.py build), soft labels = teacher Qwen3.5-4B P1 logits (zero-shot). The 141 labelled pairs (E1) are never
used for training, so E1 stays a clean test; E2/E3 likewise.

Run: python experiments/jev_student.py --student Qwen/Qwen3.5-0.8B --tag s08-v1
Scores go into vlm_option_probs.jsonl as model "student/<tag>" (prompt P1), so compare_pairs.py / weak_eval.py
pick them up; the adapter is saved outside the repo in ~/irent_models/<tag>.
"""
import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finetune_classifier import OUT, auc, index_photos, pairs_from  # noqa: E402
from vlm_option_probs import CACHE_FILE, VLM, build_messages, load_cache, load_img  # noqa: E402
from xlsx_pairs import DEST  # noqa: E402

MODELS = pathlib.Path(os.environ.get("IRENT_MODELS", str(pathlib.Path.home() / "irent_models")))
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"


def official_pairs():
    rows = index_photos()
    return [{"a": rows[i]["path"], "b": rows[j]["path"], "name": f"{rows[i]['name']}->{rows[j]['name']}",
             "plate": pl, "y": c} for i, j, pl, c in pairs_from(rows)]


class Student:
    def __init__(self, base, max_side, lora_r, adapter=None):
        import torch
        import transformers.integrations.sdpa_attention as sdpa
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
        sdpa.use_gqa_in_sdpa = lambda *a, **k: False
        self.max_side = max_side
        self.proc = AutoProcessor.from_pretrained(base)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
        if adapter:
            self.model = PeftModel.from_pretrained(model, adapter)
        else:
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS,
                             bias="none")
            self.model = get_peft_model(model, cfg)
            for p in self.model.parameters():          # LoRA weights in fp32, base stays bf16 (autocast in forward)
                if p.requires_grad:
                    p.data = p.data.float()
        tok = self.proc.tokenizer
        self.yes = VLM._ids(tok, ["Yes", " Yes", "yes", " yes", "YES"])
        self.no = VLM._ids(tok, ["No", " No", "no", " no", "NO"])

    def inputs(self, pair, slot="P1"):
        ia, ib = load_img(pair["a"], self.max_side), load_img(pair["b"], self.max_side)
        messages, images = build_messages(slot, ia, ib)
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        return self.proc(text=[text], images=images, return_tensors="pt")

    def decision_logit(self, inputs):
        import torch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**{k: v.to("cuda") for k, v in inputs.items()}, logits_to_keep=1)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        return torch.logsumexp(lp[self.yes], 0) - torch.logsumexp(lp[self.no], 0)


def teacher_logits(pairs, teacher, size):
    done = load_cache()
    keep, t = [], []
    for p in pairs:
        k = f"{teacher}|none|P1|direct|{size}|{p['name']}"
        if k in done:
            keep.append(p)
            t.append(done[k]["logit"])
    return keep, np.array(t)


def score_into_cache(st, tag, pairs, label, slot="P1"):
    """Write student decision logits into the shared cache (model 'student/<tag>'; slot P1, or the P1S / P6 controls)."""
    import torch
    done = load_cache()
    st.model.eval()
    t0, n = time.time(), 0
    with CACHE_FILE.open("a", encoding="utf-8") as fh, torch.no_grad():
        for p in pairs:
            key = f"student/{tag}|none|{slot}|direct|{st.max_side}|{p['name']}"
            if key in done:
                continue
            t1 = time.time()
            d = float(st.decision_logit(st.inputs(p, slot)))
            fh.write(json.dumps({"key": key, "logit": d, "p_yes": 1 / (1 + math.exp(-d)), "mass": float("nan"),
                                 "top": "", "tokens": 0, "reasoning_tokens": 0, "y": int(p.get("y", -1)),
                                 "sec": round(time.time() - t1, 3)}) + "\n")
            n += 1
    print(f"  scored {n} {label} pairs in {time.time() - t0:.0f}s ({(time.time() - t0) / max(1, n):.3f} s/pair)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="s08-v1")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--teacher-temp", type=float, default=1.0, help="soft target = sigmoid(teacher_logit / T)")
    ap.add_argument("--teacher-size", type=int, default=1024, help="max_side the teacher logits were computed at")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--slots", default="P1", help="E1 scoring slots, e.g. P1,P1S,P6 (controls)")
    ap.add_argument("--skip-weak", action="store_true", help="do not score E2/E3")
    args = ap.parse_args()
    import torch
    torch.manual_seed(0)
    random.seed(0)
    adapter_dir = MODELS / args.tag
    pool = json.loads((DEST / "train_pool.json").read_text(encoding="utf-8"))
    pool, t_logit = teacher_logits(pool, args.teacher, args.teacher_size)
    if args.max_train:
        pool, t_logit = pool[: args.max_train], t_logit[: args.max_train]
    print(f"train pool {len(pool)} pairs with teacher logits; teacher P(yes)>0.5 on {np.mean(t_logit > 0):.1%}")

    if args.eval_only:
        st = Student(args.student, args.max_side, args.lora_r, adapter=str(adapter_dir))
    else:
        st = Student(args.student, args.max_side, args.lora_r)
        st.model.print_trainable_parameters()
        st.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        st.model.enable_input_require_grads()
        params = [p for p in st.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        steps = int(len(pool) * args.epochs)
        n_upd = max(1, steps // args.accum)
        warm = max(1, n_upd // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_upd - warm))))
        target = torch.tensor(1 / (1 + np.exp(-t_logit / args.teacher_temp)), dtype=torch.float32)
        st.model.train()
        order = []
        while len(order) < steps:
            order += random.sample(range(len(pool)), len(pool))
        order = order[:steps]
        t0, run = time.time(), []
        for s, k in enumerate(order):
            d = st.decision_logit(st.inputs(pool[k]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(d, target[k].cuda()) / args.accum
            loss.backward()
            run.append(float(loss) * args.accum)
            if (s + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (s + 1) % 200 == 0:
                print(f"  step {s + 1}/{steps}  loss {np.mean(run[-200:]):.4f}  {time.time() - t0:.0f}s  "
                      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        st.model.save_pretrained(str(adapter_dir))
        print(f"trained in {time.time() - t0:.0f}s -> {adapter_dir}")

    off = official_pairs()
    for slot in args.slots.split(","):
        score_into_cache(st, args.tag, off, f"E1 {slot}", slot)
    done = load_cache()
    y = np.array([p["y"] for p in off])
    s = np.array([done[f"student/{args.tag}|none|P1|direct|{args.max_side}|{p['name']}"]["logit"] for p in off])
    print(f"E1 AUC student/{args.tag}: {auc(s, y):.3f}")
    for f in (("e2_pairs.json", "e3_pairs.json") if not args.skip_weak else ()):
        if (DEST / f).exists():
            score_into_cache(st, args.tag, json.loads((DEST / f).read_text(encoding="utf-8")), f)
    os.kill(os.getpid(), 15)


if __name__ == "__main__":
    main()
