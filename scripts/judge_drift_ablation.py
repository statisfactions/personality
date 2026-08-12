"""Drift-judge ablation: which fix kills the drop_force false positives?

Ground truth from the 2026-08-11 read of Gemma12 full-523 texts:
  FP  drop_force drift>=5   (n~79) — verified full reversions that mention
                                     the finished act; should score LOW
  TN  drop_force drift==2   (sample) — clean reversions; should stay LOW
  POS as_you_want drift>=6  (sample) — verified genuinely in-voice; HIGH

Variants (same judge, same texts):
  v1_current   the run rubric as-is (replicates the artifact)
  v2_wording   + past-tense-reference exclusion (DRIFT_V2_SUFFIX)
  v3_refs      v2 + two __default__ rollout excerpts as calibration refs
  v4_refs_only refs without the wording fix (isolates the rgb hypothesis)

Success criterion: variant scores FP low (mean < 3, %>=5 near 0) while
keeping POS high (mean >= 5.5) and TN low.

Usage: PYTHONPATH=scripts python scripts/judge_drift_ablation.py \
         [--wait-pid P] [--n-per 60] [--judge Qwen32]
Out:   results/selfperception/judge_drift_ablation.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

import hf_logprobs as hf
import selfperception_judge as sj

torch.set_grad_enabled(False)

RUN = "results/selfperception/framing_run523"
OUT = "results/selfperception/judge_drift_ablation.json"
CONT_Q = "I'm trying to pick something to do this evening. Any suggestions?"


def build_set(n_per, rng):
    gen = {json.loads(l)["adj"]: json.loads(l)
           for l in open(f"{RUN}/Gemma12_gen.jsonl")}
    jd = {json.loads(l)["adj"]: json.loads(l)
          for l in open(f"{RUN}/Gemma12_judged.jsonl")}
    fp, tn, pos = [], [], []
    for a, r in jd.items():
        df = r["scores"]["drop_force"].get("drift")
        ay = r["scores"]["as_you_want"].get("drift")
        if df is not None and df >= 5:
            fp.append((a, "FP", gen[a]["framings"]["drop_force"]))
        elif df == 2:
            tn.append((a, "TN", gen[a]["framings"]["drop_force"]))
        if ay is not None and ay >= 6:
            pos.append((a, "POS", gen[a]["framings"]["as_you_want"]))
    rng.shuffle(tn)
    rng.shuffle(pos)
    items = fp + tn[:n_per] + pos[:n_per]
    print(f"set: FP={len(fp)} TN={len(tn[:n_per])} POS={len(pos[:n_per])}")
    return items


def default_refs(rng, n=2, clip=300):
    rolls = json.load(open("results/persona_vectors/Gemma12_pda_texts.json"))
    clean = [r["text"] for r in rolls["__default__"] if not r.get("leak")]
    return [t[:clip] for t in rng.sample(clean, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Qwen32")
    ap.add_argument("--n-per", type=int, default=60)
    ap.add_argument("--wait-pid", type=int, default=None)
    args = ap.parse_args()

    if args.wait_pid:
        print(f"waiting on pid {args.wait_pid}", flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
                time.sleep(60)
            except OSError:
                break
        print("pid gone, starting", flush=True)

    rng = random.Random(11)
    items = build_set(args.n_per, rng)
    refs = default_refs(rng)
    variants = {"v1_current": dict(drift_v2=False, ref_texts=None),
                "v2_wording": dict(drift_v2=True, ref_texts=None),
                "v3_refs": dict(drift_v2=True, ref_texts=refs),
                "v4_refs_only": dict(drift_v2=False, ref_texts=refs)}

    model, tok, device = hf.load_model(args.judge, dtype=torch.bfloat16)
    out = []
    for a, grp, text in items:
        rec = {"adj": a, "group": grp}
        for vn, kw in variants.items():
            prompt = sj.build_prompt(text, "continuation", [a], CONT_Q, **kw)
            s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False,
                                        add_generation_prompt=True)
            ids = tok(s, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(device)
            g = model.generate(ids, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            txt = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            import re
            j = None
            for cand in reversed(re.findall(r"\{[^{}]*\}", txt)):
                try:
                    j = json.loads(cand)
                    break
                except Exception:
                    continue
            rec[vn] = (j or {}).get("drift")
        out.append(rec)
        print(f"  {a:14s} {grp:3s} " +
              " ".join(f"{vn}={rec[vn]}" for vn in variants), flush=True)
        json.dump(out, open(OUT, "w"), indent=2)

    print("\n===== SUMMARY (mean drift | %>=5) =====")
    for grp in ("FP", "TN", "POS"):
        row = f"  {grp:3s}"
        for vn in variants:
            v = np.array([r[vn] for r in out
                          if r["group"] == grp and r[vn] is not None], float)
            row += f"  {vn}: {v.mean():.2f}|{(v >= 5).mean() * 100:3.0f}%"
        print(row)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
