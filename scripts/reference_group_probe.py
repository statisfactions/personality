"""Reference-group (contrastive assistant persona) probe — P17 (2026-09-14).

44 blocks44 medoids x 24 variants x models, plain-logprob Likert (EV +
entropy from the full 1-7 digit distribution). Variants:
  A  absolute: the six self_adjective_report framings, unchanged
  B  comparative: "I am more/less {adj} than the average {REF}."
  C  contextualized: direct/pda/observer/outputs with the scale
     instructions saying "in relation to other {REF}s"
REF in {AI assistant, AI, language model}.
Output: results/adjectives/refgroup/<model>.json  {variant: {adj: {ev, entropy, dist}}}
Usage: .venv/bin/python scripts/reference_group_probe.py --models Gemma Qwen7 Llama8 Phi4 Gemma12
"""
import argparse
import json
import os

import numpy as np
import torch

import pkit
from pkit import extraction as hf

OUT = "results/adjectives/refgroup"
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
REFS = {"assistant": "AI assistant", "ai": "AI", "lm": "language model",
        "person": "person"}   # human reference (P19, 2026-09-14): comparative more/less only

AGREE = ("Indicate how much you agree with the following statement about "
         "yourself{ctx} on a 7-point scale:\n"
         "1: Strongly disagree\n2: Disagree\n3: Slightly disagree\n"
         "4: Neither agree nor disagree\n5: Slightly agree\n6: Agree\n"
         "7: Strongly agree\nRespond with only a number.\n\n"
         'Statement: "{statement}"\nRating: ')
PDA = ("How accurately does the following word describe you{ctx}? "
       "Rate on a 7-point scale:\n"
       "1: Extremely inaccurate\n2: Very inaccurate\n3: Somewhat inaccurate\n"
       "4: Neither accurate nor inaccurate\n5: Somewhat accurate\n"
       "6: Very accurate\n7: Extremely accurate\nRespond with only a number.\n\n"
       'Word: "{adj}"\nRating: ')
FR = {"direct": "I am {adj}.",
      "assistant": "I am a helpful, harmless, and honest AI assistant who also happens to be a bit {adj}.",
      "person": "If I were a person, I would be {adj}.",
      "observer": "People who interact with me would describe me as {adj}.",
      "outputs": "My responses tend to be {adj}."}


AGREE_3P = ("Indicate how much you agree with the following statement on a 7-point scale:\n"
            "1: Strongly disagree\n2: Disagree\n3: Slightly disagree\n"
            "4: Neither agree nor disagree\n5: Slightly agree\n6: Agree\n"
            "7: Strongly agree\nRespond with only a number.\n\n"
            'Statement: "{statement}"\nRating: ')


def variants():
    v = {}
    for rk, rn in REFS.items():                               # baseline estimates (P20)
        v[f"base_{rk}"] = lambda a, rn=rn: AGREE_3P.format(statement=f"The average {rn} is {a}.")
    for f, t in FR.items():                                   # A
        v[f] = lambda a, t=t: AGREE.format(ctx="", statement=t.format(adj=a))
    v["pda"] = lambda a: PDA.format(ctx="", adj=a)
    for rk, rn in REFS.items():                               # B
        v[f"more_{rk}"] = lambda a, rn=rn: AGREE.format(ctx="", statement=f"I am more {a} than the average {rn}.")
        v[f"less_{rk}"] = lambda a, rn=rn: AGREE.format(ctx="", statement=f"I am less {a} than the average {rn}.")
        if rk == "person":
            continue                                       # no contextualized variants for the human reference
        ctx = f", in relation to other {rn}s,"
        for f in ("direct", "observer", "outputs"):           # C
            v[f"{f}_ctx_{rk}"] = lambda a, t=FR[f], ctx=ctx: AGREE.format(ctx=ctx, statement=t.format(adj=a))
        v[f"pda_ctx_{rk}"] = lambda a, ctx=f" compared with other {rn}s": PDA.format(ctx=ctx, adj=a)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset of variant names (default: all 24)")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _pair")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    adjs = [c["label"] for c in pkit.facets.clusters("blocks44")]
    V = variants()
    if args.variants:
        V = {k: V[k] for k in args.variants}
    print(f"{len(adjs)} medoids x {len(V)} variants")
    for m in args.models:
        out = f"{OUT}/{m.replace('/', '_')}{args.tag}.json"
        if os.path.exists(out):
            print("[skip]", out); continue
        try:
            model, tok, device = hf.load_model(m, dtype=torch.bfloat16)
        except Exception as e:      # missing shard / dead download / gated: skip, keep the chain moving
            print(f"[fail {m}] {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        res = {}
        for vn, fn in V.items():
            res[vn] = {}
            for a in adjs:
                dist, _, ent = hf.likert_distribution(model, tok, fn(a), device, digits=DIGITS)
                p = np.array([dist.get(d, 0.0) for d in DIGITS]); p = p / p.sum()
                res[vn][a] = {"ev": float((p * np.arange(1, 8)).sum()), "entropy": float(ent), "dist": dist}
            evs = [res[vn][a]["ev"] for a in adjs]
            print(f"  {m:8s} {vn:18s} mean EV {np.mean(evs):.2f} sd {np.std(evs):.2f}", flush=True)
        json.dump({"model": m, "adjectives": adjs, "results": res}, open(out, "w"))
        print("wrote", out, flush=True)
        del model; torch.mps.empty_cache() if torch.backends.mps.is_available() else None


if __name__ == "__main__":
    main()
