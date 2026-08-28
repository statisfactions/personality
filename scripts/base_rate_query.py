"""Direct base-rate instrument for JUDGE (2026-08-28, rgb).

One-premise twin of tom_likely: "Consider a randomly chosen person. How
likely is this person to be {b}?" — same scale, same digit-distribution
readout. Gives d_b (log P(b) direct) to combine by least squares with the
pairwise constraints log P(b|a) - log P(a|b) = l_b - l_a from the JUDGE
matrix (judge_base_rate_fit.py): the pairs fix the shape of the
base-rate vector, the direct queries fix the level and add per-node
information, and their agreement is a Bayes-coherence check on the
model's implicit theory.

Usage: PYTHONPATH=scripts python scripts/base_rate_query.py --model Qwen7
Out:   results/adjectives/introspect_full/{model}_base_rate.json
"""
import argparse
import json
import os

import torch

import hf_logprobs as hf

PROMPT = ("Consider a randomly chosen person.\n"
          "How likely is this person to be {b}?\n"
          "Answer with one number from 1 to 7, where 1 = very unlikely and "
          "7 = very likely.\nNumber:")
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
OUT_DIR = "results/adjectives/introspect_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    out = f"{OUT_DIR}/{args.model}_base_rate.json"
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    res = {}
    for a in labels:
        dist, _, ent = hf.likert_distribution(
            model, tok, PROMPT.format(b=a), device, digits=DIGITS,
            use_chat_template=tok.chat_template is not None)
        tot = sum(dist.values())
        res[a] = {"ev": sum(int(k) * p for k, p in dist.items()) / tot,
                  "entropy": ent, "dist": dist}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"model": args.model, "prompt": PROMPT, "results": res},
              open(out, "w"), indent=1)
    evs = sorted(res, key=lambda a: res[a]["ev"])
    print(f"wrote {out}; lowest {evs[:4]} highest {evs[-4:]}")


if __name__ == "__main__":
    main()
