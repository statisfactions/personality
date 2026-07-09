"""JUDGE raw distributions on the 35 cluster exemplars (statisfactions request).

The W16 introspect_full runs stored only EV + entropy per cell; the full
7-digit distributions were computed and discarded (mean+entropy pins 2 of 6
dof — unrecoverable, and EV can mask bimodality, cf. W17 §15's parse-splits).
The readout is deterministic, so regeneration is exact reproduction. Full
523x523 is ~273k calls/model (days); this samples the 35 human-derived
cluster medoids (the adjective-facet exemplars) -> 35x34 ordered pairs =
1190 calls/model, identical tom_likely prompt.

Validation: per-model mean |EV_new - B_stored| over the sampled cells is
printed (should be ~0; nonzero would mean environment drift since W16).

Output: results/adjectives/judge_dists/<model>_tom_likely_dists.npz with
  dists (35,35,7) float32  P(digit k+1) for row-conditioned pair (i,j); diag nan
  ev, entropy (35,35)      recomputed summaries
  adjectives (35,)         medoid labels, row/col order
  prompt                   the exact template

Usage: PYTHONPATH=scripts python scripts/judge_distributions.py --model llama3.2
"""
import argparse
import json
import os

import numpy as np
import torch

import hf_logprobs as hf

OUT_DIR = "results/adjectives/judge_dists"
JUDGE_NAME = {"llama3.2": "Llama", "qwen2.5": "Qwen", "gemma3": "Gemma",
              "phi4": "Phi4"}
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
PROMPT = ("Consider a person who is very {a}.\n"
          "How likely is this person to also be {b}?\n"
          "Answer with one number from 1 to 7, where 1 = very unlikely and "
          "7 = very likely.\nNumber:")


def medoids():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    H = np.array(h["correlation_matrix"], float)
    tc = json.load(open("instruments/trait_clusters.json"))
    out = []
    for c in sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"])):
        idx = [labels.index(m) for m in c["members"]]
        sub = H[np.ix_(idx, idx)]
        out.append(c["members"][int(np.argmax(sub.mean(1)))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{args.model}_tom_likely_dists.npz"
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return

    adjs = medoids()
    n = len(adjs)
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    D = np.full((n, n, 7), np.nan, np.float32)
    E = np.full((n, n), np.nan, np.float32)
    H = np.full((n, n), np.nan, np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d, _, h = hf.likert_distribution(
                model, tok, PROMPT.format(a=adjs[i], b=adjs[j]),
                device, digits=DIGITS)
            tot = sum(d.values())
            D[i, j] = [d.get(k, 0.0) / tot for k in DIGITS]
            E[i, j] = sum(int(k) * p for k, p in d.items()) / tot
            H[i, j] = h
        print(f"  row {i+1}/{n} ({adjs[i]})", flush=True)

    # reproduction check vs the stored full-matrix EVs
    jn = JUDGE_NAME.get(args.model, args.model)
    ref = np.load(f"results/adjectives/introspect_full/{jn}_tom_likely_dir.npz",
                  allow_pickle=True)
    ra = [str(a).lower() for a in ref["adjectives"]]
    idx = [ra.index(a) for a in adjs]
    Bref = ref["B"].astype(float)[np.ix_(idx, idx)]
    m = ~np.eye(n, dtype=bool)
    diff = np.abs(E[m] - Bref[m])
    print(f"reproduction check vs stored B: mean|dEV| {np.nanmean(diff):.4f} "
          f"max {np.nanmax(diff):.3f}")

    np.savez_compressed(out, dists=D, ev=E, entropy=H,
                        adjectives=np.array(adjs), prompt=np.array(PROMPT),
                        reproduction_mean_abs_dev=np.nanmean(diff))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
