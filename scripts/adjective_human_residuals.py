#!/usr/bin/env python3
"""W16: which human trait-associations does the model's judgment refuse to reproduce?

rgb's observation: the tom_likely-vs-human DIFFERENCES tend to land where the HUMAN
behaviour is surprising — the model gives the trait-coherent answer, the human carries a
cultural/idiosyncratic association. This ranks those divergences.

Both matrices on one footing: prevalence-corrected (off-diagonal double-centered) then
z-scored on the off-diagonal. Residual = z_model - z_human.
  residual << 0 : HUMAN links it, model doesn't  -> "association the model won't reproduce"
  residual >> 0 : MODEL links it, humans don't    -> model over-links (stereotype / halo)
Cohort-mean judgment (wisdom of crowds). Restricts to dispositional words by default
(physical/demographic/status pairs are orthogonal, not "surprising").

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_human_residuals.py
"""
import argparse
import glob
import json
import os

import numpy as np

from adjective_dispositional_curl import DROP

H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))


def dc(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, 1); g = np.nanmean(A)
    D = A - m[:, None] - m[None, :] + g; return D


def zoff(M):
    A = M.copy(); np.fill_diagonal(A, np.nan)
    iu = np.triu_indices_from(A, 1); o = A[iu]
    return (A - np.nanmean(o)) / (np.nanstd(o) + 1e-9)


def judge(s):
    z = np.load(f"results/adjectives/introspect_full/{s}_tom_likely_dir.npz",
                allow_pickle=True)
    adj = list(z["adjectives"]); B = z["B"].astype(float)
    ri = [adj.index(w) for w in LABELS]; B = B[np.ix_(ri, ri)]
    return (B + B.T) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--all-words", action="store_true",
                    help="include physical/demographic words (default: dispositional only)")
    args = ap.parse_args()

    names = [os.path.basename(p).replace("_tom_likely_dir.npz", "") for p in
             sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz"))]
    TOM = np.nanmean([dc(judge(n)) for n in names], 0)

    zH = zoff(dc(HUMAN)); zM = zoff(TOM)
    R = zM - zH

    keep = np.array([w not in DROP for w in LABELS]) if not args.all_words \
        else np.ones(len(LABELS), bool)

    iu = np.triu_indices(len(LABELS), 1)
    pairs = []
    for a, b in zip(*iu):
        if not (keep[a] and keep[b]):
            continue
        if np.isnan(R[a, b]):
            continue
        pairs.append((R[a, b], zH[a, b], zM[a, b], HUMAN[a, b], a, b))
    pairs = np.array(pairs, dtype=object)
    res = np.array([p[0] for p in pairs], float)

    def show(idx, title):
        print(f"\n{title}")
        print(f"  {'adjective A':>16}  {'adjective B':<16} "
              f"{'resid':>6} {'human_z':>7} {'model_z':>7} {'human_r':>7}")
        for k in idx:
            r, zh, zm, hr, a, b = pairs[k]
            print(f"  {LABELS[a]:>16}  {LABELS[b]:<16} "
                  f"{r:6.2f} {zh:7.2f} {zm:7.2f} {hr:7.2f}")

    # human links it, model breaks it: most negative residual, among human-positive pairs
    humanpos = np.array([p[1] > 0.5 for p in pairs])    # human_z > 0.5
    cand = np.where(humanpos)[0]
    neg = cand[np.argsort(res[cand])[:args.n]]
    show(neg, "HUMAN links it, MODEL won't reproduce  (humans associate; model judges unlikely)")

    # model links it, humans don't: most positive residual among human-low pairs
    humanlow = np.array([p[1] < 0.0 for p in pairs])
    cand2 = np.where(humanlow)[0]
    pos = cand2[np.argsort(-res[cand2])[:args.n]]
    show(pos, "MODEL links it, HUMANS don't  (model over-links; halo / stereotype)")

    print(f"\n(dispositional words only: {keep.sum()}/{len(LABELS)}; "
          f"--all-words to include physical/demographic)")


if __name__ == "__main__":
    main()
