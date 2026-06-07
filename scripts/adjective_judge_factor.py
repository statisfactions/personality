#!/usr/bin/env python3
"""Factor structure of the tom_likely JUDGMENT geometry vs representation vs human (W16).

W14 found the *representation* geometry collapses to a ~2-factor evaluative core and
varimax goes degenerate on the weak factors (goofy). But the judgment (tom_likely)
matches the human correlation structure at r~0.8 (W16 §5.8). Does the judgment side
therefore carry a richer, more human-like factor structure — and does varimax behave?

Three correlation matrices, each in its natural form:
  HUMAN : 525-PDA inter-adjective correlations (gold; direct).
  REPR  : cohort-mean adaptive_denoise cosine (W14's object).
  TOM   : cohort-mean tom_likely, prevalence-corrected, then PROFILE-CORRELATION
          (corr of each adjective's co-occurrence profile) — the 1-7 EV scale is not
          a correlation, so this is the natural way to put it on a correlation footing.
          [robustness: also the direct diag->1 route, --tom-direct]

Reports: (1) scree + PC1/PC2 general-factor ratio; (2) top words on unrotated PC1-6;
(3) Kaiser varimax k=6 top words + whether rotation is stable (over-extraction curve,
identical metric to adjective_overextraction.rotation_profile).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_judge_factor.py
"""
import argparse
import glob
import json
import os

import numpy as np

from adjective_overextraction import (_phi, rotate, rotation_profile,
                                       _signfix_sort, KMAX)
from adjective_geom import model_matrix
from adjective_human_match import repr_matrix

H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))


def dc_off(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, 1); g = np.nanmean(A)
    Dc = A - m[:, None] - m[None, :] + g; np.fill_diagonal(Dc, 0.0)
    return Dc


def cov2corr(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d)


def judge_sym(short):
    p = f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz"
    z = np.load(p, allow_pickle=True); adj = list(z["adjectives"])
    if int(z["done"]) < int(z["n_pairs"]):
        return None
    B = z["B"].astype(float); ri = [adj.index(w) for w in LABELS]
    B = B[np.ix_(ri, ri)]
    return (B + B.T) / 2


def cohort_names():
    out = []
    for p in sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz")):
        z = np.load(p, allow_pickle=True)
        if int(z["done"]) >= int(z["n_pairs"]):
            out.append(os.path.basename(p).replace("_tom_likely_dir.npz", ""))
    return out


def topwords(L, n=6):
    """For each factor column, the top +/- loading adjectives."""
    rows = []
    for j in range(L.shape[1]):
        col = L[:, j]
        pos = [LABELS[i] for i in np.argsort(-col)[:n]]
        neg = [LABELS[i] for i in np.argsort(col)[:n]]
        rows.append((pos, neg))
    return rows


def scree(C, p=8):
    Cc = np.nan_to_num(C).copy(); np.fill_diagonal(Cc, 1.0)
    ev = np.sort(np.linalg.eigvalsh(Cc))[::-1]
    pct = 100 * ev / Cc.shape[0]
    return ev[:p], pct[:p]


def show(tag, C, kfac=6):
    ev, pct = scree(C)
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    print("scree %var:", "  ".join(f"{x:4.1f}" for x in pct),
          f"   PC1/PC2 = {ev[0]/ev[1]:.2f}")

    U = _phi(C, kfac)
    U, _ = _signfix_sort(U)
    print(f"\n-- unrotated PC1-{kfac} (top +/- words) --")
    for j, (pos, neg) in enumerate(topwords(U), 1):
        print(f"  PC{j}:  +{', '.join(pos[:5])}   |  -{', '.join(neg[:5])}")

    R = rotate(_phi(C, kfac))
    R, _ = _signfix_sort(R)
    print(f"\n-- Kaiser varimax k={kfac} (top +/- words) --")
    for j, (pos, neg) in enumerate(topwords(R), 1):
        print(f"  F{j}:  +{', '.join(pos[:5])}   |  -{', '.join(neg[:5])}")

    prof = rotation_profile(C, kaiser=True)
    meancong = {k: float(np.mean(v)) for k, v in prof.items()}
    print("\n-- rotation-stability (mean |Tucker| of unrotated->varimax, by k) --")
    print("   k:   " + "  ".join(f"{k:4d}" for k in range(2, KMAX + 1)))
    print("  cong:" + "  ".join(f"{meancong[k]:.2f}" for k in range(2, KMAX + 1)))
    # tail congruence at k=6 = how well the weak factors survive rotation
    print(f"   weak-factor survival (min congruence @k=6): {prof[6].min():.2f}")
    return pct, meancong


def build():
    names = cohort_names()
    print(f"cohort: {len(names)} judges -> {', '.join(names)}")
    # repr cohort: mean cosine (cov2corr each), then average
    Rs = [repr_matrix(n) for n in names]
    REPR = np.nanmean([cov2corr(np.nan_to_num(R)) for R in Rs if R is not None], 0)
    # tom cohort: mean prevalence-corrected judgment
    TOMdc = np.nanmean([dc_off(judge_sym(n)) for n in names
                        if judge_sym(n) is not None], 0)
    return REPR, TOMdc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tom-direct", action="store_true",
                    help="factor TOM via diag->1 direct (apples-to-apples w/ repr) "
                         "instead of profile-correlation")
    args = ap.parse_args()

    REPR, TOMdc = build()
    TOM = np.nan_to_num(TOMdc.copy())
    np.fill_diagonal(TOM, 1.0)
    if not args.tom_direct:
        TOM = np.corrcoef(np.nan_to_num(TOMdc))   # profile correlation

    show("HUMAN  (525-PDA correlations, direct)", HUMAN)
    show("REPR   (cohort-mean adaptive_denoise cosine)", REPR)
    show("TOM    (cohort tom_likely, " +
         ("direct diag->1" if args.tom_direct else "profile-correlation") + ")", TOM)


if __name__ == "__main__":
    main()
