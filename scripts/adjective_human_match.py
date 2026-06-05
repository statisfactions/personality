#!/usr/bin/env python3
"""Does the model's implicit personality covariance match humans? (W16 §5, open thread)

Compares three readouts to the human 525-PDA inter-adjective CORRELATION matrix:
  representation : adaptive_denoise cosine (already ~marginal-free)
  semantic       : tom-similarity judgment EV (symmetric)
  tom_likely     : dispositional co-occurrence judgment EV (asymmetric -> symmetrized)

rgb's correction: mean-symmetrizing only cancels the antisymmetric (directional)
part; the symmetric part still carries the PREVALENCE marginals (r[A]+r[B], etc.),
which a human correlation matrix lacks by construction. So strip additive marginals
from both via off-diagonal DOUBLE-CENTERING before comparing. Report raw vs corrected
to show how much prevalence was distorting the match.

  B = prevalence marginals (strip) + symmetric interaction (compare to human r)
      + antisymmetric interaction (the curl/cycles - no human-correlation analog)

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_human_match.py
"""
import json
import os

import numpy as np

from adjective_geom import model_matrix
from hf_logprobs import MODELS

H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.array(H["correlation_matrix"], float)


def double_center_off(M):
    """Off-diagonal additive-marginal removal: Mdc[i,j]=M[i,j]-m[i]-m[j]+grand,
    where m,grand are over off-diagonal (NaN-safe). Diagonal set to nan."""
    A = M.copy().astype(float)
    np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, axis=1)
    grand = np.nanmean(A)
    Dc = A - m[:, None] - m[None, :] + grand
    np.fill_diagonal(Dc, np.nan)
    return Dc


def match(M, Href):
    """Pearson r between upper-triangles (NaN-masked) of M and Href."""
    iu = np.triu_indices_from(M, 1)
    a, b = M[iu], Href[iu]
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def repr_matrix(short):
    base = f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers"
    p = f"{base}_bare.pt" if os.path.exists(f"{base}_bare.pt") else f"{base}.pt"
    _, C, _, _ = model_matrix(p, np.array(LABELS))
    return C


def judge_matrix(short, mode):
    base = "results/adjectives/introspect_full"
    if mode == "semantic":
        cands = [f"{base}/{short}.npz"]
    else:  # both-directions deep-dive file, else upper-triangle cohort file
        cands = [f"{base}/{short}_tom_likely_dir.npz", f"{base}/{short}_tom_likely.npz"]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        return None
    z = np.load(path, allow_pickle=True)
    adj = list(z["adjectives"]); B = z["B"].astype(float)
    ri = [adj.index(w) for w in LABELS]; B = B[np.ix_(ri, ri)]
    return (B + B.T) / 2.0          # symmetrize (cancels the directional part)


def main():
    Hdc = double_center_off(HUMAN)
    print(f"{'model':<9}{'readout':<14}{'raw r':>9}{'corrected r':>13}{'Δ (prevalence)':>16}")
    print("-" * 61)
    for short in ("Qwen7", "Gemma12"):
        mats = {"representation": repr_matrix(short),
                "semantic": judge_matrix(short, "semantic"),
                "tom_likely": judge_matrix(short, "tom_likely")}
        for name, M in mats.items():
            raw = match(M, HUMAN)
            cor = match(double_center_off(M), Hdc)
            print(f"{short:<9}{name:<14}{raw:>9.3f}{cor:>13.3f}{cor-raw:>+16.3f}")
        print()
    print("corrected = both sides off-diagonal double-centered (prevalence stripped).")
    print("hypothesis: tom_likely (dispositional co-occurrence) matches human best.")


if __name__ == "__main__":
    main()
