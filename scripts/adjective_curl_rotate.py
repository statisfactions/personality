#!/usr/bin/env python3
"""Re-rotate the curl planes to simple structure (W16 §5 follow-up).

The curl SVD returns each rotational plane in an ARBITRARY basis (singular values
pair exactly, so any rotation of a pair is an equally valid singular-vector basis).
Qwen's planes happened to land readable; Gemma's didn't. So rotate each plane to
varimax simple structure and re-read — if Gemma cleans up it was just the basis;
if it stays smeared the organization is genuinely different.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_curl_rotate.py
"""
import numpy as np

from adjective_dispositional_curl import DROP, LABELS
from adjective_factor_heatmap import varimax


def main():
    keep = [i for i, w in enumerate(LABELS) if w not in DROP]
    sub = np.array([LABELS[i] for i in keep])
    for short in ("Qwen7", "Gemma12"):
        z = np.load(f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz",
                    allow_pickle=True)
        adj = list(z["adjectives"]); B = z["B"].astype(float)
        ri = [adj.index(w) for w in sub]; B = B[np.ix_(ri, ri)]
        D = np.nan_to_num(B - B.T, nan=0.0); np.fill_diagonal(D, 0.0)
        p = D.mean(1); R = D - (p[:, None] - p[None, :]); np.fill_diagonal(R, 0.0)
        U, S, _ = np.linalg.svd(R)
        print(f"\n===== {short}  (singular values {', '.join(f'{s:.0f}' for s in S[:4])}) =====")
        for plane in range(2):
            L = U[:, 2 * plane:2 * plane + 2]            # n x 2 plane basis
            Lr = varimax(L)                              # rotate to simple structure
            print(f"  plane {plane+1} (varimax-rotated):")
            for ax in range(2):
                v = Lr[:, ax]; o = np.argsort(v)
                hi = ", ".join(sub[o[::-1][:7]]); lo = ", ".join(sub[o[:7]])
                print(f"    axis{ax+1}:  + {hi}")
                print(f"             - {lo}")


if __name__ == "__main__":
    main()
