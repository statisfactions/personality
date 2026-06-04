#!/usr/bin/env python3
"""Strongest intransitive triples (rock-paper-scissors loops) in tom_likely (W16 §5).

The asymmetry flow D[A,B]=P(B|A)-P(A|B) decomposes into gradient (a single ranking
~ prevalence) + curl (cycles). An intransitive triple is a triangle with large
CIRCULATION D[A,B]+D[B,C]+D[C,A]. Because the gradient contributes 0 to every
triangle sum, circulation = pure curl = base-rate-free by construction.

Brute-forcing all C(523,3)~24M triples is wasteful; the strongest cycles involve
the highest-curl-energy words, so rank words by curl-row-energy and brute-force
triples among the top K. Circulation uses D directly (= the curl's circulation).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_cycles.py
"""
import json
from itertools import combinations

import numpy as np

LABELS = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]


def main():
    for short in ("Qwen7", "Gemma12"):
        z = np.load(f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz",
                    allow_pickle=True)
        adj = list(z["adjectives"]); B = z["B"].astype(float)
        ri = [adj.index(w) for w in LABELS]; B = B[np.ix_(ri, ri)]
        n = len(LABELS)
        D = B - B.T
        np.fill_diagonal(D, 0.0)
        D = np.nan_to_num(D, nan=0.0)
        # curl part (for ranking words by cycle-involvement)
        p = D.mean(axis=1)
        R = D - (p[:, None] - p[None, :])
        np.fill_diagonal(R, 0.0)
        energy = (R ** 2).sum(axis=1)
        K = 110
        top = np.argsort(energy)[::-1][:K]

        best = []
        for a, b, c in combinations(top, 3):
            # both orientations; keep the positive one
            circ = D[a, b] + D[b, c] + D[c, a]
            if circ < 0:
                a, b, c, circ = a, c, b, -circ
            best.append((circ, a, b, c))
        best.sort(reverse=True)

        print(f"\n===== {short}: strongest intransitive triples (A->B->C->A) =====")
        print(f"(curl-energy concentrated in: {', '.join(LABELS[i] for i in top[:8])} ...)")
        seen = set(); shown = 0
        for circ, a, b, c in best:
            key = frozenset((a, b, c))
            if key in seen:
                continue
            seen.add(key)
            A, Bn, C = LABELS[a], LABELS[b], LABELS[c]
            print(f"\n  circulation {circ:+.2f}:  {A} -> {Bn} -> {C} -> {A}")
            print(f"    P({Bn}|{A})={B[a,b]:.1f} > P({A}|{Bn})={B[b,a]:.1f}   "
                  f"P({C}|{Bn})={B[b,c]:.1f} > P({Bn}|{C})={B[c,b]:.1f}   "
                  f"P({A}|{C})={B[c,a]:.1f} > P({C}|{A})={B[a,c]:.1f}")
            shown += 1
            if shown >= 6:
                break


if __name__ == "__main__":
    main()
