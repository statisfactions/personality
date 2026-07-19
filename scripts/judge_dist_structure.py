"""Structure of the raw JUDGE distributions (W18 §4).

Consolidates the analyses run on the full-523 regenerated distribution
tensors (results/adjectives/judge_dists/<model>_tom_likely_dists_full.npz):

1. Bimodality census — fraction of cells with >=0.25 mass >=2 digits from
   the argmax (phi4 0.56, Aya 0.26, gemmas ~0).
2. Bernoulli character — per-cell variance as a fraction of the endpoint-
   Bernoulli maximum (ev-1)(7-ev); mean endpoint mass P(1)+P(7).
3. Mode-pair census — which (argmax, second-mode) digit pairs dominate the
   bimodal cells (phi4: (4,6)/(1,3)/(4,7) commit-vs-hedge splits; pure (1,7)
   coin flips only ~9%, concentrated on ill-posed category-mismatch cells:
   col guilty/fat/unfaithful, row left-handed/blind/tall; asymmetric).
4. Tail signal — human-match of argmax-only vs full-EV vs tail-only-EV
   (argmax digit deleted, renormalized): gemma's 9% tail alone carries
   r=0.587 of 0.621 — sharpening is a rendering choice, the graded belief
   survives underneath. EV >= argmax for every model.

Usage: PYTHONPATH=scripts python scripts/judge_dist_structure.py [--models m1,m2]
"""
import argparse
import json
from collections import Counter

import numpy as np

DIR = "results/adjectives/judge_dists"
DEFAULT = "gemma3,qwen2.5,llama3.2,phi4,Llama8,Qwen7,Aya,Gemma12,Gemma27,Qwen32"
DIGITS = np.arange(1, 8)


def zoff(S):
    off = S[~np.eye(S.shape[0], dtype=bool)]
    return (S - off.mean()) / off.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    args = ap.parse_args()

    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    hl = [l.lower() for l in h["labels"]]
    H = np.array(h["correlation_matrix"], float)
    Hm = H.copy()
    np.fill_diagonal(Hm, 0)

    for m in args.models.split(","):
        try:
            z = np.load(f"{DIR}/{m}_tom_likely_dists_full.npz", allow_pickle=True)
        except FileNotFoundError:
            print(f"{m}: no full dists yet, skipping")
            continue
        adjs = [str(a).lower() for a in z["adjectives"]]
        idx = [adjs.index(l) for l in hl]
        D = z["dists"].astype(np.float64)[np.ix_(idx, idx)]
        n = len(hl)
        eye = np.eye(n, dtype=bool)
        P = D[~eye]

        ev = P @ DIGITS
        var = P @ DIGITS**2 - ev**2
        frac = var / np.maximum((ev - 1) * (7 - ev), 1e-9)
        endm = P[:, 0] + P[:, 6]

        arg = P.argmax(1)
        far = np.zeros(len(P))
        for k in range(7):
            far += np.where(np.abs(k - arg) >= 2, P[:, k], 0)
        bim = far >= 0.25
        Pfar = P.copy()
        for off in (-1, 0, 1):
            j = np.clip(arg + off, 0, 6)
            Pfar[np.arange(len(P)), j] = 0
        second = Pfar.argmax(1)
        pairs = Counter(
            (min(a, b) + 1, max(a, b) + 1)
            for a, b in zip(arg[bim], second[bim]))
        top_pairs = ", ".join(f"({a},{b}):{c/max(bim.sum(),1):.2f}"
                              for (a, b), c in pairs.most_common(4))

        # tail signal vs human
        am = np.where(~eye, D.argmax(-1) + 1.0, 0.0)
        E = np.where(~eye, np.einsum("ijk,k->ij", D, DIGITS.astype(float)), 0.0)
        Pt = D.copy()
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        Pt[ii, jj, D.argmax(-1)] = 0.0
        tot = Pt.sum(-1)
        Et = np.where(tot > 1e-6,
                      np.einsum("ijk,k->ij", Pt, DIGITS.astype(float))
                      / np.maximum(tot, 1e-9), np.nan)

        zH = zoff(Hm)

        def rH(M):
            S = 0.5 * (M + M.T)          # nan propagates: cell needs both dirs
            v, hv = S[~eye], zH[~eye]
            ok = np.isfinite(v)
            v = (v[ok] - v[ok].mean()) / v[ok].std()
            return np.corrcoef(v, hv[ok])[0, 1]

        print(f"{m:9s} bimodal {bim.mean():.3f}  var/varmax "
              f"frac>0.5 {(frac > 0.5).mean():.3f}  P(1)+P(7) {endm.mean():.3f}  "
              f"tail-mass {(1 - P.max(1)).mean():.3f}")
        print(f"          mode-pairs: {top_pairs}")
        print(f"          r(HUMAN): argmax {rH(am):+.3f}  EV {rH(E):+.3f}  "
              f"tail-only {rH(Et):+.3f}")


if __name__ == "__main__":
    main()
