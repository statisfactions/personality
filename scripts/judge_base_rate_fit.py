"""Joint least-squares base rates for JUDGE (2026-08-28, rgb).

Unknown l = log P over 523 adjectives. Two data sources:
  pairs : Y[a,b] = log P(b|a) - log P(a|b) ~ l_b - l_a   (from tom_likely)
  direct: d_b = log P_direct(b)              ~ l_b        (base_rate_query)
Minimize sum_ab (Y_ab - (l_b - l_a))^2 + lam * sum_b (d_b - l_b)^2.
Reports the fitted l, the pairs-only potential psi (level-free), the
coherence r(d, psi) between stated and implied base rates, and the phi
correlation matrix implied by the fit, vs HUMAN.

Usage: PYTHONPATH=scripts python scripts/judge_base_rate_fit.py --model Qwen7 [--lam 1.0]
"""
import argparse
import json

import numpy as np

EV2P = lambda ev: np.clip((np.asarray(ev, float) - 1) / 6, 0.02, 0.98)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    Hm = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(Hm, 0)
    z = np.load(f"results/adjectives/introspect_full/{args.model}_tom_likely_dir.npz",
                allow_pickle=True)
    B = np.asarray(z["B"], float)
    adj = [str(a).lower() for a in z["adjectives"]]
    if adj != labels:
        ix = [adj.index(l) for l in labels]
        B = B[np.ix_(ix, ix)]
    n = len(B)
    off = ~np.eye(n, dtype=bool)
    Pc = EV2P(B)
    Y = np.log(Pc) - np.log(Pc.T)
    np.fill_diagonal(Y, 0)
    psi = Y.mean(0)                                   # pairs-only, level-free
    d = np.log(EV2P([json.load(open(
        f"results/adjectives/introspect_full/{args.model}_base_rate.json"))
        ["results"][a]["ev"] for a in labels]))
    # normal equations: pairs give Laplacian n*I - 11^T (complete graph),
    # rhs = column sums of Y; direct adds lam*I and lam*d
    L = n * np.eye(n) - np.ones((n, n)) + args.lam * np.eye(n)
    rhs = Y.sum(0) + args.lam * d
    l = np.linalg.solve(L, rhs)
    P = np.clip(np.exp(l), 0.01, 0.99)
    J = 0.5 * (Pc * P[:, None] + Pc.T * P[None, :])
    cov = J - P[:, None] * P[None, :]
    phi = cov / np.sqrt((P * (1 - P))[:, None] * (P * (1 - P))[None, :])
    np.fill_diagonal(phi, 0)
    print(f"{args.model}: coherence r(direct, pairs-implied) = "
          f"{np.corrcoef(d, psi)[0, 1]:+.2f}; fitted median P = {np.median(P):.2f}; "
          f"item-level r(phi, HUMAN) = {np.corrcoef(phi[off], Hm[off])[0, 1]:.3f}")
    o = np.argsort(l)
    print("  lowest:", [labels[i] for i in o[:5]], "| highest:", [labels[i] for i in o[-5:]])
    np.savez(f"results/adjectives/introspect_full/{args.model}_base_rate_fit.npz",
             l=l, psi=psi, d=d, phi=phi.astype(np.float32), adjectives=np.array(labels))


if __name__ == "__main__":
    main()
