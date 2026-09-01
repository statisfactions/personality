"""JUDGE cooking: EV -> P map, centering, base-rate least squares, phi.

Formulas extracted verbatim from scripts/judge_base_rate_fit.py
(2026-08-28..30 arc). Column-centering is the b-incidence correction
(2026-08-27 decision: column-centering > double-centering for the
mechanism row; phi with the EV/8 map + fitted base rates for the
construct row).
"""
import json

import numpy as np

from . import load, paths


def EV2P(ev):
    """EV/8: embed the 1-7 scale in 0-8 -> P in [.125, .875], no saturation
    (2026-08-30; fixes log blowup for extreme-committed models)."""
    return np.asarray(ev, float) / 8


def column_center(B, skip_diag=True):
    """Remove per-column (target-b incidence) means from a JUDGE matrix.
    skip_diag: exclude the diagonal from the column means (convention)."""
    B = np.asarray(B, float).copy()
    n = B.shape[0]
    if skip_diag:
        mask = ~np.eye(n, dtype=bool)
        mu = np.array([B[mask[:, j], j].mean() for j in range(n)])
    else:
        mu = B.mean(0)
    return B - mu[None, :]


def pairs_log_asym(B):
    """Y[a,b] = log P(b|a) - log P(a|b) under EV/8; zero diagonal."""
    Pc = EV2P(B)
    Y = np.log(Pc) - np.log(Pc.T)
    np.fill_diagonal(Y, 0)
    return Y


def pairs_potential(B):
    """psi: pairs-only, level-free implied base-rate potential (Y col-mean)."""
    return pairs_log_asym(B).mean(0)


def implied_phi(Pc, P):
    """Phi matrix implied by conditionals Pc and base rates P
    (symmetrized joint, zero diagonal)."""
    J = 0.5 * (Pc * P[:, None] + Pc.T * P[None, :])
    cov = J - P[:, None] * P[None, :]
    phi = cov / np.sqrt((P * (1 - P))[:, None] * (P * (1 - P))[None, :])
    np.fill_diagonal(phi, 0)
    return phi


def base_rate_fit(B, d_log, lam=1.0):
    """Joint least squares for log base rates l over n adjectives.

    pairs : Y[a,b] ~ l_b - l_a   (complete-graph Laplacian n*I - 11^T)
    direct: d_log_b ~ l_b        (weight lam)

    Returns dict(l, P, psi, phi).
    """
    B = np.asarray(B, float)
    n = B.shape[0]
    Y = pairs_log_asym(B)
    psi = Y.mean(0)
    L = n * np.eye(n) - np.ones((n, n)) + lam * np.eye(n)
    rhs = Y.sum(0) + lam * np.asarray(d_log, float)
    l = np.linalg.solve(L, rhs)
    P = np.clip(np.exp(l), 0.01, 0.99)
    phi = implied_phi(EV2P(B), P)
    return {"l": l, "P": P, "psi": psi, "phi": phi}


def cook(model, lam=1.0):
    """Load a model's JUDGE matrix + direct base rates and run the joint fit.

    Returns the base_rate_fit dict plus adjectives, the direct log rates d,
    and coherence r(d, psi)."""
    j = load.load_judge(model)
    adj = list(j["B"].index)
    d_ev = load.base_rate(model).reindex(adj)
    d = np.log(EV2P(d_ev.values))
    out = base_rate_fit(j["B"].values, d, lam=lam)
    out["adjectives"] = adj
    out["d"] = d
    out["coherence"] = float(np.corrcoef(d, out["psi"])[0, 1])
    return out
