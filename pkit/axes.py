"""Factor/axis utilities: eigendecomposition, Horn PA, varimax, congruence.

Consolidated from human_axis_stability.py, self_framing_sensitivity.py,
adjective_factor_heatmap.py, facet_slides.py (verbatim formulas).
"""
import numpy as np


def eig_axes(X, kmax=60):
    """Top-kmax eigenpairs of corrcoef(X.T): (eigenvalues, item loading vecs)."""
    w, v = np.linalg.eigh(np.corrcoef(X.T))
    o = np.argsort(-w)[:kmax]
    return w[o], v[:, o]


def participation_ratio(w):
    """PR = (sum w)^2 / sum w^2 over positive eigenvalues."""
    p = np.asarray(w, float)
    p = p[p > 0]
    return float(p.sum() ** 2 / (p ** 2).sum())


def horn_k(R, n_perm=20, seed=0):
    """Horn's parallel analysis on respondents-x-items R: retained k at the
    95th-percentile permutation null."""
    rng = np.random.default_rng(seed)
    w = np.sort(np.linalg.eigvalsh(np.corrcoef(R.T)))[::-1]
    null = []
    for _ in range(n_perm):
        P = np.array([rng.permutation(R[:, j]) for j in range(R.shape[1])]).T
        null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1])
    thr = np.percentile(null, 95, axis=0)
    k = 0
    while k < len(w) and w[k] > thr[k]:
        k += 1
    return k


def varimax(Phi, gamma=1.0, q=50, tol=1e-6, normalize=False):
    """Varimax rotation of a (p x k) loading matrix (Kaiser's varimax criterion).

    normalize=True additionally applies Kaiser *normalization* (row-scale each
    variable by its communality before rotating, rescale back after) — the SPSS
    default, which Cutler & Condon (2022, supp. pp.8-9) matched via Weide &
    Beauducel (2019). normalize=False (our historical default) rotates the raw
    loadings. The two diverge most on weakly-defined factors. NB: "Kaiser's
    varimax criterion" (gamma=1, always on here) is a separate thing from
    "Kaiser normalization" (the normalize flag).
    """
    Phi = np.asarray(Phi, float)
    if normalize:
        h = np.linalg.norm(Phi, axis=1, keepdims=True)
        h = np.where(h < 1e-9, 1.0, h)
        return varimax(Phi / h, gamma, q, tol, normalize=False) * h
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vt
        d = float(s.sum())
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def tucker(a, b):
    """Tucker congruence of two loading vectors: a.b / (|a||b|)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
