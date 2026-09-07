"""Distribution readouts and matrix conventions.

The centering conventions are the named 3-row specification (raw /
entry-z / ipsatized): entry-centering of a similarity grid is ~equivalent
to elevation removal (the elevation eigenvector is near-uniform, cv=0.17;
grid top component = ipsatized PC1 at r=0.90) — see the W20 population
arc ledger.
"""
import numpy as np


def ev_from_dist(dist):
    """Expected value of a {digit-string: prob} distribution (renormalized)."""
    ks = np.array([float(k) for k in dist])
    ps = np.array([dist[k] for k in dist], float)
    # guard: zero total mass (or empty dist) can't renormalize -> treat as
    # uniform over the keys (mean of ks); 0.0 when there are no keys at all
    if ps.sum() <= 0:
        return float(ks.mean()) if ks.size else 0.0
    return float((ks * ps).sum() / ps.sum())


def entropy_from_dist(dist):
    """Shannon entropy (nats) of a {label: prob} distribution."""
    p = np.array(list(dist.values()), float)
    p = p[p > 0]
    # guard: no positive mass -> a degenerate dist carries no uncertainty (0.0)
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def ipsatize(M, eps=1e-12):
    """C&C's within-person z: standardize each ROW (person) across items.

    eps guards the constant-row case (sd = 0 -> a flat profile maps to
    zeros, i.e. "no shape", rather than NaN). Note the identity: the
    result equals sqrt(k) times the unit-normalized mean-centered row
    (the C&G "shape" component) — ipsatizing IS the shape extraction,
    up to that constant.
    """
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    return (M - mu) / (sd + eps)


def zscore_offdiag(S):
    """Z-score a square matrix by its off-diagonal mean/std (entry-z)."""
    off = S[~np.eye(S.shape[0], dtype=bool)]
    sd = off.std()
    # guard: constant off-diagonal (sd=0) -> centered zeros ("no shape"), not NaN
    return (S - off.mean()) / (sd if sd > 0 else 1.0)


entry_z = zscore_offdiag  # alias: the 3-row spec's middle row


def cos_sim(X):
    """Row-cosine similarity after column-centering (acts convention)."""
    Xc = X - X.mean(0)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    # guard: a zero-norm row (flat after centering) has no direction -> map it
    # to the zero vector (cos 0 to everything), not NaN; nonzero rows untouched
    Xn = Xc / np.where(n == 0, 1.0, n)
    return Xn @ Xn.T


def winsorize(X, massive):
    """Cap massive-dim columns at the max std of the non-massive columns."""
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    # guard: every column flagged massive -> no reference spread to cap
    # against; return X unchanged instead of crashing on an empty max
    if keep.size == 0:
        return X
    std = X.std(0)
    cap = std[keep].max()
    for m_ in massive:
        if std[m_] > cap:
            # guard: cap=0 (all reference columns constant) -> flatten the
            # column to 0 (the finite limit of X / (std/cap) as cap -> 0)
            if cap == 0:
                X[:, m_] = 0.0
            else:
                X[:, m_] /= (std[m_] / cap)
    return X


def remove_pc1(M):
    """Subtract the top eigencomponent of the zero-diagonal matrix
    (four_grid_compare convention)."""
    A = M.copy()
    np.fill_diagonal(A, 0.0)
    w, v = np.linalg.eigh(A)
    k = np.argmax(np.abs(w))
    return A - w[k] * np.outer(v[:, k], v[:, k])


def offdiag(S):
    """Flattened off-diagonal entries of a square matrix."""
    return S[~np.eye(S.shape[0], dtype=bool)]


def offdiag_corr(A, B):
    """Pearson r between the off-diagonals of two square matrices."""
    a = offdiag(np.asarray(A, float))
    b = offdiag(np.asarray(B, float))
    # guard: a constant off-diagonal has zero variance -> r is undefined;
    # report 0.0 (no measurable linear association), not NaN
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
