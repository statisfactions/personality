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
    return float((ks * ps).sum() / ps.sum())


def entropy_from_dist(dist):
    """Shannon entropy (nats) of a {label: prob} distribution."""
    p = np.array(list(dist.values()), float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def ipsatize(M):
    """C&C's within-person z: standardize each ROW (person) across items."""
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    return (M - mu) / sd


def zscore_offdiag(S):
    """Z-score a square matrix by its off-diagonal mean/std (entry-z)."""
    off = S[~np.eye(S.shape[0], dtype=bool)]
    return (S - off.mean()) / off.std()


entry_z = zscore_offdiag  # alias: the 3-row spec's middle row


def cos_sim(X):
    """Row-cosine similarity after column-centering (acts convention)."""
    Xc = X - X.mean(0)
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xn @ Xn.T


def winsorize(X, massive):
    """Cap massive-dim columns at the max std of the non-massive columns."""
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    std = X.std(0)
    cap = std[keep].max()
    for m_ in massive:
        if std[m_] > cap:
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
    return float(np.corrcoef(offdiag(np.asarray(A, float)),
                             offdiag(np.asarray(B, float)))[0, 1])
