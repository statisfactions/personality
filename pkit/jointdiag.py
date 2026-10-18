"""Common axes across several item x item matrices (rgb 2026-10-18).

    S_t ~ Q diag(w_t) Q',  min sum_t sum_{i!=j} (S_t - Q W_t Q')_ij^2

Orthonormal Q (n x K); off-diagonal least squares only (diagonals imputed
from the model each step, MINRES-style); Q updated by an MM polar step,
Q <- polar([(A_k + cI) q_k]_k), A_k = sum_t w_tk S_t, which never increases
the loss. Also the comparison methods (DISTATIS compromise, eigenvectors of
the mean or of the summed squares) and the shared scoring used to compare
them: whatever produced Q, weights come from the same off-diagonal regression.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment


def offmask(n):
    return ~np.eye(n, dtype=bool)


def unit_offdiag(A):
    """Zero diagonal, unit off-diagonal Frobenius norm."""
    A = np.array(A, float); np.fill_diagonal(A, 0.0)
    return A / np.linalg.norm(A[offmask(len(A))])


def unit_lambda1(A):
    """Zero diagonal, largest |eigenvalue| = 1 (DISTATIS/MFA normalization)."""
    A = np.array(A, float); np.fill_diagonal(A, 0.0)
    return A / np.abs(np.linalg.eigvalsh(A)).max()


def gram_offdiag(Q):
    """G_kl = sum_{i!=j} q_ik q_jk q_il q_jl = (q_k.q_l)^2 - sum_i q_ik^2 q_il^2."""
    Q2 = Q ** 2
    return (Q.T @ Q) ** 2 - Q2.T @ Q2


def weights(Q, S):
    """Off-diagonal least-squares channel weights W (T x K) for fixed axes Q.
    S: T x n x n with zero diagonals."""
    b = np.einsum("ik,tij,jk->tk", Q, S, Q)
    return np.linalg.solve(gram_offdiag(Q), b.T).T


def loss(Q, W, S):
    off = offmask(S.shape[1])
    return float(sum((((S[t] - (Q * W[t]) @ Q.T)[off]) ** 2).sum() for t in range(len(S))))


def share(Q, W, S):
    """Per-matrix share of off-diagonal sum of squares captured by the fit."""
    off = offmask(S.shape[1])
    return np.array([1 - (((S[t] - (Q * W[t]) @ Q.T)[off]) ** 2).sum() / (S[t][off] ** 2).sum()
                     for t in range(len(S))])


def fit(S, Q, iters=5000, tol=1e-10):
    """Joint off-diagonal diagonalization from start Q. Returns Q, W, loss, iterations."""
    n, K = Q.shape
    W = weights(Q, S)
    prev = f = np.inf; it = 0
    for it in range(iters):
        Sf = S.copy()
        for t in range(len(S)):
            Sf[t][np.diag_indices(n)] = ((Q ** 2) * W[t]).sum(1)
        A = np.einsum("tk,tij->kij", W, Sf)
        c = max(0.0, -min(eigh(A[k], eigvals_only=True, subset_by_index=[0, 0])[0] for k in range(K))) + 1e-9
        M = np.stack([A[k] @ Q[:, k] + c * Q[:, k] for k in range(K)], 1)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
        W = weights(Q, S)
        f = loss(Q, W, S)
        if prev - f < tol * max(1.0, prev):
            break
        prev = f
    return Q, W, f, it + 1


def top_eig(M, K, by="abs"):
    w, V = np.linalg.eigh(M)
    o = np.argsort(-np.abs(w) if by == "abs" else -w)[:K]
    return V[:, o]


def sumsq_axes(S, K):
    """Top-K eigenvectors of sum_t S_t^2: the SVD of all tables' eigenvectors
    stacked, weighted by eigenvalue (sign-agnostic 'PCA of the spectra')."""
    return top_eig(np.einsum("tij,tjk->ik", S, S), K, by="signed")


def mean_axes(S, K):
    """Top-K (|eigenvalue|) eigenvectors of the plain average matrix."""
    return top_eig(S.mean(0), K)


def distatis_axes(S, K):
    """DISTATIS (Abdi et al.): RV matrix between tables (off-diagonal entries),
    weights = its leading eigenvector normalized to sum 1, compromise =
    weighted sum, axes = top-K eigenvectors of the compromise. Pass S already
    normalized (unit_lambda1 for textbook DISTATIS). Returns axes, alpha, RV."""
    off = offmask(S.shape[1])
    X = np.array([s[off] for s in S])
    G = X @ X.T; d = np.sqrt(np.diag(G)); RV = G / np.outer(d, d)
    w, V = np.linalg.eigh(RV); a = V[:, -1]; a = a * np.sign(a.sum()); a = a / a.sum()
    return top_eig(np.einsum("t,tij->ij", a, S), K), a, RV


def order_and_sign(Q, W):
    """Axes by sum_t w_tk^2; each signed so its largest |loading| is positive."""
    o = np.argsort(-(W ** 2).sum(0)); Q, W = Q[:, o], W[:, o]
    return Q * np.sign(Q[np.abs(Q).argmax(0), np.arange(Q.shape[1])]), W


def match(Qref, Q):
    """Hungarian match of Q's axes to Qref's by |cos|. Returns perm (index into
    Q per reference axis) and the matched |cos|."""
    C = np.abs(Qref.T @ Q)
    r, c = linear_sum_assignment(-C)
    perm = np.empty(Qref.shape[1], int); perm[r] = c
    return perm, C[np.arange(Qref.shape[1]), perm]


def subspace_overlap(Q1, Q2):
    """Mean squared cosine of the principal angles (1 = same span)."""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float((s ** 2).mean())
