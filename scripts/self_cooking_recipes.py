"""SELF cooking recipes side by side (2026-09-22, rgb: one uniform recipe for
comparison AND factor analysis; explain uncooked vs grand-mean vs double-
centering vs ipsatization, and why removing the UNCOOKED PC1 leaves SELF with
no human similarity).

Score channels (HUMAN, SELF) admit five treatments of the item x item matrix:
  raw          item Pearson r across respondents, zero diagonal
  grand-mean   raw minus its off-diagonal mean (pkit.channels.center)
  double       Gower double-centering  J S J,  J = I - 11'/n
  ips-center   respondents row-centered, THEN item r   (== double-centering
               of the item COVARIANCE, exactly; approx for the correlation)
  ips-z        respondents row-standardized (C&C), then item r
For each: spectrum, what PC1 is (cosine with the constant vector and with the
human desirability axis), congruence with HUMAN under the same treatment
before and after PC1 removal, and SELF's split-half (odd/even models)
reliability of the PC1-removed grid. Also the rank-2 (elevation, gain x
desirability) account of the raw SELF covariance.
Usage: .venv/bin/python scripts/self_cooking_recipes.py
"""
import numpy as np
import pkit
from pkit import channels as chn, measures

labels = pkit.load.adjectives(); n = len(labels)
Hm = chn.human_members(labels)
Sm, names = chn.self_members(labels)
print(f"HUMAN respondents {Hm.shape[0]}, SELF core models {Sm.shape[0]}, items {n}")

H = chn.cohort_matrix("respondents", Hm)
w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
one = np.ones(n) / np.sqrt(n)
t_perp = t - one * (one @ t); t_perp /= np.linalg.norm(t_perp)
J = np.eye(n) - np.ones((n, n)) / n
m44 = ~np.eye(44, dtype=bool)


def zd(S):
    S = S.copy(); np.fill_diagonal(S, 0); return S


def corr0(X):
    return zd(np.corrcoef(np.asarray(X, float).T))


TREAT = {
    "raw":         lambda X: corr0(X),
    "grand-mean":  lambda X: chn.center(corr0(X)),
    "double":      lambda X: zd(J @ corr0(X) @ J),
    "ips-center":  lambda X: corr0(X - X.mean(1, keepdims=True)),
    "ips-z":       lambda X: corr0(measures.ipsatize(np.asarray(X, float))),
}


def pc1(S):
    w, v = np.linalg.eigh(S); i = np.argmax(w)
    return w, v[:, i]


def rm1(S):
    return measures.remove_pc1(S)


def g(S):
    return chn.blockify(S)


def r44(A, B):
    return np.corrcoef(g(A)[m44], g(B)[m44])[0, 1]


print("\n--- what PC1 is, per treatment (|cos| with constant vector 1 and with human desirability t_perp) ---")
print(f"{'treatment':11s} {'chan':6s} {'off-mean':>8s} {'l1/sum|l|':>9s} {'l2/sum|l|':>9s} {'cos(v1,1)':>9s} {'cos(v1,t)':>9s} {'cos(v2,1)':>9s} {'cos(v2,t)':>9s}")
mats = {}
for name, f in TREAT.items():
    for chan, X in [("HUMAN", Hm), ("SELF", Sm)]:
        S = f(X); mats[(name, chan)] = S
        w, v1 = pc1(S); o = np.argsort(w)[::-1]; v2 = np.linalg.eigh(S)[1][:, o[1]]
        sa = np.abs(w).sum()
        print(f"{name:11s} {chan:6s} {S[~np.eye(n,dtype=bool)].mean():8.3f} {w[o[0]]/sa:9.3f} {w[o[1]]/sa:9.3f} "
              f"{abs(v1@one):9.2f} {abs(v1@t_perp):9.2f} {abs(v2@one):9.2f} {abs(v2@t_perp):9.2f}")

print("\n--- congruence SELF vs HUMAN, same treatment both sides (44-block off-diagonal r) ---")
print(f"{'treatment':11s} {'before PC1 rm':>13s} {'after PC1 rm':>12s} {'SELF split-half after rm':>24s} {'HUMAN split-half after rm':>25s}")
rng = np.random.default_rng(0)
for name, f in TREAT.items():
    Hs, Ss = mats[(name, "HUMAN")], mats[(name, "SELF")]
    before = r44(Ss, Hs); after = r44(rm1(Ss), rm1(Hs))
    # split-half reliability of the PC1-removed grid (odd/even respondents, 5 shuffles)
    def sh(X):
        rs = []
        for _ in range(5):
            p = rng.permutation(X.shape[0]); a, b = p[::2], p[1::2]
            rs.append(r44(rm1(f(X[a])), rm1(f(X[b]))))
        return np.mean(rs)
    print(f"{name:11s} {before:13.3f} {after:12.3f} {sh(Sm):24.3f} {sh(Hm):25.3f}")

print("\n--- cross-treatment: how different are the PC1-removed SELF grids from each other? (44-block r) ---")
keys = list(TREAT)
R = np.array([[r44(rm1(mats[(a, 'SELF')]), rm1(mats[(b, 'SELF')])) for b in keys] for a in keys])
print(" " * 12 + "".join(f"{k:>11s}" for k in keys))
for a, row in zip(keys, R):
    print(f"{a:11s} " + "".join(f"{x:11.2f}" for x in row))

print("\n--- rank-2 account of the RAW item covariance: x_ma = lambda_m + beta_m t_a + e ---")
for chan, X in [("HUMAN", Hm), ("SELF", Sm)]:
    X = np.asarray(X, float)
    A = np.c_[np.ones(n), t]                       # per-respondent OLS on (1, t)
    coef = np.linalg.lstsq(A, X.T, rcond=None)[0]  # 2 x m
    fit = (A @ coef).T
    r2 = 1 - ((X - fit) ** 2).sum() / ((X - X.mean()) ** 2).sum()
    C = np.cov(X.T)                                # item covariance across respondents
    P = A @ np.linalg.pinv(A)                      # projector onto span{1, t}
    share = np.linalg.norm(P @ C @ P) / np.linalg.norm(C)
    lam, beta = coef
    print(f"{chan:6s} R2 of ratings from (elevation, gain x desirability): {r2:.3f}; "
          f"share of item-covariance Frobenius norm in span(1,t): {share:.3f}; "
          f"sd(elevation) {lam.std():.2f}, sd(gain) {beta.std():.2f}, r(elev,gain) {np.corrcoef(lam,beta)[0,1]:+.2f}")
