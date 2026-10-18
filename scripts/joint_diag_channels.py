"""Common axes across the five channels: one orthonormal basis Q (525 x K),
one diagonal weight vector per channel (rgb 2026-10-18).

    S_t ~ Q diag(w_t) Q',   min sum_t sum_{i!=j} (S_t - Q W_t Q')_ij^2

on the LEVEL-REMOVED matrices (no top removal: the evaluative axis is just a
row of the table; "level & top removed" = reading from row 2 down). Each S_t
is scaled to unit off-diagonal Frobenius norm, so each channel counts once
and w_tk^2 is the share of channel t's off-diagonal structure on axis k.

Fit: alternating least squares on off-diagonal entries only (JUDGE has no
meaningful diagonal; ones would soak up fit elsewhere):
  * diagonal imputation (MINRES-style): fill diag(S_t) with diag(Q W_t Q');
  * W step: exact per-channel regression of off-diagonal S_t on q_k q_k';
  * Q step: MM polar update, Q <- polar([ (A_k + cI) q_k ]_k), A_k = sum_t w_tk S_t.
Axes ordered by sum_t w_tk^2; each signed so its largest |loading| is
positive (w is invariant to the sign of q_k).

PREDICTIONS (registered 2026-10-18 before running; K=8):
 P1 axis 1 = evaluative, w > 0 in all five channels; largest share in SELF,
    smallest in REPRESENT (mirrors the unit-diag PC1 shares).
 P2 one agency/dynamism axis (bold/extraordinary vs ordinary/cautious) with
    non-trivial weight in HUMAN, JUDGE, SELF; ENACT loads instead (or also)
    on a dominance-vs-warmth axis, which JUDGE also uses.
 P3 REPRESENT puts most of its non-axis-1 weight on axes no other channel
    uses much (|w| < .1 elsewhere).
 P4 joint-fit share of each channel's off-diagonal structure with K=8 is
    within ~.1 of what its own top-8 eigencomponents capture for HUMAN,
    JUDGE, ENACT; SELF close too; REPRESENT loses the most.
 P5 level removal (raw vs level-removed congruence): HUMAN r > .9 (harmless),
    SELF r < .7 (removes a lot), JUDGE in between.

Usage: .venv/bin/python scripts/joint_diag_channels.py [--k 8] [--iters 500]
"""
import argparse, json, os
import numpy as np
from scipy.linalg import eigh
import pkit
from pkit import channels as C

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--out", default="results/persona_vectors/joint_diag")
args = ap.parse_args()

labels = pkit.load.adjectives(); CH = C.CHANNELS; n = len(labels)
OFF = ~np.eye(n, dtype=bool)
mem = {c: C.members(c, labels) for c in CH}
L = C.channel_level_removed(labels, mem)

# ---- level removal: how much does it change each channel? ----
raw = {c: C.cohort_matrix(mem[c][0], mem[c][1]) for c in CH}
lvl_r = {c: C.congruence(raw[c], L[c]) for c in CH}
print("raw vs level-removed congruence:", {c: round(v, 3) for c, v in lvl_r.items()})

S = []
for c in CH:
    A = C._zero_diag(L[c]); A /= np.linalg.norm(A[OFF]); S.append(A)
S = np.array(S)  # T x n x n, zero diag, unit off-diag norm


def gram_offdiag(Q):
    # G_kl = sum_{i!=j} q_ik q_jk q_il q_jl = (q_k.q_l)^2 - sum_i q_ik^2 q_il^2
    Q2 = Q ** 2
    return (Q.T @ Q) ** 2 - Q2.T @ Q2


def w_step(Q, St):
    G = gram_offdiag(Q)
    b = np.einsum("ik,tij,jk->tk", Q, St, Q)  # q_k' S_t q_k with zero diag
    return np.linalg.solve(G, b.T).T


def loss(Q, W):
    tot = 0.0
    for t in range(len(S)):
        M = (Q * W[t]) @ Q.T
        tot += ((S[t] - M)[OFF] ** 2).sum()
    return tot


def fit(Q):
    K = Q.shape[1]
    W = w_step(Q, S)
    prev = f = np.inf; it = 0
    for it in range(args.iters):
        # impute diagonals from the model, then MM polar step on Q
        Sf = S.copy()
        for t in range(len(S)):
            Sf[t][np.diag_indices(n)] = ((Q ** 2) * W[t]).sum(1)
        A = np.einsum("tk,tij->kij", W, Sf)
        c = max(0.0, -min(eigh(A[k], eigvals_only=True, subset_by_index=[0, 0])[0] for k in range(K))) + 1e-9
        M = np.stack([A[k] @ Q[:, k] + c * Q[:, k] for k in range(K)], 1)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
        W = w_step(Q, S)
        f = loss(Q, W)
        if prev - f < 1e-10 * max(1.0, prev):
            break
        prev = f
    return Q, W, f, it + 1


rng = np.random.default_rng(0)
# init 0: top-K eigenvectors of the summed (sign-agnostic: summed squares) matrices
w0, V0 = np.linalg.eigh(np.einsum("tij,tjk->ik", S, S))
inits = [V0[:, ::-1][:, :args.k]] + [np.linalg.qr(rng.standard_normal((n, args.k)))[0] for _ in range(args.seeds)]
fits = [fit(Q0) for Q0 in inits]
for i, (_, _, f, its) in enumerate(fits):
    print(f"init {i}: loss {f:.5f} ({its} iters)")
Q, W, f, _ = min(fits, key=lambda r: r[2])

# order axes by total weight^2; sign each axis so its largest |loading| is positive
order = np.argsort(-(W ** 2).sum(0)); Q, W = Q[:, order], W[:, order]
sg = np.sign(Q[np.abs(Q).argmax(0), np.arange(args.k)]); Q, W = Q * sg, W  # w invariant to q sign

# cross-init stability: best |cos| match of each axis in the other fits
# (only fits that reached the best loss within 1%: unconverged/local fits say nothing)
stab = np.array([np.abs(Q.T @ Qo).max(1) for Qo, _, fo, _ in fits if fo <= 1.01 * f])
print(f"{len(stab)} of {len(fits)} inits reached the best loss (within 1%)")

# per-channel share captured: joint fit vs own top-K eigencomponents
joint_share = 1 - np.array([((S[t] - (Q * W[t]) @ Q.T)[OFF] ** 2).sum() for t in range(len(S))])
own_share = []
for t in range(len(S)):
    w, V = np.linalg.eigh(S[t]); o = np.argsort(-np.abs(w))[:args.k]
    Vk = V[:, o]; ww = w_step(Vk, S[t:t + 1])[0]
    own_share.append(1 - ((S[t] - (Vk * ww) @ Vk.T)[OFF] ** 2).sum())

print(f"\nweights w_tk (rows: axes, cols: channels); share = w^2")
print("axis  " + " ".join(f"{c[:6]:>7s}" for c in CH) + "   stab   top + / top -")
axes = []
for k in range(args.k):
    o = np.argsort(Q[:, k])
    pos = [labels[i] for i in o[::-1][:8]]; neg = [labels[i] for i in o[:8]]
    axes.append({"pos": pos, "neg": neg, "w": dict(zip(CH, W[:, k].round(4).tolist())),
                 "stab_min": float(stab[:, k].min())})
    print(f"{k + 1:>4d}  " + " ".join(f"{W[t, k]:7.3f}" for t in range(len(CH)))
          + f"   {stab[:, k].min():.2f}   {', '.join(pos[:5])} / {', '.join(neg[:5])}")
print("\nshare of off-diag structure: joint K-axis fit vs own top-K")
for t, c in enumerate(CH):
    print(f"  {c:9s} joint {joint_share[t]:.3f}  own {own_share[t]:.3f}  axis1 {W[t, 0] ** 2:.3f}")

# axis 1 vs each channel's own top component
print("\n|cos| axis 1 vs each channel's own top component:",
      {c: round(abs(Q[:, 0] @ C.top_component(L[c], labels)[0]), 3) for c in CH})

os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, f"joint_diag_k{args.k}.npz"), Q=Q, W=W, channels=CH, labels=labels)
json.dump({"k": args.k, "loss": float(f), "n_converged_inits": int(len(stab)), "axes": axes, "joint_share": dict(zip(CH, joint_share.round(4).tolist())),
           "own_share": dict(zip(CH, np.round(np.array(own_share), 4).tolist())), "level_congruence": lvl_r},
          open(os.path.join(args.out, f"joint_diag_k{args.k}.json"), "w"), indent=1)
