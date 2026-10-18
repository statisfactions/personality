"""How much do the common-axes table's design choices matter? (rgb 2026-10-18)

Reference: joint off-diagonal diagonalization, unit off-diagonal norm, K=8
(scripts/joint_diag_channels.py). Alternatives:
  JD-l1      same fit, DISTATIS normalization (largest |eigenvalue| = 1)
  JD-K6/K10  same fit, other K
  DISTATIS   textbook (l1 normalization, RV weights, compromise eigenvectors)
  DISTATIS-f same, unit-norm tables
  SUMSQ      top eigenvectors of sum_t S_t^2 ("PCA of the stacked spectra")
  MEAN       top |eigenvalue| eigenvectors of the plain average matrix
Shared scoring: every method's axes get weights from the same off-diagonal
regression on the unit-norm tables, then are Hungarian-matched to the
reference axes by |cos|.

PREDICTIONS (registered 2026-10-18 before running):
 Q1 axis 1 (evaluative) matched |cos| > .95 in every method.
 Q2 8-dim subspace overlap with the reference > .8 for every K=8 method,
    but the eigen methods (DISTATIS, SUMSQ, MEAN) rotate axes 2-8: matched
    |cos| .5-.9 there, since variance-ordered eigenvectors aren't pinned down
    by channels weighting the axes differently.
 Q3 MEAN loses at least one REPRESENT-private axis (2, 6, 7): averaging
    dilutes structure only one channel has.
 Q4 JD-l1 upweights REPRESENT (smallest l1 relative to its norm) but the
    table is substantively the same: matched |cos| > .9 on axes 1-5.
 Q5 JD-K6 keeps axes 1-5 (matched |cos| > .9); JD-K10 keeps all 8.
 Q6 the agency (3) and strict-cheerful (4) axes appear as separate axes in
    every JD variant; in the eigen methods at least one is a mixture of the two.

Usage: .venv/bin/python scripts/joint_diag_robustness.py
"""
import json, os
import numpy as np
import pkit
from pkit import channels as C, jointdiag as J

OUT = "results/persona_vectors/joint_diag"
labels = pkit.load.adjectives(); CH = C.CHANNELS; n = len(labels)
mem = {c: C.members(c, labels) for c in CH}
L = C.channel_level_removed(labels, mem)
Sf = np.array([J.unit_offdiag(L[c]) for c in CH])
Sl = np.array([J.unit_lambda1(L[c]) for c in CH])
print("relative weight under l1 vs unit norm (off-diag norm after l1 scaling):",
      {c: round(float(np.linalg.norm(Sl[t][J.offmask(n)])), 2) for t, c in enumerate(CH)})

ref = np.load(os.path.join(OUT, "joint_diag_k8.npz"))
Qr = ref["Q"]; Wr = J.weights(Qr, Sf)

rng = np.random.default_rng(1)
methods = {}
Q, W, f, it = J.fit(Sl, J.sumsq_axes(Sl, 8)); methods["JD-l1"] = Q
Qx, _, fx, _ = J.fit(Sl, np.linalg.qr(rng.standard_normal((n, 8)))[0])
print(f"JD-l1 loss {f:.5f} (eig start) vs {fx:.5f} (random start); same optimum: {abs(f - fx) < 1e-3 * f}")
for K in (6, 10):
    methods[f"JD-K{K}"] = J.fit(Sf, J.sumsq_axes(Sf, K))[0]
Qd, a, RV = J.distatis_axes(Sl, 8); methods["DISTATIS"] = Qd
print("DISTATIS alpha:", dict(zip(CH, a.round(3))), "\nRV:\n", RV.round(2))
methods["DISTATIS-f"] = J.distatis_axes(Sf, 8)[0]
methods["SUMSQ"] = J.sumsq_axes(Sf, 8)
methods["MEAN"] = J.mean_axes(Sf, 8)

rep = {}
print("\nmatched |cos| with reference axes 1-8 | 8-dim overlap | table r (matched w^2) | per-channel share")
print(f"{'REF':10s}  {'':47s}  share " + " ".join(f"{s:.2f}" for s in J.share(Qr, Wr, Sf)))
for name, Q in methods.items():
    W = J.weights(Q, Sf); Q, W = J.order_and_sign(Q, W)
    K = Q.shape[1]
    if K >= 8:
        perm, mc = J.match(Qr, Q)
        tr = float(np.corrcoef((Wr ** 2).ravel(), (W[:, perm] ** 2).ravel())[0, 1])
    else:
        permT, mcT = J.match(Q, Qr)          # each of its K axes to a reference axis
        mc = np.full(8, np.nan); mc[permT] = mcT
        perm = None; tr = float("nan")
    ov = J.subspace_overlap(Qr, Q[:, :8] if K >= 8 else Q)
    sh = J.share(Q, W, Sf)
    print(f"{name:10s}  " + " ".join("  -- " if np.isnan(x) else f"{x:.2f}" for x in mc)
          + f"   {ov:.2f}   {tr:5.2f}   share " + " ".join(f"{s:.2f}" for s in sh))
    rep[name] = {"matched_cos": [None if np.isnan(x) else round(float(x), 3) for x in mc],
                 "overlap": round(ov, 3), "table_r": None if np.isnan(tr) else round(tr, 3),
                 "share": dict(zip(CH, sh.round(3).tolist())),
                 "axes": [{"pos": [labels[i] for i in np.argsort(-Q[:, k])[:6]],
                           "neg": [labels[i] for i in np.argsort(Q[:, k])[:6]],
                           "w": dict(zip(CH, W[:, k].round(3).tolist()))} for k in range(K)]}
    if name in ("DISTATIS", "SUMSQ", "MEAN", "JD-l1"):
        for k in range(min(K, 8)):
            o = np.argsort(Q[:, k])
            print(f"     {k + 1}: " + " ".join(f"{W[t, k]:5.2f}" for t in range(len(CH)))
                  + f"  {', '.join(labels[i] for i in o[::-1][:4])} / {', '.join(labels[i] for i in o[:4])}")

json.dump(rep, open(os.path.join(OUT, "joint_diag_robustness.json"), "w"), indent=1)
