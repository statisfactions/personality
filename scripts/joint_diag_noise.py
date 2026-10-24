"""Noise level of the joint-axes shares (rgb 2026-10-24: "what's the noise
level in this context?"). Three yardsticks, all in the figure's units
(percent of item variance, pkit.jointdiag.item_shares):

  floor     share of an arbitrary direction: (q'M0q + 1)/n with E[q'M0q] = 0,
            so 1/n = 0.19% for n = 525.
  sampling  each channel's members (people / models) split in random halves;
            the K=7 axes held fixed; full-sample SD per cell ~ SD(A - B) / 2
            (a half has twice the full-sample variance).
  crossfit  axes re-fitted on half A of every channel, scored on half B:
            matched |cos| with the reference axes, and the share drop from
            fitted (A on A) to held out (A on B) = overfitting of the
            axis choice.

Usage: .venv/bin/python scripts/joint_diag_noise.py [--splits 10] [--drop REPRESENT]
Output: results/persona_vectors/joint_diag/joint_diag_noise_k7[_noCH].json
"""
import argparse, json
import numpy as np
import pkit
from pkit import channels as C, jointdiag as J

ap = argparse.ArgumentParser()
ap.add_argument("--splits", type=int, default=10)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--drop", nargs="*", default=[])
args = ap.parse_args()
SUF = "".join(f"_no{c}" for c in args.drop)
D = "results/persona_vectors/joint_diag/"
ref = np.load(D + "joint_diag_k7" + SUF + ".npz", allow_pickle=True)
Qr = ref["Q"]; CH = [str(c) for c in ref["channels"]]; K = Qr.shape[1]
labels = pkit.load.adjectives(); n = len(labels)
mem = {c: C.members(c, labels) for c in CH}
full = {c: C.level_removed(c, mem[c][0], mem[c][1]) for c in CH}
S_full = np.array([100 * J.item_shares(Qr, full[c])[0] for c in CH])   # T x K percent


def halves(rng):
    A, B = {}, {}
    for c in CH:
        kind, X, _ = mem[c]; X = np.asarray(X)
        p = rng.permutation(len(X)); h = len(X) // 2
        A[c] = C.level_removed(c, kind, X[p[:h]]); B[c] = C.level_removed(c, kind, X[p[h:2 * h]])
    return A, B


rng = np.random.default_rng(0)
diffs, cosA, fitA, heldB = [], [], [], []
for s in range(args.splits):
    A, B = halves(rng)
    sA = np.array([100 * J.item_shares(Qr, A[c])[0] for c in CH])
    sB = np.array([100 * J.item_shares(Qr, B[c])[0] for c in CH])
    diffs.append(sA - sB)
    SA = np.array([J.unit_offdiag(A[c]) for c in CH])
    Qa, Wa, f, it = J.fit(SA, J.sumsq_axes(SA, K), args.iters)
    perm, cos = J.match(Qr, Qa); Qa = Qa[:, perm]          # reference order
    cosA.append(cos)
    fitA.append([100 * J.item_shares(Qa, A[c])[0] for c in CH])
    heldB.append([100 * J.item_shares(Qa, B[c])[0] for c in CH])
    print(f"split {s}: loss {f:.4f} ({it} it)  matched |cos| " + " ".join(f"{x:.2f}" for x in cos), flush=True)

diffs, cosA, fitA, heldB = map(np.array, (diffs, cosA, fitA, heldB))
sd_full = diffs.std(0, ddof=1) / 2
print(f"\nfloor (arbitrary direction): {100 / n:.2f}%")
print("\nfull-data share +- sampling SD (axes fixed), percent")
print("axis  " + " ".join(f"{c[:9]:>14s}" for c in CH))
for k in range(K):
    print(f"{k + 1:>4d}  " + " ".join(f"{S_full[t, k]:6.1f} +-{sd_full[t, k]:4.1f}  " for t in range(len(CH))))
print("\naxis replication across halves: matched |cos| to reference, median [min]")
for k in range(K):
    print(f"  axis {k + 1}: {np.median(cosA[:, k]):.2f} [{cosA[:, k].min():.2f}]")
print("\ncross-fit: share on half A's own axes, fitted on A vs held-out B (mean over splits)")
for t, c in enumerate(CH):
    print(f"  {c:9s} fitted " + " ".join(f"{x:5.1f}" for x in fitA[:, t].mean(0))
          + "\n            held  " + " ".join(f"{x:5.1f}" for x in heldB[:, t].mean(0)))
json.dump({"floor_pct": 100 / n, "channels": CH, "share_full": S_full.round(2).tolist(),
           "sd_full": sd_full.round(2).tolist(), "cos_median": np.median(cosA, 0).round(3).tolist(),
           "cos_min": cosA.min(0).round(3).tolist(), "fitted_A": fitA.mean(0).round(2).tolist(),
           "heldout_B": heldB.mean(0).round(2).tolist(), "splits": args.splits},
          open(D + "joint_diag_noise_k7" + SUF + ".json", "w"), indent=1)
