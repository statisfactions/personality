"""Spectral profiles of the five channels under the adopted recipe
(2026-09-22): similarity (native coefficient) vs level-removed (rows /
none / double, pkit.channels.LEVEL) vs residual (PC1 also removed).
Per channel: top eigenvalue shares, participation ratio (positive part),
and for the respondent channels Horn parallel analysis (column-permutation
null on the design matrix, 95th percentile, 20 draws).
Usage: .venv/bin/python scripts/spectral_profiles.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pkit
from pkit import channels as chn, measures

labels = pkit.load.adjectives(); n = len(labels)
CH = chn.CHANNELS
mats, mem = chn.channel_matrices(labels, with_members=True)
lev = {c: chn.level_removed(c, mem[c][0], mem[c][1]) for c in CH}
res = {c: chn.residual(c, mem[c][0], mem[c][1]) for c in CH}


def spectrum(S):
    w = np.sort(np.linalg.eigvalsh(S))[::-1]; p = w[w > 0]
    return w, p / p.sum(), p.sum() ** 2 / (p ** 2).sum()


def horn_k(X, n_perm=20, seed=0):
    rng = np.random.default_rng(seed); X = np.asarray(X, float)
    w = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]; null = []
    for _ in range(n_perm):
        P = np.array([rng.permutation(X[:, j]) for j in range(X.shape[1])]).T
        null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1])
    thr = np.percentile(null, 95, axis=0); k = 0
    while k < len(w) and w[k] > thr[k]: k += 1
    return k, thr[0]


print(f"{'channel':10s} {'stage':14s} {'l1%':>6s} {'l2%':>6s} {'l3%':>6s} {'l4-10%':>7s} {'PR':>6s} {'Horn k':>7s} {'Horn thr1':>9s}")
rows = {}
for c in CH:
    kind, X, _ = mem[c]
    for stage, S in [("similarity", mats[c]), ("level-removed", lev[c]), ("residual", res[c])]:
        w, sh, pr = spectrum(S); rows[(c, stage)] = sh
        hk = ""
        if kind == "respondents" and stage != "residual":
            Xs = np.asarray(X, float) if stage == "similarity" else np.asarray(X, float) - np.asarray(X, float).mean(1, keepdims=True)
            k, thr = horn_k(Xs); hk = f"{k:7d} {thr/n*100:9.2f}"
        print(f"{c:10s} {stage:14s} {100*sh[0]:6.1f} {100*sh[1]:6.1f} {100*sh[2]:6.1f} {100*sh[3:10].sum():7.1f} {pr:6.1f} {hk}")

fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
for ax, c in zip(axes, CH):
    for stage, ls in [("similarity", "-"), ("level-removed", "--"), ("residual", ":")]:
        ax.plot(np.arange(1, 21), 100 * rows[(c, stage)][:20], ls, marker="o", ms=2.5, label=stage)
    ax.set_title(c, fontsize=9); ax.set_xlabel("component", fontsize=8); ax.tick_params(labelsize=7)
axes[0].set_ylabel("% of positive spectrum", fontsize=8); axes[0].set_yscale("log"); axes[-1].legend(fontsize=7, frameon=False)
out = "results/persona_vectors/figs/fig_spectral_profiles"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
print("wrote", out + ".pdf/.png")
