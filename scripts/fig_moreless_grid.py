"""Correlation grids on the 44 medoids (rgb 2026-09-14): HUMAN | standard SELF
(framing-mean, same 5 models) | more-or-less self d (same 5) | standard SELF
from the full core cohort (n=50) as the well-powered reference. Raw units,
shared colorbars; raw + top-removed rows; congruence r vs HUMAN and between
the two n=5 grids. Rank of an n=5 grid is <= 4 — read as a sketch.
Usage: .venv/bin/python scripts/fig_moreless_grid.py
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit
from pkit import measures

REFS = ["assistant", "ai", "lm"]; ABS = ["direct", "assistant", "person", "pda", "observer", "outputs"]
labels = pkit.load.adjectives()
runs = {json.load(open(p))["model"]: json.load(open(p)) for p in sorted(glob.glob("results/adjectives/refgroup/*.json"))}
A = next(iter(runs.values()))["adjectives"]; ms = list(runs); idx = [labels.index(a) for a in A]
E = lambda m, vn: np.array([runs[m]["results"][vn][a]["ev"] for a in A])
m44 = ~np.eye(44, dtype=bool)
def grid(X):                      # X: respondents x 44
    S = np.corrcoef(X.T); np.fill_diagonal(S, 0); return S
def center(M):
    B = M.copy(); B[m44] -= B[m44].mean(); return B
def r(a, b): return np.corrcoef(a[m44], b[m44])[0, 1]

Hc = pkit.load.human_corr().values[np.ix_(idx, idx)].copy(); np.fill_diagonal(Hc, 0)
Xstd = np.array([np.mean([E(m, f) for f in ABS], 0) for m in ms])
Xd = np.array([np.mean([(E(m, f"more_{k}") - E(m, f"less_{k}")) / 2 for k in REFS], 0) for m in ms])
R = pkit.load.self_matrix(which="cohort"); core = R.values.std(1) >= 0.5
Xcoh = R.values[core][:, idx]
G = {"HUMAN": Hc, "SELF std (n=5)": grid(Xstd), "more-or-less d (n=5)": grid(Xd), f"SELF std core (n={core.sum()})": grid(Xcoh)}
Gp = {k: measures.remove_pc1(center(v)) for k, v in G.items()}
cl = pkit.facets.clusters("blocks44")
bb = np.cumsum([sum(1 for c in cl if c["branch"] == b) for b in sorted({c["branch"] for c in cl})])[:-1]
VL = {"raw": 1.0, "top comp. removed": 0.5}
fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.0))
for ri, (tag, GG) in enumerate([("raw", G), ("top comp. removed", Gp)]):
    for ci, k in enumerate(GG):
        ax = axes[ri, ci]; ax.imshow(GG[k], cmap="RdBu_r", vmin=-VL[tag], vmax=VL[tag])
        for b in bb: ax.axhline(b - .5, color="k", lw=.3, alpha=.5); ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(k, fontsize=8)
        if ci == 0: ax.set_ylabel(tag, fontsize=8)
        else: ax.set_xlabel(f"r vs HUMAN = {r(GG[k], GG['HUMAN']):.2f}", fontsize=7)
    fig.colorbar(axes[ri, 0].images[0], ax=axes[ri, :], shrink=0.85).ax.tick_params(labelsize=6)
out = "results/persona_vectors/figs/fig_moreless_grid"; fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", bbox_inches="tight", dpi=170); print("wrote", out)
print("\noff-diagonal r between grids (raw / top-removed):")
ks = list(G)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"  {ks[i]:22s} vs {ks[j]:22s}: {r(G[ks[i]], G[ks[j]]):.2f} / {r(Gp[ks[i]], Gp[ks[j]]):.2f}")
print("\noff-diag mean / sd:", {k: (round(v[m44].mean(), 2), round(v[m44].std(), 2)) for k, v in G.items()})
# spectral sketch: eigenvalue shares of the centered grids (rank <= 4 for n=5)
print("\ntop-4 |eigenvalue| shares of the centered grid:")
for k, v in G.items():
    w = np.linalg.eigvalsh(center(v)); w = np.sort(np.abs(w))[::-1]; print(f"  {k:22s} " + " ".join(f"{x:.2f}" for x in w[:4] / w.sum()))
# split-half of the n=50 cohort on the medoids, for the ceiling of a 44-medoid self grid
rng = np.random.default_rng(0); rs = []
for _ in range(200):
    p = rng.permutation(core.sum()); a, b = p[:25], p[25:]
    rs.append(r(grid(Xcoh[a]), grid(Xcoh[b])))
print(f"\ncohort split-half (25 vs 25) medoid-grid r = {np.mean(rs):.2f}; random 5-model subsets vs the full n=50 grid: ", end="")
rs5 = [r(grid(Xcoh[rng.choice(core.sum(), 5, replace=False)]), G[f'SELF std core (n={core.sum()})']) for _ in range(200)]
print(f"mean {np.mean(rs5):.2f} sd {np.std(rs5):.2f}  (what an n=5 grid can be expected to recover)")
