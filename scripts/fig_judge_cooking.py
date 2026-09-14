"""JUDGE cooking-stage and size-tier grids (2026-09-13, rgb request).

Two 2x4 figures in raw correlation units on the fig_cluster_grids_paper
colorbars (raw +-.6 / top-removed +-.3), blocks44, congruence r vs HUMAN:
  fig_judge_cooking.pdf — HUMAN | phi directed | phi symmetric (adopted,
      clipped) | nearest correlation matrix (Higham) of the symmetric phi
  fig_judge_size.pdf    — HUMAN | JUDGE <=4B | 7-12B | >=27B
Per-model matrices are cooked then averaged in raw units.
Usage: .venv/bin/python scripts/fig_judge_cooking.py
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pkit
from pkit import cooking, measures

labels = pkit.load.adjectives()
cl = pkit.facets.clusters("blocks44")
m44 = ~np.eye(44, dtype=bool)
m525 = ~np.eye(len(labels), dtype=bool)
TIER = {"Aya": "7-12B", "Qwen": "<=4B", "Gemma": "<=4B", "Llama": "<=4B",
        "Phi4": "<=4B", "Gemma12": "7-12B", "Llama8": "7-12B", "Qwen7": "7-12B",
        "FalconMamba": "7-12B", "Gemma27": ">=27B", "Qwen32": ">=27B", "Gemma4": ">=27B"}


def blockify(M):
    return pkit.facets.block(M, cl).values


def center(M):
    A = M.copy(); A[m525] -= A[m525].mean(); return A


def r(a, b):
    return np.corrcoef(a[m44], b[m44])[0, 1]


def directed_phi(Pc, P):
    """Row-conditional joint P(b|a)P(a) without symmetrization."""
    J = Pc * P[:, None]
    cov = J - P[:, None] * P[None, :]
    return cov / np.sqrt((P * (1 - P))[:, None] * (P * (1 - P))[None, :])


H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
per = {"phi directed": [], "phi symmetric": [], "nearest corr.": []}
tier = {t: [] for t in ["<=4B", "7-12B", ">=27B"]}
for p in sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz")):
    name = p.split("/")[-1].split("_tom")[0]
    zj = np.load(p, allow_pickle=True)
    ja = [str(a).lower() for a in zj["adjectives"]]
    B = np.asarray(zj["B"], float)
    B = B[np.ix_([ja.index(l) for l in labels], [ja.index(l) for l in labels])]
    psi = cooking.pairs_potential(B)
    P = np.clip(np.exp(psi - np.median(psi) + np.log(0.5)), 0.01, 0.99)
    Pc = cooking.EV2P(B)
    d = np.clip(directed_phi(Pc, P), -1, 1); np.fill_diagonal(d, 0)
    s = np.clip(cooking.implied_phi(Pc, P), -1, 1); np.fill_diagonal(s, 0)
    S1 = s.copy(); np.fill_diagonal(S1, 1.0)
    n = measures.nearest_corr(S1); np.fill_diagonal(n, 0)
    per["phi directed"].append(d); per["phi symmetric"].append(s); per["nearest corr."].append(n)
    tier[TIER[name]].append(s)
    w = np.linalg.eigvalsh(S1)
    print(f"{name:12s} sym-phi min eig {w.min():+.2f} ({(w < 0).sum()} neg of 525); "
          f"NCM moved {np.linalg.norm(n - s) / np.linalg.norm(s):.3f} (rel Frob); "
          f"dir-vs-sym off-diag r {measures.offdiag_corr(d, s):.3f}")

A = {"HUMAN": H} | {k: np.mean(v, 0) for k, v in per.items()}
Bt = {"HUMAN": H} | {f"JUDGE {t} (n={len(v)})": np.mean(v, 0) for t, v in tier.items()}
branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b)
                           for b in sorted({c["branch"] for c in cl})])[:-1]
VLIM = {"raw": 0.6, "top comp. removed": 0.3}


def render(mats, fname):
    CH = list(mats)
    G = {"raw": {c: blockify(mats[c]) for c in CH},
         "top comp. removed": {c: blockify(measures.remove_pc1(center(mats[c]))) for c in CH}}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9))
    for ri, tag in enumerate(G):
        for ci, c in enumerate(CH):
            ax = axes[ri, ci]
            ax.imshow(G[tag][c], cmap="RdBu_r", vmin=-VLIM[tag], vmax=VLIM[tag])
            for b in branch_breaks:
                ax.axhline(b - .5, color="k", lw=.3, alpha=.5)
                ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(c, fontsize=8, pad=3)
            if ci == 0:
                ax.set_ylabel(tag, fontsize=8)
            if c != "HUMAN":
                ax.set_xlabel(f"r = {r(G[tag][c], G[tag]['HUMAN']):.2f}", fontsize=7, labelpad=2)
        fig.colorbar(axes[ri, 0].images[0], ax=axes[ri, :], shrink=0.85).ax.tick_params(labelsize=6)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=200)
    print("wrote", fname)
    for c in CH[1:]:
        print(f"  {c:22s} raw r={r(G['raw'][c], G['raw']['HUMAN']):.3f}  "
              f"top-removed r={r(G['top comp. removed'][c], G['top comp. removed']['HUMAN']):.3f}  "
              f"off-diag mean {G['raw'][c][m44].mean():+.3f} sd {G['raw'][c][m44].std():.3f}")


render(A, "results/persona_vectors/figs/fig_judge_cooking")
render(Bt, "results/persona_vectors/figs/fig_judge_size")
