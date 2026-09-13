"""Paper-friendly five-channel cluster grids (2026-09-12).

Rebuilds every channel's cohort-mean 525 similarity matrix from FRESH
artifacts (replacing the stale 523-era facet_channel_sims cache),
applies the core-only standing rule where a roster exists, aggregates
to blocks44, and renders:
  fig_cluster_grids.pdf  — main: one row, HUMAN labeled + 4 channels,
                           pc1-removed entry-z, congruence + ceiling
  fig_cluster_grids_full.pdf — appendix: 2x5 raw + pc1-removed
Usage: .venv/bin/python scripts/fig_cluster_grids_paper.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pkit
from pkit import measures

labels = pkit.load.adjectives()
FR = pkit.load.FRAMINGS
iu = np.triu_indices(len(labels), 1)

# ---- channel matrices (525, fresh) ----------------------------------
mats = {}
H = pkit.load.human_corr().values.copy()
np.fill_diagonal(H, 0)
mats["HUMAN"] = measures.zscore_offdiag(H)

R = pkit.load.self_matrix(which="cohort")
core = R.values.std(1) >= 0.50
Sc = np.corrcoef(R.values[core].T)
np.fill_diagonal(Sc, 0)
mats["SELF"] = measures.zscore_offdiag(Sc)

z = np.load("results/adjectives/represent_model_grids.npz", allow_pickle=True)
rn = [str(x) for x in z["names"]]
sd_by = {n: None for n in rn}
keepR = []
for i, n in enumerate(rn):
    try:
        d = json.load(open(pkit.load._self_file(n.replace("_", "/", 1))))["results"]
        X = np.array([[d[f][a]["ev"] for a in labels] for f in FR])
        if X.mean(0).std() >= 0.50:
            keepR.append(i)
    except Exception:
        pass
mats["REPRESENT"] = z["grids"][keepR].astype(np.float32).mean(0)

acc = []
for p in glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz"):
    zj = np.load(p, allow_pickle=True)
    ja = [str(a).lower() for a in zj["adjectives"]]
    B = np.asarray(zj["B"], float)
    B = 0.5 * (B + B.T)
    B = B[np.ix_([ja.index(l) for l in labels], [ja.index(l) for l in labels])]
    np.fill_diagonal(B, 0)
    acc.append(measures.zscore_offdiag(B))
mats["JUDGE"] = np.mean(acc, 0)

acc = []
for p in glob.glob("results/persona_vectors/enact_mid/*.npz"):
    m = os.path.basename(p).replace(".npz", "")
    ze = np.load(p, allow_pickle=True)
    ea = [str(a).lower() for a in ze["adjectives"]]
    E = np.asarray(ze["dir"], np.float64)[[ea.index(l) for l in labels]]
    meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
    Xe = measures.cos_sim(measures.winsorize(E, np.asarray(meta["massive_dims"], int)))
    np.fill_diagonal(Xe, 0)
    acc.append(measures.zscore_offdiag(Xe))
mats["ENACT"] = np.mean(acc, 0)

CH = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
cl = pkit.facets.clusters("blocks44")
def blockify(M):
    return pkit.facets.block(M, cl).values
grids = {c: blockify(mats[c]) for c in CH}
grids_p = {c: blockify(measures.zscore_offdiag(measures.remove_pc1(mats[c]))) for c in CH}
def congr(g):
    k = g.shape[0]; m_ = ~np.eye(k, dtype=bool)
    return np.corrcoef(g[m_], grids_p["HUMAN"][m_])[0, 1]
CEIL = 0.92  # human split-half external-match ceiling, pc1-removed (ledger 2026-09-12)
branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b)
                           for b in sorted({c["branch"] for c in cl})])[:-1]
names44 = [c["label"] for c in cl]

def render(rows, fname, labeled_first=True, figh=None):
    nr = len(rows)
    fig = plt.figure(figsize=(7.0, (1.62 if not labeled_first else 2.05) * nr))
    gs = fig.add_gridspec(nr, 6, width_ratios=[1.45, 1, 1, 1, 1, 0.06],
                          wspace=0.06, hspace=0.25)
    for ri, (tag, G) in enumerate(rows):
        for ci, c in enumerate(CH):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(G[c], cmap="RdBu_r", vmin=-2, vmax=2)
            for b in branch_breaks:
                ax.axhline(b - .5, color="k", lw=.3, alpha=.5)
                ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(c, fontsize=8, pad=3)
            if c != "HUMAN":
                r = np.corrcoef(G[c][~np.eye(44, dtype=bool)],
                                G["HUMAN"][~np.eye(44, dtype=bool)])[0, 1]
                ax.set_xlabel(f"r = {r:.2f}", fontsize=7, labelpad=2)
            if ci == 0:
                # 8 branch-band labels (44 cluster labels can't fit at
                # 5-up scale; the full key is an appendix table). Band
                # named by its largest cluster.
                edges = [0] + list(branch_breaks) + [44]
                for bi in range(len(edges) - 1):
                    lo, hi = edges[bi], edges[bi + 1]
                    band = [c_ for c_ in cl][lo:hi]
                    nm = max(band, key=lambda c_: len(c_["members"]))["label"]
                    ax.text(-1.2, (lo + hi - 1) / 2, nm, fontsize=5.5,
                            ha="right", va="center")
                if nr > 1:
                    ax.set_ylabel(tag, fontsize=8, labelpad=34)
        cax = fig.add_subplot(gs[ri, 5])
        plt.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    print("wrote", fname)

render([("top comp. removed", grids_p)], "results/persona_vectors/figs/fig_cluster_grids.pdf")
render([("raw", grids), ("top comp. removed", grids_p)],
       "results/persona_vectors/figs/fig_cluster_grids_full.pdf", labeled_first=False)
for c in CH[1:]:
    m_ = ~np.eye(44, dtype=bool)
    print(f"{c}: raw r={np.corrcoef(grids[c][m_], grids['HUMAN'][m_])[0,1]:.3f}  "
          f"pc1-removed r={np.corrcoef(grids_p[c][m_], grids_p['HUMAN'][m_])[0,1]:.3f}  (ceiling {CEIL})")
