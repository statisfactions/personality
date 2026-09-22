"""Top-removed cluster grids under grand-mean centering (current paper
recipe) vs double centering (projection off the constant direction, then
the largest remaining eigencomponent). 2026-09-22, rgb: "I think self will
look less stupid". Same blocks44 machinery, raw units, shared colorbar.
Usage: .venv/bin/python scripts/fig_cluster_grids_dc.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pkit
from pkit import channels as chn, measures

labels = pkit.load.adjectives(); n = len(labels)
J = np.eye(n) - np.ones((n, n)) / n
CH = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
mats = chn.channel_matrices(labels)
cl = pkit.facets.clusters("blocks44"); m44 = ~np.eye(44, dtype=bool)


def zd(S):
    S = S.copy(); np.fill_diagonal(S, 0); return S


def dc_top_removed(S):
    return zd(measures.remove_pc1(J @ S @ J))


rows = [("raw", {c: chn.blockify(mats[c]) for c in CH}),
        ("grand-mean\ncentered", {c: chn.blockify(chn.center(mats[c])) for c in CH}),
        ("double-\ncentered", {c: chn.blockify(zd(J @ mats[c] @ J)) for c in CH}),
        ("top removed\n(grand-mean)", {c: chn.blockify(chn.top_removed(mats[c])) for c in CH}),
        ("top removed\n(double-centered)", {c: chn.blockify(dc_top_removed(mats[c])) for c in CH})]
VL = [0.6, 0.6, 0.6, 0.3, 0.3]


def r(a, b):
    return np.corrcoef(a[m44], b[m44])[0, 1]


branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b)
                           for b in sorted({c["branch"] for c in cl})])[:-1]
fig = plt.figure(figsize=(7.0, 1.75 * len(rows)))
gs = fig.add_gridspec(len(rows), 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.06, hspace=0.3)
for ri, (tag, G) in enumerate(rows):
    for ci, c in enumerate(CH):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(G[c], cmap="RdBu_r", vmin=-VL[ri], vmax=VL[ri])
        for b in branch_breaks:
            ax.axhline(b - .5, color="k", lw=.3, alpha=.5); ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(c, fontsize=8, pad=3)
        if c != "HUMAN": ax.set_xlabel(f"r = {r(G[c], G['HUMAN']):.2f}", fontsize=7, labelpad=2)
        if ci == 0: ax.set_ylabel(tag, fontsize=7.5)
    cax = fig.add_subplot(gs[ri, 5])
    plt.colorbar(fig.axes[-2].images[0], cax=cax).ax.tick_params(labelsize=6)
out = "results/persona_vectors/figs/fig_cluster_grids_dc"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
print("wrote", out + ".pdf/.png")
for tag, G in rows:
    print(tag.replace("\n", " "), {c: round(r(G[c], G["HUMAN"]), 3) for c in CH[1:]},
          "| SELF off-diag sd %.3f" % G["SELF"][m44].std())
