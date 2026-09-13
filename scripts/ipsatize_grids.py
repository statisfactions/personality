"""Ipsatization poke on the cluster grids (2026-09-13, rgb probe; P16).

Question: does within-respondent standardization (C&C ipsative z) move
the HUMAN grid much? The SELF grid? Same blocks44 machinery as
fig_cluster_grids_paper.py; only HUMAN and SELF live in score space, so
only they can be ipsatized. Renders a 2x4 figure (raw / top-removed x
HUMAN raw, HUMAN ips, SELF raw, SELF ips) and prints the grid-vs-grid
correlation table.
Usage: .venv/bin/python scripts/ipsatize_grids.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pkit
from pkit import measures

labels = pkit.load.adjectives()
cl = pkit.facets.clusters("blocks44")
m44 = ~np.eye(44, dtype=bool)


def corr_grid(X):
    """X: respondents x 525 -> zero-diag entry-z corr matrix."""
    S = np.corrcoef(X.T)
    np.fill_diagonal(S, 0)
    return measures.zscore_offdiag(S)


def blockify(M):
    return pkit.facets.block(M, cl).values


# ---- HUMAN: raw respondent matrix, aligned to the 525 label order ------
Hm, hlab = pkit.load.load_human()
hlab = [l.lower() for l in hlab]
idx = [hlab.index(l) for l in labels]
Hm = Hm[:, idx]
mats = {"HUMAN raw": corr_grid(Hm),
        "HUMAN ips": corr_grid(measures.ipsatize(Hm))}
# sanity: rebuilt raw should match the cached corr json
Hc = pkit.load.human_corr().values.copy(); np.fill_diagonal(Hc, 0)
print("HUMAN raw rebuild vs cached corr json: r = %.4f" %
      measures.offdiag_corr(mats["HUMAN raw"], measures.zscore_offdiag(Hc)))

# ---- SELF: core instruct models (sd >= .5), framing-mean EVs -----------
R = pkit.load.self_matrix(which="cohort")
core = R.values.std(1) >= 0.50
Sm = R.values[core]
mats["SELF raw"] = corr_grid(Sm)
mats["SELF ips"] = corr_grid(measures.ipsatize(Sm))
print("core models n=%d" % core.sum())

CH = ["HUMAN raw", "HUMAN ips", "SELF raw", "SELF ips"]
grids = {c: blockify(mats[c]) for c in CH}
grids_p = {c: blockify(measures.zscore_offdiag(measures.remove_pc1(mats[c])))
           for c in CH}


def r(a, b):
    return np.corrcoef(a[m44], b[m44])[0, 1]


print("\n--- P16 grid-vs-grid correlations (44-block off-diagonal) ---")
for pop in ["HUMAN", "SELF"]:
    print(f"{pop}: raw vs ips          r = {r(grids[pop+' raw'], grids[pop+' ips']):.3f}")
    print(f"{pop}: ips vs top-removed  r = {r(grids[pop+' ips'], grids_p[pop+' raw']):.3f}")
    print(f"{pop}: raw vs top-removed  r = {r(grids[pop+' raw'], grids_p[pop+' raw']):.3f}")
    print(f"{pop}: ips(top-removed) vs raw(top-removed) r = "
          f"{r(grids_p[pop+' ips'], grids_p[pop+' raw']):.3f}")
print("\ncross congruence SELF vs HUMAN:")
for a in ["raw", "ips"]:
    for b in ["raw", "ips"]:
        print(f"  SELF {a} vs HUMAN {b}: no top removal r = "
              f"{r(grids['SELF '+a], grids['HUMAN '+b]):.3f}   "
              f"both top-removed r = {r(grids_p['SELF '+a], grids_p['HUMAN '+b]):.3f}")

# eigen-spectrum of the 525 matrices: how big is the top component?
print("\ntop-eigenvalue share of |spectrum| (525 zero-diag entry-z):")
for c in CH:
    w = np.linalg.eigvalsh(mats[c])
    print(f"  {c:10s} top/sum|w| = {np.abs(w).max()/np.abs(w).sum():.3f}")

# which blocks move most under ipsatization (row-wise r, raw vs ips)?
names44 = [c["label"] for c in cl]
print("\nblocks whose row profile moves most under ipsatization (row r raw vs ips):")
for pop in ["HUMAN", "SELF"]:
    a, b = grids[pop + " raw"], grids[pop + " ips"]
    rows = [np.corrcoef(np.delete(a[i], i), np.delete(b[i], i))[0, 1] for i in range(44)]
    o = np.argsort(rows)
    print(f"  {pop}: median row r = {np.median(rows):.2f}; lowest: " +
          ", ".join(f"{names44[i]} ({rows[i]:.2f})" for i in o[:6]))

# ---- figure ------------------------------------------------------------
branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b)
                           for b in sorted({c["branch"] for c in cl})])[:-1]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9))
for ri, (tag, G) in enumerate([("raw", grids), ("top comp. removed", grids_p)]):
    for ci, c in enumerate(CH):
        ax = axes[ri, ci]
        im = ax.imshow(G[c], cmap="RdBu_r", vmin=-2, vmax=2)
        for b in branch_breaks:
            ax.axhline(b - .5, color="k", lw=.3, alpha=.5)
            ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(c, fontsize=8, pad=3)
        if ci == 0:
            ax.set_ylabel(tag, fontsize=8)
        # HUMAN ips is compared to HUMAN raw; each SELF grid to the
        # like-treated HUMAN grid (raw<->raw, ips<->ips)
        if c != "HUMAN raw":
            refname = "HUMAN raw" if c in ("HUMAN ips", "SELF raw") else "HUMAN ips"
            ax.set_xlabel(f"r = {r(G[c], G[refname]):.2f} vs {refname}",
                          fontsize=6.5, labelpad=2)
fig.colorbar(axes[0, 0].images[0], ax=axes, shrink=0.6).ax.tick_params(labelsize=6)
out = "results/persona_vectors/figs/fig_cluster_grids_ipsatize"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
print("wrote", out + ".pdf/.png")
