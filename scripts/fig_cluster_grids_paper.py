"""Paper-friendly five-channel cluster grids (2026-09-12; raw units 2026-09-13).

Rebuilds every channel's cohort-mean 525 similarity matrix from FRESH
artifacts, applies the core-only standing rule where a roster exists,
aggregates to blocks44, and renders:
  fig_cluster_grids.pdf  — main: one row, HUMAN labeled + 4 channels,
                           top component removed, congruence + ceiling
  fig_cluster_grids_full.pdf — appendix: 2x5 raw + top-removed
Units: every channel is now a correlation-like coefficient in [-1, 1]
(HUMAN/SELF Pearson r; REPRESENT/ENACT column-centered cosine; JUDGE
implied phi, clipped to [-1, 1]), so the grids are drawn in RAW units on
a shared colorbar — no entry-z. Per-model matrices are averaged in raw
units (previously entry-z per model, then averaged). Congruence r is
affine-invariant, so the only numbers that move are from the averaging
change and the phi clip; both old and new are printed.
Usage: .venv/bin/python scripts/fig_cluster_grids_paper.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pkit
from pkit import measures

labels = pkit.load.adjectives()
FR = pkit.load.FRAMINGS
PHI_CLIP = 1.0  # phi outside [-1,1] = incoherent implied joint (Aya/Qwen-3B tails)

# ---- channel matrices from pkit.channels (single source of the cooking) ----
from pkit import channels as chn
mats, mem = chn.channel_matrices(labels, with_members=True)
core_n = mem["SELF"][1].shape[0]; keepR = mem["REPRESENT"][2]; nJ = mem["JUDGE"][1].shape[0]; nE = mem["ENACT"][1].shape[0]
PHI_CLIP = chn.PHI_CLIP
# legacy entry-z-averaged cohort means, printed for comparison only
mats_z = {c: np.mean([measures.zscore_offdiag(m) for m in mem[c][1]], 0) for c in ["JUDGE", "ENACT"]}
_z = np.load(chn.REPRESENT_CACHE, allow_pickle=True); _rn = [str(x) for x in _z["names"]]
mats_z["REPRESENT"] = _z["grids"][[_rn.index(n) for n in keepR]].astype(np.float32).mean(0)

CH = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
cl = pkit.facets.clusters("blocks44")
m44 = ~np.eye(44, dtype=bool)


blockify, center = chn.blockify, chn.center
grids = {c: blockify(mats[c]) for c in CH}
# top-component removal on the grand-mean-centered matrix (pkit.channels.top_removed)
grids_p = {c: blockify(chn.top_removed(mats[c])) for c in CH}


def r(a, b):
    return np.corrcoef(a[m44], b[m44])[0, 1]


CEIL = 0.92  # human split-half external-match ceiling, top-removed (ledger 2026-09-12)
branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b)
                           for b in sorted({c["branch"] for c in cl})])[:-1]
names44 = [c["label"] for c in cl]
VLIM = {"raw": 0.6, "top comp. removed": 0.3}


def render(rows, fname, labeled_first=True):
    nr = len(rows)
    fig = plt.figure(figsize=(7.0, (1.62 if not labeled_first else 2.05) * nr))
    gs = fig.add_gridspec(nr, 6, width_ratios=[1.45, 1, 1, 1, 1, 0.06],
                          wspace=0.06, hspace=0.25)
    for ri, (tag, G) in enumerate(rows):
        v = VLIM[tag]
        for ci, c in enumerate(CH):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(G[c], cmap="RdBu_r", vmin=-v, vmax=v)
            for b in branch_breaks:
                ax.axhline(b - .5, color="k", lw=.3, alpha=.5)
                ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(c, fontsize=8, pad=3)
            if c != "HUMAN":
                ax.set_xlabel(f"r = {r(G[c], G['HUMAN']):.2f}", fontsize=7, labelpad=2)
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
                ax.set_ylabel(tag, fontsize=8, labelpad=40)
        cax = fig.add_subplot(gs[ri, 5])
        plt.colorbar(fig.axes[-2].images[0], cax=cax).ax.tick_params(labelsize=6)
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    print("wrote", fname)


render([("top comp. removed", grids_p)], "results/persona_vectors/figs/fig_cluster_grids.pdf")
render([("raw", grids), ("top comp. removed", grids_p)],
       "results/persona_vectors/figs/fig_cluster_grids_full.pdf", labeled_first=False)
RECIPES = """Per-channel cooking (all 525 adjectives). UNITS: every grid is a
correlation-like coefficient in [-1, 1], drawn in raw units on a shared
colorbar (raw row +-%.1f, top-removed row +-%.1f). No entry-z anywhere
(2026-09-13; congruence r is affine-invariant, so this changes colors, not
numbers, except that per-model matrices are now averaged in raw units).
NOTE: no within-respondent (ipsative) centering anywhere — HUMAN and SELF are
  correlations of RAW ratings across respondents (correlation standardizes
  each item across respondents, not within them). Elevation's influence is
  handled at the matrix level by the top-component-removal row, identically
  for both populations; ipsatizing first would impose the Clemans constraint
  on both matrices (see fig_cluster_grids_ipsatize for what that does).
HUMAN: raw item Pearson r over 700 ESCS respondents; zero diagonal.
SELF: framing-mean EVs, core instruct models (within-model SD >= 0.5, n=%d);
  item Pearson r over models-as-respondents; zero diagonal.
REPRESENT: per-model mid-layer activations at the read position (pers framing),
  massive dims winsorized (meta or 20x-median rule); column-centered cosine;
  zero diagonal; mean over core models (n=%d).
JUDGE: per-model tom_likely EV matrix B -> pairs-only potential psi ->
  level pinned to medP=.5 -> implied phi (EV/8 map), CLIPPED to [-1, 1]
  (an implied phi outside the coefficient's range marks an incoherent
  implied joint; affects 4-5%% of entries for Aya and Qwen-3B, <0.3%%
  elsewhere); mean (n=%d).
ENACT: per-model persona vectors, meta massive dims winsorized;
  column-centered cosine; zero diagonal; mean (n=%d).
Top-component removal: subtract the off-diagonal mean, then the largest
eigencomponent of each 525^2 matrix, BEFORE block aggregation (same
eigenvector as the former entry-z path). Blocks: 44 pole-respecting Ward
clusters (blocks44), branch-ordered; band labels = largest cluster's human
medoid. Congruence r: off-diagonal Pearson vs HUMAN at block level.
Human split-half external-match ceiling: raw .975 / top-removed .92.""" % (
    VLIM["raw"], VLIM["top comp. removed"], core_n, len(keepR), nJ, nE)
open("results/persona_vectors/figs/fig_cluster_grids_recipe.txt", "w").write(RECIPES)
print(RECIPES)
print("\ncongruence vs HUMAN (44-block). 'legacy' = entry-z per model then averaged:")
for c in CH[1:]:
    line = f"{c:10s} raw r={r(grids[c], grids['HUMAN']):.3f}  top-removed r={r(grids_p[c], grids_p['HUMAN']):.3f}"
    if c in mats_z:
        gz = blockify(mats_z[c]); gzp = blockify(measures.remove_pc1(center(mats_z[c])))
        line += f"   | legacy raw {r(gz, grids['HUMAN']):.3f}  top-removed {r(gzp, grids_p['HUMAN']):.3f}"
    print(line + f"  (ceiling {CEIL})")
print("\nraw-unit off-diagonal summaries (44-block): mean / sd / range")
for c in CH:
    g = grids[c][m44]
    print(f"  {c:10s} {g.mean():+.3f} / {g.std():.3f} / [{g.min():+.2f}, {g.max():+.2f}]")
