"""Does each channel's top component match independent desirability norms, and
does self-report match MORE? (rgb 2026-10-11)

Norms: Hampson, Goldberg & John (1987, EJP 1:241-258), British social
desirability (SDY, 1-9, 55 raters, alpha .99), OCR'd from the scanned Table 1
(projects.ori.org/lrg/PDFs_papers/Category-breadth.pdf). Every value was checked
against the table's own SDY-Z column (z = 51.71*SDY - 248.85); disagreements
were read off the page images by hand. 264 of the 525 PDA terms overlap
(including spelling variants: nosey, easy-going, self-centred, extroverted).
data/hampson_1987/sdy_british_pda525.csv.

Note: our earlier "desirability" vector (rDes, partials) was HUMAN PC1 itself,
not an external rating; this is the first independent check.

Two readouts on the level-removed 525 matrices (pkit.channels.channel_level_removed):
  1. r(PC1 loadings, SDY) over the overlap: is the top component desirability?
  2. r(M_ij, z_i z_j) over overlap pairs: how much of the matrix IS desirability?
Word-bootstrap CIs (terms resampled; respondent/model sampling not included).
Usage: PYTHONPATH=scripts .venv/bin/python scripts/desirability_pc1_check.py
"""
import numpy as np, pandas as pd
import pkit
from pkit import channels as chn
labels = pkit.load.adjectives()
mats, mem = chn.channel_matrices(labels, with_members=True)
lv = chn.channel_level_removed(labels, mem)
sdy = pd.read_csv("data/hampson_1987/sdy_british_pda525.csv").set_index("adjective")["sdy_british"]
idx = [labels.index(a) for a in sdy.index]; d = sdy.values
L = {c: chn.top_component(lv[c], labels)[0][idx] for c in chn.CHANNELS}
r = {c: np.corrcoef(L[c], d)[0, 1] for c in L}
rng = np.random.default_rng(0); B = 5000; n = len(d)
boot = {c: [] for c in L}
for _ in range(B):
    s = rng.integers(0, n, n)
    for c in L: boot[c].append(np.corrcoef(L[c][s], d[s])[0, 1])
boot = {c: np.array(v) for c, v in boot.items()}
print(f"n = {n} overlapping terms")
for c in L:
    lo, hi = np.percentile(boot[c], [2.5, 97.5]); print(f"  {c:10s} r = {r[c]:.3f}  [{lo:.3f}, {hi:.3f}]")
for c in L:
    if c == "HUMAN": continue
    dd = boot[c] - boot["HUMAN"]; lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"  {c:10s} - HUMAN  {r[c]-r['HUMAN']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
# Spearman too (SDY is bounded, loadings skew)
from scipy.stats import spearmanr
print("spearman:", {c: round(spearmanr(L[c], d)[0], 3) for c in L})
print("\nmatrix-level: r(M_ij, z_i z_j) over off-diagonal pairs of the 263 terms; r^2 = share of entry variance that is desirability")
z = (d - d.mean()) / d.std(); O = np.outer(z, z); m = ~np.eye(n, dtype=bool)
for c in chn.CHANNELS:
    S = lv[c][np.ix_(idx, idx)]; rr = np.corrcoef(S[m], O[m])[0, 1]
    share = chn.top_component(lv[c], labels)[2]
    print(f"  {c:10s} r = {rr:.3f}  r^2 = {rr**2:.2f}   (PC1 share of 525 spectrum {share:.0%})")
Ss = {c: lv[c][np.ix_(idx, idx)] for c in chn.CHANNELS}
bd = {c: [] for c in chn.CHANNELS}
for _ in range(1000):
    s = np.unique(rng.integers(0, n, n)); mm = ~np.eye(len(s), dtype=bool); Ob = np.outer(z[s], z[s])[mm]
    for c in chn.CHANNELS: bd[c].append(np.corrcoef(Ss[c][np.ix_(s, s)][mm], Ob)[0, 1] ** 2)
for c in chn.CHANNELS[1:]:
    dd = np.array(bd[c]) - np.array(bd["HUMAN"]); print(f"  r^2 {c:10s} - HUMAN  95% [{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}]")
