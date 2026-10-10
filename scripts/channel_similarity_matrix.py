"""Headline: pairwise similarity of the five channel grids (rgb 2026-09-17).

Builds the five 525 matrices via pkit.channels (the single source of the
adopted cooking; raw correlation-like units, core rosters, phi clipped), then reports:
  * the 5x5 off-diagonal Pearson congruence, level-removed and top-removed
    (raw native coefficients printed for reference),
    at the 525-adjective level and the 44-block level;
  * Mantel permutation p-values (adjective/block relabeling of one matrix);
  * per-channel split-half reliability (respondents/models halves) and the
    disattenuated congruence r / sqrt(rel_i * rel_j);
  * a figure: two adjacent lower-triangle tables, level removed | top removed (44-block).
Usage: .venv/bin/python scripts/channel_similarity_matrix.py [--perms 2000]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit
from pkit import measures

ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--halves", type=int, default=100)
args = ap.parse_args()
labels = pkit.load.adjectives(); FR = pkit.load.FRAMINGS; n = len(labels)
cl = pkit.facets.clusters("blocks44"); m44 = ~np.eye(44, dtype=bool); m525 = ~np.eye(n, dtype=bool)
rng = np.random.default_rng(0)
CH = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]

from pkit.channels import center, top_removed, blockify, congruence as r_off
def zero_diag(S): S = np.array(S, float); np.fill_diagonal(S, 0); return S

# ---- the five channels: members + cohort matrices from pkit.channels ----
from pkit import channels as chn
mats, mem = chn.channel_matrices(labels, with_members=True)
per = {c: (mem[c][0], mem[c][1]) for c in CH}
ns = {c: per[c][1].shape[0] for c in CH}
print("cohort sizes:", ns)
cohort_matrix = chn.cohort_matrix

# ---- congruence matrices at both levels, level-removed and top-removed ----
res = chn.channel_residuals(labels, mem)   # adopted recipe: level removed per channel, then PC1
lv = chn.channel_level_removed(labels, mem)  # summary first row = level removed (rgb 2026-10-10)
G525 = {"lvl": lv, "top": {c: res[c] for c in CH}}
G44 = {"lvl": {c: blockify(lv[c]) for c in CH}, "top": {c: blockify(res[c]) for c in CH}}
G44_native = {c: blockify(mats[c]) for c in CH}  # raw native coefficients, printout only
def congr(G):
    return np.array([[r_off(G[a], G[b]) for b in CH] for a in CH])
C = {(lvl, t): congr(G) for lvl, GG in [("525", G525), ("44", G44)] for t, G in GG.items()}

# ---- Mantel p (relabel one matrix, both axes) ----
def mantel_p(A, B, perms):
    obs = r_off(A, B); k = len(A); cnt = 0
    for _ in range(perms):
        p = rng.permutation(k); cnt += r_off(A, B[np.ix_(p, p)]) >= obs
    return (cnt + 1) / (perms + 1)
P44 = {t: np.array([[mantel_p(G44[t][a], G44[t][b], args.perms) if i < j else np.nan for j, b in enumerate(CH)] for i, a in enumerate(CH)]) for t in ["lvl", "top"]}
P525 = {t: np.array([[mantel_p(G525[t][a], G525[t][b], max(200, args.perms // 10)) if i < j else np.nan for j, b in enumerate(CH)] for i, a in enumerate(CH)]) for t in ["lvl", "top"]}

# ---- split-half reliability per channel (44-block), level-removed and top-removed ----
def half_matrices(kind, X, idx): return zero_diag(np.corrcoef(X[idx].T)) if kind == "respondents" else X[idx].astype(np.float64).mean(0)
rel = {}
for c in CH:
    kind, X = per[c]; N = X.shape[0]; rs = {"lvl": [], "top": []}
    for _ in range(args.halves if c != "HUMAN" else 20):
        p = rng.permutation(N); a, b = p[: N // 2], p[N // 2:]
        Ma, Mb = chn.level_removed(c, kind, X[a]), chn.level_removed(c, kind, X[b])
        rs["lvl"].append(r_off(blockify(Ma), blockify(Mb))); rs["top"].append(r_off(blockify(chn.residual(c, kind, X[a])), blockify(chn.residual(c, kind, X[b]))))
    rel[c] = {t: 2 * np.mean(v) / (1 + np.mean(v)) for t, v in rs.items()}      # Spearman-Brown to full n
def disatt(Cm, t): return np.array([[Cm[i, j] / np.sqrt(rel[a][t] * rel[b][t]) if i != j else 1.0 for j, b in enumerate(CH)] for i, a in enumerate(CH)])

def show(M, title, fmt="{:7.3f}"):
    print(f"\n{title}\n{'':10s}" + "".join(f"{c:>10s}" for c in CH))
    for i, a in enumerate(CH): print(f"{a:10s}" + "".join(("       nan" if np.isnan(M[i, j]) else " " * (10 - len(fmt.format(M[i, j]))) + fmt.format(M[i, j])) for j in range(5)))
for lvl, Cd in [("44-block", {t: C[("44", t)] for t in ["lvl", "top"]}), ("525-adjective", {t: C[("525", t)] for t in ["lvl", "top"]})]:
    for t in ["lvl", "top"]: show(Cd[t], f"{lvl} congruence, {'level removed' if t == 'lvl' else 'top removed (level + top component removed)'}")
show(P44["lvl"], f"Mantel p, 44-block level-removed ({args.perms} relabelings; floor {1/(args.perms+1):.4f})", "{:.4f}")
show(P44["top"], f"Mantel p, 44-block top-removed ({args.perms} relabelings)", "{:.4f}")
show(P525["top"], f"Mantel p, 525 top-removed ({max(200, args.perms // 10)} relabelings; floor {1/(max(200, args.perms // 10)+1):.4f})", "{:.4f}")
print("\nsplit-half reliability (Spearman-Brown to full n), 44-block:", {c: (round(rel[c]["lvl"], 2), round(rel[c]["top"], 2)) for c in CH})
show(disatt(C[("44", "lvl")], "lvl"), "44-block congruence DISATTENUATED (r / sqrt(rel_i rel_j)), level removed")
show(congr(G44_native), "44-block congruence, RAW native coefficients (no level removal; reference only)")
show(disatt(C[("44", "top")], "top"), "44-block congruence DISATTENUATED, top component removed")

# ---- figure: two adjacent tables (rgb 2026-10-10: one packed triangle was too hard
# to explain). Left = level removed, right = top component removed; lower triangle only,
# diagonal = that version's split-half reliability (44-block). ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, (t, title) in zip(axes, [("lvl", "Level removed"), ("top", "Top removed")]):
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i):
            M[i, j] = C[("44", t)][i, j]
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(i + 1):
            if i == j:
                ax.text(j, i, f"{rel[CH[i]][t]:.2f}", ha="center", va="center", fontsize=8, style="italic", color="0.35")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if abs(M[i, j]) > .6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(CH, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CH, fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=10)
fig.text(0.44, 0.01, "Off-diagonal Pearson congruence between channel matrices (44 blocks). Diagonal, italic: split-half reliability.", ha="center", fontsize=8)
fig.colorbar(im, ax=axes, shrink=.7, pad=.02); out = "results/persona_vectors/figs/fig_channel_similarity"
fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight"); fig.savefig(out + ".png", dpi=160, bbox_inches="tight"); print("\nwrote", out)

# ---- reasonableness of top-component removal: are the removed axes the same axis? ----
print("\ntop component of each centered 525 matrix: eigenvalue share, cosine with the others, with the human evaluation axis, and with uniform")
tops, shares = {}, {}
for c in CH:
    A = center(mats[c]); w, V = np.linalg.eigh(A); k = np.argmax(np.abs(w)); v = V[:, k] * np.sign(V[labels.index("kind"), k])
    tops[c] = v; shares[c] = abs(w[k]) / np.abs(w).sum()
Hfull = zero_diag(pkit.load.human_corr().values.copy()); wh, Vh = np.linalg.eigh(Hfull); evax = Vh[:, np.argmax(wh)]; evax *= np.sign(evax[labels.index("kind")])
uni = np.ones(n) / np.sqrt(n)
print(f"{'':10s}" + "".join(f"{c:>10s}" for c in CH) + f"{'human-eval':>11s}{'uniform':>9s}{'share':>7s}")
for a in CH:
    print(f"{a:10s}" + "".join(f"{abs(float(tops[a] @ tops[b])):10.2f}" for b in CH) + f"{abs(float(tops[a] @ evax)):11.2f}{abs(float(tops[a] @ uni)):9.2f}{shares[a]:7.2f}")
# second components too (is anything else shared?)
print("\nsecond component cosines (|cos|):")
sec = {}
for c in CH:
    A = center(mats[c]); w, V = np.linalg.eigh(A); o = np.argsort(-np.abs(w)); sec[c] = V[:, o[1]]
print(f"{'':10s}" + "".join(f"{c:>10s}" for c in CH))
for a in CH: print(f"{a:10s}" + "".join(f"{abs(float(sec[a] @ sec[b])):10.2f}" for b in CH))
