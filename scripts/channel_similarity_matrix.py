"""Headline: pairwise similarity of the five channel grids (rgb 2026-09-17).

Builds the five 525 matrices via pkit.channels (the single source of the
adopted cooking; raw correlation-like units, core rosters, phi clipped), then reports:
  * the 5x5 off-diagonal Pearson congruence, level-removed and top-removed
    (raw native coefficients printed for reference),
    at the 525-adjective level and the 44-block level;
  * Mantel permutation p-values (adjective/block relabeling of one matrix);
  * per-channel split-half reliability (respondents/models halves) and the
    disattenuated congruence r / sqrt(rel_i * rel_j);
  * figures: level removed | (PC1 cosine |) level & top removed, all at the
    525-adjective level (rgb 2026-10-10; blocks only for display grids).
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
rel, rel525 = {}, {}
for c in CH:
    kind, X = per[c]; N = X.shape[0]; rs = {"lvl": [], "top": []}; rf = {"lvl": [], "top": []}
    for _ in range(args.halves if c != "HUMAN" else 20):
        p = rng.permutation(N); a, b = p[: N // 2], p[N // 2:]
        Ma, Mb = chn.level_removed(c, kind, X[a]), chn.level_removed(c, kind, X[b])
        Ra, Rb = chn.residual(c, kind, X[a]), chn.residual(c, kind, X[b])
        rs["lvl"].append(r_off(blockify(Ma), blockify(Mb))); rs["top"].append(r_off(blockify(Ra), blockify(Rb)))
        rf["lvl"].append(r_off(Ma, Mb)); rf["top"].append(r_off(Ra, Rb))
    rel[c] = {t: 2 * np.mean(v) / (1 + np.mean(v)) for t, v in rs.items()}      # Spearman-Brown to full n
    rel525[c] = {t: 2 * np.mean(v) / (1 + np.mean(v)) for t, v in rf.items()}
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
print("split-half reliability (Spearman-Brown to full n), 525:", {c: (round(rel525[c]["lvl"], 2), round(rel525[c]["top"], 2)) for c in CH})
_d525 = lambda Cm, t: np.array([[Cm[i, j] / np.sqrt(rel525[a][t] * rel525[b][t]) if i != j else 1.0 for j, b in enumerate(CH)] for i, a in enumerate(CH)])
show(_d525(C[("525", "lvl")], "lvl"), "525 congruence DISATTENUATED, level removed")
show(_d525(C[("525", "top")], "top"), "525 congruence DISATTENUATED, level & top removed")
show(disatt(C[("44", "lvl")], "lvl"), "44-block congruence DISATTENUATED (r / sqrt(rel_i rel_j)), level removed")
show(congr(G44_native), "44-block congruence, RAW native coefficients (no level removal; reference only)")
show(disatt(C[("44", "top")], "top"), "44-block congruence DISATTENUATED, top component removed")

# ---- figure: two adjacent tables (rgb 2026-10-10: one packed triangle was too hard
# to explain). Left = level removed, right = top component removed; lower triangle only,
# diagonal = that version's split-half reliability (525). ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, (t, title) in zip(axes, [("lvl", "A: Level removed"), ("top", "B: Level & top removed")]):
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i):
            M[i, j] = C[("525", t)][i, j]
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(i + 1):
            if i == j:
                ax.text(j, i, f"{rel525[CH[i]][t]:.2f}", ha="center", va="center", fontsize=8, style="italic", color="0.35")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if abs(M[i, j]) > .6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(CH, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CH, fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=10)
fig.text(0.44, 0.01, "Off-diagonal Pearson congruence between channel matrices (525 adjectives). Diagonal, italic: split-half reliability.", ha="center", fontsize=8)
fig.colorbar(im, ax=axes, shrink=.7, pad=.02); out = "results/persona_vectors/figs/fig_channel_similarity"
fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight"); fig.savefig(out + ".png", dpi=160, bbox_inches="tight"); print("\nwrote", out)

# ---- three-panel variant (rgb 2026-10-10): what top removal takes out, between
# the before/after tables. Middle panel is a DIFFERENT measure (cosine between
# the channels' PC1 loading vectors on 525 adjectives, level-removed matrices,
# same component remove_pc1 removes), so it gets its own sequential colormap
# and its diagonal shows PC1's share of the positive spectrum. ----
def _pc1(M):
    A = np.array(M, float); np.fill_diagonal(A, 0.0); w, V = np.linalg.eigh(A); k = np.argmax(np.abs(w))
    return V[:, k] * np.sign(V[labels.index("kind"), k]), w[k] / w[w > 0].sum()
PCv = {c: _pc1(lv[c]) for c in CH}
PCC = np.array([[PCv[a][0] @ PCv[b][0] for b in CH] for a in CH])
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), gridspec_kw={"wspace": 0.42})
panels = [("lvl", "A: Level removed", "congruence"), (None, "B: Removed component (PC1)", "pc1"), ("top", "C: Level & top removed", "congruence")]
for ax, (t, title, kind_) in zip(axes, panels):
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i):
            M[i, j] = PCC[i, j] if kind_ == "pc1" else C[("525", t)][i, j]
    cmap, vmin = ("Purples", 0) if kind_ == "pc1" else ("RdBu_r", -1)
    im_ = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=1)
    for i in range(5):
        for j in range(i + 1):
            if i == j:
                d = f"{100 * PCv[CH[i]][1]:.0f}%" if kind_ == "pc1" else f"{rel525[CH[i]][t]:.2f}"
                ax.text(j, i, d, ha="center", va="center", fontsize=8, style="italic", color="0.35")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if M[i, j] > .6 or M[i, j] < -.6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(CH, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CH, fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=10, pad=24)
    sub = "matrix congruence, 525 adjectives; diagonal: split-half reliability" if kind_ == "congruence" else "PC1 loading cosine, 525 adjectives; diagonal: PC1 share"
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="0.3")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    if t == "lvl": cax.axis("off")   # keep panel sizes equal; one congruence bar serves both tables
    else: fig.colorbar(im_, cax=cax).ax.tick_params(labelsize=7)
out3 = "results/persona_vectors/figs/fig_channel_similarity_3panel"
fig.savefig(out3 + ".pdf", dpi=300, bbox_inches="tight"); fig.savefig(out3 + ".png", dpi=160, bbox_inches="tight"); print("wrote", out3)

# ---- subfigure panels (rgb 2026-10-10): one standalone image per panel, no
# titles or letters (Quarto subcaptions carry them), identical geometry so the
# three line up in a layout-ncol=3 figure. ----
def panel(M, diag, cmap, vmin, cbar_label, out):
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_axes([0.23, 0.21, 0.56, 0.75]); cax = fig.add_axes([0.80, 0.25, 0.03, 0.67])  # fixed geometry: panels align
    im_ = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=1)
    for i in range(5):
        for j in range(i + 1):
            if i == j:
                ax.text(j, i, diag[i], ha="center", va="center", fontsize=7, style="italic", color="0.35")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > .6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(CH, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CH, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im_, cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label(cbar_label, fontsize=7)
    fig.savefig(out + ".pdf", dpi=300); fig.savefig(out + ".png", dpi=200)
    plt.close(fig); print("wrote", out)

def lower(src):
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i): M[i, j] = src[i, j]
    return M

base = "results/persona_vectors/figs/fig_channel_similarity_"
panel(lower(C[("525", "lvl")]), [f"{rel525[c]['lvl']:.2f}" for c in CH], "RdBu_r", -1, "congruence", base + "level")
panel(lower(PCC), [f"{100 * PCv[c][1]:.0f}%" for c in CH], "Purples", 0, "PC1 cosine", base + "pc1")
panel(lower(C[("525", "top")]), [f"{rel525[c]['top']:.2f}" for c in CH], "RdBu_r", -1, "congruence", base + "top")

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
