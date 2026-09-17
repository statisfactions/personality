"""Headline: pairwise similarity of the five channel grids (rgb 2026-09-17).

Rebuilds the five 525 matrices exactly as fig_cluster_grids_paper.py does
(raw correlation-like units, core rosters, phi clipped), then reports:
  * the 5x5 off-diagonal Pearson congruence, raw and top-component-removed,
    at the 525-adjective level and the 44-block level;
  * Mantel permutation p-values (adjective/block relabeling of one matrix);
  * per-channel split-half reliability (respondents/models halves) and the
    disattenuated congruence r / sqrt(rel_i * rel_j);
  * a figure: lower triangle raw, upper triangle top-removed (44-block).
Usage: .venv/bin/python scripts/channel_similarity_matrix.py [--perms 2000]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit
from pkit import cooking, measures

ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--halves", type=int, default=100)
args = ap.parse_args()
labels = pkit.load.adjectives(); FR = pkit.load.FRAMINGS; n = len(labels)
cl = pkit.facets.clusters("blocks44"); m44 = ~np.eye(44, dtype=bool); m525 = ~np.eye(n, dtype=bool)
rng = np.random.default_rng(0)
CH = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]

def zero_diag(S): S = np.array(S, float); np.fill_diagonal(S, 0); return S
def center(M): B = M.copy(); B[~np.eye(len(M), dtype=bool)] -= B[~np.eye(len(M), dtype=bool)].mean(); return B
def top_removed(M): return measures.remove_pc1(center(M))
def blockify(M): return pkit.facets.block(M, cl).values
def r_off(A, B):
    m = ~np.eye(len(A), dtype=bool); return np.corrcoef(A[m], B[m])[0, 1]

# ---- per-respondent / per-model matrices (units as in the paper figure) ----
per = {}
Hm, hlab = pkit.load.load_human(); hlab = [l.lower() for l in hlab]; Hm = Hm[:, [hlab.index(l) for l in labels]]
per["HUMAN"] = ("respondents", Hm)                      # respondents x 525 -> corr
R = pkit.load.self_matrix(which="cohort"); core = R.values.std(1) >= 0.50
per["SELF"] = ("respondents", R.values[core])           # models x 525 -> corr
z = np.load("results/adjectives/represent_model_grids.npz", allow_pickle=True); rn = [str(x) for x in z["names"]]
keepR = []
for i, nm in enumerate(rn):
    try:
        d = json.load(open(pkit.load._self_file(nm.replace("_", "/", 1))))["results"]
        if np.array([[d[f][a]["ev"] for a in labels] for f in FR]).mean(0).std() >= 0.50: keepR.append(i)
    except Exception: pass
per["REPRESENT"] = ("matrices", z["cos"][keepR].astype(np.float32))
acc = []
for p in glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz"):
    zj = np.load(p, allow_pickle=True); ja = [str(a).lower() for a in zj["adjectives"]]
    B = np.asarray(zj["B"], float)[np.ix_([ja.index(l) for l in labels], [ja.index(l) for l in labels])]
    psi = cooking.pairs_potential(B); P = np.clip(np.exp(psi - np.median(psi) + np.log(0.5)), 0.01, 0.99)
    acc.append(zero_diag(np.clip(cooking.implied_phi(cooking.EV2P(B), P), -1, 1)))
per["JUDGE"] = ("matrices", np.array(acc))
acc = []
for p in glob.glob("results/persona_vectors/enact_mid/*.npz"):
    m = os.path.basename(p).replace(".npz", ""); ze = np.load(p, allow_pickle=True); ea = [str(a).lower() for a in ze["adjectives"]]
    E = np.asarray(ze["dir"], np.float64)[[ea.index(l) for l in labels]]; meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
    acc.append(zero_diag(measures.cos_sim(measures.winsorize(E, np.asarray(meta["massive_dims"], int)))))
per["ENACT"] = ("matrices", np.array(acc))

def cohort_matrix(kind, X):
    return zero_diag(np.corrcoef(X.T)) if kind == "respondents" else X.astype(np.float64).mean(0)
mats = {c: cohort_matrix(*per[c]) for c in CH}
ns = {c: per[c][1].shape[0] for c in CH}
print("cohort sizes:", ns)

# ---- congruence matrices at both levels, raw and top-removed ----
G525 = {"raw": {c: mats[c] for c in CH}, "top": {c: top_removed(mats[c]) for c in CH}}
G44 = {"raw": {c: blockify(mats[c]) for c in CH}, "top": {c: blockify(top_removed(mats[c])) for c in CH}}
def congr(G):
    return np.array([[r_off(G[a], G[b]) for b in CH] for a in CH])
C = {(lvl, t): congr(G) for lvl, GG in [("525", G525), ("44", G44)] for t, G in GG.items()}

# ---- Mantel p (relabel one matrix, both axes) ----
def mantel_p(A, B, perms):
    obs = r_off(A, B); k = len(A); cnt = 0
    for _ in range(perms):
        p = rng.permutation(k); cnt += r_off(A, B[np.ix_(p, p)]) >= obs
    return (cnt + 1) / (perms + 1)
P44 = {t: np.array([[mantel_p(G44[t][a], G44[t][b], args.perms) if i < j else np.nan for j, b in enumerate(CH)] for i, a in enumerate(CH)]) for t in ["raw", "top"]}
P525 = {t: np.array([[mantel_p(G525[t][a], G525[t][b], max(200, args.perms // 10)) if i < j else np.nan for j, b in enumerate(CH)] for i, a in enumerate(CH)]) for t in ["raw", "top"]}

# ---- split-half reliability per channel (44-block), raw and top-removed ----
def half_matrices(kind, X, idx): return zero_diag(np.corrcoef(X[idx].T)) if kind == "respondents" else X[idx].astype(np.float64).mean(0)
rel = {}
for c in CH:
    kind, X = per[c]; N = X.shape[0]; rs = {"raw": [], "top": []}
    for _ in range(args.halves if c != "HUMAN" else 20):
        p = rng.permutation(N); a, b = p[: N // 2], p[N // 2:]
        Ma, Mb = half_matrices(kind, X, a), half_matrices(kind, X, b)
        rs["raw"].append(r_off(blockify(Ma), blockify(Mb))); rs["top"].append(r_off(blockify(top_removed(Ma)), blockify(top_removed(Mb))))
    rel[c] = {t: 2 * np.mean(v) / (1 + np.mean(v)) for t, v in rs.items()}      # Spearman-Brown to full n
def disatt(Cm, t): return np.array([[Cm[i, j] / np.sqrt(rel[a][t] * rel[b][t]) if i != j else 1.0 for j, b in enumerate(CH)] for i, a in enumerate(CH)])

def show(M, title, fmt="{:7.3f}"):
    print(f"\n{title}\n{'':10s}" + "".join(f"{c:>10s}" for c in CH))
    for i, a in enumerate(CH): print(f"{a:10s}" + "".join(("       nan" if np.isnan(M[i, j]) else " " * (10 - len(fmt.format(M[i, j]))) + fmt.format(M[i, j])) for j in range(5)))
for lvl, Cd in [("44-block", {t: C[("44", t)] for t in ["raw", "top"]}), ("525-adjective", {t: C[("525", t)] for t in ["raw", "top"]})]:
    for t in ["raw", "top"]: show(Cd[t], f"{lvl} congruence, {'raw' if t == 'raw' else 'top component removed'}")
show(P44["raw"], f"Mantel p, 44-block raw ({args.perms} relabelings; floor {1/(args.perms+1):.4f})", "{:.4f}")
show(P44["top"], f"Mantel p, 44-block top-removed ({args.perms} relabelings)", "{:.4f}")
show(P525["top"], f"Mantel p, 525 top-removed ({max(200, args.perms // 10)} relabelings; floor {1/(max(200, args.perms // 10)+1):.4f})", "{:.4f}")
print("\nsplit-half reliability (Spearman-Brown to full n), 44-block:", {c: (round(rel[c]["raw"], 2), round(rel[c]["top"], 2)) for c in CH})
show(disatt(C[("44", "raw")], "raw"), "44-block congruence DISATTENUATED (r / sqrt(rel_i rel_j)), raw")
show(disatt(C[("44", "top")], "top"), "44-block congruence DISATTENUATED, top component removed")

# ---- figure: lower = raw, upper = top-removed (44-block), diagonal = reliability (raw/top) ----
fig, ax = plt.subplots(figsize=(6.2, 5.4)); M = np.zeros((5, 5))
for i in range(5):
    for j in range(5): M[i, j] = C[("44", "raw")][i, j] if i > j else (C[("44", "top")][i, j] if i < j else np.nan)
im = ax.imshow(np.nan_to_num(M, nan=1.0), cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(5):
    for j in range(5):
        txt = f"{M[i,j]:.2f}" if i != j else f"{rel[CH[i]]['raw']:.2f}\n{rel[CH[i]]['top']:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9 if i != j else 7, color="white" if (i != j and abs(M[i, j]) > .6) else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(CH, fontsize=8); ax.set_yticks(range(5)); ax.set_yticklabels(CH, fontsize=8)
ax.set_title("Channel congruence on 44 blocks: lower = raw, upper = top component removed,\ndiagonal = split-half reliability (raw / top-removed)", fontsize=8)
fig.colorbar(im, shrink=.75); fig.tight_layout(); out = "results/persona_vectors/figs/fig_channel_similarity"
fig.savefig(out + ".pdf", dpi=300); fig.savefig(out + ".png", dpi=160); print("\nwrote", out)

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
