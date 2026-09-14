"""More-or-less self vs absolute self on the CORE roster, 44 medoids
(rgb 2026-09-14, second pass with the _pair run). Grids over models-as-
respondents: HUMAN | direct (same models) | PDA (same models) | more-or-less
d (same models) | six-framing mean SELF (core, from self_matrix). Raw units,
raw + top-removed rows, congruence vs HUMAN, split-half reliability of each
grid, spectra (eigenvalue shares, participation ratio), and the desirability
residual's between-model agreement.
Usage: .venv/bin/python scripts/fig_moreless_grid_core.py
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit
from pkit import measures

REFS = ["assistant", "ai", "lm"]
labels = pkit.load.adjectives()
files = sorted(glob.glob("results/adjectives/refgroup/*_pair.json"))
runs, seen = [], set()
for p in files:                       # dedupe short-name reruns of repo-named models
    d = json.load(open(p)); repo = pkit.roster.MODELS.get(d["model"], d["model"]).split("/")[-1]
    if repo in seen: continue
    seen.add(repo); d["repo"] = repo; runs.append(d)
A = runs[0]["adjectives"]; idx = [labels.index(a) for a in A]; n = len(runs)
E = lambda d, vn: np.array([d["results"][vn][a]["ev"] for a in A])
X = {"direct": np.array([E(d, "direct") for d in runs]),
     "PDA": np.array([E(d, "pda") for d in runs]),
     "more-or-less d": np.array([np.mean([(E(d, f"more_{k}") - E(d, f"less_{k}")) / 2 for k in REFS], 0) for d in runs])}
R = pkit.load.self_matrix(which="cohort"); core = R.values.std(1) >= 0.5
X["6-framing SELF (core)"] = R.values[core][:, idx]
# drop flat respondents per instrument (sd < .5 on the medoids), the standing core rule
names = [d["repo"] for d in runs]
for k in X:
    keep = X[k].std(1) >= 0.5; print(f"{k:22s} n={X[k].shape[0]:2d}, sd>=.5 keeps {keep.sum():2d}")
    if k == "more-or-less d":
        flat = [(names[i], round(X["direct"][i].std(), 2), round(X[k][i].std(), 2)) for i in np.where(~keep)[0]]
        print("   flat under comparison (name, direct sd, d sd):", flat)
    X[k] = X[k][keep]
m44 = ~np.eye(44, dtype=bool)
def grid(M): S = np.corrcoef(M.T); np.fill_diagonal(S, 0); return S
def center(M): B = M.copy(); B[m44] -= B[m44].mean(); return B
def r(a, b): return np.corrcoef(a[m44], b[m44])[0, 1]
Hc = pkit.load.human_corr().values[np.ix_(idx, idx)].copy(); np.fill_diagonal(Hc, 0)
G = {"HUMAN": Hc} | {k: grid(v) for k, v in X.items()}
Gp = {k: measures.remove_pc1(center(v)) for k, v in G.items()}
rng = np.random.default_rng(0)
def split_half(M, reps=200):
    rs = []
    for _ in range(reps):
        p = rng.permutation(M.shape[0]); a, b = p[: len(p) // 2], p[len(p) // 2:]
        rs.append(r(grid(M[a]), grid(M[b])))
    return np.mean(rs)
def split_half_top(M, reps=100):
    rs = []
    for _ in range(reps):
        p = rng.permutation(M.shape[0]); a, b = p[: len(p) // 2], p[len(p) // 2:]
        rs.append(r(measures.remove_pc1(center(grid(M[a]))), measures.remove_pc1(center(grid(M[b])))))
    return np.mean(rs)
print(f"\n{'grid':22s} {'n':>3s} {'r vs HUMAN raw':>14s} {'top-removed':>11s} | {'split-half raw':>14s} {'top':>5s} | {'offdiag mean':>12s} {'sd':>5s} | eig shares 1-3   PR")
for k in G:
    w = np.abs(np.linalg.eigvalsh(center(G[k]))); w = np.sort(w)[::-1]; sh = w / w.sum(); pr = w.sum() ** 2 / (w ** 2).sum()
    if k == "HUMAN":
        print(f"{k:22s} {700:3d} {'':14s} {'':11s} | {'':14s} {'':5s} | {G[k][m44].mean():12.3f} {G[k][m44].std():5.2f} | {sh[0]:.2f} {sh[1]:.2f} {sh[2]:.2f}  {pr:5.1f}")
    else:
        print(f"{k:22s} {X[k].shape[0]:3d} {r(G[k], G['HUMAN']):14.3f} {r(Gp[k], Gp['HUMAN']):11.3f} | {split_half(X[k]):14.2f} {split_half_top(X[k]):5.2f} | {G[k][m44].mean():12.3f} {G[k][m44].std():5.2f} | {sh[0]:.2f} {sh[1]:.2f} {sh[2]:.2f}  {pr:5.1f}")
CEIL = 0.92
print("\ndisattenuated top-removed congruence  r / sqrt(split-half_top * human ceiling .92):")
for k in [k for k in G if k != "HUMAN"]:
    sh_t = split_half_top(X[k]); print(f"  {k:22s} {r(Gp[k], Gp['HUMAN']) / np.sqrt(max(sh_t, 1e-6) * CEIL):.2f}   (raw split-half top {sh_t:.2f})")
print("\ngrid-vs-grid (raw / top-removed):")
ks = [k for k in G if k != "HUMAN"]
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"  {ks[i]:22s} vs {ks[j]:22s}: {r(G[ks[i]], G[ks[j]]):.2f} / {r(Gp[ks[i]], Gp[ks[j]]):.2f}")
# desirability residual agreement per instrument (profiles): how much non-halo structure is shared
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w_, v_ = np.linalg.eigh(H); ev = v_[:, np.argmax(w_)]; ev *= np.sign(ev[labels.index("kind")]); des = ev[idx]
def resid_agree(M):
    Rm = np.array([x - np.polyval(np.polyfit(des, x, 1), des) for x in M]); C = np.corrcoef(Rm); k = len(Rm)
    return C[~np.eye(k, dtype=bool)].mean()
print("\nprofile-level: between-model agreement of the desirability RESIDUAL:", {k: round(resid_agree(v), 2) for k, v in X.items()})
print("profile-level: mean r(profile, desirability):", {k: round(np.mean([np.corrcoef(x, des)[0, 1] for x in v]), 2) for k, v in X.items()})
# figure
cl = pkit.facets.clusters("blocks44"); bb = np.cumsum([sum(1 for c in cl if c["branch"] == b) for b in sorted({c["branch"] for c in cl})])[:-1]
VL = {"raw": 1.0, "top comp. removed": 0.5}
fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.0))
for ri, (tag, GG) in enumerate([("raw", G), ("top comp. removed", Gp)]):
    for ci, k in enumerate(GG):
        ax = axes[ri, ci]; ax.imshow(GG[k], cmap="RdBu_r", vmin=-VL[tag], vmax=VL[tag])
        for b in bb: ax.axhline(b - .5, color="k", lw=.3, alpha=.5); ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(k + ("" if k == "HUMAN" else f" (n={X[k].shape[0]})"), fontsize=8)
        if ci == 0: ax.set_ylabel(tag, fontsize=8)
        else: ax.set_xlabel(f"r vs HUMAN = {r(GG[k], GG['HUMAN']):.2f}", fontsize=7)
    fig.colorbar(axes[ri, 0].images[0], ax=axes[ri, :], shrink=0.85).ax.tick_params(labelsize=6)
out = "results/persona_vectors/figs/fig_moreless_grid_core"; fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", bbox_inches="tight", dpi=170); print("wrote", out)
