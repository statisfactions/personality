"""Summary of the self-like instrument frames on the 44 medoids (rgb 2026-09-16,
scope close-out). Within-model profile correlations across medoids, median over
models, for: six absolute framings (plain arm), comparative directions d vs
assistant/AI/LM/person, baseline estimates ("the average X is"), synthetic
reconstructions base + d, and the self-enhancement profile direct - base_person.
Usage: .venv/bin/python scripts/self_frames_summary.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit

labels = pkit.load.adjectives(); H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w, v = np.linalg.eigh(H); ev = v[:, np.argmax(w)]; ev *= np.sign(ev[labels.index("kind")])
F = {}; seen = set()
for p in sorted(glob.glob("results/adjectives/refgroup/*_base.json")):
    b = json.load(open(p)); m = b["model"]; repo = pkit.roster.MODELS.get(m, m).split("/")[-1]
    pp, ph = p.replace("_base.json", "_pair.json"), p.replace("_base.json", "_human.json")
    if repo in seen or not (os.path.exists(pp) and os.path.exists(ph)): continue
    seen.add(repo); q, h = json.load(open(pp)), json.load(open(ph)); A = b["adjectives"]
    E = lambda R, vn: np.array([R["results"][vn][a]["ev"] for a in A])
    fr = {"direct": E(q, "direct"), "PDA": E(q, "pda")}
    try:
        sf = pkit.load.load_self(m)
        for f in ["assistant", "person", "observer", "outputs"]:
            s = sf[sf.framing == f].set_index("adjective")["ev"]; fr[f] = np.array([s.get(a, np.nan) for a in A])
    except Exception:
        continue
    for k in ["assistant", "ai", "lm"]:
        fr[f"d vs {k}"] = (E(q, f"more_{k}") - E(q, f"less_{k}")) / 2
    fr["d vs person"] = (E(h, "more_person") - E(h, "less_person")) / 2
    fr["avg assistant"] = E(b, "base_assistant"); fr["avg person"] = E(b, "base_person")
    fr["avg assistant + d"] = fr["avg assistant"] + fr["d vs assistant"]; fr["avg person + d"] = fr["avg person"] + fr["d vs person"]
    fr["direct - avg person"] = fr["direct"] - fr["avg person"]
    if any(np.isnan(x).any() for x in fr.values()): continue
    F[repo] = fr
names = list(next(iter(F.values())).keys()); A = json.load(open(glob.glob("results/adjectives/refgroup/*_base.json")[0]))["adjectives"]
des = np.array([ev[labels.index(a)] for a in A]); k = len(names); n = len(F)
C = np.full((k, k), np.nan); cnt = np.zeros((k, k), int)
for i in range(k):
    for j in range(k):
        rs = [np.corrcoef(fr[names[i]], fr[names[j]])[0, 1] for fr in F.values() if fr[names[i]].std() >= .5 and fr[names[j]].std() >= .5]
        C[i, j] = np.median(rs) if rs else np.nan; cnt[i, j] = len(rs)
print(f"n = {n} models; entries = median within-model r across the 44 medoids (both frames sd >= .5)\n")
print(f"{'':22s}" + "".join(f"{nm[:9]:>10s}" for nm in names))
for i, nm in enumerate(names): print(f"{nm:22s}" + "".join(f"{C[i,j]:10.2f}" for j in range(k)))
print("\nper frame: models with sd >= .5 | median r with desirability | median level | median sd")
for i, nm in enumerate(names):
    ok = [fr for fr in F.values() if fr[nm].std() >= .5]
    print(f"  {nm:22s} {len(ok):3d}/{n}  rDes {np.median([np.corrcoef(fr[nm], des)[0,1] for fr in ok]):5.2f}  level {np.median([fr[nm].mean() for fr in ok]):5.2f}  sd {np.median([fr[nm].std() for fr in ok]):4.2f}")
fig, ax = plt.subplots(figsize=(8.5, 7.5)); im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(k)); ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8); ax.set_yticks(range(k)); ax.set_yticklabels(names, fontsize=8)
for i in range(k):
    for j in range(k):
        if i != j: ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(C[i, j]) > .6 else "black")
ax.set_title(f"Self-like frames on the 44 medoids: median within-model r (n={n} core models)", fontsize=9)
fig.colorbar(im, shrink=.7); fig.tight_layout(); out = "results/persona_vectors/figs/fig_self_frames_summary"
fig.savefig(out + ".pdf", dpi=300); fig.savefig(out + ".png", dpi=150); print("wrote", out)
