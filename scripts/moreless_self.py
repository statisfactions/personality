"""'More-or-less self' from the comparative pair (rgb 2026-09-14): d = (more - less)/2
per adjective, averaged over the three reference labels, vs the absolute SELF
profiles (direct, PDA, six-framing mean) on the 44 medoids. Sense checks:
halo (r with human evaluation axis), residual structure after removing it,
between-model agreement, reference-label consistency, top/bottom adjectives.
Usage: .venv/bin/python scripts/moreless_self.py
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit

REFS = ["assistant", "ai", "lm"]
ABS = ["direct", "assistant", "person", "pda", "observer", "outputs"]
labels = pkit.load.adjectives()
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w, v = np.linalg.eigh(H); ev = v[:, np.argmax(w)]; ev *= np.sign(ev[labels.index("kind")])
runs = {json.load(open(p))["model"]: json.load(open(p)) for p in sorted(glob.glob("results/adjectives/refgroup/*.json"))}
A = next(iter(runs.values()))["adjectives"]; des = np.array([ev[labels.index(a)] for a in A]); des = (des - des.mean()) / des.std()
E = lambda m, vn: np.array([runs[m]["results"][vn][a]["ev"] for a in A])
ms = list(runs)
def resid(x):            # remove the desirability projection
    b = np.polyfit(des, x, 1); return x - np.polyval(b, des)
D, DIR, FM = {}, {}, {}
print(f"{'model':8s} {'ref-consist':>11s} {'sd d':>5s} {'sd dir':>6s} | r(d,·): {'direct':>6s} {'fmean':>5s} {'pda':>5s} | halo r(·,des): {'d':>5s} {'direct':>6s} {'fmean':>5s} | resid r(d,direct) {'':>0s} | sd resid: {'d':>4s} {'dir':>4s}")
for m in ms:
    ds = [(E(m, f"more_{r}") - E(m, f"less_{r}")) / 2 for r in REFS]
    cons = np.mean([np.corrcoef(ds[i], ds[j])[0, 1] for i in range(3) for j in range(i + 1, 3)])
    d = np.mean(ds, 0); di = E(m, "direct"); fm = np.mean([E(m, f) for f in ABS], 0); pd_ = E(m, "pda")
    D[m], DIR[m], FM[m] = d, di, fm
    print(f"{m:8s} {cons:11.2f} {d.std():5.2f} {di.std():6.2f} | {'':8s}{np.corrcoef(d, di)[0,1]:6.2f} {np.corrcoef(d, fm)[0,1]:5.2f} {np.corrcoef(d, pd_)[0,1]:5.2f} | {'':14s}{np.corrcoef(d, des)[0,1]:5.2f} {np.corrcoef(di, des)[0,1]:6.2f} {np.corrcoef(fm, des)[0,1]:5.2f} | {np.corrcoef(resid(d), resid(di))[0,1]:17.2f} | {'':9s}{resid(d).std():4.2f} {resid(di).std():4.2f}")
k = len(ms); off = ~np.eye(k, dtype=bool)
ag = lambda Z: np.corrcoef(np.array([Z[m] for m in ms]))[off].mean()
print(f"\nbetween-model agreement: d {ag(D):.2f}  direct {ag(DIR):.2f}  fmean {ag(FM):.2f}  | desirability-residual: d {ag({m: resid(D[m]) for m in ms}):.2f}  direct {ag({m: resid(DIR[m]) for m in ms}):.2f}  fmean {ag({m: resid(FM[m]) for m in ms}):.2f}")
print("\ntop/bottom 5 under d (more-or-less self) vs direct:")
for m in ms:
    o = np.argsort(-D[m]); od = np.argsort(-DIR[m])
    print(f"  {m:8s} d   +: {', '.join(A[i] for i in o[:5])}  |  -: {', '.join(A[i] for i in o[-5:])}")
    print(f"  {'':8s} dir +: {', '.join(A[i] for i in od[:5])}  |  -: {', '.join(A[i] for i in od[-5:])}")
print("\nlargest |d - centered direct| disagreements (where the two selves differ):")
for m in ms:
    x = (DIR[m] - DIR[m].mean()); b = np.polyfit(x, D[m], 1); g = D[m] - np.polyval(b, x); o = np.argsort(-np.abs(g))
    print(f"  {m:8s} " + ", ".join(f"{A[i]} {g[i]:+.1f}" for i in o[:6]))
# figure: medoids ordered by desirability; centered direct vs d
fig, axes = plt.subplots(k, 1, figsize=(9, 2.0 * k), sharex=True)
o = np.argsort(des)
for ax, m in zip(axes, ms):
    ax.axhline(0, color="k", lw=.4)
    ax.plot((DIR[m] - DIR[m].mean())[o], "o-", ms=3, lw=.8, label="direct (centered)")
    ax.plot(D[m][o], "s-", ms=3, lw=.8, label="more-or-less self d")
    ax.set_ylabel(m, fontsize=8); ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7, loc="upper left"); axes[-1].set_xticks(range(len(A))); axes[-1].set_xticklabels(np.array(A)[o], rotation=90, fontsize=6)
fig.suptitle("44 medoids ordered by human desirability", fontsize=9); fig.tight_layout()
fig.savefig("results/persona_vectors/figs/moreless_self.png", dpi=150); print("wrote results/persona_vectors/figs/moreless_self.png")
