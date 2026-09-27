"""SELF is rank-2 across models (paper point 2, 2026-09-27). Two panels:
A. the SELF similarity grid decomposed into its first component (the
   constant = level), its second (desirability), and the rest;
B. respondents in the (elevation, gain) plane: 700 humans as a gray cloud,
   49 models as points; dot area = the share of each respondent's
   row-centered profile variance NOT explained by the shared desirability
   profile (1 - r^2), i.e. their idiosyncratic residual.
Usage: .venv/bin/python scripts/fig_self_rank2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pkit
from pkit import channels as chn

labels = pkit.load.adjectives(); n = len(labels)
cl = pkit.facets.clusters("blocks44"); m44 = ~np.eye(44, dtype=bool)
branch_breaks = np.cumsum([sum(1 for c in cl if c["branch"] == b) for b in sorted({c["branch"] for c in cl})])[:-1]
Hm = np.asarray(chn.human_members(labels), float); Sm, names = chn.self_members(labels); Sm = np.asarray(Sm, float)
H = chn.cohort_matrix("respondents", Hm); w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
one = np.ones(n) / np.sqrt(n)

# ---- A: decomposition of the SELF similarity ----
S = chn.cohort_matrix("respondents", Sm)
w, V = np.linalg.eigh(S); o = np.argsort(w)[::-1]
comps = [w[o[k]] * np.outer(V[:, o[k]], V[:, o[k]]) for k in range(2)]
rest = S - comps[0] - comps[1]
share = np.abs(w[o[:2]]) / np.abs(w).sum()
cos1, cos2 = abs(V[:, o[0]] @ one), abs(V[:, o[1]] @ t)
panels = [("SELF similarity", S), (f"component 1 = level\n{100*share[0]:.0f}% · cos {cos1:.2f} w/ constant", comps[0]),
          (f"component 2 = desirability\n{100*share[1]:.0f}% · cos {cos2:.2f} w/ human axis", comps[1]), ("rest", rest)]

fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.55], width_ratios=[1, 1, 1, 1, 0.06], wspace=0.08, hspace=0.35)
for i, (title, M) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i]); G = np.array(chn.blockify(M), float)
    for j in range(44): G[j, j] = np.nan
    ax.imshow(G, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    for b in branch_breaks: ax.axhline(b - .5, color="k", lw=.3, alpha=.5); ax.axvline(b - .5, color="k", lw=.3, alpha=.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6.8, pad=3)
cax = fig.add_subplot(gs[0, 4]); plt.colorbar(fig.axes[-2].images[0], cax=cax).ax.tick_params(labelsize=6)

# ---- B: elevation x gain, dot area = idiosyncratic share ----
def stats(X):
    Xc = X - X.mean(1, keepdims=True)
    gain = (Xc @ t) / np.sqrt(n)                   # rating SD attributable to the shared desirability profile
    r2 = np.array([np.corrcoef(Xc[m], t)[0, 1] ** 2 for m in range(len(X))])
    return X.mean(1), gain, 1 - r2
he, hg, hres = stats(Hm); me, mg, mres = stats(Sm)
ax = fig.add_subplot(gs[1, :4])
ax.scatter(he, hg, s=90 * hres, c="#9a9a9a", alpha=.35, lw=0, label="human respondents (n=700)")
ax.scatter(me, mg, s=90 * mres, c="#1f4e9a", alpha=.9, lw=.6, edgecolor="white", label="models (n=49)")
for m, dx, dy in [(int(np.argmax(mg)), 4, 2), (int(np.argmin(mg)), 4, -8), (int(np.argmax(me)), 4, 2), (int(np.argmin(me)), 4, 2)]:
    ax.annotate(names[m].split("/")[-1].replace("-Instruct", "").replace("-it", "").replace("-instruct", ""), (me[m], mg[m]), fontsize=6, xytext=(dx, dy), textcoords="offset points")
for r_ in [0.2, 0.5, 0.8]:
    ax.scatter([], [], s=90 * r_, c="#1f4e9a", alpha=.9, label=f"idiosyncratic share {r_:.0%}")
ax.set_ylim(-1.8, 2.7); ax.set_xlabel("elevation (mean self-rating, 1-7)", fontsize=8); ax.set_ylabel("gain (rating SD along the shared\ndesirability profile, scale units)", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=1)
ax.text(0.02, 0.03, f"models: r² of profile on desirability {np.mean(1-mres):.2f} (min {np.min(1-mres):.2f}); humans {np.mean(1-hres):.2f} (5th pct {np.percentile(1-hres,5):.2f})",
        transform=ax.transAxes, fontsize=6.5, va="bottom")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
out = "results/persona_vectors/figs/fig_self_rank2"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300); fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
print("wrote", out, "| shares", np.round(share, 3), "| cos", round(cos1, 2), round(cos2, 2),
      "| models elev %.2f±%.2f gain %.2f±%.2f | humans elev %.2f±%.2f gain %.2f±%.2f" % (me.mean(), me.std(), mg.mean(), mg.std(), he.mean(), he.std(), hg.mean(), hg.std()))
