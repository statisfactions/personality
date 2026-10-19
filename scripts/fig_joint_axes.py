"""Figure: common axes across the five channels (joint diagonalization, K=7;
rgb 2026-10-18). Reads results/persona_vectors/joint_diag/joint_diag_k7.{npz,json}
from scripts/joint_diag_channels.py --k 7.

(a) channel weight w on each axis (rows: axes, named by their poles; cell
    text = w; share of the channel's off-diagonal structure = w^2).
(b) each channel's off-diagonal structure split by axis (w^2), plus the part
    the seven shared axes leave unexplained (1 - sum w^2; within .01 of
    1 - joint fit share).

Design variations (scripts/joint_diag_robustness.py): DISTATIS (lambda1 or
unit-norm tables), eigenvectors of the mean matrix, l1 normalization and
K = 6/8/10 all recover axes 1, 3, 4, 5 (|cos| >= .92); the averaging methods
lose the REPRESENT-only axes (6, 7); eigenvectors of the summed squares mix
axes 3 and 4. Axis names are descriptive labels, not fitted quantities.

Usage: .venv/bin/python scripts/fig_joint_axes.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = "results/persona_vectors/joint_diag/joint_diag_k7"
OUT = "results/persona_vectors/figs/fig_joint_axes"
z = np.load(SRC + ".npz", allow_pickle=True); meta = json.load(open(SRC + ".json"))
Q, W, labels = z["Q"], z["W"], [str(s) for s in z["labels"]]
CH = [str(c) for c in z["channels"]]; CHD = [c.title() for c in CH]
K = W.shape[1]

# descriptive names (rgb may rename); poles oriented so the first-named pole reads first
NAMES = ["Evaluation", "Praise words", "Agency", "Strict vs. cheerful",
         "Worry", "Self-/well- compounds", "Body"]
FLIP = [True, False, False, False, False, False, False]   # axis 1: kind pole first
assert len(NAMES) == K
# categorical slots in fixed order (validated palette, adjacent pairlist), gray remainder
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
REST = "#c9c9c9"; INK = "#222222"; MUTED = "#666666"


def poles(k, m=3):
    v = -Q[:, k] if FLIP[k] else Q[:, k]
    o = np.argsort(v)
    return ", ".join(labels[i].lower() for i in o[::-1][:m]), ", ".join(labels[i].lower() for i in o[:m])


fig = plt.figure(figsize=(12.5, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

# ---- (a) weights ----
A = np.abs(W.T)                                  # K x T (weights are all >= 0 here)
assert (W > -0.02).all(), "negative weight: switch (a) to a diverging scale"
im = axa.imshow(A, cmap="Purples", vmin=0, vmax=0.9, aspect="auto")
for k in range(K):
    for t in range(len(CH)):
        axa.text(t, k, f"{W[t, k]:.2f}", ha="center", va="center", fontsize=8.5,
                 color="white" if A[k, t] > 0.55 else INK)
axa.set_xticks(range(len(CH)), CHD, fontsize=9)
axa.xaxis.tick_top()
axa.set_yticks(range(K), [""] * K)
axa.tick_params(length=0)
for s in axa.spines.values():
    s.set_visible(False)
for k in range(K):
    p, n_ = poles(k)
    axa.add_patch(Rectangle((-0.5 - 0.16, k - 0.32), 0.1, 0.64, color=PAL[k], clip_on=False,
                            transform=axa.transData))
    axa.text(-0.75, k - 0.13, f"{k + 1}  {NAMES[k]}", ha="right", va="center", fontsize=9,
             color=INK, fontweight="bold")
    axa.text(-0.75, k + 0.2, f"{p}  /  {n_}", ha="right", va="center", fontsize=7, color=MUTED)
fig.text(0.015, 0.95, "(a) channel weight on each shared axis", fontsize=10, ha="left", va="top")

# ---- (b) shares ----
S = W ** 2                                        # T x K
joint = np.array([meta["joint_share"][c] for c in CH])
y = np.arange(len(CH))[::-1]
for t in range(len(CH)):
    left = 0.0
    for k in range(K):
        axb.barh(y[t], S[t, k], left=left, color=PAL[k], height=0.62,
                 edgecolor="white", linewidth=1.0)
        if S[t, k] >= 0.05:
            axb.text(left + S[t, k] / 2, y[t], str(k + 1), ha="center", va="center",
                     fontsize=7.5, color="white")
        left += S[t, k]
    # remainder from sum w^2 (within .01 of 1 - joint fit share: the off-diagonal
    # regressors q_k q_k' are only nearly orthogonal)
    axb.barh(y[t], 1 - left, left=left, color=REST, height=0.62, edgecolor="white", linewidth=1.0)
axb.set_yticks(y, CHD, fontsize=9)
axb.set_xlim(0, 1); axb.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"], fontsize=8)
axb.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    axb.spines[s].set_visible(False)
axb.spines["bottom"].set_color(MUTED)
axb.text(1.0, -0.95, "gray: not captured by the seven shared axes", ha="right", fontsize=7.5, color=MUTED)

cax = fig.add_axes((0.12, 0.045, 0.16, 0.022))
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([0, .3, .6, .9])
cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
cax.set_title("weight w (share = w²)", fontsize=7.5, color=MUTED, pad=3)

fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.15)
fig.text(axb.get_position().x0 - 0.06, 0.95, "(b) each channel's structure, by axis", fontsize=10, ha="left", va="top")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)
print("wrote", OUT, "| sum w^2 vs joint share:",
      {c: (round(float(S[t].sum()), 3), round(float(joint[t]), 3)) for t, c in enumerate(CH)})
