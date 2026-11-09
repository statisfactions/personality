"""Figure: common axes across the five channels (joint diagonalization, K=6;
rgb 2026-10-24, was K=7). Reads results/persona_vectors/joint_diag/joint_diag_k6.npz
from scripts/joint_diag_channels.py --k 6. K=6 because axis 7 of the K=7 fit
(physical vs character, mostly SELF) does not replicate across respondent
halves (median |cos| .54; scripts/joint_diag_noise.py); axes 1-6 do (>= .97).
Names live on the K=7 fit (NAMES below) and are carried to any other fit by
matching (|cos| >= .9), with signs aligned.

Units (rgb 2026-10-18): percent of item variance, the PC1 table's convention
(pkit.channels.SHARE_DIAG: unit diagonal for all five since 2026-10-18), via
pkit.jointdiag.item_shares. HUMAN axis 1 = 15.6 vs its own
PC1 16.4. (The fit's own weights w give shares of the squared off-diagonal
structure, ~lambda^2 / sum lambda^2, which inflate the dominant axis: 70% for
HUMAN axis 1. Not used here.)
(a) each channel's share on each shared axis (rows: axes, named by poles).
(b) the same shares stacked, then light gray = in the channel's own top-K
    components but not on the shared axes, mid gray = beyond its own top K
    (item uniqueness, sampling error, minor structure).

Design variations (scripts/joint_diag_robustness.py; K=6 axes = its K=8
reference axes 1-6 at |cos| >= .99): DISTATIS (lambda1 or unit-norm tables),
eigenvectors of the mean matrix, lambda1 normalization and K = 8/10 all recover
axes 1, 3, 4, 5 (|cos| >= .92); the averaging methods lose the REPRESENT-only
axis 6 (.18-.24) and weaken axis 2 (.78-.87); eigenvectors of the summed
squares mix axes 3 and 4 (.72 / .58). Axis names are descriptive labels, not fitted quantities.

Variant (rgb 2026-10-24, "without the oddball represent"): --drop REPRESENT
reads joint_diag_k7_noREPRESENT (joint_diag_channels.py --drop REPRESENT).
Axes matching the five-channel fit at |cos| >= .9 keep its names and signs;
the others are named in VARIANT_NAMES.

--ips (rgb 2026-11-08): adds ipsatized Human/Self (superscript dagger) after the
fitted channels, set off by whitespace; scored on the fixed axes, not in the fit.
Writes fig_joint_axes_ips.

Usage: .venv/bin/python scripts/fig_joint_axes.py [--k 6] [--drop REPRESENT] [--ips]
(--k 7 writes fig_joint_axes_k7, the previous main figure.)
"""
import argparse
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Patch, Rectangle
import pkit
from pkit import channels as C, jointdiag as J

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=6)
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--ips", action="store_true",
                help="add ipsatized Human/Self, scored on the fixed axes (not in the fit)")
args = ap.parse_args()
SUF = "".join(f"_no{c}" for c in args.drop)
REF = "results/persona_vectors/joint_diag/joint_diag_k7"       # where the names live
SRC = f"results/persona_vectors/joint_diag/joint_diag_k{args.k}" + SUF
OUT = ("results/persona_vectors/figs/fig_joint_axes" + ("" if args.k == 6 else f"_k{args.k}") + SUF
       + ("_ips" if args.ips else ""))
z = np.load(SRC + ".npz", allow_pickle=True)
Q, W, labels = z["Q"], z["W"], [str(s) for s in z["labels"]]
CH = [str(c) for c in z["channels"]]
CHD = [c.title() + ("*" if c in C.NOT_PSD else "") for c in CH]
K = W.shape[1]
mem = {c: C.members(c, labels) for c in C.CHANNELS}; LV = C.channel_level_removed(labels, mem)
SH, OWNK = [], []
for c in CH:
    per, own = J.item_shares(Q, LV[c], C.SHARE_DIAG[c]); SH.append(per); OWNK.append(own)
if args.ips:   # ipsatized (row z-scored) Human/Self after their row-centered columns
    for c in ("HUMAN", "SELF"):
        if c not in CH:
            continue
        kind, X, _ = mem[c]; X = np.asarray(X, float)
        Xz = X - X.mean(1, keepdims=True); Xz /= Xz.std(1, keepdims=True)
        per, own = J.item_shares(Q, C.cohort_matrix(kind, Xz), C.SHARE_DIAG[c])
        CH.append(c + "_ips"); CHD.append(c.title() + r"$^{\dagger}$"); SH.append(per); OWNK.append(own)
NFIT = sum(not c.endswith("_ips") for c in CH)          # fitted channels come first
GAP = 0.45                                               # whitespace before the scored-only ones
XPOS = np.array([t + (GAP if t >= NFIT else 0) for t in range(len(CH))])
SH = np.clip(np.array(SH), 0, None); OWNK = np.array(OWNK)        # T x K; JUDGE -0.000 clipped

# descriptive names (rgb may rename); poles oriented so the first-named pole reads first
NAMES = ["Evaluation", "Generic vs. behavioral", "Bold vs. ordinary", "Serious vs. playful",
         "Sensitive vs. shallow", "Compound vs. simple words", "Physical vs. character"]
FLIP = [True, False, False, False, False, False, False]   # axis 1: kind pole first
VARIANT_NAMES = {"_noREPRESENT": {4: "Glamorous vs. blunt", 5: "Physical vs. damning",
                                  6: "Easygoing vs. controlling"}}   # 0-based new-axis index
if SRC != REF:   # carry names/signs over from the K=7 five-channel fit where the axis matches
    Qr = np.load(REF + ".npz", allow_pickle=True)["Q"]
    perm, cos = J.match(Q, Qr); Kr = Qr.shape[1]   # reference axis per axis of this fit
    names_r, flip_r = NAMES, FLIP
    flip = [False] * K
    names = [VARIANT_NAMES.get(SUF, {}).get(j) for j in range(K)]
    for j, (k, c) in enumerate(zip(perm, cos)):
        if c >= 0.9:
            names[j] = names_r[k]; flip[j] = flip_r[k]
            if Qr[:, k] @ Q[:, j] < 0:
                Q[:, j] = -Q[:, j]
    NAMES, FLIP = names, flip
    # color follows the axis: matched axes keep their five-channel color
    col = [None] * K
    for j, (k, c) in enumerate(zip(perm, cos)):
        if c >= 0.9:
            col[j] = k
    spare = iter(k for k in range(max(K, Kr)) if k not in col)
    COLIDX = [c if c is not None else next(spare) for c in col]
else:
    COLIDX = list(range(K))
assert len(NAMES) == K and None not in NAMES
# categorical slots in fixed order (validated palette, adjacent pairlist), gray remainder
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
PAL = [PAL[i] for i in COLIDX]
OWN = "#dedede"; REST = "#a9a9a9"; INK = "#222222"; MUTED = "#666666"


def poles(k, m=3):
    v = -Q[:, k] if FLIP[k] else Q[:, k]
    o = np.argsort(v)
    return ", ".join(labels[i].lower() for i in o[::-1][:m]), ", ".join(labels[i].lower() for i in o[:m])


fig = plt.figure(figsize=(12.5, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

# ---- (a) shares ----
A = 100 * SH.T                                   # K x T, percent
# white -> purple -> red, lightness monotone (sequential); square-root norm so
# 0.5% vs 5% differ visibly under SELF's 41% (rgb 2026-10-18)
CMAP = LinearSegmentedColormap.from_list("wpr", ["#ffffff", "#c9b8e8", "#8e6cc9", "#b8336a", "#6e0f1f"])
NORM = PowerNorm(gamma=0.5, vmin=0, vmax=45)
for k in range(K):
    for t in range(len(CH)):
        axa.add_patch(Rectangle((XPOS[t] - 0.5, k - 0.5), 1, 1, color=CMAP(NORM(A[k, t])), lw=0))
        axa.text(XPOS[t], k, f"{A[k, t]:.1f}", ha="center", va="center", fontsize=8.5,
                 color="white" if NORM(A[k, t]) > 0.5 else INK)
axa.set_xlim(-0.5, XPOS[-1] + 0.5); axa.set_ylim(K - 0.5, -0.5)
im = matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP)
axa.set_xticks(XPOS, CHD, fontsize=9)
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
fig.text(0.015, 0.95, "(a) percent of each channel's variance on each shared axis", fontsize=10, ha="left", va="top")

# ---- (b) shares ----
y = (len(CH) - 1 + GAP if NFIT < len(CH) else len(CH) - 1) - XPOS
for t in range(len(CH)):
    left = 0.0
    for k in range(K):
        axb.barh(y[t], SH[t, k], left=left, color=PAL[k], height=0.62,
                 edgecolor="white", linewidth=1.0)
        if SH[t, k] >= 0.05:
            axb.text(left + SH[t, k] / 2, y[t], str(k + 1), ha="center", va="center",
                     fontsize=7.5, color="white")
        left += SH[t, k]
    axb.barh(y[t], OWNK[t] - left, left=left, color=OWN, height=0.62, edgecolor="white", linewidth=1.0)
    axb.barh(y[t], 1 - OWNK[t], left=OWNK[t], color=REST, height=0.62, edgecolor="white", linewidth=1.0)
axb.set_yticks(y, CHD, fontsize=9)
axb.set_xlim(0, 1); axb.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"], fontsize=8)
axb.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    axb.spines[s].set_visible(False)
axb.spines["bottom"].set_color(MUTED)
axb.legend(handles=[Patch(color=OWN, label=f"own top {K}, not shared"),
                    Patch(color=REST, label=f"beyond own top {K}")],
           loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=7.5,
           handlelength=1.2, columnspacing=1.2)
if any(c in C.NOT_PSD for c in CH):
    fig.text(0.98, 0.012, C.NOT_PSD_NOTE, ha="right", fontsize=7, color=MUTED)

cax = fig.add_axes((0.12, 0.045, 0.16, 0.022))
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([0, 1, 5, 15, 30, 45])
cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
cax.set_title("percent of variance (square-root scale)", fontsize=7.5, color=MUTED, pad=3)

fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.15)
fig.text(axb.get_position().x0 - 0.06, 0.95, "(b) each channel's variance, by axis", fontsize=10, ha="left", va="top")
if args.ips:
    fig.text(axa.get_position().x0, 0.035, r"$^{\dagger}$ipsatized (each respondent's profile z-scored); scored on the fixed axes, not in the fit",
             ha="left", fontsize=7, color=MUTED)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)
print("wrote", OUT, "| names:", NAMES, f"| shared sum / own top-{K}:",
      {c: (round(float(SH[t].sum()), 3), round(float(OWNK[t]), 3)) for t, c in enumerate(CH)})
