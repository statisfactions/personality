#!/usr/bin/env python3
"""5x5 grid of adjectives by model PC1 x PC2 loading bins (W14, exploratory).

The model's top-2 unrotated axes are two near-orthogonal evaluative POLES, not an
intensity+valence pair:
  PC1+ = negative-evaluation (Disgusting/Awful)  PC1- = warmth/prosocial (Caring)
  PC2+ = positive-evaluation (Wonderful/Excellent)  PC2- = dysregulation (Impulsive)
This grid bins each adjective into high+/med+/~0/med-/low- on each axis and drops
representative words into the cells, so the 2-pole layout (and why Wonderful≈Awful)
is legible.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_pc_grid.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go

from adjective_geom import model_matrix
from adjective_factor_heatmap import varimax

BINS = [("high +", 1.5), ("med +", 0.65), ("~0", 0.0), ("med −", -0.65), ("low −", -1.5)]
EDGES = [1.0, 0.35, -0.35, -1.0]   # z thresholds between the 5 bins (desc)
PERCELL = 7


def binidx(z):
    for i, e in enumerate(EDGES):
        if z > e:
            return i
    return 4


def structure_diagnostics(H, M):
    """The W14 §4 numbers: human-vs-model unrotated PC congruence, how much of the
    relational match the shared PC1 carries, and what the model PCs align with in
    the human Big Five (with vs without the evaluative axis projected out)."""
    np.fill_diagonal(H, 1.0); np.fill_diagonal(M, 1.0)
    eM, VM = np.linalg.eigh(M); oM = np.argsort(eM)[::-1]
    mPC = VM[:, oM[:2]] * np.sqrt(np.clip(eM[oM[:2]], 0, None))
    eH, VH = np.linalg.eigh(H); oH = np.argsort(eH)[::-1]
    hPC1 = VH[:, oH[0]] * np.sqrt(eH[oH[0]])
    Hvm = varimax(VH[:, oH[:5]] * np.sqrt(np.clip(eH[oH[:5]], 0, None)), normalize=True)
    for L in (mPC, Hvm):
        for j in range(L.shape[1]):
            if L[np.argmax(np.abs(L[:, j])), j] < 0:
                L[:, j] *= -1

    def cong(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print("human PC1 × model PC1 / PC2 congruence: "
          f"{cong(hPC1, mPC[:,0]):+.2f} / {cong(hPC1, mPC[:,1]):+.2f}")

    iu = np.triu_indices_from(H, 1)
    def r(a, b): return float(np.corrcoef(a, b)[0, 1])
    r_full = r(H[iu], M[iu])
    H1 = H - eH[oH[0]] * np.outer(VH[:, oH[0]], VH[:, oH[0]])     # drop shared PC1
    M1 = M - eM[oM[0]] * np.outer(VM[:, oM[0]], VM[:, oM[0]])
    r_noPC1 = r(H1[iu], M1[iu])
    print(f"relational match r {r_full:.3f}; remove shared PC1 -> {r_noPC1:.3f} "
          f"(PC1 carries {(r_full**2 - r_noPC1**2)/r_full**2*100:.0f}% of shared variance)")

    big = ["N", "E", "A", "C", "O"]
    print("model PC vs human Big-Five (signed r) | eval-projected-out:")
    def resid(x, e): e = e / np.linalg.norm(e); return x - (x @ e) * e
    for j, nm in ((0, "mPC1"), (1, "mPC2")):
        raw = "  ".join(f"{b}{r(mPC[:,j], Hvm[:,k]):+.2f}" for k, b in enumerate(big))
        res = "  ".join(f"{b}{r(resid(mPC[:,j], hPC1), resid(Hvm[:,k], hPC1)):+.2f}"
                        for k, b in enumerate(big))
        print(f"  {nm}: {raw}   |   {res}")


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = np.array(human["labels"])
    Cs = []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        _, C, _, _ = model_matrix(p, labels); Cs.append(C)
    M = np.mean(Cs, 0); np.fill_diagonal(M, 1.0)

    structure_diagnostics(np.array(human["correlation_matrix"], float), M.copy())

    ev, V = np.linalg.eigh(M); o = np.argsort(ev)[::-1]
    PC = V[:, o[:2]] * np.sqrt(np.clip(ev[o[:2]], 0, None))
    for j in range(2):                                   # sign-fix: peak loads positive
        if PC[np.argmax(np.abs(PC[:, j])), j] < 0:
            PC[:, j] *= -1
    Z = (PC - PC.mean(0)) / PC.std(0)                    # standardize for binning

    b1 = np.array([binidx(Z[i, 0]) for i in range(len(labels))])
    b2 = np.array([binidx(Z[i, 1]) for i in range(len(labels))])

    cells = {}
    for r in range(5):
        for c in range(5):
            m = (b1 == r) & (b2 == c)
            idx = np.where(m)[0]
            if len(idx) == 0:
                cells[(r, c)] = ("", 0); continue
            # show the most salient members (largest |loading|) so the iconic
            # extremes (Disgusting/Wonderful) surface; for the ~0/~0 cell that
            # just gives the most-structured near-neutral words.
            sal = Z[idx, 0] ** 2 + Z[idx, 1] ** 2
            pick = idx[np.argsort(-sal)][:PERCELL]
            cells[(r, c)] = ("<br>".join(labels[pick]), len(idx))

    # text grid to stdout
    print("rows = PC1 (neg-eval ↔ prosocial), cols = PC2 (pos-eval ↔ dysregulation)\n")
    for r in range(5):
        print(f"PC1 {BINS[r][0]:<7}", end="")
        for c in range(5):
            txt = cells[(r, c)][0].replace("<br>", ",")
            print(f" | {txt[:34]:<34}", end="")
        print()

    # figure: go.Table, first col = PC1 bin label, 5 PC2 cols
    header = ["PC1 ＼ PC2"] + [f"PC2 {b[0]}" for b in BINS]
    col0 = [f"PC1 {BINS[r][0]}" for r in range(5)]
    cols = [col0]
    for c in range(5):
        cols.append([f"{cells[(r, c)][0]}<br><i>(n={cells[(r, c)][1]})</i>" for r in range(5)])
    # tint columns by PC2 valence: pos-eval (left) green -> dysregulation (right) red
    coltint = ["#eef6f0", "#f6fbf8", "#ffffff", "#fdf6f6", "#fbecec"]
    fillcol = [["#eef3f8"] * 5] + [[coltint[c]] * 5 for c in range(5)]
    fig = go.Figure(go.Table(
        columnwidth=[60] + [110] * 5,
        header=dict(values=header, fill_color="#2c3e50",
                    font=dict(color="white", size=11), align="center", height=28),
        cells=dict(values=cols, fill_color=fillcol, align="center",
                   font=dict(size=9), height=104)))
    fig.update_layout(
        title=dict(text="<b>Adjectives by model PC1 × PC2 loading (cohort-mean, "
            "z-binned)</b><br><sub>Two near-orthogonal evaluative axes. "
            "<b>PC1 (rows):</b> pejorative/disgust (Awful, top) ↔ warmth/prosocial "
            "(Caring, bottom). <b>PC2 (cols):</b> admirable/excellent (Wonderful, left) ↔ "
            "dysregulation (Impulsive/Anxious, right). The neg-eval pole (Disgusting/Awful, "
            "top-left) and the pos-eval pole (Wonderful/Amazing, mid-left) BOTH sit on the "
            "positive-PC2 side — that shared salience is why Wonderful≈Awful (+0.41), not "
            "antonyms.</sub>", x=0.01),
        width=1300, height=720, font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/adjective_pc_grid.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1300, height=720, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
