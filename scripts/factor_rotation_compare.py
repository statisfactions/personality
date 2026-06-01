#!/usr/bin/env python3
"""Pretty 2x2: human vs model, unrotated PCA vs varimax adjective loadings.

Rows = 523 adjectives in a single shared order (human varimax nearest-factor
blocks) so the four panels are directly comparable. Columns = 5 components.
Unrotated shows the big general first component; varimax rotates to simple
structure. Model = cohort-mean adaptive-denoised geometry. Varimax columns of the
model are matched to the human Big Five (greedy congruence) and sign-aligned.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/factor_rotation_compare.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_factor_heatmap import varimax
from adjective_geom import model_matrix

K = 5
HUMAN_LABELS = ["A", "E", "N", "C", "O"]  # human varimax factor identities (from top words)


def phi_var(C):
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    Phi = V[:, o][:, :K] * np.sqrt(np.clip(ev[o][:K], 0, None))
    L = varimax(Phi)
    for M in (Phi, L):
        for j in range(K):
            if M[np.argmax(np.abs(M[:, j])), j] < 0:
                M[:, j] *= -1
    return Phi, L


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    Hphi, Hvar = phi_var(H)

    Cs = [model_matrix(p, labels)[1]
          for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt"))]
    Mphi, Mvar = phi_var(np.mean(Cs, 0))

    # row order: human varimax nearest-factor, then by signed loading
    assign = np.argmax(np.abs(Hvar), axis=1)
    prim = Hvar[np.arange(len(labels)), assign]
    order = sorted(range(len(labels)), key=lambda i: (assign[i], -prim[i]))
    bounds = [i for i in range(1, len(order)) if assign[order[i]] != assign[order[i-1]]]

    # match model varimax cols -> human varimax cols (greedy |congruence|), sign-align
    def tuck(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    T = np.array([[tuck(Hvar[:, i], Mvar[:, j]) for j in range(K)] for i in range(K)])
    colmap = {}; usedj = set()
    for _, i, j in sorted(((abs(T[i, j]), i, j) for i in range(K) for j in range(K)), reverse=True):
        if i in colmap or j in usedj:
            continue
        colmap[i] = j; usedj.add(j)
    Mvar_a = np.column_stack([Mvar[:, colmap[i]] * np.sign(T[i, colmap[i]]) for i in range(K)])
    # sign-align model unrotated PCs to human PCs too (for visual comparability)
    Mphi_a = np.column_stack([Mphi[:, j] * (1 if tuck(Hphi[:, j], Mphi[:, j]) >= 0 else -1)
                              for j in range(K)])

    panels = [(1, 1, "Human — unrotated PCA", Hphi, [f"PC{i+1}" for i in range(K)]),
              (1, 2, "Human — varimax", Hvar, HUMAN_LABELS),
              (2, 1, "LLM cohort — unrotated PCA", Mphi_a, [f"PC{i+1}" for i in range(K)]),
              (2, 2, "LLM cohort — varimax (cols matched to human)", Mvar_a, HUMAN_LABELS)]
    vmax = 0.6
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[2] for p in panels],
                        horizontal_spacing=0.08, vertical_spacing=0.07)
    for r, c, _, Lmat, cols in panels:
        Z = Lmat[order]
        fig.add_trace(go.Heatmap(z=Z, x=cols, y=list(range(len(order))),
            zmin=-vmax, zmax=vmax, colorscale="RdBu", reversescale=False,
            showscale=(r == 2 and c == 2),
            colorbar=dict(title="loading", thickness=12)), row=r, col=c)
        for b in bounds:
            fig.add_hline(y=b - 0.5, line_width=0.6, line_color="rgba(0,0,0,0.35)",
                          row=r, col=c)
        fig.update_yaxes(autorange="reversed", showticklabels=False, row=r, col=c)
        fig.update_xaxes(side="top", row=r, col=c, tickfont=dict(size=11))
    fig.update_layout(
        title=dict(text="<b>Adjective factor loadings: human vs LLM, unrotated vs varimax</b>"
                   "<br><sub>523 adjectives (rows, ordered by human Big-Five blocks). "
                   "Unrotated: a big general 1st component. Varimax: simple structure. "
                   "Human varimax = clean Big Five; LLM varimax recovers A/E/N, loses C/O.</sub>",
                   x=0.02),
        width=1080, height=1080, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/factor_rotation_compare.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1080, height=1080, scale=2)
    print(f"wrote {out} and .png")
    print("model varimax col -> human factor:",
          {HUMAN_LABELS[i]: f"M{colmap[i]+1} (cong {T[i,colmap[i]]:+.2f})" for i in range(K)})


if __name__ == "__main__":
    main()
