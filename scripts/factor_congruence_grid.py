#!/usr/bin/env python3
"""C&C-style factor congruence grid (W13 §3.11).

Tucker congruence (= cosine between loading vectors, ~corr between factors) among
factor solutions. 2x2 of 5x5 heatmaps:
  [Human PCs × Model PCs (unrotated)]   [Human Big Five × Model factors (varimax)]
  [Human: PCs × Big Five]               [Model: PCs × own factors]
Top row = human-vs-model alignment at each rotation; bottom row = how each side's
unrotated PCs map onto its varimax factors (the cor(PC1 vs A) C&C reports).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/factor_congruence_grid.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_geom import model_matrix
from factor_rotation_compare import phi_var

AENCO = ["A", "E", "N", "C", "O"]


def tuck(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cong(Lr, Lc):
    return np.array([[tuck(Lr[:, i], Lc[:, j]) for j in range(5)] for i in range(5)])


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    Hphi, Hvar = phi_var(H)
    Cs = [model_matrix(p, labels)[1]
          for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt"))]
    Mphi, Mvar = phi_var(np.mean(Cs, 0))

    hPC = [f"hPC{i+1}" for i in range(5)]; mPC = [f"mPC{i+1}" for i in range(5)]
    Mlab = ["M1", "M2", "M3", "M4", "M5"]
    panels = [
        (1, 1, "Human PCs × Model PCs (unrotated)", cong(Hphi, Mphi), mPC, hPC),
        (1, 2, "Human Big Five × Model factors (varimax)", cong(Hvar, Mvar), Mlab, AENCO),
        (2, 1, "Human: unrotated PCs × Big Five", cong(Hphi, Hvar), AENCO, hPC),
        (2, 2, "Model: unrotated PCs × own factors", cong(Mphi, Mvar), Mlab, mPC),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[2] for p in panels],
                        horizontal_spacing=0.13, vertical_spacing=0.13)
    for r, c, _, Z, xlab, ylab in panels:
        fig.add_trace(go.Heatmap(
            z=Z, x=xlab, y=ylab, zmin=-1, zmax=1, colorscale="RdBu",
            reversescale=False, showscale=(r == 1 and c == 2),
            colorbar=dict(title="congruence", thickness=12),
            text=np.round(Z, 2), texttemplate="%{text}",
            textfont=dict(size=10)), row=r, col=c)
        fig.update_yaxes(autorange="reversed", row=r, col=c)
        fig.update_xaxes(side="top", row=r, col=c)
    fig.update_layout(
        title=dict(text="<b>Factor congruence: human vs LLM, unrotated vs varimax</b>"
                   "<br><sub>Tucker congruence (cosine of loadings). Top-right is the "
                   "C&C metric: A/E/N find a matching LLM factor (~.5–.7), C/O do not "
                   "(|cong|≤.35). Bottom row: each side's general PC1 vs its Big Five.</sub>",
                   x=0.02),
        width=1040, height=1000, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/factor_congruence_grid.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1040, height=1000, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
