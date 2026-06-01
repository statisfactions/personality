#!/usr/bin/env python3
"""Factor ladder figure (W13 §3.11): varimax factors k=1→10, human vs model.

Two stacked tables (human, model). Rows = number of factors extracted (k=1..10);
columns = factor index (variance-sorted); cells = top-2 adjectives. Shows the
hierarchy build-up: human general factor → clean Big Five; model general factor =
raw evaluation → A/E/N overlay, C only at k=10, O never.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/factor_ladder_figure.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_factor_heatmap import varimax
from adjective_geom import model_matrix

KMAX = 10


def factors(C, labels, k):
    C = np.nan_to_num(C); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    Phi = V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None))
    L = varimax(Phi) if k > 1 else Phi
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    order = np.argsort(-(L ** 2).sum(0))
    return [", ".join(labels[i] for i in np.argsort(-L[:, j])[:2]) for j in order]


def table_columns(C, labels):
    """Return column-lists for a go.Table: ['k', F1..F10] each length KMAX."""
    rows = []  # rows[k-1] = [k, f1, f2, ..., padded to KMAX]
    for k in range(1, KMAX + 1):
        fs = factors(C, labels, k)
        rows.append([str(k)] + fs + [""] * (KMAX - k))
    return [list(col) for col in zip(*rows)]  # transpose to columns


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    Cs = [model_matrix(p, labels)[1]
          for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt"))]
    M = np.mean(Cs, 0)

    header = ["k"] + [f"F{i}" for i in range(1, KMAX + 1)]
    widths = [22] + [88] * KMAX
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                        specs=[[{"type": "table"}], [{"type": "table"}]],
                        subplot_titles=(
        "HUMAN 525-PDA — varimax factors (top-2 adjectives), k = 1 … 10",
        "LLM cohort — varimax factors (top-2 adjectives), k = 1 … 10"))
    for row, C in ((1, H), (2, M)):
        cols = table_columns(C, labels)
        # light blue header for the k column; alternate row shading via cell fill
        cellcolor = [["#eef3f8"] * KMAX] + [
            [("#ffffff" if cols[c][r] else "#f5f5f5") for r in range(KMAX)]
            for c in range(1, KMAX + 1)]
        fig.add_trace(go.Table(
            columnwidth=widths,
            header=dict(values=header, fill_color="#2c3e50",
                        font=dict(color="white", size=11), align="center", height=24),
            cells=dict(values=cols, fill_color=cellcolor,
                       font=dict(size=9), align="left", height=22)),
            row=row, col=1)
    fig.update_layout(
        title=dict(text="<b>Factor ladders: how the varimax structure builds up "
                   "(k = 1 → 10)</b><br><sub>Human general factor → clean Big Five "
                   "(O last). Model general factor = raw evaluation (Disgusting/Awful) → "
                   "A/E/N overlay; Conscientiousness only at k=10, Openness never.</sub>",
                   x=0.02),
        width=1400, height=960, font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/factor_ladders.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1400, height=960, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
