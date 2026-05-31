#!/usr/bin/env python3
"""Re-clustered affect-region heatmap from the 525-PDA human correlation matrix.

Reads results/adjectives/escs_525pda_corr_raw.json, selects an affect/valence +
prosocial-warmth region, orders it by hierarchical clustering, and renders the
correlation submatrix. The point: in HUMAN ratings the region splits by VALENCE
(positive-affect vs negative-affect, anti-correlated) plus a prosocial sub-block
— it does NOT merge into one undifferentiated "affect" cluster the way the
model/encoder lexical geometry does (the W13 §3.9 affect-axis merge).

Usage: .venv/bin/python scripts/adjective_affect_heatmap.py
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

SRC = Path("results/adjectives/escs_525pda_corr_raw.json")
OUT = Path("results/adjectives/affect_block_heatmap.html")

# Selected by full adjective label (resolved case-insensitively against the
# matrix). Positive-affect / negative-affect / prosocial-warmth seeds.
WANT = [
    "Cheerful", "Joyful", "Happy", "Glad", "Good-humored", "Good-natured",
    "Positive", "Pleased", "Laughing",
    "Angry", "Sad", "Unhappy", "Irritated", "Moody", "Crabby", "Negative",
    "Nervous", "Tense",
    "Sympathetic", "Kind", "Warm", "Warm-hearted", "Kind-hearted",
]


def main():
    d = json.load(open(SRC))
    C = np.array(d["correlation_matrix"])
    labels = d["labels"]
    lab_lower = {l.lower(): i for i, l in enumerate(labels)}

    idx, names = [], []
    for w in WANT:
        i = lab_lower.get(w.lower())
        if i is None:
            print(f"  (not found, skipping: {w})")
            continue
        idx.append(i); names.append(w)
    S = C[np.ix_(idx, idx)]

    # Hierarchical order on 1-r distance.
    dist = 1 - S
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    So = S[np.ix_(order, order)]
    names_o = [names[i] for i in order]

    fig = go.Figure(go.Heatmap(
        z=So, x=names_o, y=names_o, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=False,
        colorbar=dict(title="r", thickness=14, len=0.85),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        title=dict(text=(f"<b>525-PDA human affect region, hierarchically ordered "
                         f"(N={d['n_respondents']})</b><br><sub>Raw self-ratings. "
                         f"Positive- and negative-affect blocks ANTI-correlate "
                         f"(valence-signed) — human behavior splits where the model's "
                         f"lexical geometry merges (W13 §3.9).</sub>"),
                   x=0.02, xanchor="left"),
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=760, height=720,
        margin=dict(l=120, r=40, t=110, b=120),
        font=dict(family="Helvetica, Arial, sans-serif"),
        plot_bgcolor="white",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="cdn")
    fig.write_image(OUT.with_suffix(".png"), width=760, height=720, scale=2)
    print(f"wrote {OUT} and .png ({len(names_o)} adjectives)")


if __name__ == "__main__":
    main()
