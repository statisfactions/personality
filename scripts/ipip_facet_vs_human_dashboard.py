#!/usr/bin/env python3
"""W9 §7 visualization: side-by-side comparison of model facet cosine matrices
(meandiff-pcs extraction) against the Johnson IPIP-NEO-300 human facet
correlation matrix (N=145,388).

8 panels in a 2×4 grid: the human matrix in position (0,0) as the anchor, and
the 7 cohort models in the remaining positions, all displayed at matched
color scale. Each panel title shows the model's Pearson r against the human
matrix (computed on flattened upper-triangle entries).

Output: results/facets/ipip_facet_vs_human_dashboard.html

Usage: .venv/bin/python scripts/ipip_facet_vs_human_dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COHORT_ORDER = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]


def main():
    human = json.load(open("instruments/ipip300_human_facet_correlations.json"))
    H = np.array(human["correlation_matrix"])
    facet_lbls = human["facet_order"]
    n_human = human["n"]
    iu = np.triu_indices(30, k=1)
    human_off = H[iu]

    model_data = json.load(open("results/facets/ipip_facet_cluster.json"))

    # Build per-model reordered cosine matrices + r-vs-human
    panels = []  # list of (label, matrix, subtitle)
    panels.append((
        f"Human IPIP-NEO-300 (N={n_human:,})",
        H,
        "Johnson via NeuroQuestAi mirror",
    ))
    for m in COHORT_ORDER:
        if m not in model_data: continue
        model_lbls = model_data[m]["facet_names"]
        reorder = [model_lbls.index(lbl) for lbl in facet_lbls]
        cos = np.array(model_data[m]["cosine_matrix"])[np.ix_(reorder, reorder)]
        r = float(np.corrcoef(human_off, cos[iu])[0, 1])
        panels.append((f"{m}", cos, f"r vs human = {r:+.3f}"))

    # Determine common color scale
    all_off = np.concatenate([p[1][iu] for p in panels])
    vmin = float(np.percentile(all_off, 2))
    vmax = float(np.percentile(all_off, 98))
    # Make symmetric and clip to a reasonable range
    vabs = max(abs(vmin), abs(vmax), 0.3)
    zmin, zmax = -vabs, vabs

    # Short facet labels for axis (e.g. drop "X:" prefix to fit)
    short_lbls = [lbl.split(":")[1] if ":" in lbl else lbl for lbl in facet_lbls]

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "heatmap"}] * 4, [{"type": "heatmap"}] * 4],
        subplot_titles=[f"<b>{p[0]}</b><br><sub>{p[2]}</sub>" for p in panels],
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
    )

    for i, (label, mat, _) in enumerate(panels):
        row = i // 4 + 1
        col = i % 4 + 1
        fig.add_trace(go.Heatmap(
            z=mat, x=short_lbls, y=short_lbls,
            colorscale="RdBu_r", zmin=zmin, zmax=zmax,
            showscale=(i == 0),
            colorbar=dict(title="r / cos", thickness=10, len=0.45, y=0.78, x=1.02) if i == 0 else None,
            hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>value = %{z:+.3f}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=row, col=col)

    # Cohort-wide summary annotation
    rs = [float(np.corrcoef(human_off, p[1][iu])[0, 1]) for p in panels[1:]]
    cohort_r = np.mean(rs)

    fig.update_layout(
        title=dict(
            text=(f"W9 §7: Model facet cosine geometry (meandiff-pcs) vs human IPIP-NEO-300 facet correlations "
                  f"(N={n_human:,}, cohort mean r vs human = {cohort_r:+.3f})"),
            x=0.5, font=dict(size=13),
        ),
        height=750, width=1500,
        margin=dict(t=110),
    )

    # Trait-grouping annotation strip under each heatmap is hard; instead add small
    # trait-block separators via shapes. The facet order is A(6) C(6) E(6) N(6) O(6),
    # so block boundaries are at 6, 12, 18, 24.
    for i in range(len(panels)):
        row = i // 4 + 1
        col = i % 4 + 1
        for bb in [6, 12, 18, 24]:
            fig.add_shape(
                type="line",
                x0=bb - 0.5, x1=bb - 0.5, y0=-0.5, y1=29.5,
                line=dict(color="black", width=0.5),
                xref=f"x{i+1 if i > 0 else ''}",
                yref=f"y{i+1 if i > 0 else ''}",
            )
            fig.add_shape(
                type="line",
                x0=-0.5, x1=29.5, y0=bb - 0.5, y1=bb - 0.5,
                line=dict(color="black", width=0.5),
                xref=f"x{i+1 if i > 0 else ''}",
                yref=f"y{i+1 if i > 0 else ''}",
            )

    out = Path("results/facets/ipip_facet_vs_human_dashboard.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")
    print(f"\nCohort mean r vs human (meandiff-pcs extraction): {cohort_r:+.3f}")
    for label, _, sub in panels[1:]:
        print(f"  {label:>8s}: {sub}")


if __name__ == "__main__":
    main()
