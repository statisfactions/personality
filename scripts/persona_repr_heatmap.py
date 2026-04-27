#!/usr/bin/env python3
"""Heatmap of persona representation back-mapping (W7 §11.5.4 #4).

Two figures:

1. Per-model 5x5 cross-correlation heatmap: sampled z (rows) vs projection
   (cols), one panel per model in the 7-model cohort. Shows the diagonal
   (per-trait reconstruction) and off-diagonal coupling structure side-by-
   side. Phi4 should visibly deviate from the rest on the A row/col.

2. Cross-model agreement heatmap per trait: 7x7 model x model agreement
   on persona projections, one panel per trait. Phi4's anti-correlation
   on A (and weak coupling on N) shows up as cool-colored row/column.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_heatmap.py
"""

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
TRAITS = ["A", "C", "E", "N", "O"]
SUFFIX = "response-position"


def load(model):
    path = Path(f"results/persona_repr_mapping_{model}_{SUFFIX}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    data = {m: load(m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    models_present = list(data.keys())
    print(f"Loaded {len(models_present)} models: {models_present}")

    # ----- Figure 1: per-model 5x5 cross-correlation heatmap -----
    n_cols = min(len(models_present), 4)
    n_rows = math.ceil(len(models_present) / n_cols)

    fig1 = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[
            f"{m}<br>"
            f"<sub>diag mean = {np.mean([data[m]['diagonal_correlations'][t] for t in TRAITS]):+.3f}</sub>"
            for m in models_present
        ],
        horizontal_spacing=0.07, vertical_spacing=0.18,
    )

    for i, m in enumerate(models_present):
        row, col = i // n_cols + 1, i % n_cols + 1
        cross = np.array(data[m]["cross_correlation"])
        fig1.add_trace(
            go.Heatmap(
                z=cross,
                x=TRAITS, y=TRAITS,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                showscale=(i == 0),
                colorbar=dict(title="Pearson r", thickness=12, len=0.6, x=1.0)
                    if i == 0 else None,
                text=[[f"{v:+.2f}" for v in r] for r in cross],
                texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate=(
                    "<b>sampled z[%{y}]</b><br>"
                    "<b>projection[%{x}]</b><br>"
                    "r = %{z:+.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
        fig1.update_yaxes(autorange="reversed", row=row, col=col,
                          title_text="sampled z" if col == 1 else "")
        fig1.update_xaxes(row=row, col=col,
                          title_text="projection" if row == n_rows else "")

    fig1.update_layout(
        title=dict(
            text=("Persona representation back-mapping (W7 §11.5.9): "
                  "sampled z (rows) → projection on trait directions (cols), "
                  "across 7-model cohort. "
                  "Diagonal = per-trait reconstruction r. "
                  "Off-diagonal = cross-trait projection coupling."),
            x=0.5, font=dict(size=14),
        ),
        height=380 * n_rows + 80,
        width=420 * n_cols,
    )
    out1 = Path("results/persona_repr_heatmap_per_model.html")
    fig1.write_html(str(out1))
    print(f"Wrote {out1}")

    # ----- Figure 2: per-trait cross-model agreement heatmap -----
    # For each trait, build models × models matrix of agreement on projections
    proj = {m: {t: np.array([pd["projections"][t] for pd in data[m]["persona_data"]])
                for t in TRAITS}
            for m in models_present}

    fig2 = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"Trait {t}" for t in TRAITS] + [""],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for ti, t in enumerate(TRAITS):
        row, col = positions[ti]
        m_arr = np.zeros((len(models_present), len(models_present)))
        for i, a in enumerate(models_present):
            for j, b in enumerate(models_present):
                if i == j:
                    m_arr[i, j] = 1.0
                else:
                    m_arr[i, j] = float(np.corrcoef(proj[a][t], proj[b][t])[0, 1])
        fig2.add_trace(
            go.Heatmap(
                z=m_arr,
                x=models_present, y=models_present,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                showscale=(ti == 0),
                colorbar=dict(title="r", thickness=12, len=0.45, x=1.0, y=0.78)
                    if ti == 0 else None,
                text=[[f"{v:+.2f}" for v in r] for r in m_arr],
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
            ),
            row=row, col=col,
        )
        fig2.update_yaxes(autorange="reversed", row=row, col=col)

    fig2.update_layout(
        title=dict(
            text=("Cross-model agreement on persona projections (W7 §11.5.9, cohort): "
                  "Pearson r between model pairs' projection vectors across 50 personas, "
                  "per trait. Phi4 should be visibly anti-correlated on A."),
            x=0.5, font=dict(size=14),
        ),
        height=850, width=1380,
    )
    out2 = Path("results/persona_repr_heatmap_cross_model.html")
    fig2.write_html(str(out2))
    print(f"Wrote {out2}")

    # ----- Figure 3: scatter of sampled z vs projection per trait, all models -----
    # 5 traits × 7 models = 35 panels would be too many; just do 5 traits combined
    fig3 = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"Trait {t}" for t in TRAITS] + [""],
        horizontal_spacing=0.08, vertical_spacing=0.15,
    )
    colors = {
        "Gemma": "#4daf4a", "Llama": "#377eb8", "Phi4": "#e41a1c",
        "Qwen": "#984ea3", "Gemma12": "#a6d96a", "Llama8": "#74add1", "Qwen7": "#cab2d6",
    }
    for ti, t in enumerate(TRAITS):
        row, col = positions[ti]
        for m in models_present:
            zs = [pd["z_scores"][t] for pd in data[m]["persona_data"]]
            prs = [pd["projections"][t] for pd in data[m]["persona_data"]]
            fig3.add_trace(
                go.Scatter(
                    x=zs, y=prs,
                    mode="markers",
                    name=m,
                    marker=dict(size=6, color=colors.get(m, "#888"), opacity=0.65),
                    legendgroup=m, showlegend=(ti == 0),
                    hovertemplate=f"<b>{m}</b><br>z=%{{x:.2f}}<br>proj=%{{y:.2f}}<extra></extra>",
                ),
                row=row, col=col,
            )
        fig3.update_xaxes(title_text=f"sampled z[{t}]", row=row, col=col)
        fig3.update_yaxes(title_text=f"projection[{t}]", row=row, col=col)

    fig3.update_layout(
        title=dict(
            text="Persona z-score → projection scatter, per trait (all 7 models)",
            x=0.5, font=dict(size=14),
        ),
        height=850, width=1380,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
    )
    out3 = Path("results/persona_repr_heatmap_scatter.html")
    fig3.write_html(str(out3))
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
