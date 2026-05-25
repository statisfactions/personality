#!/usr/bin/env python3
"""W12 §8.1b: per-facet row-correlation heatmap (30 facets × 10 cohort models).

For each facet F and model M, computes Pearson(H[F, ¬F], M[F, ¬F]) — the
correlation between F's row of human off-diagonal facet correlations and F's
row of model facet cosines, with the self-cell excluded. This is the same
diagnostic narrated as a table in W9 §7.6; the heatmap makes the cohort
homogeneity (and the Cheerf/Liberal divergences) citable at a glance.

Rows are sorted by mean row-r across the 10 cohort models (descending), so
the tightest-agreement facets (N cluster) appear at the top and the
worst-recovered (A:Sympath, O:Liberal) at the bottom. An eleventh "Mean"
column on the right shows the row mean.

Inputs:
  instruments/ipip300_human_facet_correlations.json
  results/facets/ipip_facet_cluster.json

Output:
  results/facets/facet_row_corr_heatmap.html

Usage: .venv/bin/python scripts/facet_row_corr_heatmap.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


COHORT_ORDER = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7",
                "Gemma27", "Qwen32", "Gemma4"]
# W13 §8.2 axis-of-the-models outliers — rendered to the right of the cohort
# mean separator so they don't dilute the cohort baseline statistics.
EXTRA_MODELS = ["Aya", "FalconMamba"]

OUT_PATH = Path("results/facets/facet_row_corr_heatmap.html")


def row_correlation(H, M, i):
    """Pearson(H[i, ¬i], M[i, ¬i]) — self-cell excluded."""
    mask = np.ones(H.shape[0], dtype=bool)
    mask[i] = False
    return float(np.corrcoef(H[i, mask], M[i, mask])[0, 1])


def main():
    human = json.load(open("instruments/ipip300_human_facet_correlations.json"))
    H = np.array(human["correlation_matrix"])
    facet_lbls = human["facet_order"]
    n_human = human["n"]
    n_facets = len(facet_lbls)
    assert n_facets == 30 and H.shape == (30, 30)

    model_data = json.load(open("results/facets/ipip_facet_cluster.json"))

    def row_corrs(model_short):
        """Return per-facet row correlations for model, in facet_lbls order."""
        model_lbls = model_data[model_short]["facet_names"]
        reorder = [model_lbls.index(lbl) for lbl in facet_lbls]
        M = np.array(model_data[model_short]["cosine_matrix"])[np.ix_(reorder, reorder)]
        return np.array([row_correlation(H, M, i) for i in range(n_facets)])

    # rows = facets, cols = cohort models
    R = np.full((n_facets, len(COHORT_ORDER)), np.nan)
    for j, m in enumerate(COHORT_ORDER):
        if m in model_data:
            R[:, j] = row_corrs(m)

    # Extra models (not part of cohort mean). NaN-fill if missing.
    extra_present = [m for m in EXTRA_MODELS if m in model_data]
    R_extra = (np.full((n_facets, len(extra_present)), np.nan) if not extra_present
               else np.column_stack([row_corrs(m) for m in extra_present]))

    row_mean = np.nanmean(R, axis=1)              # cohort-only mean
    row_sd = np.nanstd(R, axis=1)
    order = np.argsort(-row_mean)

    R_sorted = R[order]
    R_extra_sorted = R_extra[order]
    mean_sorted = row_mean[order]
    sd_sorted = row_sd[order]
    labels_sorted = [facet_lbls[i] for i in order]

    # Layout: [10 cohort cols | cohort mean | 2 extra cols].
    Z = np.column_stack([R_sorted, mean_sorted, R_extra_sorted])
    col_labels = list(COHORT_ORDER) + ["mean"] + extra_present

    # Hover text per cell.
    hover = []
    for ri, lbl in enumerate(labels_sorted):
        row_h = []
        for ci, m in enumerate(col_labels):
            v = Z[ri, ci]
            if m == "mean":
                row_h.append(f"{lbl}<br>cohort mean: {v:+.3f}<br>SD: {sd_sorted[ri]:.3f}")
            elif m in extra_present:
                row_h.append(f"{lbl} @ {m} (§8.2 outlier)<br>row r: {v:+.3f}")
            else:
                row_h.append(f"{lbl} @ {m}<br>row r: {v:+.3f}")
        hover.append(row_h)

    # Diverging colorscale centered at 0; clip to ±1 range.
    vabs = max(0.6, float(np.nanmax(np.abs(Z))))
    zmin, zmax = -vabs, vabs

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=col_labels,
        y=labels_sorted,
        zmin=zmin, zmax=zmax,
        colorscale="RdBu",
        reversescale=False,
        colorbar=dict(title="row r", thickness=14, len=0.85),
        hoverinfo="text",
        text=hover,
        xgap=1, ygap=1,
    ))

    # Vertical separators: before the "mean" column AND before §8.2 outliers.
    fig.add_vline(x=len(COHORT_ORDER) - 0.5, line_width=2, line_color="white")
    if extra_present:
        fig.add_vline(x=len(COHORT_ORDER) + 0.5, line_width=2, line_color="white")

    fig.update_layout(
        title=dict(
            text=(f"<b>Per-facet row correlation: model cosines vs human (N={n_human:,})</b>"
                  f"<br><sub>For each facet F, Pearson(H[F,¬F], M[F,¬F]) — self-cell excluded. "
                  f"Rows sorted by cohort mean. Cohort grand mean = {np.nanmean(R):+.3f}."
                  f"</sub>"),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(tickangle=-30, side="top"),
        yaxis=dict(title="facet (high → low cohort r)",
                   tickfont=dict(size=11), autorange="reversed"),
        width=900, height=900,
        margin=dict(l=110, r=40, t=170, b=40),
        font=dict(family="Helvetica, Arial, sans-serif"),
        plot_bgcolor="white",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_PATH, include_plotlyjs="cdn")
    png_path = OUT_PATH.with_suffix(".png")
    fig.write_image(png_path, width=900, height=900, scale=2)
    print(f"wrote {OUT_PATH}")
    print(f"wrote {png_path}")
    print(f"  cohort grand mean row-r: {np.nanmean(R):+.3f}")
    print(f"  top 5 facets:    {[(l, f'{m:+.3f}') for l, m in zip(labels_sorted[:5], mean_sorted[:5])]}")
    print(f"  bottom 5 facets: {[(l, f'{m:+.3f}') for l, m in zip(labels_sorted[-5:], mean_sorted[-5:])]}")


if __name__ == "__main__":
    main()
