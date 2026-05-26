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


# Display order: all 12 models by parameter count (ascending) so column
# position tracks scale. The two W13 §8.2 outliers (Aya, FalconMamba) sit
# among the 7–8B models and are marked with "*"; they are EXCLUDED from the
# cohort-mean column so they don't dilute the baseline.
DISPLAY_ORDER = ["Qwen", "Llama", "Phi4", "Gemma",         # 3–4B
                 "FalconMamba", "Qwen7", "Aya", "Llama8",   # 7–8B
                 "Gemma12", "Gemma27", "Gemma4", "Qwen32"]  # 12–32B
COHORT_FOR_MEAN = ["Qwen", "Llama", "Phi4", "Gemma", "Qwen7", "Llama8",
                   "Gemma12", "Gemma27", "Gemma4", "Qwen32"]   # the original 10
OUTLIERS = {"Aya", "FalconMamba"}

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

    # rows = facets, cols = all 12 models (param-sorted display order)
    present = [m for m in DISPLAY_ORDER if m in model_data]
    R_all = np.column_stack([row_corrs(m) for m in present])  # (30, n_models)

    # Cohort mean/SD computed from the original 10 only (outliers excluded).
    cohort_cols = [j for j, m in enumerate(present) if m in COHORT_FOR_MEAN]
    row_mean = np.nanmean(R_all[:, cohort_cols], axis=1)
    row_sd = np.nanstd(R_all[:, cohort_cols], axis=1)
    order = np.argsort(-row_mean)

    R_sorted = R_all[order]
    mean_sorted = row_mean[order]
    sd_sorted = row_sd[order]
    labels_sorted = [facet_lbls[i] for i in order]

    # Layout: [12 model cols (param-sorted) | cohort mean]. Outliers marked "*".
    Z = np.column_stack([R_sorted, mean_sorted])
    col_labels = [m + ("*" if m in OUTLIERS else "") for m in present] + ["mean"]

    hover = []
    for ri, lbl in enumerate(labels_sorted):
        row_h = []
        for ci, m in enumerate(present + ["mean"]):
            v = Z[ri, ci]
            if m == "mean":
                row_h.append(f"{lbl}<br>cohort mean (n=10): {v:+.3f}<br>SD: {sd_sorted[ri]:.3f}")
            elif m in OUTLIERS:
                row_h.append(f"{lbl} @ {m} (§8.2 outlier, excl. from mean)<br>row r: {v:+.3f}")
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

    # Vertical separator before the "mean" column.
    fig.add_vline(x=len(present) - 0.5, line_width=2, line_color="white")

    fig.update_layout(
        title=dict(
            text=(f"<b>Per-facet row correlation: model cosines vs human (N={n_human:,})</b>"
                  f"<br><sub>For each facet F, Pearson(H[F,¬F], M[F,¬F]) — self-cell excluded. "
                  f"Models ordered by size. Cohort grand mean (n=10) = "
                  f"{np.nanmean(row_mean):+.3f}. * = §8.2 outlier (excl. from mean).</sub>"),
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
    print(f"  cohort grand mean row-r (n=10): {np.nanmean(row_mean):+.3f}")
    print(f"  top 5 facets:    {[(l, f'{m:+.3f}') for l, m in zip(labels_sorted[:5], mean_sorted[:5])]}")
    print(f"  bottom 5 facets: {[(l, f'{m:+.3f}') for l, m in zip(labels_sorted[-5:], mean_sorted[-5:])]}")


if __name__ == "__main__":
    main()
