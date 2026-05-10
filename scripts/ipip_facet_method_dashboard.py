#!/usr/bin/env python3
"""W9 §1 dashboard: compare 5 extraction methods × 7-model cohort.

Three sections:
  1. Per-method cohort summary (within / across cosine, NN-within, purity)
     as grouped bars across models, one column per method.
  2. Cross-model 7×7 cosine-matrix correlation heatmaps for the two
     informative methods (`meandiff-pcs` and `single-ipip-mean`) side
     by side. The 3 degenerate methods (single-zero/single-neutral/
     single-pcs) are summarized in the first section only.
  3. Per-model 30×30 facet cosine matrices for `single-ipip-mean`
     (the new W9 method) — 7-panel grid.

Input: results/facets/ipip_facet_cluster*.json
Output: results/facets/ipip_facet_method_dashboard.html

Usage: .venv/bin/python scripts/ipip_facet_method_dashboard.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COHORT_ORDER = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
METHODS = ["meandiff-pcs", "single-zero", "single-neutral", "single-pcs", "single-ipip-mean"]
METHOD_COLORS = {
    "meandiff-pcs":     "#377eb8",
    "single-zero":      "#999999",
    "single-neutral":   "#bbbbbb",
    "single-pcs":       "#cccccc",
    "single-ipip-mean": "#e41a1c",
}


def load_method(name):
    fname = "ipip_facet_cluster.json" if name == "meandiff-pcs" else f"ipip_facet_cluster_{name}.json"
    p = Path("results/facets") / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def cross_model_corr(data):
    """Build a per-model agreement matrix: pairwise r on flattened upper-triangle cosines."""
    models = [m for m in COHORT_ORDER if m in data]
    n = len(models)
    R = np.eye(n)
    for i, j in combinations(range(n), 2):
        a = np.array(data[models[i]]["cosine_matrix"])
        b = np.array(data[models[j]]["cosine_matrix"])
        iu = np.triu_indices(a.shape[0], k=1)
        r = float(np.corrcoef(a[iu], b[iu])[0, 1])
        R[i, j] = R[j, i] = r
    return models, R


def main():
    method_data = {m: load_method(m) for m in METHODS}

    # ---- Section 1: per-method cohort summary stats ----
    # Build a wide table: rows = method, cols = (model, metric)
    summary = {}
    for method, data in method_data.items():
        if data is None: continue
        for model in COHORT_ORDER:
            if model not in data: continue
            d = data[model]
            summary.setdefault(method, {})[model] = {
                "within": d["within_mean"], "across": d["across_mean"],
                "nn_pct": d["nn_within_trait"] / d["n_facets"],
                "purity": d["purity_5"],
            }

    # ---- Section 2: cross-model correlation for the two informative methods ----
    mpcs_models, mpcs_R = cross_model_corr(method_data["meandiff-pcs"])
    sipm_models, sipm_R = cross_model_corr(method_data["single-ipip-mean"])
    mpcs_off = mpcs_R[np.triu_indices(len(mpcs_models), k=1)]
    sipm_off = sipm_R[np.triu_indices(len(sipm_models), k=1)]

    # ---- Section 3: per-model 30×30 cosine matrices for single-ipip-mean ----
    sipm_data = method_data["single-ipip-mean"]
    facet_names = sipm_data[COHORT_ORDER[0]]["facet_names"]

    # Build figure
    fig = make_subplots(
        rows=4, cols=4,
        specs=[
            # Row 1: 4 stat bars (within, across, NN-pct, purity) × all 5 methods × 7 models
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 2}, None, {"colspan": 2}, None],
            # Row 3: cross-model heatmaps for meandiff-pcs (W8 §9) and single-ipip-mean
            [{"type": "heatmap", "colspan": 2}, None, {"type": "heatmap", "colspan": 2}, None],
            # Row 4: per-model 30×30 single-ipip-mean cosine matrices (just 4 of 7; cohort shape comparison)
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
        ],
        subplot_titles=[
            "Within-trait cosine — by model × method",
            "Across-trait cosine — by model × method",
            "NN-within fraction — by model × method",
            "5-cluster purity — by model × method",
            f"meandiff-pcs cross-model agreement (mean r = {np.mean(mpcs_off):+.3f})",
            f"single-ipip-mean cross-model agreement (mean r = {np.mean(sipm_off):+.3f})",
            "single-ipip-mean: Qwen (3B)", "single-ipip-mean: Phi4",
            "single-ipip-mean: Llama8", "single-ipip-mean: Gemma12",
        ],
        row_heights=[0.2, 0.2, 0.30, 0.30],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    # Section 1: 4 grouped bar charts
    for col_idx, (metric, title, row, col) in enumerate([
        ("within", "Within-trait cosine", 1, 1),
        ("across", "Across-trait cosine", 1, 3),
        ("nn_pct", "NN-within fraction", 2, 1),
        ("purity", "5-cluster purity", 2, 3),
    ]):
        for method in METHODS:
            ys = [summary.get(method, {}).get(m, {}).get(metric) for m in COHORT_ORDER]
            fig.add_trace(go.Bar(
                x=COHORT_ORDER, y=ys, name=method,
                marker_color=METHOD_COLORS[method],
                showlegend=(col_idx == 0),  # legend once
                legendgroup=method,
                hovertemplate=f"<b>%{{x}}</b> ({method})<br>{metric} = %{{y:.3f}}<extra></extra>",
            ), row=row, col=col)

    # Chance lines for NN and purity
    fig.add_hline(y=5/29, line_dash="dot", line_color="black", opacity=0.4,
                  annotation_text="NN chance 17.2%", annotation_position="top right",
                  row=2, col=1)
    fig.add_hline(y=0.20, line_dash="dot", line_color="black", opacity=0.4,
                  annotation_text="5-clust chance 20%", annotation_position="top right",
                  row=2, col=3)
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=3)
    fig.update_yaxes(range=[0, 0.8], row=2, col=1)
    fig.update_yaxes(range=[0, 0.8], row=2, col=3)

    # Section 2: cross-model heatmaps
    fig.add_trace(go.Heatmap(
        z=mpcs_R, x=mpcs_models, y=mpcs_models,
        colorscale="RdBu_r", zmin=0.0, zmax=1.0,
        text=[[f"{v:+.2f}" for v in r] for r in mpcs_R],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="r", thickness=10, len=0.22, y=0.40, x=0.46),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
        showlegend=False,
    ), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    fig.add_trace(go.Heatmap(
        z=sipm_R, x=sipm_models, y=sipm_models,
        colorscale="RdBu_r", zmin=0.0, zmax=1.0,
        text=[[f"{v:+.2f}" for v in r] for r in sipm_R],
        texttemplate="%{text}", textfont=dict(size=9),
        showscale=False,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
        showlegend=False,
    ), row=3, col=3)
    fig.update_yaxes(autorange="reversed", row=3, col=3)

    # Section 3: per-model 30×30 facet cosine matrices for 4 representative models
    panel_models = ["Qwen", "Phi4", "Llama8", "Gemma12"]
    for ci, m in enumerate(panel_models):
        cos = np.array(sipm_data[m]["cosine_matrix"])
        fig.add_trace(go.Heatmap(
            z=cos, x=facet_names, y=facet_names,
            colorscale="RdBu_r", zmin=-0.5, zmax=1.0,
            showscale=(ci == 0),
            colorbar=dict(title="cos", thickness=8, len=0.22, y=0.10, x=1.02) if ci == 0 else None,
            hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>cos = %{z:+.3f}<extra></extra>",
            showlegend=False,
        ), row=4, col=ci + 1)
        fig.update_xaxes(showticklabels=False, row=4, col=ci + 1)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=4, col=ci + 1)

    fig.update_layout(
        title=dict(
            text=("W9 §1: IPIP-NEO facet geometry — 5 extraction methods × 7-model cohort. "
                  "single-zero/neutral/pcs are anisotropy-degenerate; "
                  "single-ipip-mean is the matched-format baseline."),
            x=0.5, font=dict(size=14),
        ),
        height=1500, width=1500,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    out = Path("results/facets/ipip_facet_method_dashboard.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
