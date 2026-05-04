#!/usr/bin/env python3
"""W8 §9 dashboard: facet-level cosine geometry, IPIP-NEO vs HEXACO.

Single HTML page with three rows:
  1. Per-model within-trait vs across-trait facet cosine (bar chart, both
     instruments).
  2. Per-model nearest-neighbor within-trait + cluster purity (bar chart).
  3. Cross-model 7×7 cosine-matrix correlation heatmaps (one per
     instrument).

Reads from `results/ipip_facet_cluster.json` (W8 §11) and
`results/facet_cluster.json` (W7 §11.5.6 / §11.5.7-style HEXACO output).
Outputs `results/ipip_facet_dashboard.html`.

Usage:
    .venv/bin/python scripts/ipip_facet_dashboard.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COHORT_ORDER = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def cross_model_corr(per_model_matrices):
    models = list(per_model_matrices.keys())
    n = len(models)
    R = np.eye(n)
    for i, j in combinations(range(n), 2):
        a = per_model_matrices[models[i]]
        b = per_model_matrices[models[j]]
        ai = np.array(a)[np.triu_indices(np.array(a).shape[0], k=1)]
        bi = np.array(b)[np.triu_indices(np.array(b).shape[0], k=1)]
        r = float(np.corrcoef(ai, bi)[0, 1])
        R[i, j] = R[j, i] = r
    return models, R


def main():
    ipip = load("results/ipip_facet_cluster.json")
    hex_raw = load("results/facet_cluster.json")
    if ipip is None:
        print("Missing results/ipip_facet_cluster.json")
        return
    if hex_raw is None:
        print("Missing results/facet_cluster.json")
        return

    # Normalize HEXACO data to {model: entry}
    hex_entries = {e["model"]: e for e in hex_raw["summary"]}

    # Build per-model stat rows in cohort order
    rows = []
    for m in COHORT_ORDER:
        ipip_e = ipip.get(m)
        hex_e = hex_entries.get(m)
        rows.append({
            "model": m,
            "ipip_within": ipip_e["within_mean"] if ipip_e else None,
            "ipip_across": ipip_e["across_mean"] if ipip_e else None,
            "ipip_nn_within": ipip_e["nn_within_trait"] if ipip_e else None,
            "ipip_n": ipip_e["n_facets"] if ipip_e else 30,
            "ipip_purity": ipip_e["purity_5"] if ipip_e else None,
            "hex_within": hex_e["within_mean"] if hex_e else None,
            "hex_across": hex_e["across_mean"] if hex_e else None,
            "hex_nn_within": hex_e["nn_within_trait"] if hex_e else None,
            "hex_n": hex_e["n_facets"] if hex_e else 24,
            "hex_purity": hex_e["purity_6"] if hex_e else None,
        })

    # Cross-model correlations
    ipip_mats = {m: ipip[m]["cosine_matrix"] for m in ipip}
    ipip_models, ipip_R = cross_model_corr(ipip_mats)
    ipip_off = ipip_R[np.triu_indices(len(ipip_models), k=1)]

    hex_mats = {e["model"]: e["cosine_matrix"] for e in hex_raw["summary"]
                if "cosine_matrix" in e}
    hex_models, hex_R = cross_model_corr(hex_mats)
    hex_off = hex_R[np.triu_indices(len(hex_models), k=1)]

    # ---- Build figure ----
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}], [{}, {}],
               [{"type": "heatmap"}, {"type": "heatmap"}]],
        subplot_titles=[
            "IPIP-NEO 30 facets — within vs across-trait cosine",
            "HEXACO 24 facets — within vs across-trait cosine",
            "IPIP-NEO — nearest-neighbor within-trait %  +  5-cluster purity",
            "HEXACO — nearest-neighbor within-trait %  +  6-cluster purity",
            f"IPIP-NEO 30×30 cross-model agreement — mean off-diag r = {np.mean(ipip_off):+.3f}",
            f"HEXACO 24×24 cross-model agreement — mean off-diag r = {np.mean(hex_off):+.3f}",
        ],
        row_heights=[0.28, 0.28, 0.44],
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
    )

    models = [r["model"] for r in rows]

    # Row 1: within vs across-trait cosine, IPIP
    fig.add_trace(go.Bar(
        x=models, y=[r["ipip_within"] for r in rows],
        name="within-trait", marker_color="#377eb8",
        text=[f"{r['ipip_within']:+.3f}" if r["ipip_within"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=models, y=[r["ipip_across"] for r in rows],
        name="across-trait", marker_color="#999999",
        text=[f"{r['ipip_across']:+.3f}" if r["ipip_across"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
    ), row=1, col=1)
    fig.update_yaxes(title_text="cosine", range=[-0.05, 0.30], row=1, col=1)

    # Row 1: within vs across-trait cosine, HEXACO
    fig.add_trace(go.Bar(
        x=models, y=[r["hex_within"] for r in rows],
        name="within-trait (HEXACO)", marker_color="#4daf4a",
        text=[f"{r['hex_within']:+.3f}" if r["hex_within"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=models, y=[r["hex_across"] for r in rows],
        name="across-trait (HEXACO)", marker_color="#cccccc",
        text=[f"{r['hex_across']:+.3f}" if r["hex_across"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
        showlegend=False,
    ), row=1, col=2)
    fig.update_yaxes(title_text="cosine", range=[-0.05, 0.30], row=1, col=2)

    # Row 2: NN-within % and cluster purity, IPIP (5-cluster)
    fig.add_trace(go.Bar(
        x=models,
        y=[(r["ipip_nn_within"] / r["ipip_n"]) if r["ipip_nn_within"] is not None else None
           for r in rows],
        name="NN-within %", marker_color="#e41a1c",
        text=[f"{r['ipip_nn_within']}/{r['ipip_n']}" if r["ipip_nn_within"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=models, y=[r["ipip_purity"] for r in rows],
        name="5-cluster purity", marker_color="#984ea3",
        text=[f"{r['ipip_purity']:.3f}" if r["ipip_purity"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
    ), row=2, col=1)
    fig.add_hline(y=0.20, line_dash="dot", line_color="black", opacity=0.4,
                  annotation_text="5-clust chance (20%)",
                  annotation_position="top left", row=2, col=1)
    fig.update_yaxes(range=[0, 1.0], row=2, col=1)

    # Row 2: NN-within % and cluster purity, HEXACO (6-cluster)
    fig.add_trace(go.Bar(
        x=models,
        y=[(r["hex_nn_within"] / r["hex_n"]) if r["hex_nn_within"] is not None else None
           for r in rows],
        name="NN-within %", marker_color="#e41a1c",
        text=[f"{r['hex_nn_within']}/{r['hex_n']}" if r["hex_nn_within"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
        showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=models, y=[r["hex_purity"] for r in rows],
        name="6-cluster purity", marker_color="#984ea3",
        text=[f"{r['hex_purity']:.3f}" if r["hex_purity"] is not None else ""
              for r in rows],
        textposition="outside", textfont=dict(size=10),
        showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=0.167, line_dash="dot", line_color="black", opacity=0.4,
                  annotation_text="6-clust chance (16.7%)",
                  annotation_position="top left", row=2, col=2)
    fig.update_yaxes(range=[0, 1.0], row=2, col=2)

    # Row 3: cross-model heatmaps
    fig.add_trace(go.Heatmap(
        z=ipip_R, x=ipip_models, y=ipip_models,
        colorscale="RdBu_r", zmin=0.7, zmax=1.0,
        text=[[f"{v:+.2f}" for v in r] for r in ipip_R],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="r", thickness=10, len=0.30, y=0.18, x=1.02),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
        showlegend=False,
    ), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    fig.add_trace(go.Heatmap(
        z=hex_R, x=hex_models, y=hex_models,
        colorscale="RdBu_r", zmin=0.7, zmax=1.0,
        text=[[f"{v:+.2f}" for v in r] for r in hex_R],
        texttemplate="%{text}", textfont=dict(size=10),
        showscale=False,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
        showlegend=False,
    ), row=3, col=2)
    fig.update_yaxes(autorange="reversed", row=3, col=2)

    fig.update_layout(
        title=dict(
            text=("W8 §9: Facet-level cosine geometry — IPIP-NEO 30 facets vs HEXACO 24 facets "
                  "(7-model cohort, neutral-PC-projected at ~2/3 depth)"),
            x=0.5, font=dict(size=15),
        ),
        height=1200, width=1500,
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=1.04,
                    xanchor="center", x=0.25),
    )

    out = Path("results/ipip_facet_dashboard.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
