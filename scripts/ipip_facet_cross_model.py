#!/usr/bin/env python3
"""Cross-model agreement on IPIP facet cosine geometry (W8 §11).

Loads results/ipip_facet_cluster.json (the 30×30 cosine matrices per
cohort model produced by ipip_facet_cluster.py) and asks: do different
models agree on the geometric structure of IPIP-NEO facets?

Per W7 §8.4 the answer for Goldberg markers / HEXACO contrast pairs /
IPIP Likert items was YES (within-stimulus-type cross-model r=0.93-0.99).
W8 §11 question: does this hold for IPIP-NEO behavioral-item cosine
geometry too, and how does it compare to the HEXACO facet geometry
agreement?

Output:
- results/ipip_facet_cross_model.json — pairwise model cosine-matrix
  correlations (Pearson r on flattened upper triangle excluding diagonal)
- results/ipip_facet_cross_model_heatmap.html — pairwise correlation
  heatmap, one panel for IPIP, with HEXACO comparison if available

Usage:
    .venv/bin/python scripts/ipip_facet_cross_model.py
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_cluster_json(path):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def upper_triangle(mat):
    """Return flattened upper-triangle entries (excluding diagonal)."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def cross_model_correlations(per_model_matrices):
    """Pairwise Pearson r between flattened upper-triangle off-diagonals."""
    models = list(per_model_matrices.keys())
    n = len(models)
    R = np.eye(n)
    for i, j in combinations(range(n), 2):
        a = upper_triangle(per_model_matrices[models[i]])
        b = upper_triangle(per_model_matrices[models[j]])
        r = float(np.corrcoef(a, b)[0, 1])
        R[i, j] = R[j, i] = r
    return models, R


def build_heatmap(models, R, title):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=R, x=models, y=models,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in r] for r in R],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="r", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
    ))
    off = R[np.triu_indices(len(models), k=1)]
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Pearson r between flattened upper-triangle off-diagonals "
                 f"(mean off-diag r = {np.mean(off):+.3f})</sub>",
            x=0.5, font=dict(size=14),
        ),
        yaxis=dict(autorange="reversed"),
        height=560, width=720,
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipip", default="results/ipip_facet_cluster.json")
    parser.add_argument("--hexaco", default="results/facet_cluster.json",
                        help="HEXACO comparison data (from facet_cluster.py).")
    args = parser.parse_args()

    ipip_data = load_cluster_json(args.ipip)
    if not ipip_data:
        print(f"No data at {args.ipip} — run scripts/ipip_facet_cluster.py first.")
        return

    # Build per-model matrices for IPIP
    ipip_mats = {m: np.array(s["cosine_matrix"]) for m, s in ipip_data.items()}
    if len(ipip_mats) < 2:
        print(f"Only {len(ipip_mats)} model(s) in IPIP data; need ≥2 for cross-model.")
        return

    ipip_models, ipip_R = cross_model_correlations(ipip_mats)
    ipip_off = ipip_R[np.triu_indices(len(ipip_models), k=1)]
    print(f"\n=== IPIP-NEO 30×30 cosine matrices, cross-model agreement ===")
    print(f"  Models: {ipip_models}")
    print(f"  Mean pairwise r: {np.mean(ipip_off):+.3f}  (range {np.min(ipip_off):+.3f} to {np.max(ipip_off):+.3f})")

    # Build comparison panel
    panels = [("IPIP-NEO 30 facets", ipip_models, ipip_R)]

    # HEXACO comparison if available
    hex_data = load_cluster_json(args.hexaco)
    if hex_data and "summary" in hex_data:
        # facet_cluster.json saves cosine matrices via raw 'cosine_matrix' key
        # if present in the per-model summary entries, otherwise we can't compare.
        # Some versions saved separately; check for what's there.
        # Try to load: each summary entry might have "cosine_matrix" or not.
        hex_mats = {}
        for entry in hex_data.get("summary", []):
            if "cosine_matrix" in entry:
                hex_mats[entry["model"]] = np.array(entry["cosine_matrix"])
        if len(hex_mats) >= 2:
            hex_models, hex_R = cross_model_correlations(hex_mats)
            hex_off = hex_R[np.triu_indices(len(hex_models), k=1)]
            print(f"\n=== HEXACO 24×24 cosine matrices, cross-model agreement (REF) ===")
            print(f"  Models: {hex_models}")
            print(f"  Mean pairwise r: {np.mean(hex_off):+.3f}  (range {np.min(hex_off):+.3f} to {np.max(hex_off):+.3f})")
            panels.append(("HEXACO 24 facets", hex_models, hex_R))
        else:
            print(f"\n  (HEXACO cosine matrices not stored in {args.hexaco}; skipping comparison.)")

    # Save JSON
    out = {
        "ipip": {
            "models": ipip_models,
            "matrix": ipip_R.tolist(),
            "mean_pairwise_r": float(np.mean(ipip_off)),
        }
    }
    if len(panels) > 1:
        out["hexaco"] = {
            "models": panels[1][1],
            "matrix": panels[1][2].tolist(),
            "mean_pairwise_r": float(np.mean(panels[1][2][np.triu_indices(len(panels[1][1]), k=1)])),
        }
    Path("results").mkdir(exist_ok=True)
    json_out = Path("results/ipip_facet_cross_model.json")
    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {json_out}")

    # Heatmap figure
    if len(panels) == 1:
        fig = build_heatmap(panels[0][1], panels[0][2], panels[0][0] + " — cross-model agreement")
    else:
        # Compute subtitles with off-diagonal means baked in
        subtitles = []
        for title, models, R in panels:
            off = R[np.triu_indices(len(models), k=1)]
            subtitles.append(f"{title}<br><sub>mean off-diag r = {np.mean(off):+.3f}</sub>")
        fig = make_subplots(rows=1, cols=2, subplot_titles=subtitles,
                            horizontal_spacing=0.15)
        for ci, (_, models, R) in enumerate(panels):
            fig.add_trace(go.Heatmap(
                z=R, x=models, y=models,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=[[f"{v:+.2f}" for v in r] for r in R],
                texttemplate="%{text}",
                textfont=dict(size=10),
                showscale=(ci == 0),
                hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
            ), row=1, col=ci + 1)
            fig.update_yaxes(autorange="reversed", row=1, col=ci + 1)
        fig.update_layout(
            title=dict(
                text="Cross-model agreement on facet cosine geometry — IPIP-NEO vs HEXACO",
                x=0.5, font=dict(size=14),
            ),
            height=560, width=1280,
        )

    html_out = Path("results/ipip_facet_cross_model_heatmap.html")
    fig.write_html(str(html_out))
    print(f"Saved {html_out}")


if __name__ == "__main__":
    main()
