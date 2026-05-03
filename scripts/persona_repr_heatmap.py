#!/usr/bin/env python3
"""Heatmaps for persona representation / instrument-response tracks (W7 §11.5.4 #4).

Generates three figures per mode (representation back-mapping vs Likert
instrument response):

1. Per-model 5x5 cross-correlation grid: ground-truth Σ (empirical sample
   correlation of the 50 sampled z's, seed=42) + one panel per model
   showing sampled z (rows) vs measured value (cols).

2. Cross-model 7x7 agreement per trait: each model's measurement vector
   across 50 personas, correlated with every other model's measurement
   vector. Phi4's anti-correlation on A persona projections is the
   striking find here.

3. Per-trait scatter of sampled z vs measured value, all models color-
   coded. Z-scored within model so cross-model magnitude differences
   (Gemma's residual norms ~100x Llama's) don't squash the visual.

Modes are autodetected — runs whichever has data in results/.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_heatmap.py
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_heatmap.py --mode rep
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_heatmap.py --mode likert
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
TRAITS = ["A", "C", "E", "N", "O"]

# Per-variant suffix applied to result-file patterns + output filenames.
# "markers" reproduces the W7 §11.5.9-10 figures (no suffix).
# "ipip_raw" uses the W8 IPIP-NEO-300 behavioral persona compositions
# (rep against marker-derived directions; Likert with marker rating target).
# "ipip_raw_target_ipip" additionally uses IPIP rating items as Likert targets
# (W8 §4 — only meaningful for the likert mode; rep is identical to ipip_raw).
VARIANTS = {
    "markers": {
        "rep_suffix":    "",
        "likert_suffix": "",
        "title_tag":     "W7 §11.5.9-10 (Goldberg-marker personas)",
        "out_tag":       "",
    },
    "ipip_raw": {
        "rep_suffix":    "_ipip_raw",
        "likert_suffix": "_ipip_raw",
        "title_tag":     "W8 §3 (IPIP-NEO-300 behavioral personas, marker rating target)",
        "out_tag":       "_ipip_raw",
    },
    "ipip_raw_target_ipip": {
        "rep_suffix":    "_ipip_raw",  # same rep file as ipip_raw (marker dirs)
        "likert_suffix": "_ipip_raw_target-ipip",
        "title_tag":     "W8 §4 (IPIP personas, IPIP rating target, marker dirs)",
        "out_tag":       "_ipip_raw_target_ipip",
    },
    "ipip_full": {
        "rep_suffix":    "_ipip_raw_dir-ipip",
        "likert_suffix": "_ipip_raw_target-ipip",
        "title_tag":     "W8 §5 (IPIP personas, IPIP directions, IPIP rating target)",
        "out_tag":       "_ipip_full",
    },
}


def make_modes(variant):
    v = VARIANTS[variant]
    return {
        "rep": {
            "results_pattern": f"results/persona_repr_mapping_{{m}}_response-position{v['rep_suffix']}.json",
            "value_key": "projections",
            "label": "projection",
            "title": f"Persona representation back-mapping — {v['title_tag']}",
            "out_prefix": f"persona_repr_heatmap{v['out_tag']}",
        },
        "likert": {
            "results_pattern": f"results/persona_instrument_response_{{m}}{v['likert_suffix']}.json",
            "value_key": "scored_trait",
            "label": "scored trait (Likert)",
            "title": f"Persona Likert response track — {v['title_tag']}",
            "out_prefix": f"persona_likert_heatmap{v['out_tag']}",
        },
    }


def load(model, mode, modes):
    path = Path(modes[mode]["results_pattern"].format(m=model))
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_values(persona_data, value_key):
    """Return {trait: np.array of length n_personas} from a persona_data list."""
    return {t: np.array([pd[value_key][t] for pd in persona_data]) for t in TRAITS}


def compute_sampled_z_sigma(n=50, seed=42):
    with open("instruments/synthetic_personas.json") as f:
        d = json.load(f)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(d["personas"]), size=n, replace=False)
    selected = [d["personas"][int(i)] for i in idxs]
    Z = np.array([[p["z_scores"][t] for t in TRAITS] for p in selected])
    return np.corrcoef(Z.T)


def make_per_model_figure(data, sigma, mode, modes):
    """Per-model 5x5 cross-correlation: Σ + one panel per model."""
    cfg = modes[mode]
    label = cfg["label"]
    models_present = list(data.keys())

    panels = [("Σ (sampled z)", sigma, True)] + [
        (m, np.array(data[m]["cross_correlation"]), False) for m in models_present
    ]
    n_cols = min(len(panels), 4)
    n_rows = math.ceil(len(panels) / n_cols)

    titles = []
    for label_str, _, is_sigma in panels:
        if is_sigma:
            titles.append(f"<b>{label_str}</b><br><sub>empirical sample corr (N=50)</sub>")
        else:
            m = label_str
            diag = np.mean([data[m]['diagonal_correlations'][t] for t in TRAITS])
            titles.append(f"{m}<br><sub>diag mean = {diag:+.3f}</sub>")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.07, vertical_spacing=0.18,
    )

    for i, (label_str, mat, is_sigma) in enumerate(panels):
        row, col = i // n_cols + 1, i % n_cols + 1
        x_axis_label = "sampled z" if is_sigma else label
        fig.add_trace(
            go.Heatmap(
                z=mat, x=TRAITS, y=TRAITS,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                showscale=(i == 0),
                colorbar=dict(title="r", thickness=12, len=0.6, x=1.0)
                    if i == 0 else None,
                text=[[f"{v:+.2f}" for v in r] for r in mat],
                texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate=(
                    f"<b>%{{y}}</b> (sampled z)<br>"
                    f"<b>%{{x}}</b> ({x_axis_label})<br>"
                    f"r = %{{z:+.3f}}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
        fig.update_yaxes(autorange="reversed", row=row, col=col,
                         title_text="sampled z" if col == 1 else "")
        fig.update_xaxes(row=row, col=col,
                         title_text=x_axis_label if row == n_rows else "")

    fig.update_layout(
        title=dict(
            text=(f"{cfg['title']}: sampled z (rows) → {label} (cols), "
                  f"7-model cohort + ground-truth Σ"),
            x=0.5, font=dict(size=14),
        ),
        height=380 * n_rows + 80,
        width=420 * n_cols,
    )
    return fig


def make_cross_model_figure(data, mode, modes):
    """Cross-model agreement on measurement vectors, per trait."""
    cfg = modes[mode]
    label = cfg["label"]
    models_present = list(data.keys())
    if len(models_present) < 2:
        return None

    vals = {m: get_values(data[m]["persona_data"], cfg["value_key"])
            for m in models_present}

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"Trait {t}" for t in TRAITS] + [""],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    for ti, t in enumerate(TRAITS):
        row, col = positions[ti]
        m_arr = np.zeros((len(models_present), len(models_present)))
        for i, a in enumerate(models_present):
            for j, b in enumerate(models_present):
                if i == j:
                    m_arr[i, j] = 1.0
                else:
                    m_arr[i, j] = float(np.corrcoef(vals[a][t], vals[b][t])[0, 1])
        fig.add_trace(
            go.Heatmap(
                z=m_arr, x=models_present, y=models_present,
                colorscale="RdBu_r", zmin=-1, zmax=1,
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
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=dict(
            text=(f"{cfg['title']}: cross-model agreement on per-persona "
                  f"{label} vectors (Pearson r, 50 personas)"),
            x=0.5, font=dict(size=14),
        ),
        height=850, width=1380,
    )
    return fig


def make_scatter_figure(data, mode, modes, normalize=True):
    """Sampled z vs measured value, per trait, all models. Z-scored within
    model when normalize=True so cross-model magnitude differences don't
    squash the visual."""
    cfg = modes[mode]
    label = cfg["label"]
    models_present = list(data.keys())

    colors = {
        "Gemma": "#4daf4a", "Llama": "#377eb8", "Phi4": "#e41a1c",
        "Qwen": "#984ea3", "Gemma12": "#a6d96a", "Llama8": "#74add1",
        "Qwen7": "#cab2d6",
    }
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"Trait {t}" for t in TRAITS] + [""],
        horizontal_spacing=0.08, vertical_spacing=0.15,
    )

    for ti, t in enumerate(TRAITS):
        row, col = positions[ti]
        for m in models_present:
            zs = np.array([pd["z_scores"][t] for pd in data[m]["persona_data"]])
            vs = np.array([pd[cfg["value_key"]][t] for pd in data[m]["persona_data"]])
            if normalize:
                # z-score within model (this trait) so cross-model is comparable
                if vs.std() > 0:
                    vs = (vs - vs.mean()) / vs.std()
            fig.add_trace(
                go.Scatter(
                    x=zs, y=vs, mode="markers", name=m,
                    marker=dict(size=6, color=colors.get(m, "#888"), opacity=0.65),
                    legendgroup=m, showlegend=(ti == 0),
                    hovertemplate=(
                        f"<b>{m}</b><br>z=%{{x:.2f}}<br>"
                        f"{label}{'(z)' if normalize else ''}=%{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=f"sampled z[{t}]", row=row, col=col)
        ylab = f"{label}[{t}]" + (" (z within model)" if normalize else "")
        fig.update_yaxes(title_text=ylab, row=row, col=col)

    fig.update_layout(
        title=dict(
            text=(f"{cfg['title']}: sampled z → {label}, per trait (all models, "
                  f"{'within-model z-scored' if normalize else 'raw'})"),
            x=0.5, font=dict(size=14),
        ),
        height=850, width=1380,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
    )
    return fig


def render_mode(mode, sigma, modes):
    cfg = modes[mode]
    data = {m: load(m, mode, modes) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    if not data:
        print(f"  [{mode}] no data files found, skipping")
        return
    print(f"  [{mode}] loaded {len(data)} models: {list(data.keys())}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    prefix = cfg["out_prefix"]

    fig1 = make_per_model_figure(data, sigma, mode, modes)
    out1 = out_dir / f"{prefix}_per_model.html"
    fig1.write_html(str(out1))
    print(f"  [{mode}] wrote {out1}")

    fig2 = make_cross_model_figure(data, mode, modes)
    if fig2 is not None:
        out2 = out_dir / f"{prefix}_cross_model.html"
        fig2.write_html(str(out2))
        print(f"  [{mode}] wrote {out2}")
    else:
        print(f"  [{mode}] skipped cross_model (need ≥2 models)")

    fig3 = make_scatter_figure(data, mode, modes, normalize=True)
    out3 = out_dir / f"{prefix}_scatter.html"
    fig3.write_html(str(out3))
    print(f"  [{mode}] wrote {out3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rep", "likert", "all"], default="all")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="markers",
                        help="Persona/rating variant (W7 markers vs W8 IPIP forms).")
    args = parser.parse_args()

    print(f"Variant: {args.variant} ({VARIANTS[args.variant]['title_tag']})")
    modes = make_modes(args.variant)

    sigma = compute_sampled_z_sigma()
    print(f"Σ off-diagonal mean: {np.mean(sigma[~np.eye(5, dtype=bool)]):+.3f}")

    modes_to_run = ["rep", "likert"] if args.mode == "all" else [args.mode]
    for mode in modes_to_run:
        print(f"\n--- Mode: {mode} ---")
        render_mode(mode, sigma, modes)


if __name__ == "__main__":
    main()
