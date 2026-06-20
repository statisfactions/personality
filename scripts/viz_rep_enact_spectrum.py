"""Visualize the REPRESENT->ENACT dimension compression (W17).

Two figures:
  1. spectrum: cumulative variance vs PC rank for both channels, with
     participation-ratio (effective dimension) markers.
  2. clouds: side-by-side 2D PCA of the same 523 adjectives in each channel,
     colored by human evaluative score (mean human corr with pos_eval minus
     neg_eval groups), eval-antonym words labeled. Same constellation,
     different room - with the contrast knob visible.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/viz_rep_enact_spectrum.py \
      --model llama3.2 --layer 14
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_introspection import GROUPS
from represent_enact_geometry import load_pair

HUMAN_JSON = "results/adjectives/escs_525pda_corr_raw.json"
OUT_DIR = Path("results/persona_vectors/figs")


def valence_scores(common):
    h = json.load(open(HUMAN_JSON))
    M = np.array(h["correlation_matrix"], float)
    idx = {l.lower(): i for i, l in enumerate(h["labels"])}
    pos = [idx[w.lower()] for w in GROUPS["pos_eval"]]
    neg = [idx[w.lower()] for w in GROUPS["neg_eval"]]
    out = []
    for a in common:
        i = idx[a]
        out.append(np.nanmean(M[i, pos]) - np.nanmean(M[i, neg]))
    return np.array(out)


def pr(s):
    lam = s ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, Y, common = load_pair(args.model, args.layer)
    val = valence_scores(common)
    chans = {"REPRESENT": X, "ENACT": Y}
    colors = {"REPRESENT": "#4C78A8", "ENACT": "#E45756"}

    # --- Fig 1: spectrum -----------------------------------------------------
    fig = go.Figure()
    prs = {}
    for name, Z in chans.items():
        s = np.linalg.svd(Z, compute_uv=False)
        cum = np.cumsum(s ** 2) / (s ** 2).sum()
        prs[name] = pr(s)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(cum) + 1), y=cum, mode="lines",
            name=f"{name} (eff. dim ≈ {prs[name]:.0f})",
            line=dict(color=colors[name], width=3)))
        fig.add_vline(x=prs[name], line_dash="dash",
                      line_color=colors[name], opacity=0.6)
    fig.update_xaxes(type="log", title="principal component rank (log)")
    fig.update_yaxes(title="cumulative variance explained", range=[0, 1.02])
    fig.update_layout(
        title=(f"{args.model} layer {args.layer}: enactment compresses the "
               f"adjective space ({prs['REPRESENT']:.0f} → "
               f"{prs['ENACT']:.0f} effective dimensions)"),
        width=900, height=500, template="plotly_white",
        legend=dict(x=0.55, y=0.15))
    f1 = OUT_DIR / f"spectrum_{args.model}_L{args.layer}"
    fig.write_html(f"{f1}.html")
    fig.write_image(f"{f1}.png", scale=2)

    # --- Fig 2: clouds -------------------------------------------------------
    eval_words = {w.lower() for w in GROUPS["pos_eval"] + GROUPS["neg_eval"]}
    fig = make_subplots(cols=2, rows=1, subplot_titles=[
        f"REPRESENT — diffuse (eff. dim ≈ {prs['REPRESENT']:.0f})",
        f"ENACT — concentrated (eff. dim ≈ {prs['ENACT']:.0f})"])
    for col, (name, Z) in enumerate(chans.items(), start=1):
        U, s, _ = np.linalg.svd(Z, full_matrices=False)
        P = U[:, :2] * s[:2]
        # orient PC1 with valence, PC2 arbitrary-consistent
        if np.corrcoef(P[:, 0], val)[0, 1] < 0:
            P[:, 0] *= -1
        v1 = 100 * s[0] ** 2 / (s ** 2).sum()
        v2 = 100 * s[1] ** 2 / (s ** 2).sum()
        fig.add_trace(go.Scatter(
            x=P[:, 0], y=P[:, 1], mode="markers",
            marker=dict(size=5, color=val, colorscale="RdBu_r",
                        cmid=0, showscale=(col == 2),
                        colorbar=dict(title="human<br>valence", x=1.02)),
            text=common, hoverinfo="text", showlegend=False), 1, col)
        for i, a in enumerate(common):
            if a in eval_words:
                fig.add_annotation(
                    x=P[i, 0], y=P[i, 1], text=a, row=1, col=col,
                    font=dict(size=10), arrowsize=0.7, arrowwidth=1,
                    ax=18, ay=-14)
        fig.update_xaxes(title=f"PC1 ({v1:.0f}% var)", row=1, col=col)
        fig.update_yaxes(title=f"PC2 ({v2:.0f}% var)", row=1, col=col)
    fig.update_layout(
        title=(f"{args.model} layer {args.layer}: the same 523 adjectives in "
               "each channel's top-2 PCs — enactment turns up the contrast "
               "on the evaluative axis"),
        width=1200, height=560, template="plotly_white")
    f2 = OUT_DIR / f"clouds_{args.model}_L{args.layer}"
    fig.write_html(f"{f2}.html")
    fig.write_image(f"{f2}.png", scale=2)

    print(f"effective dims: REPRESENT {prs['REPRESENT']:.1f}  "
          f"ENACT {prs['ENACT']:.1f}")
    print(f"saved {f1}.png/.html and {f2}.png/.html")


if __name__ == "__main__":
    main()
