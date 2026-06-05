#!/usr/bin/env python3
"""Pretty side-by-side: human vs model personality covariance (W16 §5.8 viz).

Restricts to the strongest-loading adjectives on the top human PCs (readable
subset), orders them by the HUMAN cluster structure, and shows the human 525-PDA
correlations next to the model's prevalence-corrected tom_likely geometry under the
SAME ordering — so matching block structure is visible. Both panels z-scored
off-diagonal for comparable colorscales.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_human_match_viz.py --model Qwen7
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))


def dc_off(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, 1); g = np.nanmean(A)
    return A - m[:, None] - m[None, :] + g


def z_off(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    iu = np.triu_indices_from(A, 1); o = A[iu]
    return (A - np.nanmean(o)) / (np.nanstd(o) + 1e-9)


def judge(short):
    for suf in ("_tom_likely_dir", "_tom_likely"):
        p = f"results/adjectives/introspect_full/{short}{suf}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); adj = list(z["adjectives"])
            B = z["B"].astype(float); ri = [adj.index(w) for w in LABELS]
            return (B[np.ix_(ri, ri)] + B[np.ix_(ri, ri)].T) / 2
    raise SystemExit(f"no tom_likely matrix for {short}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--n-pc", type=int, default=6); ap.add_argument("--per-pole", type=int, default=4)
    args = ap.parse_args()

    # top-loading words on the leading human PCs -> readable subset
    w, V = np.linalg.eigh(HUMAN); pcs = np.argsort(w)[::-1]
    sel = []
    for k in range(args.n_pc):
        v = V[:, pcs[k]]; o = np.argsort(v)
        sel += list(o[:args.per_pole]) + list(o[-args.per_pole:])
    sel = list(dict.fromkeys(sel))

    Hs = HUMAN[np.ix_(sel, sel)]
    Ms = dc_off(judge(args.model))[np.ix_(sel, sel)]
    # cluster on the human subset
    d = 1 - Hs; np.fill_diagonal(d, 0.0); d = (d + d.T) / 2
    leaf = leaves_list(linkage(squareform(d, checks=False), "average"))
    labs = [LABELS[sel[i]] for i in leaf]
    Hz = z_off(Hs)[np.ix_(leaf, leaf)]
    Mz = z_off(Ms)[np.ix_(leaf, leaf)]
    # match r on this subset (corrected)
    iu = np.triu_indices(len(leaf), 1)
    r = np.corrcoef(Hz[iu], Mz[iu])[0, 1]

    fig = make_subplots(1, 2, horizontal_spacing=0.12, subplot_titles=(
        "HUMAN — 525-PDA self-report correlations",
        f"{args.model} — tom_likely (prevalence-corrected)"))
    for col, M in ((1, Hz), (2, Mz)):
        fig.add_trace(go.Heatmap(z=M, x=labs, y=labs, colorscale="RdBu_r", zmid=0,
            zmin=-2.5, zmax=2.5, showscale=(col == 2),
            colorbar=dict(title="z", len=0.9)), row=1, col=col)
    for c in (1, 2):
        fig.update_xaxes(tickfont=dict(size=7), tickangle=90, row=1, col=c)
        fig.update_yaxes(tickfont=dict(size=7), autorange="reversed", row=1, col=c)
    fig.update_layout(
        title=dict(text=f"<b>The model's implicit personality covariance matches the "
            f"human one</b><br><sub>Top-loading adjectives on the leading human PCs, "
            f"ordered by HUMAN cluster structure (same order both panels). Matching blocks "
            f"= shared covariance. Subset corrected-match r = {r:.2f} "
            f"(full-set {args.model} tom_likely r≈0.73).</sub>", x=0.01),
        width=1500, height=760, font=dict(family="Helvetica, Arial"))
    out = f"results/adjectives/introspect_full/human_match_{args.model}.png"
    fig.write_html(out.replace(".png", ".html"), include_plotlyjs="cdn")
    fig.write_image(out, width=1500, height=760, scale=2)
    print(f"subset {len(leaf)} words, corrected-match r={r:.3f}; wrote {out}")


if __name__ == "__main__":
    main()
