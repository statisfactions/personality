#!/usr/bin/env python3
"""Full 525x525 human correlation matrix, factor-ordered (standard psych viz).

Mimics C&C Fig 2/3 (and our facet charts): factor-analyze the correlation
matrix, assign each adjective to its highest-|loading| factor (nearest-factor),
order adjectives by (factor, signed loading), and render the ordered matrix so
the factor blocks fall on the diagonal. Non-circular: the ordering is derived
from the data, not a hand-picked subset.

Method: PCA on the correlation matrix (loadings = sqrt(eigval)*eigvec for the
top k), Kaiser varimax rotation (matches C&C's PCA+varimax). Factors are
auto-labeled by their top-loading adjectives so we can read what each block is —
in particular whether factor 1 of the RAW ratings is an evaluative/valence axis
(the general-desirability factor C&C ipsatized away).

  --ipsatize : use the within-person-z matrix instead of raw (suppresses the
               evaluative factor, recovers Big-Five-like structure).
  --k N      : number of factors (default 5).

Usage: .venv/bin/python scripts/adjective_factor_heatmap.py [--ipsatize] [--k 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def varimax(Phi, gamma=1.0, q=50, tol=1e-6):
    """Kaiser varimax rotation of a (p x k) loading matrix."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vt
        d = float(s.sum())
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipsatize", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    mode = "ipsatized" if args.ipsatize else "raw"

    src = Path(f"results/adjectives/escs_525pda_corr_{mode}.json")
    d = json.load(open(src))
    C = np.array(d["correlation_matrix"], dtype=float)
    labels = d["labels"]
    n = C.shape[0]
    # Guard scattered NaN (constant cols) before eigendecomposition.
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)

    # PCA on the correlation matrix -> loadings for the top-k components.
    evals, evecs = np.linalg.eigh(C)
    order_desc = np.argsort(evals)[::-1]
    evals, evecs = evals[order_desc], evecs[:, order_desc]
    var_frac = evals[:args.k] / n
    Phi = evecs[:, :args.k] * np.sqrt(np.clip(evals[:args.k], 0, None))  # (n x k)
    L = varimax(Phi)

    # Orient each factor so its dominant pole is positive (sign convention).
    for j in range(args.k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1

    # nearest-factor assignment; order by (factor, signed loading desc).
    assign = np.argmax(np.abs(L), axis=1)
    primary = L[np.arange(n), assign]
    order = sorted(range(n), key=lambda i: (assign[i], -primary[i]))
    Co = C[np.ix_(order, order)]
    assign_o = assign[order]

    # Factor labels from top-loading adjectives (signed).
    print(f"\n=== {mode} 525-PDA, {args.k}-factor varimax (var frac of top-{args.k}: "
          f"{', '.join(f'{v:.3f}' for v in var_frac)}) ===")
    block_bounds, factor_names = [], []
    for j in range(args.k):
        members = [i for i in range(n) if assign[i] == j]
        members.sort(key=lambda i: -L[i, j])
        pos = [labels[i] for i in members[:6]]
        neg = [labels[i] for i in members[::-1][:4] if L[i, j] < 0]
        nm = f"F{j+1} (n={len(members)})"
        factor_names.append(nm)
        block_bounds.append(sum(1 for a in assign_o if a <= j))
        print(f"  {nm}: +{', '.join(pos)}" + (f"   |  -{', '.join(neg)}" if neg else ""))

    # Where do the crux affect words land? (cheerful must NOT share F3 with angry)
    watch = ["Cheerful", "Happy", "Joyful", "Sympathetic", "Kind", "Warm",
             "Angry", "Irritated", "Sad", "Moody", "Nervous"]
    low = {l.lower(): i for i, l in enumerate(labels)}
    print("  --- crux affect words -> factor ---")
    for w in watch:
        i = low.get(w.lower())
        if i is not None:
            print(f"    {w:<13} -> F{assign[i]+1}  (loading {L[i, assign[i]]:+.2f})")

    fig = go.Figure(go.Heatmap(
        z=Co, zmin=-1, zmax=1, colorscale="RdBu", reversescale=False,
        colorbar=dict(title="r", thickness=14, len=0.85),
        hoverinfo="text",
        text=[[f"{labels[order[i]]} x {labels[order[k]]}<br>r={Co[i,k]:+.2f}"
               for k in range(n)] for i in range(n)],
    ))
    # Block separators + factor labels at block centers.
    starts = [0] + block_bounds[:-1]
    for b in block_bounds[:-1]:
        fig.add_vline(x=b - 0.5, line_width=1.5, line_color="black")
        fig.add_hline(y=b - 0.5, line_width=1.5, line_color="black")
    centers = [(starts[j] + block_bounds[j] - 1) / 2 for j in range(args.k)]
    fig.update_layout(
        title=dict(text=(f"<b>525-PDA human adjective correlations ({mode}), "
                         f"{args.k}-factor varimax order (N={d['n_respondents']})</b>"
                         f"<br><sub>Each diagonal block = adjectives whose top loading "
                         f"is that factor. See console for factor top-words.</sub>"),
                   x=0.02, xanchor="left"),
        xaxis=dict(tickmode="array", tickvals=centers, ticktext=factor_names,
                   side="bottom", tickangle=0),
        yaxis=dict(tickmode="array", tickvals=centers, ticktext=factor_names,
                   autorange="reversed"),
        width=820, height=800,
        margin=dict(l=80, r=40, t=110, b=60),
        font=dict(family="Helvetica, Arial, sans-serif"), plot_bgcolor="white",
    )
    out = Path(f"results/adjectives/escs_525pda_factor_heatmap_{mode}_k{args.k}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.with_suffix(".png"), width=820, height=800, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
