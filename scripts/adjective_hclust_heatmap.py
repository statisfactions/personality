#!/usr/bin/env python3
"""Full 525x525 human matrix ordered by HIERARCHICAL clustering (vs varimax).

Companion to adjective_factor_heatmap.py. Average-linkage on 1-r distance,
ordered by the dendrogram leaf order; cut at k clusters and compared against the
varimax nearest-factor assignment (adjusted Rand index) so we can see whether
the two methods carve the adjective space the same way. Prints each cluster's
most-central members and where the crux affect words land.

Usage: .venv/bin/python scripts/adjective_hclust_heatmap.py [--ipsatize] [--k 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from adjective_factor_heatmap import varimax


def factor_assignment(C, k):
    """varimax nearest-factor labels, for comparison with hclust."""
    n = C.shape[0]
    evals, evecs = np.linalg.eigh(C)
    o = np.argsort(evals)[::-1]
    Phi = evecs[:, o][:, :k] * np.sqrt(np.clip(evals[o][:k], 0, None))
    L = varimax(Phi)
    return np.argmax(np.abs(L), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipsatize", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    mode = "ipsatized" if args.ipsatize else "raw"

    d = json.load(open(f"results/adjectives/escs_525pda_corr_{mode}.json"))
    C = np.array(d["correlation_matrix"], dtype=float)
    labels = d["labels"]
    n = C.shape[0]
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)

    dist = 1 - C
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    clusters = fcluster(Z, t=args.k, criterion="maxclust")

    # Cluster sizes + agreement with varimax.
    factors = factor_assignment(C, args.k)
    ari = adjusted_rand_score(clusters, factors)
    sizes = {c: int((clusters == c).sum()) for c in sorted(set(clusters))}
    print(f"\n=== {mode} 525-PDA hierarchical (average, 1-r), cut k={args.k} ===")
    print(f"  cluster sizes: {sizes}")
    print(f"  adjusted Rand index vs varimax nearest-factor: {ari:+.3f}")

    # Most-central members per cluster (highest mean within-cluster r).
    for c in sorted(set(clusters)):
        members = np.where(clusters == c)[0]
        if len(members) == 1:
            print(f"  C{c} (n=1): {labels[members[0]]}")
            continue
        sub = C[np.ix_(members, members)]
        central = members[np.argsort(-sub.mean(axis=1))]
        print(f"  C{c} (n={len(members)}): {', '.join(labels[i] for i in central[:7])}")

    # Crux affect words -> cluster.
    low = {l.lower(): i for i, l in enumerate(labels)}
    watch = ["Cheerful", "Happy", "Joyful", "Sympathetic", "Kind", "Warm",
             "Angry", "Irritated", "Sad", "Moody", "Nervous"]
    print("  --- crux affect words -> cluster ---")
    for w in watch:
        i = low.get(w.lower())
        if i is not None:
            print(f"    {w:<13} -> C{clusters[i]}")

    # Heatmap in dendrogram order with cluster block separators.
    Co = C[np.ix_(order, order)]
    clu_o = clusters[order]
    bounds = [i for i in range(1, n) if clu_o[i] != clu_o[i-1]]
    fig = go.Figure(go.Heatmap(
        z=Co, zmin=-1, zmax=1, colorscale="RdBu", reversescale=False,
        colorbar=dict(title="r", thickness=14, len=0.85),
        hoverinfo="text",
        text=[[f"{labels[order[i]]} x {labels[order[j]]}<br>r={Co[i,j]:+.2f}"
               for j in range(n)] for i in range(n)],
    ))
    for b in bounds:
        fig.add_vline(x=b - 0.5, line_width=1.2, line_color="black")
        fig.add_hline(y=b - 0.5, line_width=1.2, line_color="black")
    fig.update_layout(
        title=dict(text=(f"<b>525-PDA human adjective correlations ({mode}), "
                         f"hierarchical order, k={args.k} (N={d['n_respondents']})</b>"
                         f"<br><sub>Average linkage on 1-r. ARI vs varimax "
                         f"nearest-factor = {ari:+.3f}. See console for cluster "
                         f"members.</sub>"),
                   x=0.02, xanchor="left"),
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False,
                                                     autorange="reversed"),
        width=820, height=800, margin=dict(l=40, r=40, t=110, b=40),
        font=dict(family="Helvetica, Arial, sans-serif"), plot_bgcolor="white",
    )
    out = Path(f"results/adjectives/escs_525pda_hclust_heatmap_{mode}_k{args.k}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.with_suffix(".png"), width=820, height=800, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
