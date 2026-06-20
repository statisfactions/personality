"""The 'different in different ways' figure (W17): negative-attribute
syndromes in the ENACT space, vs the same words clustered by the HUMAN grid.

Follow-up to the density-asymmetry result: positive attributes have no
residual structure beyond the halo, negative attributes do. Here we ask
whether the negative residual structure is *recognizable syndromes* and
whether the model's syndromes match the human ones.

  - negative subset: bottom 30% of adjectives by human valence score
  - ENACT clustering: k-means on unit-normalized persona directions (mid layer)
  - HUMAN clustering: k-means on correlation profiles across the same words
  - agreement: adjusted Rand index
  - figure: ENACT-subset PCA, colored by ENACT clusters (left) and HUMAN
    clusters (right); representative words (nearest centroid) labeled

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/viz_neg_syndromes.py \
      --model llama3.2 --k 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import four_grid_compare as fg
from adjective_introspection import GROUPS
from represent_enact_geometry import load_pair

OUT_DIR = Path("results/persona_vectors/figs")
PALETTE = ["#4C78A8", "#E45756", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.30)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    H, labels = fg.load_human()
    idx = {l: i for i, l in enumerate(labels)}
    val = np.array([
        np.nanmean(H[i, [idx[w] for w in GROUPS["pos_eval"]]]) -
        np.nanmean(H[i, [idx[w] for w in GROUPS["neg_eval"]]])
        for i in range(len(labels))])
    neg_i = np.where(val <= np.quantile(val, args.quantile))[0]

    _, Y, common = load_pair(args.model, args.layer)
    cidx = {a: i for i, a in enumerate(common)}
    words = [labels[i] for i in neg_i if labels[i].lower() in cidx]
    E = np.stack([Y[cidx[w.lower()]] for w in words])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    Hsub = np.nan_to_num(H[np.ix_([idx[w] for w in words],
                                  [idx[w] for w in words])])
    print(f"{len(words)} negative adjectives")

    for k in (4, 5, 6):
        km = KMeans(k, n_init=20, random_state=0).fit(E)
        print(f"  k={k}: ENACT silhouette={silhouette_score(E, km.labels_):.3f}")

    km_e = KMeans(args.k, n_init=20, random_state=0).fit(E)
    km_h = KMeans(args.k, n_init=20, random_state=0).fit(Hsub)
    ari = adjusted_rand_score(km_e.labels_, km_h.labels_)
    print(f"\nENACT-vs-HUMAN cluster agreement (ARI, k={args.k}): {ari:.3f}")

    # rosters: nearest-centroid words per ENACT cluster
    rosters = {}
    for c in range(args.k):
        members = np.where(km_e.labels_ == c)[0]
        d = np.linalg.norm(E[members] - km_e.cluster_centers_[c], axis=1)
        rosters[c] = [words[members[j]] for j in np.argsort(d)]
        print(f"  ENACT cluster {c} (n={len(members)}): "
              + ", ".join(rosters[c][:10]))

    # figure: subset's own top-2 PCs, two colorings
    U, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    P = U[:, :2] * s[:2]
    v = 100 * s[:2] ** 2 / (s ** 2).sum()
    fig = make_subplots(cols=2, rows=1, subplot_titles=[
        f"colored by ENACT clusters (k={args.k})",
        f"same points, colored by HUMAN clusters (ARI={ari:.2f})"])
    for col, lab in [(1, km_e.labels_), (2, km_h.labels_)]:
        for c in range(args.k):
            m = lab == c
            fig.add_trace(go.Scatter(
                x=P[m, 0], y=P[m, 1], mode="markers",
                marker=dict(size=6, color=PALETTE[c % len(PALETTE)]),
                text=[w for w, mm in zip(words, m) if mm],
                hoverinfo="text", showlegend=False), 1, col)
    for c in range(args.k):  # label 3 exemplars per ENACT cluster, both panels
        for w in rosters[c][:3]:
            i = words.index(w)
            for col in (1, 2):
                fig.add_annotation(x=P[i, 0], y=P[i, 1], text=w.lower(),
                                   row=1, col=col, font=dict(size=9),
                                   arrowsize=0.6, arrowwidth=1, ax=15, ay=-12)
    fig.update_xaxes(title=f"neg-subset PC1 ({v[0]:.0f}%)")
    fig.update_yaxes(title=f"neg-subset PC2 ({v[1]:.0f}%)")
    fig.update_layout(
        title=(f"{args.model} L{args.layer}: ways of being bad — negative-"
               "adjective syndromes in the ENACT space vs human clustering"),
        width=1200, height=560, template="plotly_white")
    base = OUT_DIR / f"neg_syndromes_{args.model}_L{args.layer}_k{args.k}"
    fig.write_html(f"{base}.html")
    fig.write_image(f"{base}.png", scale=2)

    json.dump({"model": args.model, "k": args.k, "ari": ari,
               "rosters": {c: rosters[c] for c in rosters},
               "human_clusters": {c: [w for w, l in zip(words, km_h.labels_)
                                      if l == c] for c in range(args.k)}},
              open(f"{base}.json", "w"), indent=1)
    print(f"\nsaved {base}.png/.html/.json")


if __name__ == "__main__":
    main()
