#!/usr/bin/env python3
"""2D visualization of facet directions (MDS on cosine distance) + PCA variance.

Produces results/facet_cluster.html — 2x2 scatter with 4 models, color by
parent HEXACO trait. Hover shows facet name.

Usage:
    python scripts/facet_viz.py
"""

import json
from collections import defaultdict
from pathlib import Path

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import extract_meandiff_vectors as mdx


MODELS = {
    # Small cohort (weeks 1–6).
    "Llama":   "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma":   "google/gemma-3-4b-it",
    "Phi4":    "microsoft/Phi-4-mini-instruct",
    "Qwen":    "Qwen/Qwen2.5-3B-Instruct",
    # Phase-1 larger cohort.
    "Llama8":  "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma12": "google/gemma-3-12b-it",
    "Qwen7":   "Qwen/Qwen2.5-7B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
TRAIT_COLORS = {
    "H": "#1f77b4",  # blue
    "E": "#d62728",  # red
    "X": "#ff7f0e",  # orange
    "A": "#2ca02c",  # green
    "C": "#9467bd",  # purple
    "O": "#8c564b",  # brown
}
FORMAT = "chat"
CACHE_DIR_OLD = Path("results/phase_b_cache")
CACHE_DIR_NEW = Path("results/phase_b_cache_stratified")
HOLDOUT_FILE = Path("instruments/contrast_pairs_holdout.json")
STRATIFIED_FILE = Path("instruments/contrast_pairs_stratified.json")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def extract_facet_dirs(model_tag, pair_source, method="md_proj", use_stratified=False):
    """method: 'md_proj' (mean-diff + neutral-PC project), 'md_raw', or 'lr'.
    pair_source: parsed JSON of either holdout or stratified pairs file.
    use_stratified: if True, pull ph_tr/pl_tr from stratified cache; else ph_h/pl_h from holdout cache.
    """
    cache_dir = CACHE_DIR_NEW if use_stratified else CACHE_DIR_OLD
    # Neutral always from old cache — same per-model neutral regardless of source
    neutral = torch.load(CACHE_DIR_OLD / f"{model_tag}_neutral_{FORMAT}.pt", weights_only=False)
    if isinstance(neutral, torch.Tensor):
        neutral = neutral.numpy()

    facet_names, dir_rows, parents = [], [], []
    common_layer = None

    for trait in TRAITS:
        blob = torch.load(cache_dir / f"{model_tag}_{trait}_{FORMAT}_pairs.pt", weights_only=False)
        if use_stratified:
            ph, pl = blob["ph_tr"], blob["pl_tr"]
            pairs_meta = blob["pairs"]
        else:
            ph, pl = blob["ph_h"], blob["pl_h"]
            pairs_meta = pair_source["traits"][trait]["pairs"]

        if common_layer is None:
            n_layers = ph.shape[1]
            common_layer = int(round(n_layers * 2 / 3))

        by_facet = defaultdict(list)
        for i, p in enumerate(pairs_meta):
            by_facet[p["facet"]].append(i)

        for facet, idxs in by_facet.items():
            d = (ph[idxs, common_layer, :] - pl[idxs, common_layer, :]).float().numpy()
            if method == "lr":
                n = len(idxs)
                X = np.vstack([d / 2, -d / 2])
                y = np.array([1] * n + [0] * n)
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
                direction = lr.coef_[0]
            else:
                diff = d.mean(axis=0)
                if method == "md_proj":
                    try:
                        pcs, _, _ = mdx.compute_pc_projection(
                            torch.from_numpy(neutral[:, common_layer, :]), 0.5
                        )
                        direction = mdx.project_out_pcs(diff, pcs)
                    except Exception:
                        direction = diff
                else:
                    direction = diff
            facet_names.append(facet)
            parents.append(trait)
            dir_rows.append(unit(direction))

    return facet_names, parents, np.vstack(dir_rows), common_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["md_proj", "md_raw", "lr"], default="md_proj")
    parser.add_argument("--source", choices=["holdout", "stratified"], default="holdout")
    args = parser.parse_args()
    method = args.method
    use_stratified = (args.source == "stratified")

    if use_stratified:
        with open(STRATIFIED_FILE) as f:
            pair_source = json.load(f)
    else:
        with open(HOLDOUT_FILE) as f:
            pair_source = json.load(f)

    import math
    n_models = len(MODELS)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(MODELS.keys()),
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    pca_summary = []
    for i, (short, repo) in enumerate(MODELS.items()):
        row, col = i // n_cols + 1, i % n_cols + 1
        tag = safe(repo)
        facets, parents, D, L = extract_facet_dirs(tag, pair_source, method=method, use_stratified=use_stratified)

        # PCA variance (how many dims explain the structure?)
        pca = PCA()
        pca.fit(D)
        cev = np.cumsum(pca.explained_variance_ratio_)
        pc1, pc2, pc3 = float(cev[0]), float(cev[1]), float(cev[2])
        pca_summary.append((short, L, pc1, pc2, pc3))

        # MDS on cosine distance
        cos = D @ D.T
        dist = 1 - cos
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=0, normalized_stress="auto")
        coords = mds.fit_transform(dist)

        # One scatter trace per trait so legend works cleanly
        for t in TRAITS:
            mask = [j for j, p in enumerate(parents) if p == t]
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1],
                    mode="markers+text",
                    marker=dict(size=14, color=TRAIT_COLORS[t],
                                line=dict(width=1, color="white")),
                    text=[facets[j] for j in mask],
                    textposition="top center", textfont=dict(size=9),
                    name=t, legendgroup=t, showlegend=(i == 0),
                    hovertemplate="<b>%{text}</b><br>trait: " + t + "<extra></extra>",
                ),
                row=row, col=col,
            )
        # subplot axis titles
        fig.update_xaxes(title_text=f"MDS1 (PC1={pc1:.2f}, PC2={pc2:.2f})", row=row, col=col)
        fig.update_yaxes(title_text="MDS2", row=row, col=col)

    fig.update_layout(
        title=dict(
            text="HEXACO facet directions (MDS on cosine distance, 2/3-depth layer, chat format)",
            x=0.5,
        ),
        height=450 * n_rows, width=1200,
        hovermode="closest",
        legend=dict(title="HEXACO trait", orientation="h", y=-0.05, x=0.5, xanchor="center"),
    )

    parts = []
    if method != "md_proj": parts.append(method)
    if use_stratified: parts.append("stratified")
    suffix_ = "_" + "_".join(parts) if parts else ""
    out = Path(f"results/facet_cluster{suffix_}.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")

    # Cosine heatmap per model + averaged view, ordered by trait then facet
    hm_rows = math.ceil((n_models + 1) / n_cols)  # +1 for mean panel
    last_idx = n_models  # zero-based index where the mean panel goes
    last_row, last_col = last_idx // n_cols + 1, last_idx % n_cols + 1
    # Build specs grid; mark trailing cells (after the mean panel) as None
    total_cells = hm_rows * n_cols
    specs = []
    for r in range(hm_rows):
        row_specs = []
        for c in range(n_cols):
            cell = r * n_cols + c
            row_specs.append(None if cell > n_models else {})
        specs.append(row_specs)

    hm = make_subplots(
        rows=hm_rows, cols=n_cols,
        subplot_titles=list(MODELS.keys()) + ["Mean across models"],
        specs=specs,
        horizontal_spacing=0.12, vertical_spacing=0.10,
    )
    cos_sum = None
    canonical_labels = None
    cos_per_model = {}
    for i, (short, repo) in enumerate(MODELS.items()):
        row, col = i // n_cols + 1, i % n_cols + 1
        tag = safe(repo)
        facets, parents, D, _ = extract_facet_dirs(tag, pair_source, method=method, use_stratified=use_stratified)
        order = sorted(range(len(facets)), key=lambda j: (TRAITS.index(parents[j]), facets[j]))
        labels = [f"{parents[j]}:{facets[j]}" for j in order]
        if canonical_labels is None:
            canonical_labels = labels
        cos = (D[order] @ D[order].T)
        cos_sum = cos if cos_sum is None else cos_sum + cos
        cos_per_model[short] = cos
        # Mask diagonal (trivially 1) so colorscale isn't dominated by self-similarity
        cos_display = cos.copy()
        np.fill_diagonal(cos_display, np.nan)
        hm.add_trace(
            go.Heatmap(
                z=cos_display, x=labels, y=labels,
                colorscale="RdBu_r", zmin=-0.4, zmax=0.4,
                showscale=(i == 0),
                hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>cos=%{z:+.3f}<extra></extra>",
            ),
            row=row, col=col,
        )
        hm.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=row, col=col)
        hm.update_yaxes(tickfont=dict(size=8), autorange="reversed", row=row, col=col)

    # Mean across models, plotted in the trailing slot
    cos_mean = cos_sum / len(MODELS)
    cos_mean_display = cos_mean.copy()
    np.fill_diagonal(cos_mean_display, np.nan)
    hm.add_trace(
        go.Heatmap(
            z=cos_mean_display, x=canonical_labels, y=canonical_labels,
            colorscale="RdBu_r", zmin=-0.4, zmax=0.4, showscale=False,
            hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>mean cos=%{z:+.3f}<extra></extra>",
        ),
        row=last_row, col=last_col,
    )
    hm.update_xaxes(tickangle=-60, tickfont=dict(size=9), row=last_row, col=last_col)
    hm.update_yaxes(tickfont=dict(size=9), autorange="reversed", row=last_row, col=last_col)

    hm.update_layout(
        title=dict(
            text=f"HEXACO facet cosine similarity — method={method} (rows/cols ordered H→O)",
            x=0.5,
        ),
        height=600 * hm_rows, width=1400,
    )
    hm_out = Path(f"results/facet_cosine_heatmap{suffix_}.html")
    hm.write_html(str(hm_out))
    print(f"Wrote {hm_out}")

    # Cross-model cosine-matrix similarity: how alike are the 24x24 matrices?
    # Vectorize the upper triangle (excluding diagonal) and compute pairwise Pearson r.
    print("\n=== Pairwise correlation between cosine matrices (upper-tri off-diagonal) ===")
    iu = np.triu_indices(len(canonical_labels), k=1)
    vecs = {short: cos_per_model[short][iu] for short in MODELS}
    names = list(MODELS.keys())
    print(f"{'':>10s} " + " ".join(f"{n:>8s}" for n in names))
    for a in names:
        row_vals = []
        for b in names:
            if a == b:
                row_vals.append("    1.000")
            else:
                r = float(np.corrcoef(vecs[a], vecs[b])[0, 1])
                row_vals.append(f"{r:+8.3f}")
        print(f"{a:>10s} " + " ".join(row_vals))

    print(f"\nPCA cumulative variance:")
    print(f"{'model':>6s}  {'layer':>5s}  {'PC1':>5s}  {'PC2':>5s}  {'PC3':>5s}")
    for m, L, pc1, pc2, pc3 in pca_summary:
        print(f"{m:>6s}  {L:>5d}  {pc1:>.3f}  {pc2:>.3f}  {pc3:>.3f}")


if __name__ == "__main__":
    main()
