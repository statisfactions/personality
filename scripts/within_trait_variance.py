#!/usr/bin/env python3
"""Within-trait variance structure: do the 50 pair-diffs per trait lie near
a single line (1D latent trait) or form a cloud (bundle of sub-directions)?

For each (model, trait), take all 50 training pair-diffs at ~2/3-depth layer
(MD-projected form: top neutral PCs subtracted), and run PCA. Report:
  - Explained-variance profile (PC1, PC1+2, ..., PC1+10 cumulative)
  - Coherence: variance captured by the mean-diff direction
    (vs. PC1, which is the max-variance *orthogonal-to-mean* direction under
     centered PCA)
  - Participation ratio (effective # of dimensions)

Output:
  - results/within_trait_variance.html     — scree plots per model
  - results/within_trait_variance.json     — raw numbers
  - stdout summary table

Usage:
    python scripts/within_trait_variance.py
"""

import json
from pathlib import Path

import numpy as np
import torch
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
TRAIT_COLORS = {"H": "#1f77b4", "E": "#d62728", "X": "#ff7f0e",
                "A": "#2ca02c", "C": "#9467bd", "O": "#8c564b"}
FORMAT = "chat"
CACHE_DIR_OLD = Path("results/phase_b_cache")
CACHE_DIR_NEW = Path("results/phase_b_cache_stratified")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def pca_stats(D, k=10):
    """Centered PCA + participation ratio + mean-direction coherence on row-matrix D."""
    pca = PCA(n_components=min(k, D.shape[0] - 1))
    pca.fit(D)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    total_var = float(np.sum(D ** 2))
    mean_dir = unit(D.mean(axis=0))
    projs = D @ mean_dir
    coherence = float(np.sum(projs ** 2)) / total_var if total_var > 0 else 0.0

    full = PCA()
    full.fit(D)
    eigs = full.explained_variance_
    eigs = eigs[eigs > 1e-12]
    pr = float((eigs.sum() ** 2) / (eigs ** 2).sum())

    return {
        "explained_variance_ratio": [float(x) for x in evr],
        "cumulative_explained_variance": [float(x) for x in cum],
        "coherence_mean_direction": coherence,
        "participation_ratio": pr,
    }


def analyze_trait(model_tag, trait, neutral_np, cache_dir, k_pcs_to_report=10):
    blob = torch.load(cache_dir / f"{model_tag}_{trait}_{FORMAT}_pairs.pt", weights_only=False)
    ph_tr, pl_tr = blob["ph_tr"], blob["pl_tr"]
    n_layers = ph_tr.shape[1]
    L = int(round(n_layers * 2 / 3))

    diffs = (ph_tr[:, L, :] - pl_tr[:, L, :]).float().numpy()
    try:
        pcs, _, _ = mdx.compute_pc_projection(
            torch.from_numpy(neutral_np[:, L, :]), 0.5
        )
        diffs_proj = np.array([mdx.project_out_pcs(d, pcs) for d in diffs])
    except Exception:
        diffs_proj = diffs

    # Raw (as-is, amplitude preserved) and unit-normalized variants
    raw = pca_stats(diffs_proj, k_pcs_to_report)
    norms = np.linalg.norm(diffs_proj, axis=1, keepdims=True)
    diffs_unit = diffs_proj / (norms + 1e-12)
    normed = pca_stats(diffs_unit, k_pcs_to_report)

    # Amplitude spread — how variable are the norms?
    amp_cv = float(np.std(norms) / (np.mean(norms) + 1e-12))

    return {
        "layer": L,
        "n_pairs": diffs_proj.shape[0],
        "raw": raw,
        "unit": normed,
        "amp_cv": amp_cv,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["old", "stratified"], default="old")
    args = parser.parse_args()
    cache_dir = CACHE_DIR_NEW if args.source == "stratified" else CACHE_DIR_OLD

    results = {}
    for short, repo in MODELS.items():
        tag = safe(repo)
        neutral = torch.load(CACHE_DIR_OLD / f"{tag}_neutral_{FORMAT}.pt", weights_only=False)
        if isinstance(neutral, torch.Tensor):
            neutral = neutral.numpy()
        results[short] = {}
        for trait in TRAITS:
            results[short][trait] = analyze_trait(tag, trait, neutral, cache_dir)

    # Print summary table: raw vs unit-normalized, side-by-side
    print(f"\n{'=' * 115}")
    print(f"{'model':>6s} {'trait':>2s} {'L':>3s}  "
          f"{'--- raw ---':>31s}  {'--- unit-normalized ---':>31s}  {'amp_cv':>6s}")
    print(f"{'':>12s}  {'PC1':>5s} {'PC1+5':>6s} {'MD':>5s} {'PR':>5s}  "
          f"{'PC1':>5s} {'PC1+5':>6s} {'MD':>5s} {'PR':>5s}")
    print("-" * 115)
    for short in MODELS:
        for trait in TRAITS:
            r = results[short][trait]
            rr, ru = r["raw"], r["unit"]
            print(f"{short:>6s} {trait:>2s} {r['layer']:>3d}  "
                  f"{rr['cumulative_explained_variance'][0]:>5.3f} "
                  f"{rr['cumulative_explained_variance'][4]:>6.3f} "
                  f"{rr['coherence_mean_direction']:>5.3f} "
                  f"{rr['participation_ratio']:>5.1f}  "
                  f"{ru['cumulative_explained_variance'][0]:>5.3f} "
                  f"{ru['cumulative_explained_variance'][4]:>6.3f} "
                  f"{ru['coherence_mean_direction']:>5.3f} "
                  f"{ru['participation_ratio']:>5.1f}  "
                  f"{r['amp_cv']:>6.3f}")

    # Scree plot figure — 2x2, one per model, lines per trait
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(MODELS.keys()),
                        horizontal_spacing=0.1, vertical_spacing=0.12)
    for i, short in enumerate(MODELS):
        row, col = i // 2 + 1, i % 2 + 1
        for trait in TRAITS:
            r = results[short][trait]
            # Plot both raw (dashed) and unit-normalized (solid)
            cum_raw = r["raw"]["cumulative_explained_variance"]
            cum_unit = r["unit"]["cumulative_explained_variance"]
            x = list(range(1, len(cum_raw) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x, y=cum_unit,
                    mode="lines+markers",
                    name=trait, legendgroup=trait, showlegend=(i == 0),
                    line=dict(color=TRAIT_COLORS[trait], width=2),
                    marker=dict(size=6),
                    hovertemplate=f"<b>{trait}</b> unit-norm PC%{{x}}=%{{y:.3f}}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=cum_raw,
                    mode="lines",
                    name=f"{trait} (raw)", legendgroup=trait, showlegend=False,
                    line=dict(color=TRAIT_COLORS[trait], width=1, dash="dot"),
                    hovertemplate=f"<b>{trait}</b> raw PC%{{x}}=%{{y:.3f}}<extra></extra>",
                ),
                row=row, col=col,
            )
        # Reference line: isotropic (no structure)
        fig.add_hline(y=1.0, line=dict(dash="dot", color="gray", width=1),
                      annotation_text="1.0 ceiling", annotation_position="bottom right",
                      row=row, col=col)
        fig.update_xaxes(title_text="# PCs", row=row, col=col)
        fig.update_yaxes(title_text="cumulative variance", range=[0, 1.05],
                         row=row, col=col)
    fig.update_layout(
        title=dict(
            text="Within-trait pair-diff PCA (centered) — cumulative explained variance<br>"
                 "<sup>Solid = unit-normalized pair-diffs; dotted = raw (amplitude preserved)</sup>",
            x=0.5,
        ),
        height=800, width=1200,
        legend=dict(title="trait", orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )
    suffix = "_stratified" if args.source == "stratified" else ""
    out_html = Path(f"results/within_trait_variance{suffix}.html")
    fig.write_html(str(out_html))
    print(f"\nWrote {out_html}")

    out_json = Path(f"results/within_trait_variance{suffix}.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
