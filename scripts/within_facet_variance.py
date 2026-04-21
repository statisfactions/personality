#!/usr/bin/env python3
"""Within-facet variance structure: do the ~35 pair-diffs per facet lie near
a single line (1D facet axis) or form a cloud?

Natural next step after within_trait_variance.py (which showed within-trait
pair-diffs are diffuse). With the stratified dataset we have enough pairs per
facet (median 35) to ask the question at finer granularity.

Output:
  - results/within_facet_variance.html — scree plots
  - results/within_facet_variance.json — raw numbers
  - stdout summary per (model, trait, facet) + per-facet summary across models

Usage:
    python scripts/within_facet_variance.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import extract_meandiff_vectors as mdx


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
    "Qwen":  "Qwen/Qwen2.5-3B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
TRAIT_COLORS = {"H": "#1f77b4", "E": "#d62728", "X": "#ff7f0e",
                "A": "#2ca02c", "C": "#9467bd", "O": "#8c564b"}
FORMAT = "chat"
CACHE_DIR_OLD = Path("results/phase_b_cache")
CACHE_DIR_STRAT = Path("results/phase_b_cache_stratified")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def pca_stats(D, k=10):
    if D.shape[0] < 2:
        return None
    k_eff = min(k, D.shape[0] - 1)
    pca = PCA(n_components=k_eff)
    pca.fit(D)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    total_var = float(np.sum(D ** 2))
    mean_dir = unit(D.mean(axis=0))
    coherence = float(np.sum((D @ mean_dir) ** 2)) / total_var if total_var > 0 else 0.0
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
        "n": int(D.shape[0]),
    }


def analyze_facets(model_tag, neutral_np):
    """Returns dict[trait][facet] -> pca_stats."""
    results = defaultdict(dict)
    common_layer = None
    for trait in TRAITS:
        blob = torch.load(CACHE_DIR_STRAT / f"{model_tag}_{trait}_{FORMAT}_pairs.pt",
                          weights_only=False)
        ph_tr, pl_tr = blob["ph_tr"], blob["pl_tr"]
        pairs = blob["pairs"]
        if common_layer is None:
            common_layer = int(round(ph_tr.shape[1] * 2 / 3))

        by_facet = defaultdict(list)
        for i, p in enumerate(pairs):
            by_facet[p["facet"]].append(i)

        # Neutral PCs at this layer (once per trait — same neutral)
        pcs = None
        try:
            pcs, _, _ = mdx.compute_pc_projection(
                torch.from_numpy(neutral_np[:, common_layer, :]), 0.5
            )
        except Exception:
            pass

        for facet, idxs in by_facet.items():
            diffs = (ph_tr[idxs, common_layer, :] - pl_tr[idxs, common_layer, :]).float().numpy()
            if pcs is not None:
                diffs_proj = np.array([mdx.project_out_pcs(d, pcs) for d in diffs])
            else:
                diffs_proj = diffs
            stats = pca_stats(diffs_proj)
            if stats:
                stats["layer"] = common_layer
                results[trait][facet] = stats
    return results, common_layer


def main():
    all_results = {}
    for short, repo in MODELS.items():
        tag = safe(repo)
        neutral = torch.load(CACHE_DIR_OLD / f"{tag}_neutral_{FORMAT}.pt", weights_only=False)
        if isinstance(neutral, torch.Tensor):
            neutral = neutral.numpy()
        all_results[short], L = analyze_facets(tag, neutral)
        print(f"\n--- {short} (layer {L}) ---")

    # Print per (model, trait, facet) table
    print(f"\n{'=' * 110}")
    print(f"WITHIN-FACET PCA — ~35 pair-diffs per facet, MD-projected, 2/3-depth layer")
    print(f"{'=' * 110}")
    print(f"{'model':>6s} {'trait':>2s} {'facet':>25s} {'n':>3s}  "
          f"{'PC1':>5s} {'PC1+2':>6s} {'PC1+5':>6s}  {'MD_dir':>6s}  {'PR':>5s}")
    for short in MODELS:
        for trait in TRAITS:
            for facet, s in sorted(all_results[short].get(trait, {}).items()):
                cev = s["cumulative_explained_variance"]
                print(f"{short:>6s} {trait:>2s} {facet:>25s} {s['n']:>3d}  "
                      f"{cev[0]:>5.3f} {cev[1]:>6.3f} "
                      f"{cev[4] if len(cev) >= 5 else cev[-1]:>6.3f}  "
                      f"{s['coherence_mean_direction']:>6.3f}  "
                      f"{s['participation_ratio']:>5.1f}")

    # Compare: within-facet vs within-trait MD coherence per model
    print(f"\n{'=' * 80}")
    print(f"FACET vs TRAIT MD-coherence (does aggregating across facets dilute the axis?)")
    print(f"{'=' * 80}")
    # Load within-trait results (stratified) for comparison
    trait_path = Path("results/within_trait_variance_stratified.json")
    trait_data = json.loads(trait_path.read_text()) if trait_path.exists() else {}
    print(f"{'model':>6s} {'trait':>2s}  {'trait MD':>8s}  {'mean facet MD':>13s}  "
          f"{'max facet MD':>12s}  facets sorted by MD coherence:")
    for short in MODELS:
        for trait in TRAITS:
            facets = all_results[short].get(trait, {})
            if not facets:
                continue
            facet_mds = [(f, v["coherence_mean_direction"]) for f, v in facets.items()]
            facet_mds.sort(key=lambda x: -x[1])
            trait_md = trait_data.get(short, {}).get(trait, {}).get("unit", {}).get("coherence_mean_direction")
            if trait_md is None:
                trait_md = trait_data.get(short, {}).get(trait, {}).get("raw", {}).get("coherence_mean_direction", 0.0)
            mean_f = np.mean([m for _, m in facet_mds])
            max_f = max(m for _, m in facet_mds)
            top = " | ".join(f"{f}:{m:.2f}" for f, m in facet_mds)
            print(f"{short:>6s} {trait:>2s}  {trait_md:>8.3f}  {mean_f:>13.3f}  "
                  f"{max_f:>12.3f}  {top}")

    # Scree plot
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(MODELS.keys()),
                        horizontal_spacing=0.1, vertical_spacing=0.12)
    for i, short in enumerate(MODELS):
        row, col = i // 2 + 1, i % 2 + 1
        for trait in TRAITS:
            for facet, s in sorted(all_results[short].get(trait, {}).items()):
                cev = s["cumulative_explained_variance"]
                x = list(range(1, len(cev) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x, y=cev, mode="lines",
                        line=dict(color=TRAIT_COLORS[trait], width=1.5),
                        opacity=0.6,
                        name=trait,
                        legendgroup=trait, showlegend=(i == 0 and facet == sorted(all_results[short][trait].keys())[0]),
                        hovertemplate=f"<b>{trait}:{facet}</b> PC%{{x}}=%{{y:.3f}}<extra></extra>",
                    ),
                    row=row, col=col,
                )
        fig.add_hline(y=1.0, line=dict(dash="dot", color="gray", width=1),
                      row=row, col=col)
        fig.update_xaxes(title_text="# PCs", row=row, col=col)
        fig.update_yaxes(title_text="cumulative variance", range=[0, 1.05],
                         row=row, col=col)
    fig.update_layout(
        title=dict(
            text="Within-facet pair-diff PCA — ~35 pairs per facet, 4 facets per trait<br>"
                 "<sup>One line per facet; colored by parent HEXACO trait</sup>",
            x=0.5,
        ),
        height=800, width=1200,
        legend=dict(title="trait", orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )
    out_html = Path("results/within_facet_variance.html")
    fig.write_html(str(out_html))
    print(f"\nWrote {out_html}")

    out_json = Path("results/within_facet_variance.json")
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
