#!/usr/bin/env python3
"""Facet-level clustering of HEXACO representation directions.

For each model, extract one MD-projected direction per HEXACO facet (24 total:
6 traits x 4 facets). Cluster the 24 direction vectors and check whether they
group by parent trait (HEXACO prediction) or by some other structure.

Uses cached holdout pair activations from results/phase_b_cache/ and the
facet labels in instruments/contrast_pairs_holdout.json. Extraction at a
common layer per model (2/3 through the stack, per Sofroniew et al.) so
cross-trait cosines are directly comparable.

Usage:
    python scripts/facet_cluster.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
FORMAT = "chat"
CACHE_DIR = Path("results/phase_b_cache")
HOLDOUT_FILE = Path("instruments/contrast_pairs_holdout.json")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def extract_facet_dirs(model_tag, hold):
    """Return (facet_list, dirs_matrix) for one model at ~2/3 depth."""
    neutral = torch.load(CACHE_DIR / f"{model_tag}_neutral_{FORMAT}.pt", weights_only=False)
    if isinstance(neutral, torch.Tensor):
        neutral = neutral.numpy()

    facet_names = []
    dir_rows = []
    common_layer = None

    for trait in TRAITS:
        blob = torch.load(CACHE_DIR / f"{model_tag}_{trait}_{FORMAT}_pairs.pt", weights_only=False)
        ph_h, pl_h = blob["ph_h"], blob["pl_h"]
        hold_pairs = hold["traits"][trait]["pairs"]

        if common_layer is None:
            n_layers = ph_h.shape[1]
            common_layer = int(round(n_layers * 2 / 3))

        by_facet = defaultdict(list)
        for i, p in enumerate(hold_pairs):
            by_facet[p["facet"]].append(i)

        for facet, idxs in by_facet.items():
            diff = (ph_h[idxs, common_layer, :] - pl_h[idxs, common_layer, :]).float().mean(dim=0).numpy()
            try:
                pcs, _, _ = mdx.compute_pc_projection(
                    torch.from_numpy(neutral[:, common_layer, :]), 0.5
                )
                direction = mdx.project_out_pcs(diff, pcs)
            except Exception:
                direction = diff
            facet_names.append(f"{trait}:{facet}")
            dir_rows.append(unit(direction))

    return facet_names, np.vstack(dir_rows), common_layer


def analyze(model_name, facet_names, D):
    """Report cosine matrix, within-trait vs across-trait averages, cluster struct."""
    n = len(facet_names)
    cos = D @ D.T
    parent = [f.split(":")[0] for f in facet_names]

    # Within-trait vs across-trait mean cosines
    within, across = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if parent[i] == parent[j] else across).append(cos[i, j])
    print(f"\n--- {model_name} (layer {common_layer_cache[model_name]}, {n} facets) ---")
    print(f"  within-trait cos:  mean={np.mean(within):+.3f}  median={np.median(within):+.3f}  "
          f"n={len(within)}")
    print(f"  across-trait cos:  mean={np.mean(across):+.3f}  median={np.median(across):+.3f}  "
          f"n={len(across)}")
    print(f"  ratio (within/across): {np.mean(within) / max(abs(np.mean(across)), 1e-3):+.2f}x")

    # For each facet: find its nearest neighbor — same parent trait, or different?
    right = 0
    misgrouped = []
    for i in range(n):
        sims = cos[i].copy()
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        if parent[i] == parent[j]:
            right += 1
        else:
            misgrouped.append((facet_names[i], facet_names[j], cos[i, j]))
    print(f"  nearest-neighbor within-trait: {right}/{n}")
    if misgrouped[:6]:
        print(f"  sample mis-groupings (facet -> nearest, cos):")
        for a, b, c in misgrouped[:6]:
            print(f"    {a:>35s} -> {b:<35s} ({c:+.3f})")

    # Hierarchical cluster: cluster by (1 - cos) distance, linkage=average
    from scipy.cluster.hierarchy import linkage, fcluster
    dist = 1 - cos
    np.fill_diagonal(dist, 0)
    # symmetrize numerical noise
    dist = (dist + dist.T) / 2
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # Cut into 6 clusters (matches HEXACO prediction)
    clusters = fcluster(Z, t=6, criterion="maxclust")
    # Purity: for each predicted cluster, what's its dominant true trait?
    cluster_to_traits = defaultdict(list)
    for c, par in zip(clusters, parent):
        cluster_to_traits[int(c)].append(par)
    purity_num = 0
    for c, trs in sorted(cluster_to_traits.items()):
        counts = {t: trs.count(t) for t in set(trs)}
        top_t, top_n = max(counts.items(), key=lambda x: x[1])
        purity_num += top_n
        print(f"  cluster {c}: n={len(trs):2d}  top={top_t}({top_n})  mix={counts}")
    purity = purity_num / n
    print(f"  6-cluster purity: {purity:.3f} (chance with 6 clusters ~ 1/6 = 0.167)")
    return {
        "model": model_name,
        "within_mean": float(np.mean(within)),
        "across_mean": float(np.mean(across)),
        "nn_within_trait": right,
        "n_facets": n,
        "purity_6": purity,
        "cosine_matrix": cos.tolist(),
    }


common_layer_cache = {}


def main():
    with open(HOLDOUT_FILE) as f:
        hold = json.load(f)

    summary = []
    all_names, all_dirs = {}, {}
    for short, repo in MODELS.items():
        tag = safe(repo)
        names, D, L = extract_facet_dirs(tag, hold)
        common_layer_cache[short] = L
        all_names[short] = names
        all_dirs[short] = D
        summary.append(analyze(short, names, D))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'model':>6s}  {'layer':>5s}  {'within':>7s}  {'across':>7s}  {'NN':>4s}  {'purity@6':>8s}")
    for s in summary:
        print(f"{s['model']:>6s}  {common_layer_cache[s['model']]:>5d}  "
              f"{s['within_mean']:>+7.3f}  {s['across_mean']:>+7.3f}  "
              f"{s['nn_within_trait']:>2d}/{s['n_facets']}  {s['purity_6']:>8.3f}")

    out = Path("results/facet_cluster.json")
    out.write_text(json.dumps({
        "summary": summary,
        "layer_used": {m: common_layer_cache[m] for m in MODELS},
        "facet_names": {m: all_names[m] for m in MODELS},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
