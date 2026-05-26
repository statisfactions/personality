#!/usr/bin/env python3
"""W13 §3.7: how sensitive is meandiff-pcs facet geometry to the choice of
"neutral" text whose PCs get projected out?

The facet directions are always mean(fwd) - mean(rev) over the IPIP items;
the only thing that varies is which top-k (50%-variance) PCs we subtract,
computed from a chosen neutral activation set. We sweep four neutral sets of
increasing bias and watch whether the geometry (r vs human, within/across,
purity) holds:

  scenarios        300 HEXACO scenario-setups (the W8 default)
  all_ipip         all 288 IPIP item activations
  positive_ipip    forward-keyed IPIP items only
  positive_N_ipip  forward-keyed Neuroticism items only  (adversarial:
                   removing the N axis should specifically damage N facets
                   if meandiff-pcs is fragile)

All from cached activations — no new inference. Per-trait r-vs-human is broken
out so the predicted N-specific damage is visible.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/neutral_sensitivity.py --models Qwen7 Gemma12
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ipip_facet_cluster import (
    ALL_MODELS, safe, extract_or_load_activations, load_neutral,
    build_directions, TRAIT_ORDER,
)

HUMAN = json.load(open("instruments/ipip300_human_facet_correlations.json"))
H = np.array(HUMAN["correlation_matrix"])
HUMAN_FACETS = HUMAN["facet_order"]
IU = np.triu_indices(30, k=1)

NEUTRAL_SETS = ["scenarios", "all_ipip", "positive_ipip", "positive_N_ipip"]


def make_neutral(which, acts, meta, model_name):
    """Return a neutral activation array (n, n_layers+1, hidden) for the PCs."""
    if which == "scenarios":
        return load_neutral(model_name)
    if which == "all_ipip":
        return acts
    if which == "positive_ipip":
        idx = [i for i, m in enumerate(meta) if m["pole"] == "fwd"]
        return acts[idx]
    if which == "positive_N_ipip":
        idx = [i for i, m in enumerate(meta) if m["pole"] == "fwd" and m["trait"] == "N"]
        return acts[idx]
    raise ValueError(which)


def r_vs_human(facet_names, cos):
    """Overall r (flattened upper-tri) and per-trait row r vs the human matrix."""
    # Reorder model cos to the human facet order.
    idx = [facet_names.index(lbl) for lbl in HUMAN_FACETS]
    M = cos[np.ix_(idx, idx)]
    overall = float(np.corrcoef(H[IU], M[IU])[0, 1])
    # Per-trait: mean row-correlation over facets whose label starts with that trait.
    per_trait = {}
    for t in TRAIT_ORDER:
        rows = [i for i, lbl in enumerate(HUMAN_FACETS) if lbl.startswith(t + ":")]
        rs = []
        for i in rows:
            mask = np.ones(30, dtype=bool); mask[i] = False
            rs.append(np.corrcoef(H[i, mask], M[i, mask])[0, 1])
        per_trait[t] = float(np.mean(rs))
    return overall, per_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen7"])
    args = ap.parse_args()

    results = {}
    for model_name in args.models:
        if model_name not in ALL_MODELS:
            print(f"skip unknown model {model_name}")
            continue
        acts, meta, n_layers = extract_or_load_activations(model_name)
        common_layer = int(round(n_layers * 2 / 3))
        print(f"\n========== {model_name} (layer {common_layer}/{n_layers}) ==========")

        # Build facet directions ONCE per neutral set (only the PCs change).
        rows = []
        for which in NEUTRAL_SETS:
            neutral = make_neutral(which, acts, meta, model_name)
            n_neu = neutral.shape[0]
            # build_directions returns (facet_names, D); recompute cos here.
            facet_names, D = build_directions(acts, meta, common_layer,
                                              "meandiff-pcs", neutral)
            # how many PCs were removed (recompute for logging)
            pcs, _, k = __import__("extract_meandiff_vectors").compute_pc_projection(
                torch.from_numpy(neutral[:, common_layer, :]), 0.5)
            cos = D @ D.T
            parent = [f.split(":")[0] for f in facet_names]
            within = [cos[i, j] for i in range(len(facet_names)) for j in range(i+1, len(facet_names)) if parent[i] == parent[j]]
            across = [cos[i, j] for i in range(len(facet_names)) for j in range(i+1, len(facet_names)) if parent[i] != parent[j]]
            ratio = np.mean(within) / max(abs(np.mean(across)), 1e-3)
            overall, per_trait = r_vs_human(facet_names, cos)
            rows.append((which, n_neu, k, overall, per_trait, float(np.mean(within)), float(np.mean(across)), float(ratio)))

        # Print comparison table
        print(f"\n  {'neutral set':<16s} {'n':>4s} {'kPC':>4s} | {'r_all':>6s} | "
              f"{'N':>6s} {'E':>6s} {'O':>6s} {'A':>6s} {'C':>6s} | {'w/a':>5s}")
        print("  " + "-" * 78)
        for which, n_neu, k, overall, per_trait, wm, am, ratio in rows:
            pt = " ".join(f"{per_trait[t]:>+6.3f}" for t in TRAIT_ORDER)
            print(f"  {which:<16s} {n_neu:>4d} {k:>4d} | {overall:>+6.3f} | {pt} | {ratio:>5.2f}")
        results[model_name] = [
            {"neutral_set": w, "n_neutral": n, "k_pcs": k, "r_overall": o,
             "per_trait_r": pt, "within_mean": wm, "across_mean": am, "ratio": r}
            for w, n, k, o, pt, wm, am, r in rows
        ]

    out = Path("results/facets/neutral_sensitivity.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
