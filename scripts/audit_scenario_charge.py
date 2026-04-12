#!/usr/bin/env python3
"""Audit: does per-scenario signal strength correlate with 'emotional charge'?

Uses the model's own representation as the charge detector: computes each
contrast pair's projection magnitude onto the LDA trait direction at the
best layer. If signal is driven by scenario charge, the top-K and bottom-K
pairs by projection should look qualitatively different in stakes/content.

For each model × trait: fit LDA at best-CV layer, then for every pair
compute |high_proj - low_proj| as the signal strength for that pair.
Report the distribution and the extreme quartiles.

Usage:
    python scripts/audit_scenario_charge.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

REPE_DIR = Path("results/repe")
CONTRAST_PAIRS = Path("instruments/contrast_pairs.json")

MODELS = [
    "google_gemma-3-4b-it",
    "meta-llama_Llama-3.2-3B-Instruct",
    "microsoft_Phi-4-mini-instruct",
]
TRAITS = ["H", "E", "X", "A", "C", "O"]


def best_layer_lda(diffs):
    """Return (best_layer, lda_direction) via 5-fold CV accuracy."""
    n_pairs, n_layers, _ = diffs.shape
    best_acc, best_layer = 0.0, 0
    for l in range(n_layers):
        d = diffs[:, l, :].numpy()
        if np.any(np.isnan(d)) or np.all(d == 0):
            continue
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_pairs + [0] * n_pairs)
        try:
            acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
            if acc > best_acc:
                best_acc, best_layer = acc, l
        except Exception:
            pass

    d = diffs[:, best_layer, :].numpy()
    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    direction = lda.coef_[0]
    direction = direction / np.linalg.norm(direction)
    return best_layer, best_acc, direction


def main():
    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)

    # For each (model, trait), compute per-pair signal strength
    # and aggregate across models to find robust high/low-signal pairs per trait.
    per_pair_signal = {t: np.zeros((len(cp["traits"][t]["pairs"]), len(MODELS)))
                       for t in TRAITS}

    print("Computing per-pair signal strength...\n")
    for mi, model in enumerate(MODELS):
        for trait in TRAITS:
            fpath = REPE_DIR / f"{model}_{trait}_directions.pt"
            if not fpath.exists():
                print(f"  SKIP {model} × {trait} (no file)")
                continue
            data = torch.load(fpath, weights_only=False)
            diffs = data["raw_diffs"]  # (n_pairs, n_layers, hidden_dim)

            layer, acc, direction = best_layer_lda(diffs)

            # Per-pair signal: projection magnitude of the diff onto direction
            d_best = diffs[:, layer, :].numpy()
            projections = d_best @ direction  # (n_pairs,)
            # Signed projection — positive means pair separates correctly along direction
            per_pair_signal[trait][:, mi] = projections

            print(f"  {model} × {trait}: layer={layer:2d}, acc={acc:.3f}, "
                  f"proj range=[{projections.min():+.2f}, {projections.max():+.2f}]")

    # Aggregate: mean signed projection across models (after z-scoring within model)
    print("\n\nAggregating across models (z-scored within each model-trait, then averaged)...")
    agg = {}
    for trait in TRAITS:
        sig = per_pair_signal[trait]  # (n_pairs, n_models)
        # z-score within each model
        zed = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-9)
        # mean across models — robust signal
        agg[trait] = zed.mean(axis=1)

    # For each trait, show top-5 and bottom-5 pairs
    print("\n" + "=" * 80)
    print("HIGH-SIGNAL AND LOW-SIGNAL PAIRS PER TRAIT")
    print("=" * 80)

    for trait in TRAITS:
        pairs = cp["traits"][trait]["pairs"]
        scores = agg[trait]
        # Sort descending: highest signal first
        order = np.argsort(-scores)

        print(f"\n--- {trait} ({cp['traits'][trait]['name']}) ---")
        print(f"  Signal z-score distribution: "
              f"mean={scores.mean():+.2f}, std={scores.std():.2f}, "
              f"min={scores.min():+.2f}, max={scores.max():+.2f}")

        print("\n  TOP 5 (strongest signal, representation clearly separated):")
        for rank, idx in enumerate(order[:5], 1):
            p = pairs[int(idx)]
            print(f"    [{rank}] z={scores[idx]:+.2f}  situation: {p['situation']}")
            print(f"         high: {p['high']}")
            print(f"         low:  {p['low']}")

        print("\n  BOTTOM 5 (weakest signal, representation barely separated or reversed):")
        for rank, idx in enumerate(order[-5:][::-1], 1):
            p = pairs[int(idx)]
            print(f"    [{rank}] z={scores[idx]:+.2f}  situation: {p['situation']}")
            print(f"         high: {p['high']}")
            print(f"         low:  {p['low']}")

    # Also look at the 10 pairs with NEGATIVE signal across models
    # (diff projects in the LOW-trait direction — maybe mislabeled or ambiguous)
    print("\n\n" + "=" * 80)
    print("PAIRS WITH CONSISTENTLY NEGATIVE SIGNAL (potential mislabels/ambiguity)")
    print("=" * 80)
    for trait in TRAITS:
        pairs = cp["traits"][trait]["pairs"]
        scores = agg[trait]
        negatives = [(i, scores[i]) for i in range(len(pairs)) if scores[i] < -0.5]
        if not negatives:
            continue
        print(f"\n--- {trait}: {len(negatives)} pairs with z < -0.5 ---")
        for idx, z in sorted(negatives, key=lambda x: x[1])[:5]:
            p = pairs[int(idx)]
            print(f"  z={z:+.2f}  [{idx}] {p['situation']}")
            print(f"         high: {p['high']}")
            print(f"         low:  {p['low']}")


if __name__ == "__main__":
    main()
