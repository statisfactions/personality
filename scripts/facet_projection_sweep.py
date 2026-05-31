#!/usr/bin/env python3
"""Projection-robustness sweep for the per-facet row-correlation finding.

The canonical extraction is `meandiff-itempc1` = project out the TOP-1 item PC
(the anisotropy/norm axis) from the fwd-rev contrast. Two questions this answers:

  k=0 (raw):   does meandiff with NO projection collapse to anisotropy nonsense?
               (rgb's prior — the cosine matrix is the anisotropy-sensitive
               quantity even though the fwd-rev subtraction cancels the bulk.)
  k=1,2,3:     is the affect-facet divergence vs human (A:Sympath inverts,
               E:Cheerf fails) a knife-edge artifact of the *specific* top-1
               choice, or stable across how many PCs you strip?

For each model and each k we rebuild the 30x30 facet cosine matrix and correlate
each facet's row (self-cell excluded) against the human IPIP-NEO row. We report
the cohort mean over the canonical 10 (out-of-cohort robustness probes excluded).

This does NOT touch the canonical results/facets/ipip_facet_cluster.json. It is
a notes-only diagnostic (W13 §3.9 follow-up).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/facet_projection_sweep.py
"""

import json
from pathlib import Path

import numpy as np
import torch

import extract_meandiff_vectors as mdx
from ipip_facet_cluster import (
    extract_or_load_activations, FACETS, TRAIT_ORDER, unit, ALL_MODELS,
)

COHORT = ["Qwen", "Llama", "Phi4", "Gemma", "Qwen7", "Llama8",
          "Gemma12", "Gemma27", "Gemma4", "Qwen32"]
K_VALUES = [0, 1, 2, 3]
# Facets to spotlight: the affect-axis divergences + reference points.
SPOTLIGHT = ["A:Sympath", "E:Cheerf", "O:Liberal", "C:Order", "O:Emotion",
             "N:Anger", "A:Altru"]
HUMAN_PATH = "instruments/ipip300_human_facet_correlations.json"
OUT_PATH = Path("results/facets/facet_projection_sweep.json")


def row_correlation(H, M, i):
    mask = np.ones(H.shape[0], dtype=bool)
    mask[i] = False
    return float(np.corrcoef(H[i, mask], M[i, mask])[0, 1])


def facet_directions_for_k(acts, meta, layer, k, item_pcs):
    """meandiff directions with top-k item PCs projected out (k=0 = raw)."""
    from collections import defaultdict
    by_facet = defaultdict(lambda: {"fwd": [], "rev": []})
    for i, m in enumerate(meta):
        by_facet[(m["trait"], m["facet"])][m["pole"]].append(acts[i, layer])

    names, rows = [], []
    for t in TRAIT_ORDER:
        for fname in FACETS[t]:
            p = by_facet[(t, fname)]
            if not p["fwd"] or not p["rev"]:
                continue
            diff = np.mean(p["fwd"], axis=0) - np.mean(p["rev"], axis=0)
            d = diff if k == 0 else mdx.project_out_pcs(diff, item_pcs[:k])
            names.append(f"{t}:{fname}")
            rows.append(unit(d))
    return names, np.vstack(rows)


def main():
    human = json.load(open(HUMAN_PATH))
    H = np.array(human["correlation_matrix"])
    human_order = human["facet_order"]
    n = len(human_order)

    # results[k][facet] -> list of per-model row-r ; plus full-matrix r
    per_model = {}        # model -> {k -> {"rows": {facet: r}, "matrix_r": r}}

    for model in COHORT:
        if model not in ALL_MODELS:
            print(f"skip unknown {model}")
            continue
        print(f"\n== {model} ==")
        acts, meta, n_layers = extract_or_load_activations(model)
        layer = int(round(n_layers * 2 / 3))
        item_layer_t = torch.from_numpy(acts[:, layer, :])
        item_pcs, _, _ = mdx.compute_pc_projection(item_layer_t, 0.5)  # var-sorted

        per_model[model] = {}
        for k in K_VALUES:
            names, D = facet_directions_for_k(acts, meta, layer, k, item_pcs)
            reorder = [names.index(lbl) for lbl in human_order]
            cos = (D @ D.T)[np.ix_(reorder, reorder)]
            rows = {human_order[i]: row_correlation(H, cos, i) for i in range(n)}
            # full-matrix upper-tri r to human
            iu = np.triu_indices(n, 1)
            matrix_r = float(np.corrcoef(H[iu], cos[iu])[0, 1])
            per_model[model][k] = {"rows": rows, "matrix_r": matrix_r}
            print(f"  k={k}: matrix_r={matrix_r:+.3f}  "
                  f"Sympath={rows['A:Sympath']:+.3f}  Cheerf={rows['E:Cheerf']:+.3f}")

    # Cohort means
    def cohort_mean(selector):
        vals = [selector(per_model[m]) for m in per_model]
        return float(np.mean(vals))

    print("\n" + "=" * 72)
    print("COHORT-MEAN row-r vs human, by projection depth k (n=%d models)" % len(per_model))
    print("  k=0 is RAW meandiff (no projection); k=1 is canonical meandiff-itempc1")
    print("=" * 72)
    header = "facet         " + "".join(f"   k={k}  " for k in K_VALUES)
    print(header)
    table = {}
    for facet in SPOTLIGHT:
        cells, row = [], {}
        for k in K_VALUES:
            mv = cohort_mean(lambda d, k=k, f=facet: d[k]["rows"][f])
            row[k] = mv
            cells.append(f"{mv:+6.3f}")
        table[facet] = row
        print(f"{facet:13} " + "  ".join(cells))
    # full-matrix r row
    mr = {k: cohort_mean(lambda d, k=k: d[k]["matrix_r"]) for k in K_VALUES}
    print(f"{'[full matrix]':13} " + "  ".join(f"{mr[k]:+6.3f}" for k in K_VALUES))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"k_values": K_VALUES, "cohort": list(per_model.keys()),
               "spotlight_cohort_mean": table,
               "matrix_r_cohort_mean": mr,
               "per_model": per_model}, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
