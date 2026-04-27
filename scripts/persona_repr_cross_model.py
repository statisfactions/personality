#!/usr/bin/env python3
"""Cross-model agreement on persona projections (W7 §11.5.4 #4 follow-up).

Given that 7 models each independently project 50 personas onto Big Five
trait directions, ask: do they agree on which personas project most
strongly on each trait? This is a persona-level cross-architecture fidelity
check, complementary to the per-trait diagonal r.

For each trait t and each pair of models (a, b):
  r(projection[a, t, :], projection[b, t, :])  across 50 personas

Strong cross-model agreement → models share a common persona-level
representation of trait expression. Weak agreement → models build their
representations differently per persona.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_cross_model.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
TRAITS = ["A", "C", "E", "N", "O"]
SUFFIX = "response-position"


def load(model):
    path = Path(f"results/persona_repr_mapping_{model}_{SUFFIX}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    data = {}
    for m in MODELS:
        d = load(m)
        if d is None:
            print(f"  WARNING: missing for {m}")
            continue
        data[m] = d

    if len(data) < 2:
        print("Not enough data for cross-model analysis")
        return

    models_present = list(data.keys())

    # Build (n_models, n_traits, n_personas) projection tensor
    n_personas = data[models_present[0]]["n_personas"]
    proj = {m: {t: np.array([pd["projections"][t] for pd in data[m]["persona_data"]]) for t in TRAITS}
            for m in models_present}

    # Sanity: persona ordering consistent across models
    pids = {m: [pd["persona_id"] for pd in data[m]["persona_data"]] for m in models_present}
    assert all(pids[m] == pids[models_present[0]] for m in models_present), "persona orders differ"

    # ----- Per-trait pairwise model agreement -----
    print("=" * 80)
    print("  Per-trait cross-model agreement on persona projections")
    print(f"  (Pearson r across {n_personas} personas, for each model pair)")
    print("=" * 80)

    per_trait_means = {}
    per_trait_full = {}
    for t in TRAITS:
        rs = []
        full = {}
        for a, b in combinations(models_present, 2):
            r = float(np.corrcoef(proj[a][t], proj[b][t])[0, 1])
            rs.append(r)
            full[f"{a}~{b}"] = r
        per_trait_means[t] = float(np.mean(rs))
        per_trait_full[t] = full
        print(f"\n  Trait {t}: mean pairwise r = {np.mean(rs):+.3f}  (range {min(rs):+.3f} to {max(rs):+.3f}, n_pairs={len(rs)})")

    print(f"\n  Mean over traits: {np.mean(list(per_trait_means.values())):+.3f}")
    print()

    # ----- 7×7 matrix per trait (compact) -----
    print("=" * 80)
    print("  Per-trait pairwise matrix (all 21 pairs per trait)")
    print("=" * 80)
    for t in TRAITS:
        print(f"\n  Trait {t}:")
        print(f"  {'':>10s}" + " ".join(f"{m:>9s}" for m in models_present))
        for a in models_present:
            row = [f"{a:>10s}"]
            for b in models_present:
                if a == b:
                    row.append(f"{1.0:>+9.3f}")
                else:
                    r = float(np.corrcoef(proj[a][t], proj[b][t])[0, 1])
                    row.append(f"{r:>+9.3f}")
            print("  " + " ".join(row))

    # ----- Comparison: cross-model rep agreement vs sampled-z agreement -----
    print("\n" + "=" * 80)
    print("  Sanity: how does cross-model rep agreement compare to sampled-z?")
    print("  Each model's projection should track sampled z; cross-model agreement")
    print("  upper-bound is set by how well each tracks z.")
    print("=" * 80)
    z = {t: np.array([pd["z_scores"][t] for pd in data[models_present[0]]["persona_data"]]) for t in TRAITS}

    print(f"\n  {'Trait':5s}  {'mean cross-model r':>20s}  {'mean rep~z r':>14s}  {'ratio':>8s}")
    for t in TRAITS:
        cross_r = per_trait_means[t]
        rep_z_rs = [float(np.corrcoef(proj[m][t], z[t])[0, 1]) for m in models_present]
        mean_rep_z = float(np.mean(rep_z_rs))
        ratio = cross_r / mean_rep_z if mean_rep_z != 0 else float('nan')
        print(f"  {t:5s}  {cross_r:>+20.3f}  {mean_rep_z:>+14.3f}  {ratio:>+8.3f}")

    # ----- Save -----
    out_path = Path("results/persona_repr_cross_model.json")
    payload = {
        "models": models_present,
        "traits": TRAITS,
        "n_personas": n_personas,
        "per_trait_mean_pairwise_r": per_trait_means,
        "per_trait_pairwise_r": per_trait_full,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
