#!/usr/bin/env python3
"""IPIP-NEO-300 facet-level rescoring (W7 §11.5.4 #2).

Re-aggregates the existing IPIP-300 Likert results at the 30-facet
granularity rather than the 5-trait granularity. Tests whether the W1
"rank-1 assistant-shape collapse" (E-C r=0.93 across 4 models at trait
level) survives at facet resolution or dissolves into richer structure.

Facet key: Goldberg/Johnson IPIP-NEO-300 (1999). Items in each trait scale
are listed at stride 5 globally and stride 6 within trait. So for trait
scale `IPIP300-NEU` with item list [ipip1, ipip6, ipip11, ..., ipip296],
within-trait position p maps to facet f = (p-1) % 6, with 10 items per
facet. The 5 facet labels per trait (in order) are the canonical
IPIP-NEO-300 facets:

  N: Anxiety, Anger, Depression, Self-Consciousness, Immoderation, Vulnerability
  E: Friendliness, Gregariousness, Assertiveness, Activity Level,
     Excitement-Seeking, Cheerfulness
  O: Imagination, Artistic Interests, Emotionality, Adventurousness,
     Intellect, Liberalism
  A: Trust, Morality, Altruism, Cooperation, Modesty, Sympathy
  C: Self-Efficacy, Orderliness, Dutifulness, Achievement-Striving,
     Self-Discipline, Cautiousness

(Verified against the items in admin_sessions/prod_run_01_external_rating.json
by inspection: the first three items in each facet are perfectly thematic.)

Usage:
    .venv/bin/python scripts/ipip_facet_rescore.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np


ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"

MODELS = [
    "Gemma", "Llama", "Phi4", "Qwen",
    "Gemma12", "Llama8", "Qwen7",
]

TRAIT_OF = {
    "IPIP300-NEU": "N",
    "IPIP300-EXT": "E",
    "IPIP300-OPE": "O",
    "IPIP300-AGR": "A",
    "IPIP300-CON": "C",
}

FACETS = {
    "N": ["Anxiety", "Anger", "Depression", "Self-Consciousness",
          "Immoderation", "Vulnerability"],
    "E": ["Friendliness", "Gregariousness", "Assertiveness", "Activity Level",
          "Excitement-Seeking", "Cheerfulness"],
    "O": ["Imagination", "Artistic Interests", "Emotionality", "Adventurousness",
          "Intellect", "Liberalism"],
    "A": ["Trust", "Morality", "Altruism", "Cooperation", "Modesty", "Sympathy"],
    "C": ["Self-Efficacy", "Orderliness", "Dutifulness", "Achievement-Striving",
          "Self-Discipline", "Cautiousness"],
}
TRAIT_ORDER = ["N", "E", "O", "A", "C"]
ALL_FACET_KEYS = [(t, f) for t in TRAIT_ORDER for f in FACETS[t]]


def build_facet_map(session_path):
    """Return {item_id: (trait, facet_name, reverse_keyed_bool)}."""
    with open(session_path) as f:
        s = json.load(f)
    scales = s["measures"]["IPIP300"]["scales"]
    item_meta = {}
    for scale_id, sdef in scales.items():
        trait = TRAIT_OF[scale_id]
        item_ids = sdef["item_ids"]
        rk = set(sdef["reverse_keyed_item_ids"])
        for pos, iid in enumerate(item_ids):
            facet_idx = pos % 6
            facet_name = FACETS[trait][facet_idx]
            item_meta[iid] = (trait, facet_name, iid in rk)
    return item_meta


def per_facet_scores(results_path, item_meta):
    """Return {(trait, facet): {'argmax': mean, 'ev': mean, 'n': int}}."""
    with open(results_path) as f:
        d = json.load(f)
    ir = d["item_results"]

    bucket_argmax = {k: [] for k in ALL_FACET_KEYS}
    bucket_ev = {k: [] for k in ALL_FACET_KEYS}

    for iid, meta in item_meta.items():
        if iid not in ir:
            continue
        trait, facet, reverse = meta
        am = ir[iid].get("argmax")
        ev = ir[iid].get("expected_value")
        if am is not None:
            v = int(am)
            if reverse:
                v = 6 - v
            bucket_argmax[(trait, facet)].append(v)
        if ev is not None:
            v = ev
            if reverse:
                v = 6.0 - v
            bucket_ev[(trait, facet)].append(v)

    out = {}
    for k in ALL_FACET_KEYS:
        am = bucket_argmax[k]
        ev = bucket_ev[k]
        out[k] = {
            "argmax": float(np.mean(am)) if am else None,
            "ev": float(np.mean(ev)) if ev else None,
            "n": len(am),
        }
    return out


def trait_scores(results_path):
    """Pull stored trait-level scores."""
    with open(results_path) as f:
        d = json.load(f)
    out = {}
    for sk, ss in d["scale_scores"].items():
        t = TRAIT_OF[sk]
        out[t] = {"argmax": ss.get("argmax_mean"), "ev": ss.get("ev_mean")}
    return out


def corr_matrix(matrix):
    """Pearson correlation across rows (each row = one feature, columns = samples)."""
    return np.corrcoef(matrix)


def pc1_explained_variance(matrix):
    """SVD-based PC1 fraction. matrix is (n_features, n_samples)."""
    X = matrix - matrix.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    if var.sum() == 0:
        return 0.0, S.tolist()
    return float(var[0] / var.sum()), (var / var.sum()).tolist()


def main():
    item_meta = build_facet_map(ADMIN_SESSION)

    # facet_data[model] = {(trait, facet): {argmax, ev, n}}
    facet_data = {}
    trait_data = {}
    for m in MODELS:
        path = f"results/{m}_ipip300.json"
        if not Path(path).exists():
            print(f"  WARNING: {path} missing, skipping {m}")
            continue
        facet_data[m] = per_facet_scores(path, item_meta)
        trait_data[m] = trait_scores(path)

    models_present = list(facet_data.keys())
    n_models = len(models_present)
    print(f"Cohort: {models_present}\n")

    # Build (30, n_models) facet matrix and (5, n_models) trait matrix (EV)
    facet_mat_ev = np.array(
        [[facet_data[m][k]["ev"] for m in models_present] for k in ALL_FACET_KEYS]
    )
    facet_mat_am = np.array(
        [[facet_data[m][k]["argmax"] for m in models_present] for k in ALL_FACET_KEYS]
    )
    trait_mat_ev = np.array(
        [[trait_data[m][t]["ev"] for m in models_present] for t in TRAIT_ORDER]
    )
    trait_mat_am = np.array(
        [[trait_data[m][t]["argmax"] for m in models_present] for t in TRAIT_ORDER]
    )

    # ----- PCA: rank-1 collapse intensity -----
    print("=" * 84)
    print("  RANK-1 COLLAPSE INTENSITY  (PC1 explained variance)")
    print("=" * 84)
    print()

    print("Trait level (5 features × {} models):".format(n_models))
    pc1_ev, _ = pc1_explained_variance(trait_mat_ev)
    pc1_am, _ = pc1_explained_variance(trait_mat_am)
    print(f"  EV    PC1 = {pc1_ev:.3f}    argmax PC1 = {pc1_am:.3f}")
    print()

    print("Facet level (30 features × {} models):".format(n_models))
    pc1_fev, _ = pc1_explained_variance(facet_mat_ev)
    pc1_fam, _ = pc1_explained_variance(facet_mat_am)
    print(f"  EV    PC1 = {pc1_fev:.3f}    argmax PC1 = {pc1_fam:.3f}")
    print()
    print(f"  PC1 drop (EV):     trait {pc1_ev:.3f} -> facet {pc1_fev:.3f}  "
          f"(Δ = {pc1_fev - pc1_ev:+.3f})")
    print(f"  PC1 drop (argmax): trait {pc1_am:.3f} -> facet {pc1_fam:.3f}  "
          f"(Δ = {pc1_fam - pc1_am:+.3f})")
    print()

    # ----- Trait-level cross-trait correlation matrix -----
    print("=" * 84)
    print("  TRAIT-LEVEL CROSS-TRAIT CORRELATION (across {} models)".format(n_models))
    print("=" * 84)
    print()
    trait_corr = corr_matrix(trait_mat_ev)
    print(f"        " + "".join(f"{t:>8s}" for t in TRAIT_ORDER))
    for i, t in enumerate(TRAIT_ORDER):
        print(f"  {t:5s}", end="")
        for j in range(len(TRAIT_ORDER)):
            print(f"{trait_corr[i, j]:>+8.3f}", end="")
        print()
    print()

    # E-C trait correlation specifically (W1 baseline 0.93)
    e_idx, c_idx = TRAIT_ORDER.index("E"), TRAIT_ORDER.index("C")
    n_idx = TRAIT_ORDER.index("N")
    print(f"  E ↔ C trait correlation: {trait_corr[e_idx, c_idx]:+.3f}  "
          f"(W1 baseline: +0.93)")
    print(f"  E ↔ N trait correlation: {trait_corr[e_idx, n_idx]:+.3f}")
    print(f"  C ↔ N trait correlation: {trait_corr[c_idx, n_idx]:+.3f}")
    print()

    # ----- Facet-level: within-trait vs across-trait correlations -----
    facet_corr = corr_matrix(facet_mat_ev)
    print("=" * 84)
    print("  FACET-LEVEL: within-trait vs across-trait facet correlations (EV)")
    print("=" * 84)
    print()

    parent = [k[0] for k in ALL_FACET_KEYS]  # trait letter for each of 30
    n = len(ALL_FACET_KEYS)
    within_pairs, across_pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = facet_corr[i, j]
            if np.isnan(r):
                continue
            (within_pairs if parent[i] == parent[j] else across_pairs).append(r)
    print(f"  Within-trait facet pairs (n={len(within_pairs)}):")
    print(f"    mean r = {np.mean(within_pairs):+.3f}  median = {np.median(within_pairs):+.3f}  "
          f"std = {np.std(within_pairs):.3f}")
    print(f"  Across-trait facet pairs (n={len(across_pairs)}):")
    print(f"    mean r = {np.mean(across_pairs):+.3f}  median = {np.median(across_pairs):+.3f}  "
          f"std = {np.std(across_pairs):.3f}")
    print(f"  Difference (within - across): {np.mean(within_pairs) - np.mean(across_pairs):+.3f}")
    print()

    # ----- E-vs-C facet block: does the rank-1 collapse hold per-facet? -----
    print("=" * 84)
    print("  E-FACETS × C-FACETS  (the W1 rank-1 anchor — does it dissolve?)")
    print("=" * 84)
    print()
    e_facet_idx = [i for i, k in enumerate(ALL_FACET_KEYS) if k[0] == "E"]
    c_facet_idx = [i for i, k in enumerate(ALL_FACET_KEYS) if k[0] == "C"]
    e_c_block = facet_corr[np.ix_(e_facet_idx, c_facet_idx)]

    print("        " + "".join(f"{f[:11]:>13s}" for _, f in [ALL_FACET_KEYS[i] for i in c_facet_idx]))
    for i, ei in enumerate(e_facet_idx):
        f_label = ALL_FACET_KEYS[ei][1][:11]
        print(f"  {f_label:11s}", end="")
        for j in range(len(c_facet_idx)):
            print(f"{e_c_block[i, j]:>+13.3f}", end="")
        print()
    print()
    print(f"  E×C facet pairs (n={e_c_block.size}): "
          f"mean = {np.mean(e_c_block):+.3f}  "
          f"median = {np.median(e_c_block):+.3f}  "
          f"max = {np.max(e_c_block):+.3f}  min = {np.min(e_c_block):+.3f}")
    print(f"  Trait-level E↔C: {trait_corr[e_idx, c_idx]:+.3f}")
    drop = trait_corr[e_idx, c_idx] - np.mean(e_c_block)
    print(f"  Drop trait → facet-mean: {drop:+.3f}")
    print()

    # ----- Top off-trait facet couplings (which facets violate trait separation?) -----
    print("=" * 84)
    print("  TOP CROSS-TRAIT FACET COUPLINGS (|r| > 0.85)")
    print("=" * 84)
    print()
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if parent[i] == parent[j]:
                continue
            r = facet_corr[i, j]
            if abs(r) > 0.85:
                rows.append((r, ALL_FACET_KEYS[i], ALL_FACET_KEYS[j]))
    rows.sort(key=lambda x: -abs(x[0]))
    if rows:
        for r, k1, k2 in rows[:20]:
            print(f"  {k1[0]}:{k1[1]:<22s} ↔ {k2[0]}:{k2[1]:<22s}  r = {r:+.3f}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
    else:
        print("  (none)")
    print()

    # ----- Most-discriminable within-trait facet pairs (lowest r) -----
    print("=" * 84)
    print("  MOST-DISCRIMINABLE WITHIN-TRAIT FACET PAIRS (lowest within-trait r)")
    print("=" * 84)
    print()
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if parent[i] != parent[j]:
                continue
            r = facet_corr[i, j]
            rows.append((r, ALL_FACET_KEYS[i], ALL_FACET_KEYS[j]))
    rows.sort()
    for r, k1, k2 in rows[:8]:
        print(f"  {k1[0]}:{k1[1]:<22s} ↔ {k2[0]}:{k2[1]:<22s}  r = {r:+.3f}")
    print()

    # ----- Per-model raw scores table (EV) -----
    print("=" * 84)
    print("  PER-MODEL FACET EV SCORES")
    print("=" * 84)
    print()
    header = f"{'Trait':5s} {'Facet':22s}" + "".join(f"{m:>9s}" for m in models_present)
    print(header)
    print("-" * len(header))
    for trait in TRAIT_ORDER:
        for facet in FACETS[trait]:
            row = f"{trait:5s} {facet:22s}"
            for m in models_present:
                v = facet_data[m][(trait, facet)]["ev"]
                row += f"{v:>9.3f}" if v is not None else f"{'—':>9s}"
            print(row)
        print()

    # ----- Save JSON -----
    out_path = Path("results/ipip_facet_rescore.json")
    payload = {
        "models": models_present,
        "facets": [f"{t}:{f}" for t, f in ALL_FACET_KEYS],
        "trait_pc1_ev": pc1_ev,
        "facet_pc1_ev": pc1_fev,
        "trait_pc1_argmax": pc1_am,
        "facet_pc1_argmax": pc1_fam,
        "trait_corr_ev": trait_corr.tolist(),
        "facet_corr_ev": facet_corr.tolist(),
        "within_trait_mean_r": float(np.mean(within_pairs)),
        "across_trait_mean_r": float(np.mean(across_pairs)),
        "ec_trait_r": float(trait_corr[e_idx, c_idx]),
        "ec_facet_block_mean_r": float(np.mean(e_c_block)),
        "ec_facet_block_min_r": float(np.min(e_c_block)),
        "ec_facet_block_max_r": float(np.max(e_c_block)),
        "per_model_per_facet": {
            m: {f"{t}:{f}": facet_data[m][(t, f)] for t, f in ALL_FACET_KEYS}
            for m in models_present
        },
    }
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
