#!/usr/bin/env python3
"""X per-facet decomposition (W7 §11.5.4 #1).

Tests whether the trait-level Likert↔RepE disagreement on Extraversion
(W7 §11.5.1, Likert-argmax ↔ RepE-probe r = -0.79 across 7 models under
chat-template RepE) is uniform across the four X facets or concentrated
in one.

Method (matched to cross_method_matrix.py so the trait-level number
recovers the matrix's -0.79):

  1. For each model, derive the X RepE direction the same way the matrix
     does: load `results/repe/<safe>_X_directions.pt`, pick best layer by
     5-fold LDA CV, fit logistic regression on antipodal-augmented
     training pair diffs at that layer, take coef as unit direction.

  2. Trait-level mean_proj per trait: mean of training pair diffs @
     (per-trait direction). Z-score these 6 numbers within model -> the
     model's trait-level RepE z-scores. The X entry, correlated with
     Likert-argmax across 7 models, is the matrix's -0.79.

  3. Per-facet RepE on the 24-pair X holdout: project each holdout pair
     diff at the X best-layer onto the X direction, group by facet (6
     pairs/facet), take the mean. This decomposes "where on the X axis"
     by facet, using the *same* X direction as the trait-level number.

  4. Per-facet matrix-frame z: subtract the model's 6-trait mean and
     divide by the model's 6-trait std (same affine map as trait-level
     z), so a facet's z is directly comparable to the trait-level X z
     that produces -0.79.

  5. Per-facet Likert (argmax + EV): mean of reverse-key-corrected
     scores over the 4 HEXACO-100 X items in the facet.

If the X-disagreement is the represent-vs-enact gap (§11.5.1), the
per-facet Pearson r values should all be roughly negative. If one facet
drives it (e.g. Boldness), that facet's r will be sharply negative and
the others closer to zero.

Usage:
    PYTHONPATH=scripts python scripts/x_facet_decomposition.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


MODELS = [
    # (short_name, hf_repo, hexaco_likert_path)
    ("Gemma",   "google/gemma-3-4b-it",             "results/Gemma_hexaco100.json"),
    ("Llama",   "meta-llama/Llama-3.2-3B-Instruct", "results/Llama_hexaco100.json"),
    ("Phi4",    "microsoft/Phi-4-mini-instruct",    "results/Phi4_hexaco100.json"),
    ("Qwen",    "Qwen/Qwen2.5-3B-Instruct",         "results/Qwen_hexaco100.json"),
    ("Gemma12", "google/gemma-3-12b-it",            "results/Gemma12_hexaco100.json"),
    ("Llama8",  "meta-llama/Llama-3.1-8B-Instruct", "results/Llama8_hexaco100.json"),
    ("Qwen7",   "Qwen/Qwen2.5-7B-Instruct",         "results/Qwen7_hexaco100.json"),
]

ALL_TRAITS = ["H", "E", "X", "A", "C", "O"]
TRAIT = "X"
CACHE_DIR = Path("results/phase_b_cache")
HEXACO_FILE = "instruments/hexaco100.json"
HOLDOUT_FILE = "instruments/contrast_pairs_holdout.json"

X_FACETS = ["Liveliness", "Sociability", "Social Boldness", "Social Self-Esteem"]
FORMAT = "chat"


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def fit_repe_direction(repe_pt_path):
    """Mirror cross_method_matrix.load_repe_scores: pick best layer via LDA-CV
    on antipodal-augmented training diffs, fit LR at that layer, return
    (direction, best_layer, training_diffs_at_layer)."""
    data = torch.load(repe_pt_path, weights_only=False)
    diffs = data["raw_diffs"]  # (n_pairs, n_layers, hidden_dim) tensor
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
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    direction = unit(clf.coef_[0])
    return direction, best_layer, d


def trait_level_mean_proj(model_repo):
    """Per-trait mean_proj for all 6 traits (used for within-model z-score)."""
    out = {}
    layer_used = {}
    for trait in ALL_TRAITS:
        path = Path(f"results/repe/{safe(model_repo)}_{trait}_directions.pt")
        if not path.exists():
            out[trait] = None
            continue
        direction, best_layer, train_d = fit_repe_direction(path)
        out[trait] = float(np.mean(train_d @ direction))
        layer_used[trait] = best_layer
    return out, layer_used


def per_facet_holdout_proj(model_repo, x_direction, x_best_layer, holdout):
    """Project the 24 X holdout pair diffs at x_best_layer onto x_direction,
    grouped by facet."""
    blob = torch.load(
        CACHE_DIR / f"{safe(model_repo)}_{TRAIT}_{FORMAT}_pairs.pt",
        weights_only=False,
    )
    ph_h, pl_h = blob["ph_h"], blob["pl_h"]  # (n_pairs, n_layers, hidden_dim)
    if x_best_layer >= ph_h.shape[1]:
        raise ValueError(
            f"x_best_layer={x_best_layer} >= holdout n_layers={ph_h.shape[1]}"
        )
    pair_diffs = (ph_h[:, x_best_layer, :] - pl_h[:, x_best_layer, :]).float().numpy()

    pairs = holdout["traits"][TRAIT]["pairs"]
    by_facet = defaultdict(list)
    for i, p in enumerate(pairs):
        by_facet[p["facet"]].append(i)

    out = {}
    for facet in X_FACETS:
        idxs = by_facet[facet]
        out[facet] = float(np.mean(pair_diffs[idxs] @ x_direction))
    return out


def per_facet_likert(likert_path, hexaco):
    with open(likert_path) as f:
        d = json.load(f)
    items = hexaco["items"]
    ir = d["item_results"]

    by_facet_argmax = defaultdict(list)
    by_facet_ev = defaultdict(list)
    for item_id, meta in items.items():
        if meta["scale"] != TRAIT or item_id not in ir:
            continue
        am = ir[item_id].get("argmax")
        ev = ir[item_id].get("expected_value")
        if am is not None:
            v = int(am)
            if meta.get("reverse_keyed"):
                v = 6 - v
            by_facet_argmax[meta["facet"]].append(v)
        if ev is not None:
            v = ev
            if meta.get("reverse_keyed"):
                v = 6.0 - v
            by_facet_ev[meta["facet"]].append(v)

    return (
        {f: float(np.mean(by_facet_argmax[f])) for f in X_FACETS if by_facet_argmax[f]},
        {f: float(np.mean(by_facet_ev[f])) for f in X_FACETS if by_facet_ev[f]},
    )


def trait_level_likert(likert_path):
    with open(likert_path) as f:
        d = json.load(f)
    s = d["scale_scores"][TRAIT]
    return float(s["argmax_mean"]), float(s["ev_mean"])


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), n
    return float(np.corrcoef(xs, ys)[0, 1]), n


def main():
    with open(HEXACO_FILE) as f:
        hexaco = json.load(f)
    with open(HOLDOUT_FILE) as f:
        holdout = json.load(f)

    rows = {}  # model -> dict of pieces

    for short, repo, lpath in MODELS:
        try:
            facet_argmax, facet_ev = per_facet_likert(lpath, hexaco)
            tl_argmax, tl_ev = trait_level_likert(lpath)
        except FileNotFoundError:
            print(f"  WARNING: {lpath} missing, skipping {short}")
            continue

        # Trait-level RepE for all 6 traits (for within-model z-score)
        trait_proj, trait_layer = trait_level_mean_proj(repo)
        if any(v is None for v in trait_proj.values()):
            print(f"  WARNING: missing trait directions for {short}, skipping")
            continue

        # Re-fit X direction so we can project the holdout
        x_path = Path(f"results/repe/{safe(repo)}_X_directions.pt")
        x_dir, x_layer, _ = fit_repe_direction(x_path)

        try:
            facet_repe = per_facet_holdout_proj(repo, x_dir, x_layer, holdout)
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: holdout projection failed for {short} ({e})")
            continue

        # Within-model z over the 6 traits
        trait_vals = np.array([trait_proj[t] for t in ALL_TRAITS], dtype=float)
        mu, sd = trait_vals.mean(), trait_vals.std()
        x_idx = ALL_TRAITS.index(TRAIT)
        x_trait_z = (trait_vals[x_idx] - mu) / sd if sd > 0 else 0.0
        # Per-facet matrix-frame z (same affine as trait z-score)
        facet_repe_z = {f: (v - mu) / sd if sd > 0 else 0.0 for f, v in facet_repe.items()}

        rows[short] = {
            "facet_argmax": facet_argmax,
            "facet_ev": facet_ev,
            "facet_repe_raw": facet_repe,
            "facet_repe_z_matrix": facet_repe_z,
            "trait_argmax": tl_argmax,
            "trait_ev": tl_ev,
            "trait_repe_raw": float(trait_vals[x_idx]),
            "trait_repe_z_matrix": float(x_trait_z),
            "x_layer": x_layer,
        }

    models_present = list(rows.keys())

    # ----- Sanity: recover trait-level r ≈ -0.788 (argmax) and -0.626 (EV) -----
    tl_argmax = [rows[m]["trait_argmax"] for m in models_present]
    tl_ev = [rows[m]["trait_ev"] for m in models_present]
    tl_repe_z = [rows[m]["trait_repe_z_matrix"] for m in models_present]

    r_am, n = pearson(tl_argmax, tl_repe_z)
    r_ev, _ = pearson(tl_ev, tl_repe_z)

    print("=" * 96)
    print("  X PER-FACET DECOMPOSITION  (W7 §11.5.4 #1)")
    print("=" * 96)
    print()
    print("Sanity check (should match cross_method_matrix.py output for X):")
    print(f"  Likert-argmax ↔ RepE-probe-z   r = {r_am:+.3f}  (n={n})  "
          f"[matrix says -0.788]")
    print(f"  Likert-EV     ↔ RepE-probe-z   r = {r_ev:+.3f}  (n={n})  "
          f"[matrix says -0.626]")
    print()

    # ----- Per-(model, facet) table -----
    print(f"{'Model':10s} {'Facet':22s} {'argmax':>8s} {'EV':>8s} {'RepE-raw':>14s} "
          f"{'RepE-z':>10s} {'Layer':>6s}")
    print("-" * 96)
    for m in models_present:
        r = rows[m]
        for f in X_FACETS:
            print(f"{m:10s} {f:22s} {r['facet_argmax'][f]:>8.3f} "
                  f"{r['facet_ev'][f]:>8.3f} {r['facet_repe_raw'][f]:>+14.3f} "
                  f"{r['facet_repe_z_matrix'][f]:>+10.3f} {r['x_layer']:>6d}")
        print()

    # ----- Per-facet Pearson across models -----
    print("--- Per-facet Pearson r across models ---")
    print()
    print(f"{'Facet':24s} {'argmax↔RepE-z':>16s} {'EV↔RepE-z':>14s} "
          f"{'argmax↔RepE-raw':>18s}")
    print("-" * 80)

    per_facet_r = {}
    for facet in X_FACETS:
        am = [rows[m]["facet_argmax"][facet] for m in models_present]
        ev = [rows[m]["facet_ev"][facet] for m in models_present]
        rep_raw = [rows[m]["facet_repe_raw"][facet] for m in models_present]
        rep_z = [rows[m]["facet_repe_z_matrix"][facet] for m in models_present]
        r_amz, _ = pearson(am, rep_z)
        r_evz, _ = pearson(ev, rep_z)
        r_amr, _ = pearson(am, rep_raw)
        per_facet_r[facet] = {
            "argmax_vs_repez": r_amz,
            "ev_vs_repez": r_evz,
            "argmax_vs_reperaw": r_amr,
            "n": len(models_present),
        }
        print(f"{facet:24s} {r_amz:>+16.3f} {r_evz:>+14.3f} {r_amr:>+18.3f}")

    print()

    # ----- Compare facet ordering: which facet's correlation is most negative -----
    facet_z_corrs = [(f, per_facet_r[f]["argmax_vs_repez"]) for f in X_FACETS]
    facet_z_corrs.sort(key=lambda x: x[1])
    print("--- Facet ordering (most-negative argmax↔RepE-z first) ---")
    for f, r in facet_z_corrs:
        print(f"  {f:24s}  r = {r:+.3f}")
    print()
    spread = facet_z_corrs[-1][1] - facet_z_corrs[0][1]
    print(f"  Range across facets: {spread:+.3f}")
    print(f"  Trait-level (canon): {r_am:+.3f}")
    print(f"  Mean of facet rs:    {np.mean([r for _, r in facet_z_corrs]):+.3f}")
    print()

    # ----- Save -----
    out_path = Path("results/x_facet_decomposition.json")
    payload = {
        "models": models_present,
        "facets": X_FACETS,
        "format": FORMAT,
        "method": (
            "Per-trait LR direction at LDA-CV best layer; per-trait mean_proj "
            "z-scored within model across 6 traits to define a matrix-frame z. "
            "Per-facet RepE = mean of holdout pair diffs (6 per facet) projected "
            "onto X direction at X best layer; per-facet z applies the same "
            "affine (mu, sd) as the trait-level z."
        ),
        "trait_level": {
            "argmax_vs_repez": r_am,
            "ev_vs_repez": r_ev,
            "n": n,
        },
        "per_facet_correlations": per_facet_r,
        "per_model_per_facet": {
            m: {
                "argmax": rows[m]["facet_argmax"],
                "ev": rows[m]["facet_ev"],
                "repe_raw": rows[m]["facet_repe_raw"],
                "repe_z_matrix": rows[m]["facet_repe_z_matrix"],
                "x_layer": rows[m]["x_layer"],
                "trait_argmax": rows[m]["trait_argmax"],
                "trait_ev": rows[m]["trait_ev"],
                "trait_repe_raw": rows[m]["trait_repe_raw"],
                "trait_repe_z_matrix": rows[m]["trait_repe_z_matrix"],
            } for m in models_present
        },
    }
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
