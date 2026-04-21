#!/usr/bin/env python3
"""Compare trait direction vectors extracted from n=50 (old training) and
n≈140 (new stratified) contrast pair sets.

For each (model, trait), fit LR / MD-raw / MD-projected at ~2/3-depth layer
on both caches and report cosine similarity. Also do facet-level comparison
where possible (new set has 4 facets × ~35 pairs; old set has 50 unlabeled,
so we can only get one direction out of it).

Reports:
  - Per (model, trait) cosine agreement across methods
  - Per-trait mean and range
  - Per-facet direction (new only) vs trait direction (old) — tests the
    within-trait bundle hypothesis

Usage:
    python scripts/compare_old_vs_new.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import extract_meandiff_vectors as mdx


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
    "Qwen":  "Qwen/Qwen2.5-3B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
OLD_CACHE = Path("results/phase_b_cache")
NEW_CACHE = Path("results/phase_b_cache_stratified")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def fit_directions(ph, pl, layer, neutral_at_layer=None):
    """Return dict of {method: unit direction} at given layer."""
    n = ph.shape[0]
    d = (ph[:, layer, :] - pl[:, layer, :]).float().numpy()

    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n + [0] * n)
    lr_dir = unit(LogisticRegression(C=1.0, max_iter=2000).fit(X, y).coef_[0])

    md_raw = unit(d.mean(axis=0))

    if neutral_at_layer is not None:
        try:
            pcs, _, _ = mdx.compute_pc_projection(
                torch.from_numpy(neutral_at_layer), 0.5
            )
            md_proj = unit(mdx.project_out_pcs(d.mean(axis=0), pcs))
        except Exception:
            md_proj = md_raw
    else:
        md_proj = md_raw

    return {"LR": lr_dir, "MD-raw": md_raw, "MD-proj": md_proj}


def main():
    all_rows = []
    facet_rows = []

    # Load new pairs file for facet labels
    strat = json.load(open("instruments/contrast_pairs_stratified.json"))

    for short, repo in MODELS.items():
        tag = safe(repo)
        neutral_path = OLD_CACHE / f"{tag}_neutral_chat.pt"
        if not neutral_path.exists():
            print(f"skip {short}: no neutral cache")
            continue
        neutral = torch.load(neutral_path, weights_only=False)
        if isinstance(neutral, torch.Tensor):
            neutral = neutral.numpy()

        for trait in TRAITS:
            old_path = OLD_CACHE / f"{tag}_{trait}_chat_pairs.pt"
            new_path = NEW_CACHE / f"{tag}_{trait}_chat_pairs.pt"
            if not (old_path.exists() and new_path.exists()):
                print(f"skip {short}/{trait}: missing cache")
                continue

            old = torch.load(old_path, weights_only=False)
            new = torch.load(new_path, weights_only=False)

            n_layers = old["ph_tr"].shape[1]
            L = int(round(n_layers * 2 / 3))

            old_dirs = fit_directions(old["ph_tr"], old["pl_tr"], L, neutral[:, L, :])
            new_dirs = fit_directions(new["ph_tr"], new["pl_tr"], L, neutral[:, L, :])

            row = {
                "model": short, "trait": trait, "layer": L,
                "n_old": int(old["ph_tr"].shape[0]),
                "n_new": int(new["ph_tr"].shape[0]),
            }
            for m in ["LR", "MD-raw", "MD-proj"]:
                row[f"cos_{m}"] = float(np.dot(old_dirs[m], new_dirs[m]))
            # Cross-method within new (sanity check — should match Week 6)
            row["cos_new_LR_vs_MDp"] = float(np.dot(new_dirs["LR"], new_dirs["MD-proj"]))
            all_rows.append(row)

            # Facet-level: for each facet in the new data, fit its own direction,
            # compare to the overall OLD trait direction
            pairs = strat["traits"][trait]["pairs"]
            by_facet = defaultdict(list)
            for i, p in enumerate(pairs):
                by_facet[p["facet"]].append(i)

            for facet, idxs in by_facet.items():
                ph_f = new["ph_tr"][idxs]
                pl_f = new["pl_tr"][idxs]
                facet_dirs = fit_directions(ph_f, pl_f, L, neutral[:, L, :])
                for m in ["LR", "MD-proj"]:
                    facet_rows.append({
                        "model": short, "trait": trait, "facet": facet,
                        "method": m, "n_facet": len(idxs),
                        "cos_old_trait": float(np.dot(old_dirs[m], facet_dirs[m])),
                        "cos_new_trait": float(np.dot(new_dirs[m], facet_dirs[m])),
                    })

    # Main table
    print(f"\n{'=' * 100}")
    print(f"DIRECTION AGREEMENT: old (n=50) vs new (n≈140) per (model, trait) at 2/3-depth layer")
    print(f"{'=' * 100}")
    print(f"{'model':>6s} {'trait':>2s} {'L':>3s} {'n_old':>5s} {'n_new':>5s}  "
          f"{'LR':>5s}  {'MD':>5s}  {'MDp':>5s}  {'(new:LR↔MDp)':>12s}")
    for r in all_rows:
        print(f"{r['model']:>6s} {r['trait']:>2s} {r['layer']:>3d} "
              f"{r['n_old']:>5d} {r['n_new']:>5d}  "
              f"{r['cos_LR']:>+5.3f}  {r['cos_MD-raw']:>+5.3f}  {r['cos_MD-proj']:>+5.3f}  "
              f"{r['cos_new_LR_vs_MDp']:>+12.3f}")

    # Summary
    import statistics as st
    for m in ["LR", "MD-raw", "MD-proj"]:
        vs = [r[f"cos_{m}"] for r in all_rows]
        print(f"\n{m}: mean cos(old, new) = {st.mean(vs):+.3f}  "
              f"median = {st.median(vs):+.3f}  min = {min(vs):+.3f}  max = {max(vs):+.3f}")

    # Per-trait breakdown (LR)
    print(f"\nPer-trait mean cos(old, new), LR:")
    by_trait = defaultdict(list)
    for r in all_rows:
        by_trait[r["trait"]].append(r["cos_LR"])
    for t in TRAITS:
        vs = by_trait[t]
        print(f"  {t}: mean={st.mean(vs):+.3f}  range=[{min(vs):+.3f}, {max(vs):+.3f}]")

    # Facet-level report: within each trait, does each facet's direction agree with
    # the old trait direction? If some facets have low agreement, that confirms
    # the bundle-of-axes picture.
    print(f"\n{'=' * 100}")
    print(f"FACET-LEVEL (new only): each facet's direction vs old trait direction (LR)")
    print(f"{'=' * 100}")
    print(f"{'model':>6s} {'trait':>2s} {'facet':>25s} {'n':>3s}  "
          f"{'cos_old_trait':>13s}  {'cos_new_trait':>13s}")
    for r in facet_rows:
        if r["method"] != "LR":
            continue
        print(f"{r['model']:>6s} {r['trait']:>2s} {r['facet']:>25s} {r['n_facet']:>3d}  "
              f"{r['cos_old_trait']:>+13.3f}  {r['cos_new_trait']:>+13.3f}")

    # Per-trait facet-spread summary (which traits have heterogeneous facets?)
    print(f"\nPer-trait facet spread (LR, cos to old trait direction):")
    by_trait_facet = defaultdict(list)
    for r in facet_rows:
        if r["method"] == "LR":
            by_trait_facet[(r["model"], r["trait"])].append(r["cos_old_trait"])
    # Aggregate across models: mean and spread of 4 facet cosines per (model, trait)
    print(f"{'model':>6s} {'trait':>2s}  {'mean':>6s} {'min':>6s} {'max':>6s} {'spread':>6s}")
    for short in MODELS:
        for trait in TRAITS:
            vs = by_trait_facet.get((short, trait), [])
            if not vs:
                continue
            spread = max(vs) - min(vs)
            print(f"{short:>6s} {trait:>2s}  {st.mean(vs):>+6.3f} {min(vs):>+6.3f} "
                  f"{max(vs):>+6.3f} {spread:>+6.3f}")

    # Save
    out = Path("results/compare_old_vs_new.json")
    out.write_text(json.dumps({"by_cell": all_rows, "by_facet": facet_rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
