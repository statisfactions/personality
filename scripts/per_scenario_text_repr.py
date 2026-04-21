#!/usr/bin/env python3
"""Per-scenario test of the text-vs-representation confound.

For each individual scenario s_i (N=838, across 24 facets):
  - text_emb_i: MiniLM embedding of its situation string
  - pair_dir_i: unit direction of (h_i - l_i) at common layer, MD-projected
For each of the 24 facet centers:
  - text_center_F: mean MiniLM embedding of facet F's scenarios
  - repr_center_F: MD-projected direction of facet F's pair-diffs (per model)

For each (scenario, OTHER-facet) pair (s_i's own facet excluded):
  text_sim  = cos(text_emb_i, text_center_F)
  repr_sim  = cos(pair_dir_i, repr_center_F[model])
  record a row

Correlate text_sim ↔ repr_sim:
  - Overall pooled (all scenarios × all other facets × all models)
  - Per-(scenario-facet, other-facet) cell (is the confound concentrated in
    certain facet pairs?)
  - Per-model (does one model show stronger text-confound than others?)

This answers: if a Fairness scenario is textually closer to Boldness than
average, does its representation also drift toward Boldness? If yes, the
confound operates at the fine-grained level. If no (while facet-center r
was +0.22), the facet-level correlation is an aggregation artifact, not a
per-stimulus drift.

Usage:
    python scripts/per_scenario_text_repr.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

import extract_meandiff_vectors as mdx


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
    "Qwen":  "Qwen/Qwen2.5-3B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
CACHE_OLD = Path("results/phase_b_cache")
CACHE_NEW = Path("results/phase_b_cache_stratified")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def main():
    d = json.load(open("instruments/contrast_pairs_stratified.json"))

    # Flatten all scenarios with their metadata
    scenarios = []
    for t in TRAITS:
        for i, p in enumerate(d["traits"][t]["pairs"]):
            scenarios.append({
                "trait": t, "facet": p["facet"], "idx_in_trait": i,
                "situation": p["situation"],
            })
    n_scen = len(scenarios)
    print(f"Loaded {n_scen} scenarios across 24 facets")

    # 1. Text embeddings
    print("\nEmbedding scenarios (MiniLM)...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [s["situation"] for s in scenarios]
    text_embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape: {text_embs.shape}")

    # Facet text centers
    facet_keys = sorted({(s["trait"], s["facet"]) for s in scenarios},
                        key=lambda k: (TRAITS.index(k[0]), k[1]))
    facet_text_center = {}
    for k in facet_keys:
        mask = [i for i, s in enumerate(scenarios) if (s["trait"], s["facet"]) == k]
        m = text_embs[mask].mean(axis=0)
        facet_text_center[k] = m / (np.linalg.norm(m) + 1e-12)

    # 2. Per-model: individual pair directions and facet repr centers
    rows = []
    for short, repo in MODELS.items():
        tag = safe(repo)
        neutral = torch.load(CACHE_OLD / f"{tag}_neutral_chat.pt", weights_only=False)
        if isinstance(neutral, torch.Tensor):
            neutral = neutral.numpy()

        # Load all per-trait activations
        L = None
        trait_blobs = {}
        for t in TRAITS:
            blob = torch.load(CACHE_NEW / f"{tag}_{t}_chat_pairs.pt", weights_only=False)
            trait_blobs[t] = blob
            if L is None:
                L = int(round(blob["ph_tr"].shape[1] * 2 / 3))

        # Neutral PC subtraction at layer L
        pcs, _, _ = mdx.compute_pc_projection(
            torch.from_numpy(neutral[:, L, :]), 0.5
        )

        # Per-scenario pair direction
        pair_dirs = np.zeros((n_scen, neutral.shape[-1]), dtype=np.float32)
        for i_global, s in enumerate(scenarios):
            blob = trait_blobs[s["trait"]]
            i_local = s["idx_in_trait"]
            diff = (blob["ph_tr"][i_local, L, :] - blob["pl_tr"][i_local, L, :]).float().numpy()
            projd = mdx.project_out_pcs(diff, pcs)
            pair_dirs[i_global] = unit(projd)

        # Facet repr centers (MD-projected direction from ~35 pairs)
        facet_repr_center = {}
        for k in facet_keys:
            t, f = k
            blob = trait_blobs[t]
            idxs = [i for i, s in enumerate(scenarios)
                    if (s["trait"], s["facet"]) == k]
            # Convert global idxs to local idxs within trait
            local_idxs = [scenarios[i]["idx_in_trait"] for i in idxs]
            diff = (blob["ph_tr"][local_idxs, L, :] -
                    blob["pl_tr"][local_idxs, L, :]).float().numpy().mean(axis=0)
            projd = mdx.project_out_pcs(diff, pcs)
            facet_repr_center[k] = unit(projd)

        # Record per-(scenario, other-facet) rows
        for i_scen, s in enumerate(scenarios):
            own = (s["trait"], s["facet"])
            for k in facet_keys:
                if k == own:
                    continue
                text_sim = float(text_embs[i_scen] @ facet_text_center[k])
                repr_sim = float(pair_dirs[i_scen] @ facet_repr_center[k])
                rows.append({
                    "model": short,
                    "own_facet": f"{own[0]}:{own[1]}",
                    "other_facet": f"{k[0]}:{k[1]}",
                    "text_sim": text_sim,
                    "repr_sim": repr_sim,
                })
        print(f"  {short}: {n_scen} scenarios × {len(facet_keys) - 1} other facets recorded")

    print(f"\nTotal observations: {len(rows)}")

    # --- Overall pooled correlation ---
    text_arr = np.array([r["text_sim"] for r in rows])
    repr_arr = np.array([r["repr_sim"] for r in rows])
    print(f"\n=== OVERALL POOLED (all models, scenarios, other-facets) ===")
    print(f"  text_sim:   mean={text_arr.mean():+.3f}  sd={text_arr.std():.3f}")
    print(f"  repr_sim:   mean={repr_arr.mean():+.3f}  sd={repr_arr.std():.3f}")
    print(f"  Pearson r = {np.corrcoef(text_arr, repr_arr)[0, 1]:+.4f}")
    print(f"  Spearman rho = {spearmanr(text_arr, repr_arr)[0]:+.4f}")

    # --- Per-model correlation ---
    print(f"\n=== Per-model ===")
    print(f"  {'model':>6s}  {'Pearson r':>10s}  {'Spearman rho':>12s}  N")
    for short in MODELS:
        sub = [r for r in rows if r["model"] == short]
        t = np.array([r["text_sim"] for r in sub])
        rr = np.array([r["repr_sim"] for r in sub])
        print(f"  {short:>6s}  {np.corrcoef(t, rr)[0,1]:>+10.4f}  "
              f"{spearmanr(t, rr)[0]:>+12.4f}  {len(sub)}")

    # --- Partial: correlation within specific (own_facet, other_facet) cells ---
    # This is the cleanest: among e.g. all Fairness scenarios compared to Boldness,
    # do the text-closer ones also activate the Boldness direction more?
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["own_facet"], r["other_facet"])].append(r)

    print(f"\n=== Within-cell correlations (N={n_scen // len(facet_keys) * len(MODELS)} per cell, one row per scenario-model) ===")
    print(f"Strongest 10 POSITIVE within-cell r(text, repr) — where text drives representation:")
    cell_rs = []
    for cell, rs in by_cell.items():
        t = np.array([r["text_sim"] for r in rs])
        rr = np.array([r["repr_sim"] for r in rs])
        if len(t) < 10:
            continue
        r_val = float(np.corrcoef(t, rr)[0, 1])
        cell_rs.append((cell, r_val, len(t)))
    cell_rs.sort(key=lambda x: -x[1])
    for (own, other), r_val, n in cell_rs[:10]:
        print(f"  {own:>25s} → {other:<25s}  r={r_val:+.3f}  N={n}")

    print(f"\nStrongest 10 NEGATIVE within-cell r(text, repr):")
    for (own, other), r_val, n in cell_rs[-10:][::-1]:
        print(f"  {own:>25s} → {other:<25s}  r={r_val:+.3f}  N={n}")

    # Distribution summary
    rs_only = [r for _, r, _ in cell_rs]
    print(f"\nWithin-cell r distribution (N={len(cell_rs)} cells):")
    print(f"  mean={np.mean(rs_only):+.3f}  median={np.median(rs_only):+.3f}  "
          f"min={min(rs_only):+.3f}  max={max(rs_only):+.3f}")
    print(f"  cells with |r| > 0.30: {sum(1 for r in rs_only if abs(r) > 0.30)} / {len(rs_only)}")
    print(f"  cells with |r| > 0.50: {sum(1 for r in rs_only if abs(r) > 0.50)} / {len(rs_only)}")

    # H:Fairness-specific deep dive
    print(f"\n=== H:Fairness as scenario-source — correlation with every other facet ===")
    print(f"(If text-confound is the story, expect strong positive within-cell rs)")
    fair_cells = [(c, r, n) for c, r, n in cell_rs if c[0] == "H:Fairness"]
    fair_cells.sort(key=lambda x: -x[1])
    for (own, other), r_val, n in fair_cells:
        print(f"  {own:>25s} → {other:<25s}  r={r_val:+.3f}")

    # Save
    out = Path("results/per_scenario_text_repr.json")
    out.write_text(json.dumps({
        "overall_pearson_r": float(np.corrcoef(text_arr, repr_arr)[0, 1]),
        "overall_spearman_rho": float(spearmanr(text_arr, repr_arr)[0]),
        "cell_rs": [(c[0], c[1], r, n) for c, r, n in cell_rs],
        "n_rows": len(rows),
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
