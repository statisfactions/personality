#!/usr/bin/env python3
"""Compute 30x30 IPIP-NEO-300 inter-facet correlations directly from the
PsychArchives Kajonius & Johnson 2019 data deposit (N=307,313).

Replaces the prior NeuroQuestAi-mirror-derived version (N=145,388) with
the primary-source larger-N version. Same instrument (IPIP-NEO-300),
same Johnson 1999 facet scoring, same 30 facets — just a different
sampling of Johnson's accumulated dataset.

IMPORTANT — PA .por convention: the IPIP300.por file in this deposit
stores responses with REVERSE-KEYED ITEMS ALREADY REVERSED. The Excel
"Sign" column ("+C1", "-C4") indicates the item's theoretical pole but
should NOT be re-applied to the stored values. We confirmed this
empirically: within each facet, raw inter-item correlations among
items marked "fwd" and "rev" in the canonical Johnson key are all
positive (~0.2-0.6); double-reversing flips half of them and breaks
internal consistency (e.g., Achievement-Striving alpha → -0.99).
Apply NO additional reverse-keying. Without this fix the within-trait
mean was +0.167 (the W9 §7.5 saga); with the fix it is +0.418, which
matches both the NQ pre-scored mirror (+0.405) and K&J 2019 Figure 1.

Output: instruments/ipip300_human_facet_correlations.json
"""

import json
from pathlib import Path

import numpy as np
import pyreadstat

POR_PATH = (
    "data/kajonius_johnson_2019/data/kajonius_johnson_2019/IPIP300.por"
)
FACET_MAP = "instruments/ipip300_facet_map.json"
OUT = "instruments/ipip300_human_facet_correlations.json"

TRAIT_ORDER = ["A", "C", "E", "N", "O"]
# Match abbreviations used in scripts/ipip_facet_cluster.py
_CANON_TO_ABBREV = {
    "Anxiety": "Anxiety", "Anger": "Anger", "Depression": "Depression",
    "Self-Consciousness": "Self-Cons", "Immoderation": "Immoder",
    "Vulnerability": "Vulner",
    "Friendliness": "Friend", "Gregariousness": "Gregar",
    "Assertiveness": "Assert", "Activity Level": "Activity",
    "Excitement-Seeking": "Excite", "Cheerfulness": "Cheerf",
    "Imagination": "Imagin", "Artistic Interests": "Artist",
    "Emotionality": "Emotion", "Adventurousness": "Advent",
    "Intellect": "Intell", "Liberalism": "Liberal",
    "Trust": "Trust", "Morality": "Moral", "Altruism": "Altru",
    "Cooperation": "Cooper", "Modesty": "Modest", "Sympathy": "Sympath",
    "Self-Efficacy": "Self-Eff", "Orderliness": "Order",
    "Dutifulness": "Dutiful", "Achievement-Striving": "Achieve",
    "Self-Discipline": "Discipl", "Cautiousness": "Caution",
}


def main():
    print(f"Loading {POR_PATH} ...")
    df, _meta = pyreadstat.read_por(POR_PATH)
    n_raw = len(df)
    print(f"  raw cases: {n_raw}")

    # Load canonical item → (trait, facet, pole)
    with open(FACET_MAP) as f:
        canon = json.load(f)["items"]

    # Group item indices (1..300) per (trait, facet_abbrev) and pole
    buckets = {}  # (trait, facet_abbrev) -> list of (item_idx, pole)
    for n in range(1, 301):
        info = canon[f"ipip{n}"]
        t = info["trait"]
        f_abbrev = _CANON_TO_ABBREV[info["facet"]]
        key = (t, f_abbrev)
        buckets.setdefault(key, []).append((n, info["pole"]))
    assert len(buckets) == 30, f"expected 30 buckets, got {len(buckets)}"
    for k, v in buckets.items():
        assert len(v) == 10, f"bucket {k} has {len(v)} items, expected 10"

    # Score facets per participant: simple sum across items (data is pre-reversed
    # in the .por file — see module docstring; do NOT apply reverse-keying here).
    # Each item is 1-5 Likert, so per-facet scores in [10, 50] before missing-handling.
    # IMPORTANT: Johnson IPIP-NEO-300 uses 0 to indicate MISSING (per DAT300.doc).
    items = df.loc[:, [f"I{n}" for n in range(1, 301)]].to_numpy(dtype="float64")

    # Drop rows with any missing item response (either NaN or coded-0)
    has_zero = (items == 0).any(axis=1)
    has_nan = np.isnan(items).any(axis=1)
    mask = ~(has_zero | has_nan)
    items = items[mask]
    n_complete = items.shape[0]
    print(f"  complete cases: {n_complete} "
          f"(dropped {n_raw - n_complete}; "
          f"zero-coded {has_zero.sum()}, NaN-coded {has_nan.sum()})")

    facet_names = []
    facet_scores = np.zeros((n_complete, 30), dtype="float64")
    j = 0
    # Iterate in canonical order (A, C, E, N, O × 6 facets each)
    for t in TRAIT_ORDER:
        # Get facets in canonical order from canon
        # Find facets for this trait in their canonical order via item-1 lookup
        facets_for_trait = []
        for n in range(1, 31):
            if canon[f"ipip{n}"]["trait"] == t:
                f_abbrev = _CANON_TO_ABBREV[canon[f"ipip{n}"]["facet"]]
                if f_abbrev not in facets_for_trait:
                    facets_for_trait.append(f_abbrev)
        for f_abbrev in facets_for_trait:
            facet_names.append(f"{t}:{f_abbrev}")
            for item_idx, _pole in buckets[(t, f_abbrev)]:
                # No reverse-keying: .por data is pre-reversed (see module docstring).
                facet_scores[:, j] += items[:, item_idx - 1]
            j += 1
    assert j == 30

    print(f"\nFacet score ranges (expected [10, 50]):")
    print(f"  min: {facet_scores.min():.0f}, max: {facet_scores.max():.0f}")
    print(f"  per-facet means (first 6): {facet_scores.mean(axis=0)[:6].round(2)}")

    # 30×30 correlation matrix
    corr = np.corrcoef(facet_scores, rowvar=False)
    assert corr.shape == (30, 30)

    # Stats
    within_pairs = []
    across_pairs = []
    for i in range(30):
        for k in range(i + 1, 30):
            t_i, _ = facet_names[i].split(":")
            t_k, _ = facet_names[k].split(":")
            if t_i == t_k:
                within_pairs.append(corr[i, k])
            else:
                across_pairs.append(corr[i, k])
    print(f"\n30×30 facet correlation matrix:")
    print(f"  within-trait pairs (n={len(within_pairs)}): "
          f"mean={np.mean(within_pairs):+.4f}, median={np.median(within_pairs):+.4f}")
    print(f"  across-trait pairs (n={len(across_pairs)}): "
          f"mean={np.mean(across_pairs):+.4f}, median={np.median(across_pairs):+.4f}")
    print(f"  within/across ratio (mean): {np.mean(within_pairs) / max(abs(np.mean(across_pairs)), 1e-9):.1f}×")

    # Compare with prior NeuroQuestAi-mirror version
    prior_path = OUT
    if Path(prior_path).exists():
        with open(prior_path) as f:
            prior = json.load(f)
        if prior["facet_order"] == facet_names:
            prior_mat = np.array(prior["correlation_matrix"])
            triu_i, triu_k = np.triu_indices(30, k=1)
            prior_off = prior_mat[triu_i, triu_k]
            new_off = corr[triu_i, triu_k]
            r_diff = np.corrcoef(prior_off, new_off)[0, 1]
            max_abs_diff = np.max(np.abs(prior_off - new_off))
            print(f"\nSanity check vs prior (NeuroQuestAi N=145,388):")
            print(f"  cor(new offdiag, prior offdiag) = {r_diff:.4f}")
            print(f"  max |Δ| any pair = {max_abs_diff:.4f}")
        else:
            print("Prior has different facet_order; skipping comparison")

    # Save
    out = {
        "source": ("Johnson IPIP-NEO-300 raw data, direct from PsychArchives "
                   "deposit e42a4531-1daa-4f3d-aef4-58f085c77cd8 "
                   "(Kajonius & Johnson 2019 supplementary materials; "
                   "raw item-level data for both IPIP-NEO-300 and "
                   "IPIP-NEO-120 are included in that deposit)"),
        "scoring": ("simple sum of 10 items per facet on 1-5 Likert; "
                    "data in .por file is pre-reversed for reverse-keyed "
                    "items, so NO reverse-keying is applied here"),
        "n_raw": n_raw,
        "n": n_complete,
        "complete_case_policy": "drop rows with any 0-coded missing or NaN",
        "facet_order": facet_names,
        "correlation_matrix": corr.tolist(),
        "within_mean": float(np.mean(within_pairs)),
        "across_mean": float(np.mean(across_pairs)),
        "within_median": float(np.median(within_pairs)),
        "across_median": float(np.median(across_pairs)),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
