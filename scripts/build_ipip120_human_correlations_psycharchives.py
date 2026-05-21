#!/usr/bin/env python3
"""Compute 30x30 IPIP-NEO-120 inter-facet correlations from the PsychArchives
Kajonius & Johnson 2019 data deposit (raw N=619,150).

Companion to build_ipip_human_correlations_psycharchives.py (which scores
IPIP-NEO-300). Same dataset deposit, different instrument: the 120-item
short form has 4 items per facet instead of 10.

Motivation: Kajonius & Johnson 2019 Figure 1 shows a strong block-diagonal
facet correlation structure on the IPIP-NEO-120, with e.g. cor(C1, C4) ≈ 0.68.

IMPORTANT — PA .por convention: the IPIP120.por file in this deposit
stores responses with REVERSE-KEYED ITEMS ALREADY REVERSED. The Excel
"Sign" column ("+C1", "-C4") indicates the item's theoretical pole but
should NOT be re-applied to the stored values. Confirmed empirically:
under no-reverse, within-facet alphas are ~+0.6-0.8 (Achievement-Striving
α = +0.80); under double-reverse, the same alphas go negative (α = -0.99).

Output: instruments/ipip120_human_facet_correlations.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

POR_PATH = (
    "data/kajonius_johnson_2019/data/kajonius_johnson_2019/IPIP120.por"
)
KEY_PATH = (
    "data/kajonius_johnson_2019/data/kajonius_johnson_2019/IPIP-NEO-ItemKey.xls"
)
OUT = "instruments/ipip120_human_facet_correlations.json"

# Match abbreviations used in scripts/ipip_facet_cluster.py and the 300 build
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

# Canonical Big-Five trait order (same as 300 build, NEOAC -> A,C,E,N,O)
TRAIT_ORDER = ["A", "C", "E", "N", "O"]


def main():
    # Load IPIP-NEO-ItemKey: rows with non-null Short# are the 120-item subset
    print(f"Loading scoring key {KEY_PATH} ...")
    key = pd.read_excel(KEY_PATH).dropna(subset=["Short#"]).copy()
    key["Short#"] = key["Short#"].astype(int)
    key = key.sort_values("Short#").reset_index(drop=True)
    assert len(key) == 120
    # Sign is "+N1", "-A2" etc -> pole, trait, facet_num
    key["pole"] = key["Sign"].str[0].map({"+": "fwd", "-": "rev"})
    key["trait"] = key["Sign"].str[1]
    key["facet_num"] = key["Sign"].str[2]
    # Verify 4 items per (trait, facet_num)
    cnt = key.groupby(["trait", "facet_num"]).size()
    assert (cnt == 4).all(), f"non-4 facets: {cnt[cnt!=4]}"

    print(f"Loading {POR_PATH} ...")
    df, _meta = pyreadstat.read_por(POR_PATH)
    n_raw = len(df)
    print(f"  raw cases: {n_raw}")

    # IMPORTANT: Johnson IPIP-NEO encoding uses 0 to indicate MISSING.
    items = df.loc[:, [f"I{n}" for n in range(1, 121)]].to_numpy(dtype="float64")
    has_zero = (items == 0).any(axis=1)
    has_nan = np.isnan(items).any(axis=1)
    mask = ~(has_zero | has_nan)
    items = items[mask]
    n_complete = items.shape[0]
    print(f"  complete cases: {n_complete} "
          f"(dropped {n_raw - n_complete}; "
          f"zero-coded {has_zero.sum()}, NaN-coded {has_nan.sum()})")

    # Bucket: (trait, facet_abbrev) -> list of (short_item_idx, pole)
    buckets = {}
    for _, row in key.iterrows():
        t = row["trait"]
        f_abbrev = _CANON_TO_ABBREV[row["Facet"]]
        buckets.setdefault((t, f_abbrev), []).append(
            (int(row["Short#"]), row["pole"])
        )
    assert len(buckets) == 30
    for k, v in buckets.items():
        assert len(v) == 4, f"bucket {k}: {len(v)} items"

    # Canonical order: A,C,E,N,O × 6 facets each, with facets in the order they
    # first appear in the key (matches the 300 build's iteration).
    facet_names = []
    seen_order = {t: [] for t in TRAIT_ORDER}
    for _, row in key.iterrows():
        t = row["trait"]
        f_abbrev = _CANON_TO_ABBREV[row["Facet"]]
        if f_abbrev not in seen_order[t]:
            seen_order[t].append(f_abbrev)
    for t in TRAIT_ORDER:
        for f_abbrev in seen_order[t]:
            facet_names.append(f"{t}:{f_abbrev}")
    assert len(facet_names) == 30

    # Score: simple sum, no reverse-keying (.por data is pre-reversed; see docstring).
    # Per-facet score in [4, 20].
    facet_scores = np.zeros((n_complete, 30), dtype="float64")
    for j, fn in enumerate(facet_names):
        t, f_abbrev = fn.split(":")
        for item_idx, _pole in buckets[(t, f_abbrev)]:
            facet_scores[:, j] += items[:, item_idx - 1]

    print(f"\nFacet score ranges (expected [4, 20]):")
    print(f"  min: {facet_scores.min():.0f}, max: {facet_scores.max():.0f}")

    corr = np.corrcoef(facet_scores, rowvar=False)
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
    print(f"\n30x30 facet correlation matrix (IPIP-NEO-120 raw-sum):")
    print(f"  within-trait pairs (n={len(within_pairs)}): "
          f"mean={np.mean(within_pairs):+.4f}, median={np.median(within_pairs):+.4f}")
    print(f"  across-trait pairs (n={len(across_pairs)}): "
          f"mean={np.mean(across_pairs):+.4f}, median={np.median(across_pairs):+.4f}")
    print(f"  within/across ratio (mean): "
          f"{np.mean(within_pairs) / max(abs(np.mean(across_pairs)), 1e-9):.2f}x")

    # Compare with IPIP-NEO-300 raw-sum reference (if present)
    ref300 = "instruments/ipip300_human_facet_correlations.json"
    if Path(ref300).exists():
        with open(ref300) as f:
            r300 = json.load(f)
        if r300["facet_order"] == facet_names:
            r300_mat = np.array(r300["correlation_matrix"])
            triu_i, triu_k = np.triu_indices(30, k=1)
            corr_r = np.corrcoef(r300_mat[triu_i, triu_k], corr[triu_i, triu_k])[0, 1]
            print(f"\nCorrelation with IPIP-300 raw-sum off-diagonal pattern: r={corr_r:.4f}")
            print(f"  IPIP-300 within mean: {r300['within_mean']:+.4f}, "
                  f"across mean: {r300['across_mean']:+.4f}")

    out = {
        "instrument": "IPIP-NEO-120",
        "source": ("Johnson IPIP-NEO-120 raw data, direct from PsychArchives "
                   "deposit e42a4531-1daa-4f3d-aef4-58f085c77cd8 "
                   "(Kajonius & Johnson 2019 supplementary materials)"),
        "n_raw": n_raw,
        "n": n_complete,
        "items_per_facet": 4,
        "complete_case_policy": "drop rows with any 0-coded missing item",
        "scoring": ("simple sum of 4 items per facet on 1-5 Likert; "
                    "data in .por file is pre-reversed for reverse-keyed "
                    "items, so NO reverse-keying is applied here"),
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
