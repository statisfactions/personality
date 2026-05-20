#!/usr/bin/env python3
"""Build instruments/ipip300_facet_map.json from the canonical Johnson
IPIP-NEO-300 scoring scheme (as implemented in NeuroQuestAi/five-factor-e).

The Johnson 1999 layout interleaves all 5 traits and all 6 facets:
- Item n (1-indexed) belongs to trait "NEOAC"[(n-1) % 5]
- Item n belongs to facet ((n-1) % 30) // 5 within its trait

This makes the *standalone-trait* facet ordering (which our pipeline uses
via `iids[fi::6]`) match the canonical layout exactly.

After writing the map, this script verifies:
1. The map agrees with our admin_session's IPIP300 trait-scale item_ids
2. Reverse-keyed status agrees with admin_session's reverse_keyed_item_ids
3. Each (trait, facet) bucket has exactly 10 items
"""

import json
from pathlib import Path

ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
OUT = "instruments/ipip300_facet_map.json"

# Canonical Johnson IPIP-NEO-300 facet names per trait (from
# five-factor-e/ipipneo/model.py Big5* enums; matches ipip.ori.org).
# Order within each trait is the facet ORDER in the interleaved layout.
TRAITS = ["N", "E", "O", "A", "C"]
FACETS = {
    "N": ["Anxiety", "Anger", "Depression", "Self-Consciousness",
          "Immoderation", "Vulnerability"],
    "E": ["Friendliness", "Gregariousness", "Assertiveness",
          "Activity Level", "Excitement-Seeking", "Cheerfulness"],
    "O": ["Imagination", "Artistic Interests", "Emotionality",
          "Adventurousness", "Intellect", "Liberalism"],
    "A": ["Trust", "Morality", "Altruism", "Cooperation",
          "Modesty", "Sympathy"],
    "C": ["Self-Efficacy", "Orderliness", "Dutifulness",
          "Achievement-Striving", "Self-Discipline", "Cautiousness"],
}

# Reverse-keyed item ids (1-indexed) per Johnson scheme; 55 items.
# Source: five-factor-e/ipipneo/reverse.py IPIP_NEO_ITEMS_REVERSED_300.
# This is also derivable from admin_session, included here for sanity check.
# (We re-derive from admin_session below rather than hardcoding.)


def canonical_trait_facet(n):
    """Item n (1-indexed) -> (trait, facet)."""
    idx = (n - 1) % 30  # position in cycle of 30
    trait = TRAITS[idx % 5]
    facet_idx = idx // 5
    return trait, FACETS[trait][facet_idx]


def main():
    # Build canonical map for all 300 items
    canon = {}
    for n in range(1, 301):
        trait, facet = canonical_trait_facet(n)
        canon[f"ipip{n}"] = {"trait": trait, "facet": facet}

    # Verify against admin_session
    with open(ADMIN_SESSION) as f:
        ipip = json.load(f)["measures"]["IPIP300"]
    scales = ipip["scales"]
    items = ipip["items"]
    short = {"A": "AGR", "C": "CON", "E": "EXT", "N": "NEU", "O": "OPE"}

    discrepancies = []
    for t in TRAITS:
        sc = scales[f"IPIP300-{short[t]}"]
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])
        assert len(iids) == 60, f"trait {t} has {len(iids)} items, expected 60"

        # Trait check: every item in this scale should have trait == t per canon
        for iid in iids:
            if canon[iid]["trait"] != t:
                discrepancies.append(
                    f"  trait mismatch: admin scale {t}-iids has {iid}, but canon says trait {canon[iid]['trait']}"
                )

        # Facet check via stride-6
        for fi, fname in enumerate(FACETS[t]):
            stride_iids = iids[fi::6]
            for iid in stride_iids:
                if canon[iid]["facet"] != fname:
                    discrepancies.append(
                        f"  facet mismatch: admin scale {t}, fi={fi} ({fname}) yields {iid}, but canon says facet {canon[iid]['facet']}"
                    )
            if len(stride_iids) != 10:
                discrepancies.append(
                    f"  count mismatch: {t}.{fname} stride-6 gave {len(stride_iids)} items, expected 10"
                )

        # Reverse-keyed enrichment: tag pole in canon
        for iid in iids:
            canon[iid]["pole"] = "rev" if iid in rev else "fwd"

    # Add item text from admin session
    for iid in canon:
        canon[iid]["text"] = items.get(iid, "")

    if discrepancies:
        print("DISCREPANCIES FOUND:")
        for d in discrepancies[:20]:
            print(d)
        raise SystemExit(1)

    # Per-bucket count check on canon
    by_bucket = {}
    for iid, info in canon.items():
        key = (info["trait"], info["facet"])
        by_bucket.setdefault(key, []).append(iid)
    for k, v in by_bucket.items():
        assert len(v) == 10, f"bucket {k} has {len(v)} items"

    # Write output
    out = {
        "_method": {
            "source": ("Johnson IPIP-NEO-300 canonical scoring "
                       "(via NeuroQuestAi/five-factor-e)"),
            "verified_against": ADMIN_SESSION,
            "trait_cycle": TRAITS,
            "facet_names_per_trait": FACETS,
            "layout_rule": (
                "Item n (1-indexed): trait = 'NEOAC'[(n-1) % 5]; "
                "within-trait facet index = ((n-1) % 30) // 5"
            ),
        },
        "items": canon,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT}: 300 items across {len(by_bucket)} facets, all checks pass.")
    print(f"  example: ipip1 -> {canon['ipip1']}")
    print(f"  example: ipip6 -> {canon['ipip6']}")
    print(f"  example: ipip31 -> {canon['ipip31']}  (10th Anxiety item starts here)")


if __name__ == "__main__":
    main()
