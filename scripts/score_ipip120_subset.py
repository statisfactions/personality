#!/usr/bin/env python3
"""Option (b) diagnostic: score Qwen's existing IPIP-300 result file down to
the 120-item subset and compare against the full-form IPIP-300 trait scores.

If the 120-subset and 300-full trait means agree closely, the short-form
gives a reliable estimate for any model we've already run IPIP-300 on, which
means option-(b) subsetting is a free path for the within-Qwen / cross-cohort
Mandarin comparison without re-running inference for the English side.

Output: prints a comparison table and writes
results/surveys/Qwen_ipip120e_subset.json with per-trait and per-facet means.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

QWEN_PATH = "results/surveys/Qwen_ipip300.json"
INSTR_PATH = "instruments/ipip120_english.json"
OUT_PATH = "results/surveys/Qwen_ipip120e_subset.json"

with open(QWEN_PATH) as f:
    qwen = json.load(f)
with open(INSTR_PATH) as f:
    instr = json.load(f)
with open("instruments/ipip120_english_facet_map.json") as f:
    facet_map = json.load(f)

items_meta = facet_map["items"]                  # ipip120e_k → {trait,facet,pole,text,ipip300_id}
item_results = qwen["item_results"]              # ipipN → {expected_value, distribution, ...}

# Build per-trait and per-facet lists of (item_id, scored_ev).
trait_scores = defaultdict(list)
facet_scores = defaultdict(list)
for item_id, meta in items_meta.items():
    ipip300_id = meta.get("ipip300_id")
    assert ipip300_id is not None, item_id
    ev = item_results[ipip300_id]["expected_value"]
    scored = (6.0 - ev) if meta["pole"] == "rev" else ev
    trait_scores[meta["trait"]].append((item_id, scored))
    facet_scores[(meta["trait"], meta["facet"])].append((item_id, scored))

TRAIT_NAME = {"N": "Neuroticism", "E": "Extraversion", "O": "Openness",
              "A": "Agreeableness", "C": "Conscientiousness"}
TRAIT_TO_300_SCALE = {"N": "IPIP300-NEU", "E": "IPIP300-EXT", "O": "IPIP300-OPE",
                      "A": "IPIP300-AGR", "C": "IPIP300-CON"}

# Trait means
trait_summary = {}
print(f"{'trait':<22s} {'120-subset μ':>14s} {'full IPIP-300 μ':>16s} {'Δ':>8s}")
for t in ["N", "E", "O", "A", "C"]:
    subset_mean = mean(s for _, s in trait_scores[t])
    full_scale = qwen["scale_scores"][TRAIT_TO_300_SCALE[t]]
    full_mean = full_scale["ev_mean"]
    delta = subset_mean - full_mean
    print(f"{TRAIT_NAME[t]:<22s} {subset_mean:>14.4f} {full_mean:>16.4f} {delta:>+8.4f}")
    trait_summary[t] = {
        "name": TRAIT_NAME[t],
        "n_items": len(trait_scores[t]),
        "subset_mean": round(subset_mean, 4),
        "full_form_mean": round(full_mean, 4),
        "delta": round(delta, 4),
    }

print()
print(f"{'facet':<28s} {'μ':>8s} {'n':>3s}")
facet_summary = {}
for (trait, facet), pairs in sorted(facet_scores.items()):
    fm = mean(s for _, s in pairs)
    facet_summary[f"{trait}:{facet}"] = {"mean": round(fm, 4), "n_items": len(pairs)}
    print(f"{trait}:{facet:<26s} {fm:>8.4f} {len(pairs):>3d}")

# Persist output.
out = {
    "model": qwen.get("model"),
    "hf_repo": qwen.get("hf_repo"),
    "source_full_form_path": QWEN_PATH,
    "instrument": "IPIP-NEO-120 (English) scored from IPIP-300 item results",
    "trait_summary": trait_summary,
    "facet_summary": facet_summary,
    "item_results": {
        item_id: {
            "ipip300_id": meta["ipip300_id"],
            "text": meta["text"],
            "trait": meta["trait"],
            "facet": meta["facet"],
            "pole": meta["pole"],
            "ev_raw": item_results[meta["ipip300_id"]]["expected_value"],
            "ev_scored": (6.0 - item_results[meta["ipip300_id"]]["expected_value"])
                          if meta["pole"] == "rev" else item_results[meta["ipip300_id"]]["expected_value"],
        }
        for item_id, meta in items_meta.items()
    },
}
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT_PATH}")
