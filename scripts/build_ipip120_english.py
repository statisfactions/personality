#!/usr/bin/env python3
"""Build instruments/ipip120_english.json and ipip120_english_facet_map.json
parallel to the Mandarin instrument.

English item texts come from Johnson's Admin 120 sheet (OSF /ycvdk/), prefixed
with "I " to match the IPIP-300 convention used elsewhere in this repo. Each
item is cross-referenced with our existing instruments/ipip300.json so we have
the IPIP-300 id (ipip{N}) that corresponds to each Johnson-120 item — that
mapping is what lets us score the existing Qwen IPIP-300 result file down to
the 120-item subset.
"""

import json
from pathlib import Path

import pandas as pd

ADMIN = pd.read_excel("/tmp/ipip120_scoring.xls", sheet_name="Admin 120")
assert ADMIN.shape[0] == 120

with open("instruments/ipip300.json") as f:
    IPIP300 = json.load(f)
# Build a text-stripped lookup: lowercase, no "I " prefix, no trailing dot.
def norm(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("i "):
        s = s[2:]
    return s.rstrip(".").strip()
TEXT_TO_IPIP300 = {norm(v): k for k, v in IPIP300["items"].items()}

# Two Johnson IPIP-NEO-120 items have minor wording revisions from the
# IPIP-300 original. Same construct, near-identical phrasing — mapping
# explicitly so the (b) subset diagnostic covers all 120.
MANUAL_OVERRIDES = {
    "feel comfortable around other people": "ipip62",   # IPIP-300: "feel comfortable around people"
    "enjoy wild flights of fancy":          "ipip33",   # IPIP-300: "enjoy wild flights of fantasy"
}
for text, ipip300_id in MANUAL_OVERRIDES.items():
    TEXT_TO_IPIP300[norm(text)] = ipip300_id

TRAIT_NAME = {
    "N": "Neuroticism", "E": "Extraversion", "O": "Openness",
    "A": "Agreeableness", "C": "Conscientiousness",
}
TRAIT_ORDER = ["N", "E", "O", "A", "C"]

items_flat = {}
items_meta = {}
trait_to_ids = {t: [] for t in TRAIT_ORDER}
trait_to_reverse = {t: [] for t in TRAIT_ORDER}
facet_to_ids = {}
ipip300_id_for = {}            # ipip120e_k → ipip{N}
missing = []

for i, row in ADMIN.iterrows():
    k = int(row["M5-120 #"])
    assert k == i + 1
    facet_code = str(row["Facet"]).strip()
    facet_name = str(row["Facet Name"]).strip()
    fr = str(row["Score F/R"]).strip()
    text_raw = str(row["Text"]).strip()
    reverse = (fr == "R")
    trait = facet_code[0]

    text_prefixed = f"I {text_raw[0].lower()}{text_raw[1:]}"
    item_id = f"ipip120e_{k}"

    ipip300_id = TEXT_TO_IPIP300.get(norm(text_raw))
    if ipip300_id is None:
        missing.append((k, text_raw))
    else:
        ipip300_id_for[item_id] = ipip300_id

    items_flat[item_id] = text_prefixed
    items_meta[item_id] = {
        "trait": trait,
        "facet": facet_name,
        "facet_code": facet_code,
        "pole": "rev" if reverse else "fwd",
        "text": text_prefixed,
        "ipip300_id": ipip300_id,
    }
    trait_to_ids[trait].append(item_id)
    if reverse:
        trait_to_reverse[trait].append(item_id)
    facet_to_ids.setdefault(facet_name, []).append(item_id)

print(f"matched {120 - len(missing)}/120 items to IPIP-300; missing:")
for k, txt in missing[:15]:
    print(f"  item {k}: {txt!r}")

# Build the main instrument JSON.
SOURCE_NOTE = (
    "English IPIP-NEO-120 items from Johnson, J. A. (2014). Development of the "
    "IPIP-NEO-120. Journal of Research in Personality, 51, 78-89. "
    "Item texts and scoring key from Johnson's Admin 120 sheet "
    "(https://osf.io/ycvdk/), with 'I ' prefix added to match this project's "
    "IPIP-300 convention. Parallel to instruments/ipip120_mandarin.json — same "
    "item numbering 1..120, same trait/facet/keying. Each item's "
    "ipip300_id field gives the matching IPIP-300 item for cross-referencing "
    "with existing IPIP-300 result files."
)

instrument = {
    "_meta": {
        "name": "IPIP-NEO-120 English",
        "source": SOURCE_NOTE,
        "trait_scales": [f"IPIP120E-{t}" for t in TRAIT_ORDER],
        "n_items": 120,
        "item_id_format": "ipip120e_{1..120}, matching Johnson IPIP-NEO-120 admin order",
        "facet_map_path": "instruments/ipip120_english_facet_map.json",
        "parallel_to": "instruments/ipip120_mandarin.json",
    },
    "user_readable_name": "IPIP-NEO-120 (English)",
    "items": items_flat,
    "scales": {
        f"IPIP120E-{t}": {
            "user_readable_name": f"IPIP-NEO-120 {TRAIT_NAME[t]}",
            "item_ids": trait_to_ids[t],
            "reverse_keyed_item_ids": trait_to_reverse[t],
        }
        for t in TRAIT_ORDER
    },
    "ipip300_id_map": ipip300_id_for,    # ipip120e_k → ipipN for subsetting
}

facet_names_per_trait = {t: [] for t in TRAIT_ORDER}
seen = set()
for item_id, meta in items_meta.items():
    key = (meta["trait"], meta["facet"])
    if key not in seen:
        facet_names_per_trait[meta["trait"]].append(meta["facet"])
        seen.add(key)

facet_map = {
    "_method": {
        "source": "Johnson IPIP-NEO-120 canonical scoring (Admin 120 sheet, https://osf.io/ycvdk/).",
        "trait_cycle": ["N", "E", "O", "A", "C"],
        "facet_names_per_trait": facet_names_per_trait,
    },
    "items": items_meta,
}

INSTR_OUT = Path("instruments/ipip120_english.json")
FACET_OUT = Path("instruments/ipip120_english_facet_map.json")
INSTR_OUT.write_text(json.dumps(instrument, ensure_ascii=False, indent=2))
FACET_OUT.write_text(json.dumps(facet_map, ensure_ascii=False, indent=2))

print(f"\nwrote {INSTR_OUT} ({INSTR_OUT.stat().st_size:,} bytes)")
print(f"wrote {FACET_OUT} ({FACET_OUT.stat().st_size:,} bytes)")

# Confirm trait counts match Mandarin file.
print(f"\ntrait counts:")
for t in TRAIT_ORDER:
    print(f"  {t} ({TRAIT_NAME[t]:<18s}): {len(trait_to_ids[t]):>3d} items, "
          f"{len(trait_to_reverse[t]):>3d} reverse-keyed")
