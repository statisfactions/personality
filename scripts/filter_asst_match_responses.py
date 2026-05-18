#!/usr/bin/env python3
"""Filter P=60 response JSONs to the assistant-matched / mismatched subsets
defined in psychometrics/gfc_tirt/ablation_assistant_subsets.json.

Mirrors filter_ablation_responses.py but for the W12 §6.6 follow-up.
"""

import json
from pathlib import Path

SUBSETS_PATH = Path("psychometrics/gfc_tirt/ablation_assistant_subsets.json")
SRC_DIR = Path("psychometrics/gfc_tirt")
OUT_DIR = Path("psychometrics/gfc_tirt/ablation_assistant_subsets")
OUT_DIR.mkdir(exist_ok=True, parents=True)

subsets = json.load(open(SUBSETS_PATH))
selections = {k: set(subsets[k]) for k in ("asst_top30", "asst_bot30")}

src_files = sorted(SRC_DIR.glob("*_ipipneogfc60_hf_*.json"))
# Exclude FG and any non-honest. Honest P=60 files have no _fake_good suffix.
src_files = [f for f in src_files
             if "_fake_good" not in f.name
             and "_ipipneogfc60_hf_" in f.name]
print(f"Source response files: {len(src_files)}")

for src in src_files:
    d = json.load(open(src))
    base = src.stem
    for subset_name, block_set in selections.items():
        sorted_blocks = sorted(block_set)
        block_remap = {orig: new + 1 for new, orig in enumerate(sorted_blocks)}
        filtered = []
        for r in d["results"]:
            if r["block"] not in block_set:
                continue
            r_new = dict(r)
            r_new["original_block"] = r["block"]
            r_new["block"] = block_remap[r["block"]]
            filtered.append(r_new)
        new = dict(d)
        new["results"] = filtered
        new["ablation_subset"] = subset_name
        new["ablation_blocks_original"] = sorted_blocks
        new["block_remap"] = block_remap
        out_path = OUT_DIR / f"{base}_{subset_name}.json"
        with open(out_path, "w") as f:
            json.dump(new, f)
        print(f"  {out_path.name}: {len(filtered)} records, blocks remapped to 1..{len(sorted_blocks)}")
print(f"\nDone. {len(src_files) * 2} filtered JSONs in {OUT_DIR}")
