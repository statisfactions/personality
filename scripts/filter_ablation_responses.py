#!/usr/bin/env python3
"""Filter P=60 response JSONs to a subset of blocks for the W12 TIRT ablation.

For each of the 21 P=60 inference files, write three filtered copies
(top30, bot30, rand30) keeping only records whose block is in the subset.
Output filename pattern: <base>_<subset>.json under
psychometrics/gfc_tirt/ablation_subsets/.
"""

import json
from pathlib import Path

SUBSETS_PATH = Path("psychometrics/gfc_tirt/ablation_subsets.json")
SRC_DIR = Path("psychometrics/gfc_tirt")
OUT_DIR = Path("psychometrics/gfc_tirt/ablation_subsets")
OUT_DIR.mkdir(exist_ok=True, parents=True)

subsets = json.load(open(SUBSETS_PATH))
selections = {k: set(subsets[k]) for k in ("top30", "bot30", "rand30")}

src_files = sorted(SRC_DIR.glob("*_ipipneogfc60_hf_*.json"))
src_files = [f for f in src_files if not f.name.endswith("_indep_fit.json")]
print(f"Source response files: {len(src_files)}")

for src in src_files:
    d = json.load(open(src))
    base = src.stem  # e.g., Gemma_ipipneogfc60_hf_description
    for subset_name, block_set in selections.items():
        # Renumber the selected blocks to a contiguous 1..30 range, so
        # downstream R fitter's `paste0("b", seq_len(P))` works.
        # Preserve the original order (sorted block IDs -> 1..30).
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
print(f"\nDone. {len(src_files) * 3} filtered JSONs in {OUT_DIR}")
