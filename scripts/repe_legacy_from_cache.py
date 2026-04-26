#!/usr/bin/env python3
"""Adapter: convert phase_b_cache/<tag>_<trait>_<format>_pairs.pt activations
into the legacy results/repe/<tag>_<trait>_directions.pt format that
cross_method_matrix.py expects (raw_diffs of shape n_pairs × n_layers × hidden).

Default format is "bare" — matches the Week 3 RepE protocol used to
generate the existing small-cohort legacy files. Use --format chat for the
chat-template variant.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/repe_legacy_from_cache.py
    PYTHONPATH=scripts .venv/bin/python scripts/repe_legacy_from_cache.py \
        --models Gemma12 Llama8 Qwen7 --format bare
"""

import argparse
from pathlib import Path

import torch

from hf_logprobs import MODELS as ALL_MODELS


CACHE_DIR = Path("results/phase_b_cache")
OUT_DIR = Path("results/repe")
TRAITS = ["H", "E", "X", "A", "C", "O"]
LARGE_COHORT = ["Gemma12", "Llama8", "Qwen7"]


def safe(s): return s.replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=LARGE_COHORT,
                        help="Short names from hf_logprobs.MODELS")
    parser.add_argument("--traits", nargs="+", default=TRAITS)
    parser.add_argument("--format", choices=["bare", "chat"], default="bare",
                        help="Which phase_b_cache format to read (bare matches Week 3 protocol)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    written = 0
    for short in args.models:
        if short not in ALL_MODELS:
            print(f"unknown model: {short}")
            continue
        repo = ALL_MODELS[short]
        tag = safe(repo)
        for trait in args.traits:
            src = CACHE_DIR / f"{tag}_{trait}_{args.format}_pairs.pt"
            dst = OUT_DIR / f"{tag}_{trait}_directions.pt"
            if dst.exists() and not args.overwrite:
                print(f"skip (exists): {dst.name}")
                continue
            if not src.exists():
                print(f"missing: {src.name}")
                continue
            blob = torch.load(src, weights_only=False)
            ph = blob["ph_tr"]  # (n_pairs, n_layers, hidden_dim)
            pl = blob["pl_tr"]
            raw_diffs = ph - pl
            torch.save({"raw_diffs": raw_diffs}, dst)
            print(f"wrote {dst.name}  shape={tuple(raw_diffs.shape)}")
            written += 1
    print(f"\nWrote {written} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
