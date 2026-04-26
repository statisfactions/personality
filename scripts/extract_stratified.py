#!/usr/bin/env python3
"""Extract activations for the new stratified training pairs and cache them in
the same format as results/phase_b_cache/ for drop-in reuse with the existing
analysis scripts.

Output per (model, trait):
    results/phase_b_cache_stratified/<tag>_<trait>_chat_pairs.pt
containing {"ph_tr", "pl_tr", "pairs"} — note: no ph_h/pl_h since we don't
split train/holdout here (the whole stratified set is training data for
structural analyses).

Neutral activations reuse results/phase_b_cache/<tag>_neutral_chat.pt.

Usage:
    python scripts/extract_stratified.py
    python scripts/extract_stratified.py --models Llama --traits H
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

import extract_meandiff_vectors as mdx


MODELS = {
    # Small cohort (weeks 1–6).
    "Llama":   "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma":   "google/gemma-3-4b-it",
    "Phi4":    "microsoft/Phi-4-mini-instruct",
    "Qwen":    "Qwen/Qwen2.5-3B-Instruct",
    # Phase-1 larger cohort (SAE-covered).
    "Llama8":  "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma12": "google/gemma-3-12b-it",
    "Qwen7":   "Qwen/Qwen2.5-7B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
STRATIFIED_FILE = Path("instruments/contrast_pairs_stratified.json")
CACHE_DIR = Path("results/phase_b_cache_stratified")


def safe(s): return s.replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--traits", nargs="+", default=TRAITS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(STRATIFIED_FILE) as f:
        cp = json.load(f)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    t_start = time.time()
    for short in args.models:
        if short not in MODELS:
            print(f"unknown model: {short}")
            continue
        repo = MODELS[short]
        tag = safe(repo)

        print(f"\n{'=' * 70}\nModel: {short} ({repo})\n{'=' * 70}")
        t0 = time.time()
        model, tokenizer = mdx.load_model(repo, args.device, args.dtype)
        print(f"Loaded in {time.time() - t0:.1f}s")

        for trait in args.traits:
            out_path = CACHE_DIR / f"{tag}_{trait}_chat_pairs.pt"
            if out_path.exists():
                print(f"  {trait}: already exists, skipping")
                continue

            train_td = {
                "name": cp["traits"][trait]["name"],
                "pairs": cp["traits"][trait]["pairs"],
                "high_descriptor": cp["traits"][trait]["high_descriptor"],
                "low_descriptor": cp["traits"][trait]["low_descriptor"],
            }
            n = len(train_td["pairs"])
            print(f"  {trait}: extracting {n} pairs (chat, generic prefix)...", end=" ", flush=True)
            t0 = time.time()
            ph_tr, pl_tr = mdx.extract_trait_activations(
                model, tokenizer, train_td, "generic", args.device,
                chat_template=True, verbose=False,
            )
            print(f"{time.time() - t0:.1f}s  shape={tuple(ph_tr.shape)}")

            torch.save(
                {"ph_tr": ph_tr, "pl_tr": pl_tr, "pairs": train_td["pairs"]},
                out_path,
            )

            del ph_tr, pl_tr
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()

        del model, tokenizer
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f} min)")
    print(f"Cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
