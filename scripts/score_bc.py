#!/usr/bin/env python3
"""Score binary-choice (BC) log-odds for cross_method_matrix.py.

For each (model × trait), runs all 50 train contrast pairs through
hf_logprobs.bc_logodds with the chat template (matching the old Ollama
/api/chat protocol). Single A/B ordering; NOT position-debiased — this
matches the Week 3 cross-method matrix protocol exactly. See Week 5 §8
for the ~30pt position-bias finding; debiased numbers are a future ablation.

Output: results/binary_choice_6trait_hf.json keyed `<TRAIT>_<short_name>`.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/score_bc.py
    PYTHONPATH=scripts .venv/bin/python scripts/score_bc.py --models Gemma12 --traits H
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from hf_logprobs import MODELS, load_model, bc_logodds


CONTRAST_PAIRS = Path("instruments/contrast_pairs.json")
OUT_PATH = Path("results/binary_choice_6trait.json")
TRAITS = ["H", "E", "X", "A", "C", "O"]
# Default to larger cohort only — small-cohort BC entries already live in OUT_PATH
# from the prior Ollama runs (keyed by Ollama names) and don't need re-running.
# Override with --models if you want to refresh the small cohort too.
COHORT = ["Gemma12", "Llama8", "Qwen7"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT)
    parser.add_argument("--traits", nargs="+", default=TRAITS)
    args = parser.parse_args()

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)

    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    t_start = time.time()
    for short in args.models:
        if short not in MODELS:
            print(f"unknown model: {short}")
            continue

        print(f"\n=== {short} ({MODELS[short]}) ===")
        t0 = time.time()
        model, tok, device = load_model(short)
        print(f"  loaded in {time.time() - t0:.1f}s, device={device}")

        for trait in args.traits:
            key = f"{trait}_{short}"
            pairs = cp["traits"][trait]["pairs"]
            t1 = time.time()
            logodds = []
            for p in pairs:
                bc_prompt = (
                    f"{p['situation']}\n\n"
                    f"Which would you do?\n"
                    f"A) {p['high']}\nB) {p['low']}\n\n"
                    f"Respond with just the letter."
                )
                diff, _, _ = bc_logodds(model, tok, bc_prompt, device,
                                        a="A", b="B", use_chat_template=True)
                logodds.append(float(diff))
            results[key] = logodds
            n_high = sum(1 for v in logodds if v > 0)
            print(f"  {trait}: {len(logodds)} pairs in {time.time()-t1:.1f}s  "
                  f"high-pick={n_high}/{len(logodds)} ({n_high/len(logodds)*100:.0f}%)  "
                  f"mean_lo={sum(logodds)/len(logodds):+.2f}")

            # Save incrementally so a long run is resumable
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\nTotal: {time.time()-t_start:.1f}s")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
