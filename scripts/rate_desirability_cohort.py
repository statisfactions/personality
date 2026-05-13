#!/usr/bin/env python3
"""W11 Phase A: validate cohort models as desirability raters.

For each cohort model and each of the 60 Goldberg items in Okada's
Table 3, ask the model to rate the trait's desirability on a 9-point
scale via distributional logprobs over {'1','2','3','4','5','6',
'7','8','9'}. Compute EV per item per model, then compare to Okada's
published values via Pearson r.

Per-item distributional approach (not Okada's batched-25-items approach):
deterministic, single forward pass per item, gives EV with implicit
uncertainty. Trades exact methodology fidelity for robustness on small
models that can't reliably emit a 25-integer block.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/rate_desirability_cohort.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hf_logprobs import MODELS as ALL_MODELS, load_model, likert_distribution


COHORT = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
ITEMS_FILE = "instruments/okada_goldberg_items.json"
OUT_FILE = "results/desirability/cohort_phase_a.json"

PROMPT_TEMPLATE = (
    "The following is a characteristic of people. Indicate on a scale "
    "from 1 to 9 how desirable you think this trait or characteristic "
    "is for an adult person: 1 = Very undesirable, 3 = Undesirable, "
    "5 = Neutral, 7 = Desirable, 9 = Very desirable. Use any number "
    "from 1 through 9 that best indicates your opinion.\n\n"
    "Characteristic: \"{text}\"\n\n"
    "Rating (single integer 1-9):"
)

ALPHABET = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
NUMERIC_VALUES = list(range(1, 10))


def rate_one(model, tok, device, text):
    user_body = PROMPT_TEMPLATE.format(text=text)
    dist, argmax, _ = likert_distribution(
        model, tok, user_body, device,
        digits=tuple(ALPHABET), use_chat_template=True,
    )
    # dist is a Dict[str, float] keyed by digit token
    probs = [dist[d] for d in ALPHABET]
    ev = float(np.sum(np.array(probs) * np.array(NUMERIC_VALUES)))
    return {"ev": round(ev, 4), "argmax": argmax, "dist": [round(p, 6) for p in probs]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT,
                        help="Cohort models to run. Default: all 7.")
    parser.add_argument("--output", default=OUT_FILE)
    args = parser.parse_args()

    items_data = json.load(open(ITEMS_FILE))
    items = items_data["items"]
    print(f"Loaded {len(items)} items from {ITEMS_FILE}")

    Path("results/desirability").mkdir(parents=True, exist_ok=True)

    # Allow resume: load existing partial output if present
    if Path(args.output).exists():
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming from {args.output}: {len(results.get('ratings', {}))} model(s) already done")
    else:
        results = {
            "okada_items_source": ITEMS_FILE,
            "prompt_template": PROMPT_TEMPLATE,
            "alphabet": ALPHABET,
            "ratings": {},
        }

    for short in args.models:
        if short in results["ratings"]:
            print(f"  {short}: already done, skipping")
            continue
        if short not in ALL_MODELS:
            print(f"  {short}: unknown model, skipping")
            continue
        repo = ALL_MODELS[short]
        print(f"\n========== {short} ({repo}) ==========")
        t0 = time.time()
        model, tok, device = load_model(short)
        per_item = {}
        for idx, item in enumerate(items):
            text = item["text"]
            r = rate_one(model, tok, device, text)
            per_item[f"b{item['block']}_{item['side']}"] = {
                "text": text,
                "domain": item["domain"],
                "keying": item["keying"],
                "okada_sd": item["sd"],
                **r,
            }
            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(items)}  last EV={r['ev']:.2f} for {text[:60]!r}")
        results["ratings"][short] = per_item

        # Quick correlation snapshot
        oks = [items[i]["sd"] for i in range(len(items))]
        evs = [per_item[f"b{items[i]['block']}_{items[i]['side']}"]["ev"] for i in range(len(items))]
        r = float(np.corrcoef(oks, evs)[0, 1])
        print(f"  {short} vs Okada: Pearson r = {r:+.3f}  (n={len(items)}, elapsed {time.time()-t0:.0f}s)")

        # Save after each model so partial results survive a crash
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        del model, tok
        import gc
        gc.collect()
        try:
            import torch
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    # Final summary
    print(f"\n========== Cohort vs Okada Pearson r ==========")
    oks = [it["sd"] for it in items]
    for short in args.models:
        if short not in results["ratings"]:
            continue
        per_item = results["ratings"][short]
        evs = [per_item[f"b{items[i]['block']}_{items[i]['side']}"]["ev"] for i in range(len(items))]
        r = float(np.corrcoef(oks, evs)[0, 1])
        print(f"  {short:<10s} r = {r:+.3f}")
    # Cohort mean
    if len(results["ratings"]) > 1:
        cohort_mean_evs = []
        for i in range(len(items)):
            evs_for_item = [results["ratings"][m][f"b{items[i]['block']}_{items[i]['side']}"]["ev"]
                            for m in args.models if m in results["ratings"]]
            cohort_mean_evs.append(float(np.mean(evs_for_item)))
        r_cohort = float(np.corrcoef(oks, cohort_mean_evs)[0, 1])
        print(f"  COHORT-MEAN r = {r_cohort:+.3f}")

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
