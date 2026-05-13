#!/usr/bin/env python3
"""W11 Phase B: rate the 300 IPIP-NEO items for desirability.

Per-item distributional rating across 7 cohort models, using Option A
from the phrasing pilot (de-personalized items + Okada Appendix A
prompt). Output is the input to Phase C's constrained-MIP pair
selection.

Reads:
  admin_sessions/prod_run_01_external_rating.json (IPIP-NEO-300 items
  + scales with keying + trait labels)

Writes:
  results/desirability/cohort_phase_b_ipip300.json — per-model per-item
  EV ratings, distributions, argmax, and the cohort-mean desirability
  scores ready for MIP input.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/rate_desirability_ipip300.py
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from hf_logprobs import MODELS as ALL_MODELS, load_model, likert_distribution


COHORT = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]

ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
OUT_FILE = "results/desirability/cohort_phase_b_ipip300.json"
ANNOTATIONS = "instruments/ipip300_annotations.json"  # for typo fixes

PROMPT_TEMPLATE = (
    "The following is a characteristic of people. Indicate on a scale "
    "from 1 to 9 how desirable you think this trait or characteristic "
    "is for an adult person: 1 = Very undesirable, 3 = Undesirable, "
    "5 = Neutral, 7 = Desirable, 9 = Very desirable. Use any number "
    "from 1 through 9 that best indicates your opinion.\n\n"
    "Characteristic: \"{text}\"\n\n"
    "Rating (single integer 1-9):"
)

ALPHABET = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
NUMERIC_VALUES = list(range(1, 10))

# Map IPIP scale labels to canonical single-letter trait codes
TRAIT_MAP = {
    "IPIP300-AGR": "A", "IPIP300-CON": "C", "IPIP300-EXT": "E",
    "IPIP300-NEU": "N", "IPIP300-OPE": "O",
}


def depersonalize(text: str) -> str:
    """Strip leading 'I ' / 'I'm ' to convert first-person to fragment.
    Matches the transform used in the Phase B prep pilot.
    """
    t = text.strip()
    t = re.sub(r"^I'm\b", "am", t)
    t = re.sub(r"^I\s+", "", t)
    return t


def build_item_list():
    """Load IPIP-NEO-300 items with trait/keying labels."""
    admin = json.load(open(ADMIN_SESSION))
    ipip = admin["measures"]["IPIP300"]
    items = ipip["items"]
    scales = ipip["scales"]
    # Try to load annotations for typo fixes if present
    fixes = {}
    if Path(ANNOTATIONS).exists():
        ann = json.load(open(ANNOTATIONS))
        fixes = ann.get("fix", {})

    item_list = []
    for scale_label, trait in TRAIT_MAP.items():
        sc = scales[scale_label]
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])
        for iid in iids:
            raw = fixes.get(iid, items[iid])
            item_list.append({
                "id": iid,
                "text": raw,
                "depersonalized": depersonalize(raw),
                "trait": trait,
                "keying": "-" if iid in rev else "+",
            })
    return item_list


def rate_one(model, tok, device, depersonalized_text):
    prompt = PROMPT_TEMPLATE.format(text=depersonalized_text)
    dist, argmax, _ = likert_distribution(
        model, tok, prompt, device,
        digits=ALPHABET, use_chat_template=True,
    )
    probs = [dist[d] for d in ALPHABET]
    ev = float(np.sum(np.array(probs) * np.array(NUMERIC_VALUES)))
    return {"ev": round(ev, 4), "argmax": argmax, "dist": [round(p, 6) for p in probs]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT,
                        help="Cohort models. Default: all 7.")
    parser.add_argument("--output", default=OUT_FILE)
    args = parser.parse_args()

    items = build_item_list()
    print(f"Loaded {len(items)} IPIP-NEO-300 items "
          f"({sum(1 for i in items if i['keying']=='+')}/{sum(1 for i in items if i['keying']=='-')} forward/reverse)")

    Path("results/desirability").mkdir(parents=True, exist_ok=True)

    if Path(args.output).exists():
        results = json.load(open(args.output))
        print(f"Resuming from {args.output}: {len(results.get('ratings', {}))} model(s) done")
    else:
        results = {
            "items": items,
            "prompt_template": PROMPT_TEMPLATE,
            "alphabet": list(ALPHABET),
            "ratings": {},
        }

    for short in args.models:
        if short in results["ratings"]:
            print(f"  {short}: done, skipping")
            continue
        if short not in ALL_MODELS:
            print(f"  {short}: unknown, skipping"); continue
        print(f"\n========== {short} ({ALL_MODELS[short]}) ==========")
        t0 = time.time()
        model, tok, device = load_model(short)
        per_item = {}
        for idx, it in enumerate(items):
            r = rate_one(model, tok, device, it["depersonalized"])
            per_item[it["id"]] = r
            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(items)}  last EV={r['ev']:.2f}")
        results["ratings"][short] = per_item

        # Quick sanity: positive vs negative-keyed mean EV (should be higher for +)
        pos_evs = [per_item[it["id"]]["ev"] for it in items if it["keying"] == "+"]
        neg_evs = [per_item[it["id"]]["ev"] for it in items if it["keying"] == "-"]
        print(f"  {short}: forward-keyed mean EV = {np.mean(pos_evs):.2f}, "
              f"reverse-keyed mean EV = {np.mean(neg_evs):.2f} "
              f"(should be higher for forward; elapsed {time.time()-t0:.0f}s)")

        json.dump(results, open(args.output, "w"), indent=2)
        del model, tok
        import gc; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    # Compute cohort-mean per item
    print(f"\n========== Cohort-mean desirability ==========")
    cohort_means = {}
    for it in items:
        evs = [results["ratings"][m][it["id"]]["ev"]
               for m in args.models if m in results["ratings"]]
        cohort_means[it["id"]] = float(np.mean(evs))
    results["cohort_mean_desirability"] = cohort_means
    json.dump(results, open(args.output, "w"), indent=2)

    # Per-trait sanity
    print(f"  Per-trait cohort-mean desirability (forward vs reverse):")
    for trait in "ACENO":
        pos_means = [cohort_means[it["id"]] for it in items if it["trait"]==trait and it["keying"]=="+"]
        neg_means = [cohort_means[it["id"]] for it in items if it["trait"]==trait and it["keying"]=="-"]
        print(f"    {trait}: forward mean {np.mean(pos_means):.2f} (n={len(pos_means)}), "
              f"reverse mean {np.mean(neg_means):.2f} (n={len(neg_means)})")

    # Range of cohort-mean
    cm_arr = np.array(list(cohort_means.values()))
    print(f"\n  Cohort-mean range: [{cm_arr.min():.2f}, {cm_arr.max():.2f}], "
          f"mean {cm_arr.mean():.2f}, SD {cm_arr.std():.2f}")
    print(f"  Saved {args.output}")


if __name__ == "__main__":
    main()
