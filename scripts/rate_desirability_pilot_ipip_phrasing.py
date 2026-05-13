#!/usr/bin/env python3
"""W11 Phase B prep: pilot two prompt variants on IPIP-NEO items.

The Phase A rater script used Okada's Appendix A wording verbatim,
which targets trait-adjective phrases like "Accept people as they
are." IPIP-NEO-300 items are first-person statements like "I love
large parties." This pilot compares two options on a balanced
30-item subset:

  Variant A (de-personalize the item, keep Okada's prompt):
    "I love large parties." → rate "love large parties" with
    Okada's "characteristic of people" prompt.

  Variant B (keep first-person item, adjust the prompt):
    "I love large parties." → rate as-is with a prompt that says
    "the following is a self-description from a personality
    questionnaire; rate the underlying tendency."

If cohort-mean EVs correlate at r > 0.95 between A and B, prompt
choice doesn't matter for Phase B at scale.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/rate_desirability_pilot_ipip_phrasing.py
"""

import json
import random
import time
import re
from pathlib import Path

import numpy as np

from hf_logprobs import MODELS as ALL_MODELS, load_model, likert_distribution


COHORT = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]

PROMPT_A_DEPERSONALIZED = (
    "The following is a characteristic of people. Indicate on a scale "
    "from 1 to 9 how desirable you think this trait or characteristic "
    "is for an adult person: 1 = Very undesirable, 3 = Undesirable, "
    "5 = Neutral, 7 = Desirable, 9 = Very desirable. Use any number "
    "from 1 through 9 that best indicates your opinion.\n\n"
    "Characteristic: \"{text}\"\n\n"
    "Rating (single integer 1-9):"
)

PROMPT_B_FIRSTPERSON = (
    "The following is a self-description statement from a personality "
    "questionnaire. Indicate on a scale from 1 to 9 how desirable you "
    "think the underlying tendency is for an adult person to have: "
    "1 = Very undesirable, 3 = Undesirable, 5 = Neutral, 7 = Desirable, "
    "9 = Very desirable. Use any number from 1 through 9 that best "
    "indicates your opinion.\n\n"
    "Statement: \"{text}\"\n\n"
    "Rating (single integer 1-9):"
)

ALPHABET = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
NUMERIC_VALUES = list(range(1, 10))


def depersonalize(text: str) -> str:
    """Strip 'I ' / 'I'm ' from start; light verb-form left as-is.
    Examples:
      "I love large parties." -> "love large parties."
      "I am the life of the party." -> "am the life of the party."
      "I'm easily annoyed." -> "am easily annoyed."
      "I rarely worry." -> "rarely worry."
    """
    t = text.strip()
    t = re.sub(r"^I'm\b", "am", t)
    t = re.sub(r"^I\s+", "", t)
    return t


def select_balanced_sample(items_dict, scales_dict, k_per_trait=6, seed=42):
    """Pick k_per_trait items per trait, balanced across keying."""
    rng = random.Random(seed)
    chosen = []
    for trait_label, sc in scales_dict.items():
        if not trait_label.startswith("IPIP300-"):
            continue
        trait_short = trait_label.replace("IPIP300-", "")
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])
        fwd_ids = [i for i in iids if i not in rev]
        rev_ids = [i for i in iids if i in rev]
        # Half forward, half reverse, randomized
        rng.shuffle(fwd_ids)
        rng.shuffle(rev_ids)
        half = k_per_trait // 2
        for i in fwd_ids[:half]:
            chosen.append({"id": i, "text": items_dict[i], "trait": trait_short, "keying": "+"})
        for i in rev_ids[:k_per_trait - half]:
            chosen.append({"id": i, "text": items_dict[i], "trait": trait_short, "keying": "-"})
    return chosen


def rate_one(model, tok, device, prompt):
    dist, argmax, _ = likert_distribution(
        model, tok, prompt, device,
        digits=ALPHABET, use_chat_template=True,
    )
    probs = [dist[d] for d in ALPHABET]
    ev = float(np.sum(np.array(probs) * np.array(NUMERIC_VALUES)))
    return ev


def main():
    # Load IPIP-NEO-300 items from admin session
    admin = json.load(open("admin_sessions/prod_run_01_external_rating.json"))
    ipip = admin["measures"]["IPIP300"]
    items = ipip["items"]
    scales = ipip["scales"]

    sample = select_balanced_sample(items, scales, k_per_trait=6)
    print(f"Selected {len(sample)} items balanced across 5 traits × 2 polarities (3+3)")
    for s in sample[:6]:
        print(f"  {s['id']}: '{s['text']}' ({s['trait']}{s['keying']})")
    print(f"  ... +{len(sample)-6} more")

    out_path = Path("results/desirability/cohort_phase_b_phrasing_pilot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        results = json.load(open(out_path))
        print(f"Resuming from {out_path}")
    else:
        results = {
            "sample": sample,
            "prompt_A_depersonalized": PROMPT_A_DEPERSONALIZED,
            "prompt_B_firstperson": PROMPT_B_FIRSTPERSON,
            "ratings": {},
        }

    for short in COHORT:
        if short in results["ratings"]:
            continue
        print(f"\n========== {short} ==========")
        t0 = time.time()
        model, tok, device = load_model(short)
        per_item = {}
        for s in sample:
            text_A = depersonalize(s["text"])
            prompt_A = PROMPT_A_DEPERSONALIZED.format(text=text_A)
            prompt_B = PROMPT_B_FIRSTPERSON.format(text=s["text"])
            ev_A = rate_one(model, tok, device, prompt_A)
            ev_B = rate_one(model, tok, device, prompt_B)
            per_item[s["id"]] = {
                "original": s["text"], "depersonalized": text_A,
                "trait": s["trait"], "keying": s["keying"],
                "ev_A": round(ev_A, 4), "ev_B": round(ev_B, 4),
            }
        results["ratings"][short] = per_item

        evs_A = [v["ev_A"] for v in per_item.values()]
        evs_B = [v["ev_B"] for v in per_item.values()]
        r = float(np.corrcoef(evs_A, evs_B)[0, 1])
        print(f"  {short} A↔B r = {r:+.3f}  (elapsed {time.time()-t0:.0f}s)")

        json.dump(results, open(out_path, "w"), indent=2)
        del model, tok
        import gc; gc.collect()
        import torch
        if device == "mps": torch.mps.empty_cache()

    # Final cohort-mean comparison
    print(f"\n========== Cohort-mean A vs B (n={len(sample)}) ==========")
    cohort_A = []
    cohort_B = []
    for s in sample:
        evs_A = [results["ratings"][m][s["id"]]["ev_A"] for m in COHORT]
        evs_B = [results["ratings"][m][s["id"]]["ev_B"] for m in COHORT]
        cohort_A.append(np.mean(evs_A))
        cohort_B.append(np.mean(evs_B))
    r = float(np.corrcoef(cohort_A, cohort_B)[0, 1])
    delta = float(np.mean(np.array(cohort_A) - np.array(cohort_B)))
    abs_delta = float(np.mean(np.abs(np.array(cohort_A) - np.array(cohort_B))))
    print(f"  Cohort-mean A↔B Pearson r = {r:+.3f}")
    print(f"  Cohort-mean A − B  : {delta:+.3f}")
    print(f"  Cohort-mean |A − B|: {abs_delta:.3f}")

    # Per-model summary
    print(f"\nPer-model A↔B r:")
    for m in COHORT:
        evs_A = [results["ratings"][m][s["id"]]["ev_A"] for s in sample]
        evs_B = [results["ratings"][m][s["id"]]["ev_B"] for s in sample]
        r = float(np.corrcoef(evs_A, evs_B)[0, 1])
        print(f"  {m:<10s} r = {r:+.3f}")


if __name__ == "__main__":
    main()
