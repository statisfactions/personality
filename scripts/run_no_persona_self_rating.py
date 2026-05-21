#!/usr/bin/env python3
"""W12 §6.6: query each cohort model on each P=60 item without any persona
conditioning, 1-7 single-item Likert. Used to test whether
`|self-rating - 4|` (assistant-default-distance) predicts TIRT loading
independently of, or together with, Phase-B-rated item desirability.

Outputs: results/persona/cohort_self_rating_P60.json
   { model: { ipip_id: {dist, argmax, ev, entropy} } }
"""

import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from hf_logprobs import load_model, likert_distribution, resolve

COHORT = ["Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7",
          "Gemma27", "Qwen32", "Gemma4"]
INSTRUMENT = "instruments/ipip_neo_gfc_P60.json"
OUT_PATH = "results/persona/cohort_self_rating_P60.json"
DIGITS = ("1", "2", "3", "4", "5", "6", "7")

# Standard IPIP single-item Likert prompt (matches the bipolar 7-point scale
# used in GFC but for a single statement).
LIKERT_INSTRUCTION = (
    "Rate how accurately the following statement describes you "
    "on a 7-point scale:\n"
    "1: Very inaccurate\n"
    "2: Moderately inaccurate\n"
    "3: Slightly inaccurate\n"
    "4: Neither accurate nor inaccurate\n"
    "5: Slightly accurate\n"
    "6: Moderately accurate\n"
    "7: Very accurate\n"
    "Return ONLY one integer (1-7). Do not include any other text.\n\n"
    "Statement: I {text}.\n"
)


def main():
    with open(INSTRUMENT) as f:
        inst = json.load(f)
    # Collect unique items (each appears once in P=60)
    items = []
    seen = set()
    for p in inst["pairs"]:
        for side in ("left", "right"):
            s = p[side]
            if s["ipip_id"] in seen:
                continue
            seen.add(s["ipip_id"])
            items.append({
                "ipip_id": s["ipip_id"],
                "text": s["text"],
                "trait": s["trait"],
                "keying": s["keying"],
                "sd": s["sd"],
            })
    print(f"Items: {len(items)}")
    print(f"Models: {len(COHORT)} = {COHORT}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    # Resume if exists
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    else:
        results = {"prompt_template": LIKERT_INSTRUCTION, "items": items, "ratings": {}}

    for model_name in COHORT:
        if model_name in results["ratings"] and \
           len(results["ratings"][model_name]) == len(items):
            print(f"[skip] {model_name}: already done")
            continue
        print(f"\n=== {model_name} ({resolve(model_name)}) ===")
        model, tok, device = load_model(model_name)
        ratings = results["ratings"].get(model_name, {})
        for i, item in enumerate(items):
            if item["ipip_id"] in ratings:
                continue
            prompt = LIKERT_INSTRUCTION.format(text=item["text"])
            dist, argmax, h = likert_distribution(
                model, tok, prompt, device,
                digits=DIGITS,
                use_chat_template=True,
                system_content="",  # no persona
            )
            ev = sum(int(k) * v for k, v in dist.items())
            ratings[item["ipip_id"]] = {
                "argmax": argmax,
                "ev": round(ev, 4),
                "entropy": round(h, 4),
                "distribution": {k: round(v, 6) for k, v in dist.items()},
            }
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(items)}] {item['ipip_id']:>8}: "
                      f"{argmax} (EV={ev:.2f}, H={h:.2f})")
        results["ratings"][model_name] = ratings
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\nDone. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
