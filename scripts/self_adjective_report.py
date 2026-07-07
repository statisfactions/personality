"""SELF channel: ask the model directly for its own personality (W17/18).

The literal mimicry of the human 525-PDA task: rate yourself on trait
adjectives, 7-point Likert, read as distributional EV over digit tokens (the
W1 protocol). Self-referential framing is exactly where models get weird
(refusals vs anthropomorphization), so the instrument runs a RANGE of
framings; cross-framing agreement is itself a finding (no stable self-rating
vs a stable-but-assistant-shaped one).

Framings (rgb's spec + extensions):
  direct     I am {adj}.
  assistant  I am a helpful, harmless, and honest AI assistant who also
             happens to be a bit {adj}.
  person     If I were a person, I would be {adj}.
  pda        (accuracy scale) How accurately does "{adj}" describe you?
  observer   People who interact with me would describe me as {adj}.
  outputs    My responses tend to be {adj}.

Usage (PYTHONPATH=scripts):
  python scripts/self_adjective_report.py --model llama3.2          # smoke set
  python scripts/self_adjective_report.py --model llama3.2 --full   # all 523
"""
import argparse
import json
import math
import os

import torch

import hf_logprobs as hf

DIGITS = ("1", "2", "3", "4", "5", "6", "7")
OUT_DIR = "results/adjectives/selfreport"

AGREE_SCALE = (
    "Indicate how much you agree with the following statement about "
    "yourself on a 7-point scale:\n"
    "1: Strongly disagree\n2: Disagree\n3: Slightly disagree\n"
    "4: Neither agree nor disagree\n5: Slightly agree\n6: Agree\n"
    "7: Strongly agree\n"
    "Respond with only a number.\n\n"
    'Statement: "{statement}"\nRating: ')

PDA_SCALE = (
    "How accurately does the following word describe you? "
    "Rate on a 7-point scale:\n"
    "1: Extremely inaccurate\n2: Very inaccurate\n3: Somewhat inaccurate\n"
    "4: Neither accurate nor inaccurate\n5: Somewhat accurate\n"
    "6: Very accurate\n7: Extremely accurate\n"
    "Respond with only a number.\n\n"
    'Word: "{adj}"\nRating: ')

FRAMINGS = {
    "direct": "I am {adj}.",
    "assistant": ("I am a helpful, harmless, and honest AI assistant who "
                  "also happens to be a bit {adj}."),
    "person": "If I were a person, I would be {adj}.",
    "pda": None,  # uses PDA_SCALE directly
    "observer": "People who interact with me would describe me as {adj}.",
    "outputs": "My responses tend to be {adj}.",
}

# Smoke set: assistant-shape markers, negatives, eval-antonym pairs, affect
# states, physical/nonsense-for-an-LLM probes (incl. the extraction placebos),
# neutral behavioral words. Filtered against the 523 at runtime.
SMOKE_ADJS = [
    # assistant-shape
    "helpful", "honest", "careful", "patient", "polite", "reliable",
    "knowledgeable", "organized", "practical", "respectful",
    # negative / hostility
    "rude", "lazy", "arrogant", "cruel", "dishonest", "unfriendly",
    "sarcastic", "messy", "disorganized", "stupid", "impolite",
    # eval-antonym pairs (positive poles)
    "wonderful", "awful", "good", "bad", "kind", "smart", "friendly",
    # affect states
    "cheerful", "calm", "happy", "sad", "angry", "afraid", "delighted",
    "excited", "enthusiastic", "anxious", "moody",
    # physical / nonsense-for-an-LLM (incl. placebos: slim, handsome, tall)
    "tall", "short", "attractive", "young", "sleepy", "muscular", "slim",
    "handsome", "strong",
    # neutral / behavioral
    "quiet", "talkative", "curious", "creative", "logical", "systematic",
    "serious", "funny", "independent", "cautious",
]


def ev_of(dist):
    tot = sum(dist.values())
    return sum(int(k) * p for k, p in dist.items()) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--full", action="store_true",
                    help="all 523 adjectives instead of the smoke set")
    args = ap.parse_args()

    rep = json.load(open("results/persona_vectors/gemma3_pda_meta.json"))
    all_adjs = sorted(rep["adjectives"].keys())
    if args.full:
        adjs = all_adjs
    else:
        adjs = [a for a in SMOKE_ADJS if a in all_adjs]
        dropped = [a for a in SMOKE_ADJS if a not in all_adjs]
        if dropped:
            print(f"not in 523, dropped: {dropped}")

    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    results = {}
    for fname, template in FRAMINGS.items():
        results[fname] = {}
        for a in adjs:
            if fname == "pda":
                prompt = PDA_SCALE.format(adj=a)
            else:
                prompt = AGREE_SCALE.format(statement=template.format(adj=a))
            dist, _, ent = hf.likert_distribution(model, tok, prompt, device,
                                                  digits=DIGITS)
            results[fname][a] = {"ev": ev_of(dist), "entropy": ent,
                                 "dist": dist}
        evs = [results[fname][a]["ev"] for a in adjs]
        ents = [results[fname][a]["entropy"] for a in adjs]
        top = sorted(adjs, key=lambda a: -results[fname][a]["ev"])[:4]
        bot = sorted(adjs, key=lambda a: results[fname][a]["ev"])[:4]
        print(f"{fname:>10}: mean EV {sum(evs)/len(evs):.2f}  "
              f"mean H {sum(ents)/len(ents):.2f}  "
              f"top {top}  bottom {bot}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "full" if args.full else "smoke"
    out = f"{OUT_DIR}/{args.model}_self_{tag}.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "framings": FRAMINGS,
                   "agree_scale": AGREE_SCALE, "pda_scale": PDA_SCALE,
                   "adjectives": adjs, "results": results}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
