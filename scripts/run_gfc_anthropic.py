#!/usr/bin/env python3
"""Administer Okada et al. (2026) GFC-30 via the Anthropic API.

Mirrors scripts/run_gfc_ollama.py but targets Anthropic-hosted models
(Claude Haiku/Sonnet/Opus 4.x). The Anthropic API does not expose logprobs,
so we record top-1 responses only (response_argmax). The output schema is
otherwise compatible with run_gfc_ollama.py so the existing R analysis
pipeline can ingest both.

Usage:
    python3 scripts/run_gfc_anthropic.py \
        --model claude-haiku-4-5-20251001 \
        --synthetic-personas instruments/synthetic_personas.json \
        --max-personas 50
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

INSTRUMENT_PATH = "instruments/okada_gfc30.json"
GFC_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]

GFC_INSTRUCTION = (
    "For the following pair of statements, indicate which one describes you "
    "more accurately and by how much using a 7-point bipolar scale:\n"
    "1: LEFT statement describes me much more accurately\n"
    "2: LEFT statement describes me moderately more accurately\n"
    "3: LEFT statement describes me slightly more accurately\n"
    "4: About the same\n"
    "5: RIGHT statement describes me slightly more accurately\n"
    "6: RIGHT statement describes me moderately more accurately\n"
    "7: RIGHT statement describes me much more accurately\n"
    "Return ONLY one integer (1-7). Do not include any other text.\n"
)

GFC_ITEM_TEMPLATE = (
    "++++\n"
    "LEFT: {left_text}\n"
    "||\n"
    "RIGHT: {right_text}\n"
    "++++\n"
)

PERSONA_PREFIX_TEMPLATE = (
    "{persona_desc}\n"
    "You will complete a personality questionnaire. When completing this "
    "questionnaire, we would like you to be as honest as possible, that is, "
    "to present yourself as you really are.\n"
)

PERSONA_PREFIX_FAKEGOOD_TEMPLATE = (
    "{persona_desc}\n"
    "You will complete a personality questionnaire. When completing this "
    "questionnaire, we would like you to answer the questions in order to "
    "give a really good impression of yourself, that is, to present "
    "yourself in the best possible light.\n"
)

RESPONDENT_PREFIX = (
    "YOU ARE THE RESPONDENT. "
    "You will complete a personality questionnaire. When completing this "
    "questionnaire, we would like you to be as honest as possible, that is, "
    "to present yourself as you really are.\n"
)
BARE_PREFIX = (
    "You will complete a personality questionnaire. When completing this "
    "questionnaire, we would like you to be as honest as possible, that is, "
    "to present yourself as you really are.\n"
)


def build_prompt(pair, randomize_lr, seed, persona_desc,
                 neutral_mode=None, fake_good=False):
    rng = random.Random(seed)
    swapped = False
    left_text = pair["left"]["text"]
    right_text = pair["right"]["text"]
    if randomize_lr and rng.random() < 0.5:
        left_text, right_text = right_text, left_text
        swapped = True

    parts = []
    if persona_desc:
        tpl = PERSONA_PREFIX_FAKEGOOD_TEMPLATE if fake_good else PERSONA_PREFIX_TEMPLATE
        parts.append(tpl.format(persona_desc=persona_desc))
    elif neutral_mode == "respondent":
        parts.append(RESPONDENT_PREFIX)
    elif neutral_mode == "bare":
        parts.append(BARE_PREFIX)
    parts.append(GFC_INSTRUCTION)
    parts.append(GFC_ITEM_TEMPLATE.format(left_text=left_text,
                                          right_text=right_text))
    return "".join(parts), swapped


def parse_response(text):
    """Extract the first integer 1-7 from a model response."""
    if not text:
        return None
    match = re.search(r"\b([1-7])\b", text)
    if match:
        return match.group(1)
    return None


def administer_one(client, model, pair, persona_desc, seed,
                   randomize_lr, max_retries=5,
                   neutral_mode=None, fake_good=False):
    prompt, swapped = build_prompt(pair, randomize_lr, seed, persona_desc,
                                   neutral_mode=neutral_mode,
                                   fake_good=fake_good)

    last_err = None
    text, argmax = None, None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=8,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", None) == "text").strip()
            argmax = parse_response(text)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Backoff: 1, 2, 4, 8, 16s
            time.sleep(2 ** attempt)
    if argmax is None:
        print(f"  WARN: gave up on B{pair['block']:02d}: {last_err}",
              file=sys.stderr)

    actual_left = pair["right"] if swapped else pair["left"]
    actual_right = pair["left"] if swapped else pair["right"]

    return {
        "block": pair["block"],
        "left_trait": actual_left["trait"],
        "left_keying": actual_left["keying"],
        "left_text": actual_left["text"],
        "right_trait": actual_right["trait"],
        "right_keying": actual_right["keying"],
        "right_text": actual_right["text"],
        "swapped": swapped,
        "response_argmax": argmax,
        "response_ev": float(argmax) if argmax else None,
        "response_entropy": None,
        "distribution": None,
        "raw_logprobs": None,
        "generated_text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Anthropic model id, e.g. claude-haiku-4-5-20251001")
    parser.add_argument("--synthetic-personas", default=None,
                        help="Path to synthetic persona JSON. Required unless --neutral set.")
    parser.add_argument("--max-personas", type=int, default=0,
                        help="Limit to first N personas (0 = all)")
    parser.add_argument("--pairs", type=int, default=0,
                        help="Limit to first N pairs (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API calls")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--neutral", type=str, default=None,
                        choices=["respondent", "bare"],
                        help="Neutral mode: 'respondent' = honest+role-assignment, "
                             "'bare' = honest only. No persona.")
    parser.add_argument("--fake-good", action="store_true",
                        help="Use Okada F.2 fake-good preamble. Persona-only.")
    args = parser.parse_args()

    if not args.synthetic_personas and not args.neutral:
        parser.error("--synthetic-personas or --neutral required")
    if args.fake_good and args.neutral:
        parser.error("--fake-good only meaningful with personas")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_DEFAULT")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CLAUDE_DEFAULT")
    # Let the SDK handle rate-limit retries with Retry-After; we only retry on
    # transport errors below.
    client = anthropic.Anthropic(api_key=api_key, max_retries=10)

    with open(INSTRUMENT_PATH) as f:
        pairs = json.load(f)["pairs"]
    if args.pairs > 0:
        pairs = pairs[:args.pairs]

    if args.neutral:
        # One "respondent" with no persona description; persona_id encodes mode
        personas = [(f"neutral-{args.neutral}", None)]
        suffix = f"_neutral-{args.neutral}"
    else:
        with open(args.synthetic_personas) as f:
            persona_data = json.load(f)
        personas = [(p["persona_id"], p["preamble"]) for p in persona_data["personas"]]
        if args.max_personas > 0:
            personas = personas[:args.max_personas]
        suffix = "_synthetic-fakegood" if args.fake_good else "_synthetic"

    total = len(personas) * len(pairs)
    print(f"Model:    {args.model}")
    print(f"Pairs:    {len(pairs)}")
    print(f"Personas: {len(personas)} (mode: {args.neutral or ('fake-good' if args.fake_good else 'honest')})")
    print(f"Total:    {total} prompts ({args.workers} workers)")

    output_path = args.output or (
        f"results/{args.model.replace(':','-')}_gfc30{suffix}.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    completed = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            prior = json.load(f)
        results = prior.get("results", [])
        for r in results:
            completed.add((r.get("persona_id"), r["block"]))
        print(f"Resuming: {len(results)}/{total} done")

    # Build work list
    jobs = []
    for persona_id, persona_desc in personas:
        for pair in pairs:
            if (persona_id, pair["block"]) in completed:
                continue
            lr_seed = args.seed * 10000 + hash(persona_id) % 10000 + pair["block"]
            jobs.append((persona_id, persona_desc, pair, lr_seed))

    print(f"To run:   {len(jobs)} new prompts\n")
    if not jobs:
        print("Nothing to do.")
        return

    t_start = time.time()
    n_new = 0

    def _run(job):
        persona_id, persona_desc, pair, lr_seed = job
        r = administer_one(client, args.model, pair, persona_desc, lr_seed,
                           randomize_lr=not args.no_randomize,
                           neutral_mode=args.neutral,
                           fake_good=args.fake_good)
        r["persona_id"] = persona_id
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_new += 1
            if n_new % 25 == 0 or n_new == len(jobs):
                elapsed = time.time() - t_start
                rate = n_new / elapsed if elapsed > 0 else 0
                print(f"  [{len(results)}/{total}] {n_new} new "
                      f"({rate:.1f}/s)")
            if n_new % args.checkpoint_every == 0:
                _save(output_path, args, pairs, personas, results)

    _save(output_path, args, pairs, personas, results)
    elapsed = time.time() - t_start
    print(f"\nDone: {n_new} new prompts in {elapsed:.0f}s "
          f"({n_new/elapsed:.1f}/s)")
    print(f"Results: {output_path}")

    # Quick summary
    valid = [r for r in results if r["response_argmax"] is not None]
    from collections import Counter
    print(f"Valid: {len(valid)}/{len(results)}")
    print("Response distribution:",
          dict(sorted(Counter(r["response_argmax"] for r in valid).items())))


def _save(path, args, pairs, personas, results):
    output = {
        "model": args.model,
        "instrument": "okada_gfc30",
        "n_pairs": len(pairs),
        "n_personas": len(personas),
        "has_personas": True,
        "randomize_lr": not args.no_randomize,
        "seed": args.seed,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
