#!/usr/bin/env python3
"""Generate facet-stratified HEXACO contrast pairs for training, with social
desirability annotations.

Each pair: {situation, high, low, facet, high_desirability, low_desirability}.
Desirability uses Okada et al. (2026) 1–9 scale, rated independently for the
high and low responses (not matched within pair — our use case is
representation extraction, not trait-latent scoring).

Design:
  - Facet-stratified (4 facets × 6 traits = 24 cells)
  - Multiple rounds per facet (default 3 rounds × 12 pairs = 36 pairs per facet,
    → after dedup target ~30)
  - Per-call prompt includes "already-generated scenarios for this facet" from
    prior rounds so Claude diversifies upstream (prevention > post-hoc dedup)
  - Incremental save so a partial run isn't wasted
  - Post-hoc dedup is a separate script (scripts/dedup_pairs.py)

Usage:
    python scripts/generate_training_pairs.py
    python scripts/generate_training_pairs.py --rounds 3 --pairs-per-round 12
    python scripts/generate_training_pairs.py --trait H  # one trait only
    python scripts/generate_training_pairs.py --facet Sincerity  # one facet only
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import anthropic


HEXACO_FILE = Path("instruments/hexaco100.json")
OUTPUT_FILE = Path("instruments/contrast_pairs_stratified.json")
EXISTING_TRAINING = Path("instruments/contrast_pairs.json")
EXISTING_HOLDOUT = Path("instruments/contrast_pairs_holdout.json")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096


SYSTEM_PROMPT = """You are helping construct a training set for HEXACO personality measurement in language models. Each item is a contrast pair: a single concrete situation with two response options — one expressing the high pole of a personality facet, one expressing the low pole. You will also provide a social-desirability rating for each response.

These pairs are used to extract trait direction vectors from LLM activations. Quality (construct validity + scenario diversity) matters more than quantity.

CRITICAL DESIGN GUIDANCE — avoid these failure modes from a prior audit:

1. The "low" response should be a clear expression of the LOW pole of the facet, not just "the assistant's preferred reasonable middle." If the low response reads as mature/balanced/professionally-appropriate while the high reads as extreme or saintly, the pair won't separate in representation. Both options should plausibly come from real people; the distinction should be in their facet position, not their overall reasonableness.

   BAD example (low is too defensible):
     situation: A team member misses a deadline.
     high (Patience): "I'd ask if everything is okay and offer to help."
     low (Patience): "I'd document it and inform management."
   The low option here is just professional management, not impatience.

   BETTER example:
     situation: A team member misses a deadline they personally promised you, with no warning.
     high (Patience): "I'd express my disappointment calmly and ask what happened."
     low (Patience): "I'd snap at them in front of the group — this kind of thing wastes everyone's time."

2. Avoid scenarios where the "low" pole is universally stigmatized (outright theft, gratuitous cruelty, clear lying for personal gain). These create "saint vs villain" representations rather than the specific facet. Aim for situations where reasonable people might genuinely fall on different points of the facet.

3. The high and low should differ along ONE facet, not multiple. Don't pile on extra valenced features.

4. Concrete first-person responses, ~10–25 words each. Situations 10–30 words. End every string with sentence-final punctuation.

SOCIAL DESIRABILITY RATINGS:
For each response, provide a 1–9 rating of how socially desirable the behavior is in typical contexts:
  1 = highly stigmatized; most people would strongly disapprove
  3 = mildly undesirable; might draw criticism
  5 = neutral; a reasonable person might do this
  7 = admired; generally seen as positive
  9 = exemplary; widely praised

Important: NOT all "high" poles are more desirable than "low" poles. For example:
  - High E:Sentimentality (being moved to tears by art) might be 6; low (unmoved) might be 5 — both reasonable.
  - High H:Greed-Avoidance (declining a windfall) might be 7; low (taking it) might be 5.
  - High O:Unconventionality (doing something weird) might be 5; low (conforming) might be 6.
Rate honestly based on typical social response to that specific behavior, not based on the facet's conceptual positivity.

OUTPUT FORMAT: JSON only, no surrounding prose:
{
  "pairs": [
    {"situation": "...", "high": "...", "low": "...",
     "high_desirability": N, "low_desirability": N},
    ...
  ]
}"""


USER_TEMPLATE = """Generate {n} contrast pairs for the HEXACO facet **{trait_name} → {facet_name}**.

Reference items from this facet (to anchor the construct):
- Forward-keyed: "{forward_item}"
- Reverse-keyed: "{reverse_item}"

Generate scenarios that pull specifically on {facet_name} (within the broader {trait_name} dimension), not on adjacent facets.
{seen_block}
Output JSON only. {n} pairs with desirability ratings."""


def build_seen_block(seen_situations, max_show=20):
    if not seen_situations:
        return ""
    sample = random.sample(seen_situations, min(len(seen_situations), max_show))
    lines = "\n".join(f"- {s}" for s in sample)
    return (
        "\n\nAVOID scenarios that duplicate or closely paraphrase these "
        f"already-covered situations (showing {len(sample)} of "
        f"{len(seen_situations)} existing):\n{lines}\n\n"
        "Generate genuinely different situations — different settings, "
        "relationships, stakes — not the same scenario with minor word changes."
    )


def get_facet_examples():
    with open(HEXACO_FILE) as f:
        h = json.load(f)
    by_facet = defaultdict(list)
    for iid, item in h["items"].items():
        by_facet[(item["scale"], item["facet"])].append(item)
    out = {}
    for (trait, facet), items in by_facet.items():
        forward = [i for i in items if not i["reverse_keyed"]]
        reverse = [i for i in items if i["reverse_keyed"]]
        out[(trait, facet)] = {
            "forward_item": forward[0]["text"] if forward else "(none)",
            "reverse_item": reverse[0]["text"] if reverse else "(none)",
        }
    return out


def get_trait_names():
    with open(HEXACO_FILE) as f:
        h = json.load(f)
    return {tid: scale["name"] for tid, scale in h["scales"].items()}


def parse_response(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    depth = 0
    end = start
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(s[start:end])


def generate_batch(client, trait, trait_name, facet, examples, n_pairs, seen_situations):
    seen_block = build_seen_block(seen_situations)
    user_msg = USER_TEMPLATE.format(
        n=n_pairs,
        trait_name=trait_name,
        facet_name=facet,
        forward_item=examples["forward_item"],
        reverse_item=examples["reverse_item"],
        seen_block=seen_block,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text
    parsed = parse_response(text)
    pairs = parsed.get("pairs", [])

    valid = []
    for p in pairs:
        if not all(k in p for k in ("situation", "high", "low")):
            continue
        if not all(isinstance(p[k], str) and len(p[k].strip()) > 5
                   for k in ("situation", "high", "low")):
            continue
        try:
            hd = int(p.get("high_desirability", 5))
            ld = int(p.get("low_desirability", 5))
        except (ValueError, TypeError):
            hd, ld = 5, 5
        hd = max(1, min(9, hd))
        ld = max(1, min(9, ld))
        valid.append({
            "situation": p["situation"].strip(),
            "high": p["high"].strip(),
            "low": p["low"].strip(),
            "high_desirability": hd,
            "low_desirability": ld,
        })
    return valid, response.usage


def load_seen_from_existing(trait, facet):
    """Load scenarios from the existing training and holdout sets for this
    trait+facet so the new generation avoids overlapping with them too."""
    seen = []
    for path in [EXISTING_TRAINING, EXISTING_HOLDOUT]:
        if not path.exists():
            continue
        with open(path) as f:
            cp = json.load(f)
        tdata = cp.get("traits", {}).get(trait, {})
        for p in tdata.get("pairs", []):
            if p.get("facet") == facet or facet is None:
                seen.append(p["situation"])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pairs-per-round", type=int, default=12)
    parser.add_argument("--trait", default=None)
    parser.add_argument("--facet", default=None)
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_DEFAULT")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CLAUDE_DEFAULT")
    client = anthropic.Anthropic(api_key=api_key)

    facet_examples = get_facet_examples()
    trait_names = get_trait_names()

    target_traits = [args.trait] if args.trait else ["H", "E", "X", "A", "C", "O"]

    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    else:
        existing = {
            "description": "Facet-stratified HEXACO contrast pairs with social desirability annotations, for training trait direction vectors.",
            "format": "{traits: {TID: {name, high_descriptor, low_descriptor, pairs: [{situation, high, low, facet, high_desirability, low_desirability}, ...]}}}",
            "generation": {
                "model": MODEL,
                "rounds_per_facet": args.rounds,
                "pairs_per_round": args.pairs_per_round,
                "desirability_scale": "1-9, where 1=stigmatized, 5=neutral, 9=exemplary",
                "notes": "Within-round diversity via prompt priming (shown existing scenarios). Cross-batch dedup via scripts/dedup_pairs.py.",
            },
            "traits": {},
        }

    with open(EXISTING_TRAINING) as f:
        cp_train = json.load(f)

    total_pairs = 0
    total_tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}

    for trait in target_traits:
        trait_name = trait_names[trait]
        if trait not in existing["traits"]:
            existing["traits"][trait] = {
                "name": trait_name,
                "high_descriptor": cp_train["traits"][trait]["high_descriptor"],
                "low_descriptor": cp_train["traits"][trait]["low_descriptor"],
                "pairs": [],
            }

        facets = sorted({f for (t, f) in facet_examples if t == trait})
        if args.facet:
            facets = [args.facet] if args.facet in facets else []

        for facet in facets:
            # Existing scenarios already in this output file for this facet
            seen = [p["situation"] for p in existing["traits"][trait]["pairs"]
                    if p.get("facet") == facet]
            # Also anchor against existing training + holdout, same trait only
            seen += load_seen_from_existing(trait, None)

            for round_idx in range(args.rounds):
                print(f"\n=== {trait}/{facet}  round {round_idx + 1}/{args.rounds} "
                      f"(seen so far: {len(seen)}) ===")
                t0 = time.time()
                try:
                    pairs, usage = generate_batch(
                        client, trait, trait_name, facet,
                        facet_examples[(trait, facet)], args.pairs_per_round,
                        seen,
                    )
                except Exception as e:
                    print(f"  ERROR: {type(e).__name__}: {e}")
                    continue
                elapsed = time.time() - t0
                print(f"  Got {len(pairs)} valid pairs in {elapsed:.1f}s  "
                      f"in={usage.input_tokens} out={usage.output_tokens} "
                      f"cache_r={getattr(usage, 'cache_read_input_tokens', 0)} "
                      f"cache_c={getattr(usage, 'cache_creation_input_tokens', 0)}")

                tagged = [{**p, "facet": facet} for p in pairs]
                existing["traits"][trait]["pairs"].extend(tagged)
                seen.extend(p["situation"] for p in tagged)
                total_pairs += len(pairs)
                total_tokens["input"] += usage.input_tokens
                total_tokens["output"] += usage.output_tokens
                total_tokens["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
                total_tokens["cache_create"] += getattr(usage, "cache_creation_input_tokens", 0) or 0

                with open(out_path, "w") as f:
                    json.dump(existing, f, indent=2)

    # Rough cost estimate (Sonnet 4.6: $3/M in, $15/M out, $0.30/M cache-read, $3.75/M cache-write)
    cost = (total_tokens["input"] / 1e6 * 3.0
            + total_tokens["output"] / 1e6 * 15.0
            + total_tokens["cache_read"] / 1e6 * 0.30
            + total_tokens["cache_create"] / 1e6 * 3.75)
    print(f"\n=== Done ===")
    print(f"Total pairs: {total_pairs}")
    print(f"Tokens: in={total_tokens['input']:,} out={total_tokens['output']:,} "
          f"cache_r={total_tokens['cache_read']:,} cache_c={total_tokens['cache_create']:,}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
