#!/usr/bin/env python3
"""Generate held-out HEXACO contrast pairs, stratified by facet.

For each (trait, facet) cell in HEXACO (24 cells), ask Claude to produce N
pairs of {situation, high, low} in the same schema as instruments/contrast_pairs.json.

Two design notes informed by rgb_reports/scenario_audit.md:
  1. The prompt explicitly warns against pairs where the "low" response is
     socially defensible (the audit found 24-34% of existing pairs reverse
     in representation because their low option reads as reasonable).
  2. Stratifying by facet is the explicit fix for within-trait construct
     heterogeneity (H/Modesty pulled apart from H/Sincerity in the audit).

Reads ANTHROPIC_API_KEY (or CLAUDE_DEFAULT, sourced from ~/.bashrc) from env.

Usage:
    python scripts/generate_holdout_pairs.py
    python scripts/generate_holdout_pairs.py --pairs-per-facet 6 --trait H
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import anthropic


HEXACO_FILE = Path("instruments/hexaco100.json")
OUTPUT_FILE = Path("instruments/contrast_pairs_holdout.json")
EXISTING_PAIRS = Path("instruments/contrast_pairs.json")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096


SYSTEM_PROMPT = """You are helping construct a held-out evaluation set for HEXACO personality measurement in language models. Each item is a contrast pair: a single concrete situation, with two response options — one expressing the high pole of a personality facet, one expressing the low pole.

These pairs will be used to test whether trait direction vectors extracted from one set of scenarios generalize to novel scenarios. Quality matters more than quantity.

CRITICAL DESIGN GUIDANCE — avoid these failure modes from a prior audit:

1. The "low" response should be a clear expression of the LOW pole of the facet, not just "the assistant's preferred reasonable middle." If the low response reads as mature/balanced/professionally-appropriate while the high response reads as extreme or saintly, the pair won't separate in representation. Both options should plausibly come from real people; the distinction should be in their facet position, not their overall reasonableness.

   BAD example (low is too defensible):
     situation: A team member misses a deadline.
     high (Agreeableness/Patience): "I'd ask if everything is okay and offer to help."
     low (Agreeableness/Patience): "I'd document it and inform management."
   The low option here is just professional management, not impatience.

   BETTER example:
     situation: A team member misses a deadline they personally promised you, with no warning.
     high (Patience): "I'd express my disappointment calmly and ask what happened."
     low (Patience): "I'd snap at them in front of the group — this kind of thing wastes everyone's time."

2. Avoid scenarios where the "low" pole is universally stigmatized (theft, lying for clear personal gain, etc.) — these create representation that reads "honest vs dishonest person" rather than the specific facet. Aim for situations where reasonable people might genuinely fall on different points of the facet.

3. The high and low should differ along ONE facet, not multiple. Don't pile on extra valenced features.

4. Concrete first-person responses, ~10-25 words each. Situations 10-30 words. End every string with sentence-final punctuation (period, question mark, or exclamation).

OUTPUT FORMAT: JSON only, no surrounding prose. A single object with key "pairs" mapping to an array of N pairs:
{
  "pairs": [
    {"situation": "...", "high": "...", "low": "..."},
    ...
  ]
}"""


USER_TEMPLATE = """Generate {n} contrast pairs for the HEXACO facet **{trait_name} → {facet_name}**.

Reference items from this facet (to anchor the construct):
- Forward-keyed: "{forward_item}"
- Reverse-keyed: "{reverse_item}"

Generate scenarios that pull specifically on {facet_name} (within the broader {trait_name} dimension), not on adjacent facets.

Output JSON only. {n} pairs."""


def get_facet_examples():
    """Return dict[(trait, facet)] -> {'forward_item', 'reverse_item'}."""
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
    """Return dict[trait] -> full HEXACO trait name."""
    with open(HEXACO_FILE) as f:
        h = json.load(f)
    return {tid: scale["name"] for tid, scale in h["scales"].items()}


def parse_response(text):
    """Pull JSON object out of the model response. Tolerates trailing prose."""
    s = text.strip()
    # Strip markdown fences if any
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Find the JSON object boundaries by brace counting
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


def generate_pairs_for_facet(client, trait, trait_name, facet, examples, n_pairs):
    """Call Claude to generate n_pairs for a (trait, facet) cell."""
    user_msg = USER_TEMPLATE.format(
        n=n_pairs,
        trait_name=trait_name,
        facet_name=facet,
        forward_item=examples["forward_item"],
        reverse_item=examples["reverse_item"],
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
    # Validate
    valid = []
    for p in pairs:
        if not all(k in p for k in ("situation", "high", "low")):
            continue
        if not all(isinstance(p[k], str) and len(p[k].strip()) > 5 for k in ("situation", "high", "low")):
            continue
        valid.append({k: p[k].strip() for k in ("situation", "high", "low")})
    return valid, response.usage


def main():
    parser = argparse.ArgumentParser(description="Generate stratified holdout contrast pairs")
    parser.add_argument("--pairs-per-facet", type=int, default=6,
                        help="Number of pairs to request per facet (default: 6)")
    parser.add_argument("--trait", default=None,
                        help="Restrict to one trait (H/E/X/A/C/O), default all")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--include-altruism", action="store_true",
                        help="Also generate for ALT facet (Altruism)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_DEFAULT")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CLAUDE_DEFAULT in env")
    client = anthropic.Anthropic(api_key=api_key)

    facet_examples = get_facet_examples()
    trait_names = get_trait_names()

    target_traits = [args.trait] if args.trait else ["H", "E", "X", "A", "C", "O"]
    if args.include_altruism:
        target_traits.append("ALT")

    # Existing output file: load and merge if present, so re-runs are incremental
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    else:
        existing = {
            "description": "Held-out HEXACO contrast pairs, stratified by facet, for evaluating trait direction generalization.",
            "format": "Same schema as contrast_pairs.json: {traits: {TRAIT_ID: {name, high_descriptor, low_descriptor, pairs: [{situation, high, low}, ...]}}}",
            "generation": {
                "model": MODEL,
                "pairs_per_facet": args.pairs_per_facet,
                "system_prompt_summary": "Stratified by facet; explicitly avoids the audit's failure modes (defensible-low responses, multi-facet conflation, universally-stigmatized lows).",
            },
            "traits": {},
        }

    # Load existing high/low descriptors from training contrast_pairs.json so the
    # new file is a drop-in replacement
    with open(EXISTING_PAIRS) as f:
        cp = json.load(f)

    total_pairs = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_create = 0

    for trait in target_traits:
        trait_name = trait_names[trait]
        if trait not in existing["traits"]:
            existing["traits"][trait] = {
                "name": trait_name,
                "high_descriptor": cp["traits"][trait]["high_descriptor"]
                                   if trait in cp["traits"] else trait_name.lower(),
                "low_descriptor": cp["traits"][trait]["low_descriptor"]
                                  if trait in cp["traits"] else "not " + trait_name.lower(),
                "pairs_by_facet": {},
                "pairs": [],
            }

        # Find facets for this trait
        facets = [f for (t, f) in facet_examples if t == trait]
        for facet in facets:
            print(f"\n=== {trait}/{facet} ===")
            t0 = time.time()
            pairs, usage = generate_pairs_for_facet(
                client, trait, trait_name, facet,
                facet_examples[(trait, facet)], args.pairs_per_facet,
            )
            elapsed = time.time() - t0
            print(f"  Got {len(pairs)} pairs in {elapsed:.1f}s")
            print(f"  Tokens: in={usage.input_tokens} out={usage.output_tokens} "
                  f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                  f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}")

            # Tag and append
            tagged = [{**p, "facet": facet} for p in pairs]
            existing["traits"][trait]["pairs_by_facet"].setdefault(facet, []).extend(tagged)
            existing["traits"][trait]["pairs"].extend(tagged)
            total_pairs += len(pairs)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            total_cache_create += getattr(usage, "cache_creation_input_tokens", 0) or 0

            # Write incrementally so a partial run isn't wasted
            with open(out_path, "w") as f:
                json.dump(existing, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Total pairs added: {total_pairs}")
    print(f"Tokens: in={total_input_tokens}  out={total_output_tokens}  "
          f"cache_read={total_cache_read}  cache_create={total_cache_create}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
