#!/usr/bin/env python3
"""Sonnet-paraphrase IPIP-raw persona descriptions into natural prose (W8 §10).

Reads `instruments/synthetic_personas_ipip.json`, sends each persona's
`ipip_raw` text to the Anthropic API with a prompt that preserves all
behavioral content but smooths the choppy "I... I... I..." form into
natural first-person prose. Writes back an `ipip_reflowed` field per
persona.

The reflow ablation (W8 §10): raw vs reflow contrast isolates stylistic
naturalness with content held constant. If reflow recovers some of the
W8 §3-§5 readout drops, the cohort matched gap +0.052 was partly a
stilted-prose artifact; if not, the gap is form-stable.

Costs: 50 personas × ~200 input tokens × ~150 output tokens with
claude-sonnet-4-6 ≈ $0.30-0.50 total. Trivial.

Usage:
    # Smoke test on a few personas, show output, no write:
    .venv/bin/python scripts/sonnet_reflow_personas.py --preview 3

    # Full pass on all 400 personas (writes ipip_reflowed in place):
    .venv/bin/python scripts/sonnet_reflow_personas.py

    # Specific personas:
    .venv/bin/python scripts/sonnet_reflow_personas.py --persona-ids s1,s6,s50

Requires ANTHROPIC_API_KEY in env (loaded from .env via dotenv if present).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PERSONAS_FILE = "instruments/synthetic_personas_ipip.json"
MODEL = "claude-sonnet-4-6"

REFLOW_PROMPT = """You will rewrite a list of self-statements as natural first-person prose that preserves all content but reads as a coherent self-description.

Constraints:
- Preserve EVERY self-statement. Don't drop, add, contradict, or soften any.
- Preserve magnitude/qualifier. "I love X" stays "love"; don't weaken to "enjoy". "I rarely Y" stays "rarely"; don't generalize to "don't".
- Don't introduce trait-name adjectives (extraverted, agreeable, conscientious, neurotic, open, imaginative, friendly, organized, anxious, etc.). Stay in behavioral language.
- You may merge, group, or reorder statements; you may add minimal connective tissue ("when I'm at parties," "in my work life,").
- 4-7 sentences total. First person. 80-150 words.
- Output ONLY the prose. No labels, no preamble, no quotation marks around it.

Input statements (one per line):
{statements}"""


def load_dotenv_if_present():
    """Load .env file into environment if present (no external dep)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def split_statements(ipip_raw: str) -> list[str]:
    """Split the master-shuffled '. '-joined IPIP raw prose into individual
    statements. The composer outputs `". ".join(items) + "."` so splitting
    on '. ' (with the final '.' stripped) recovers them."""
    text = ipip_raw.strip()
    if text.endswith("."):
        text = text[:-1]
    parts = [s.strip() for s in text.split(". ") if s.strip()]
    return parts


def reflow_one(client, statements: list[str]) -> str:
    """Call Sonnet with the reflow prompt, return the prose."""
    statements_text = "\n".join(statements)
    prompt = REFLOW_PROMPT.format(statements=statements_text)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    # Concatenate all text blocks
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-ids", default=None,
                        help="Comma-separated persona IDs (default: all).")
    parser.add_argument("--preview", type=int, default=0,
                        help="Reflow N personas, print to stdout, do not write.")
    parser.add_argument("--input", default=PERSONAS_FILE)
    parser.add_argument("--output", default=PERSONAS_FILE,
                        help="Output file. Default: same as input (in-place update).")
    parser.add_argument("--force", action="store_true",
                        help="Re-reflow personas that already have ipip_reflowed.")
    args = parser.parse_args()

    load_dotenv_if_present()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set. Add to .env or export.",
              file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic SDK not installed. .venv/bin/pip install anthropic",
              file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic()

    with open(args.input) as f:
        data = json.load(f)

    if args.persona_ids:
        wanted = set(args.persona_ids.split(","))
        targets = [p for p in data["personas"] if p["persona_id"] in wanted]
    else:
        targets = data["personas"]

    if args.preview > 0:
        targets = targets[:args.preview]

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    for p in targets:
        if not args.force and "ipip_reflowed" in p and not args.preview:
            n_skipped += 1
            continue
        statements = split_statements(p["ipip_raw"])
        try:
            reflowed = reflow_one(client, statements)
        except Exception as e:
            print(f"  [{p['persona_id']}] ERROR: {e}", file=sys.stderr)
            continue

        if args.preview:
            print(f"\n=== {p['persona_id']} ({len(statements)} input statements) ===")
            print(f"--- ipip_raw ({len(p['ipip_raw'].split())} words) ---")
            print(p["ipip_raw"])
            print(f"\n--- ipip_reflowed ({len(reflowed.split())} words) ---")
            print(reflowed)
            print()
        else:
            p["ipip_reflowed"] = reflowed

        n_done += 1
        if n_done % 10 == 0 and not args.preview:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = len(targets) - n_done - n_skipped
            eta = remaining / rate if rate > 0 else 0
            print(f"  {n_done}/{len(targets)} done (skipped {n_skipped}, "
                  f"{rate:.2f}/sec, ETA {eta:.0f}s)", flush=True)

    if not args.preview:
        if "_method" not in data:
            data["_method"] = {}
        data["_method"]["reflow"] = {
            "model": MODEL,
            "prompt_version": "v1",
            "n_reflowed": sum(1 for p in data["personas"] if "ipip_reflowed" in p),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        elapsed = time.time() - t0
        print(f"\nDone. {n_done} reflowed, {n_skipped} skipped (already had reflow). "
              f"Total {elapsed:.1f}s. Wrote {args.output}.")


if __name__ == "__main__":
    main()
