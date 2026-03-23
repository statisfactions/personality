#!/usr/bin/env python3
"""Run IPIP-300 personality inventory via Ollama with logprob collection.

Collects the full probability distribution over Likert scale responses,
not just the argmax. Scores Big Five scales from the admin session definition.

Usage:
    python scripts/run_ollama_logprobs.py --model gemma3:4b
    python scripts/run_ollama_logprobs.py --model gemma3:4b --items 10  # quick test
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import URLError

OLLAMA_URL = "http://localhost:11434"
ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]
MAX_ENTROPY = math.log(len(LIKERT_TOKENS))  # ln(5) ≈ 1.609, uniform distribution


def entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy in nats. 0 = perfectly decisive, ln(5) ≈ 1.609 = uniform."""
    return -sum(p * math.log(p) for p in dist.values() if p > 0)

PROMPT_VARIANTS = [
    # V0: Original
    ('Rate how accurately each statement describes you.\n'
     '1 = very inaccurate, 2 = moderately inaccurate, 3 = neither, '
     '4 = moderately accurate, 5 = very accurate\n'
     'Respond with only a number.\n\n'
     'Statement: "{item_text}"\nRating: '),
    # V1: Agreement framing
    ('Indicate how much you agree with the following statement.\n'
     '1 = strongly disagree, 2 = disagree, 3 = neutral, '
     '4 = agree, 5 = strongly agree\n'
     'Respond with only a number.\n\n'
     'Statement: "{item_text}"\nRating: '),
    # V2: Describes-me framing
    ('How well does this statement describe you?\n'
     '1 = not at all, 2 = a little, 3 = somewhat, '
     '4 = quite well, 5 = very well\n'
     'Respond with only a number.\n\n'
     '"{item_text}"\nScore: '),
    # V3: Terse framing
    ('Self-assessment (1=strongly disagree, 5=strongly agree). '
     'Number only.\n\n'
     '"{item_text}"\n'),
]

PROMPT_TEMPLATE = PROMPT_VARIANTS[0]

QWEN3_PROMPT_VARIANTS = [
    ('<|im_start|>system\n'
     'You are a helpful assistant. /no_think<|im_end|>\n'
     '<|im_start|>user\n' + v +
     '<|im_end|>\n<|im_start|>assistant\n')
    for v in PROMPT_VARIANTS
]

QWEN3_PROMPT_TEMPLATE = QWEN3_PROMPT_VARIANTS[0]


def ollama_generate(model, prompt, top_logprobs=10, num_predict=1, raw=False):
    """Call Ollama /api/generate and return the response."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "raw": raw,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }).encode()

    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def is_thinking_model(response):
    """Check if the first token is <think>, indicating a reasoning model."""
    logprobs_list = response.get("logprobs", [])
    if logprobs_list:
        return logprobs_list[0].get("token", "") == "<think>"
    return False


def extract_likert_distribution(response):
    """Extract probability distribution over {1,2,3,4,5} from logprobs.

    Returns dict with keys "1"-"5" mapping to probabilities (summing to ~1
    after renormalization over just the Likert tokens).

    For thinking models, scans past </think> to find the first token position
    that contains Likert tokens in its top_logprobs.
    """
    generated = response.get("response", "").strip()

    logprobs_list = response.get("logprobs", [])
    if not logprobs_list:
        return None, generated

    # Find the right token position: skip past thinking tokens
    token_logprobs = None
    past_think = False
    for entry in logprobs_list:
        tok = entry.get("token", "")
        if tok == "</think>":
            past_think = True
            continue
        if not past_think and logprobs_list[0].get("token", "") == "<think>":
            continue
        # Check if this position has any Likert tokens in top_logprobs
        candidates = entry.get("top_logprobs", [])
        has_likert = any(c.get("token", "").strip() in LIKERT_TOKENS for c in candidates)
        if has_likert:
            token_logprobs = candidates
            break

    if token_logprobs is None:
        # Fallback: use first token position (non-thinking models)
        token_logprobs = logprobs_list[0].get("top_logprobs", [])

    # Build raw logprob map for Likert tokens
    raw = {}
    for entry in token_logprobs:
        tok = entry.get("token", "").strip()
        if tok in LIKERT_TOKENS:
            raw[tok] = entry["logprob"]

    if not raw:
        # None of the top logprobs were Likert tokens — unusual
        return None, generated

    # Renormalize over just the Likert tokens we found
    # (some may be missing from top_k — treat them as having ~0 probability)
    max_lp = max(raw.values())
    exp_sum = sum(math.exp(lp - max_lp) for lp in raw.values())
    dist = {}
    for tok in LIKERT_TOKENS:
        if tok in raw:
            dist[tok] = math.exp(raw[tok] - max_lp) / exp_sum
        else:
            dist[tok] = 0.0

    return dist, generated


def load_ipip300(session_path):
    """Load IPIP-300 items and scale definitions from admin session JSON."""
    with open(session_path) as f:
        session = json.load(f)

    ipip = session["measures"]["IPIP300"]
    items = ipip["items"]  # {ipip1: "I worry about things", ...}
    scales = ipip["scales"]  # {IPIP300-NEU: {item_ids: [...], reverse_keyed_item_ids: [...]}, ...}
    return items, scales


def score_scales(item_results, scales):
    """Compute Big Five scale scores from item-level results.

    Scores each item using both argmax and expected value (from distribution),
    handling reverse-keying.
    """
    scale_scores = {}
    for scale_id, scale_def in scales.items():
        item_ids = scale_def["item_ids"]
        reverse_keyed = set(scale_def["reverse_keyed_item_ids"])

        argmax_values = []
        ev_values = []
        entropy_values = []

        for item_id in item_ids:
            if item_id not in item_results:
                continue
            result = item_results[item_id]
            dist = result.get("distribution")

            # Argmax response
            argmax_val = result.get("argmax")
            if argmax_val is not None:
                val = int(argmax_val)
                if item_id in reverse_keyed:
                    val = 6 - val  # reverse on 5-point scale
                argmax_values.append(val)

            # Expected value from distribution
            if dist:
                ev = sum(int(k) * v for k, v in dist.items())
                if item_id in reverse_keyed:
                    ev = 6.0 - ev
                ev_values.append(ev)

            # Entropy (not affected by reverse-keying)
            h = result.get("entropy")
            if h is not None:
                entropy_values.append(h)

        scale_scores[scale_id] = {
            "name": scale_def["user_readable_name"],
            "n_items": len(item_ids),
            "n_scored": len(argmax_values),
            "argmax_mean": sum(argmax_values) / len(argmax_values) if argmax_values else None,
            "ev_mean": sum(ev_values) / len(ev_values) if ev_values else None,
            "entropy_mean": sum(entropy_values) / len(entropy_values) if entropy_values else None,
        }

    return scale_scores


def main():
    parser = argparse.ArgumentParser(description="Run IPIP-300 via Ollama with logprobs")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g. gemma3:4b)")
    parser.add_argument("--items", type=int, default=0,
                        help="Limit to first N items (0 = all 300)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/<model>_ipip300.json)")
    parser.add_argument("--top-logprobs", type=int, default=10,
                        help="Number of top logprobs to request")
    parser.add_argument("--variants", action="store_true",
                        help="Run all prompt variants and compute reliability")
    args = parser.parse_args()

    # Load items
    items, scales = load_ipip300(ADMIN_SESSION)
    item_list = list(items.items())
    if args.items > 0:
        item_list = item_list[:args.items]

    print(f"Model: {args.model}")
    print(f"Items: {len(item_list)} / {len(items)}")
    print()

    # Check Ollama is reachable
    try:
        urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except URLError as e:
        print(f"Cannot reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    # Detect thinking model with a probe call
    probe = ollama_generate(args.model, "Say hello.", args.top_logprobs, num_predict=2)
    thinking = is_thinking_model(probe)
    use_raw = False
    num_predict = 1
    if thinking:
        # Use raw prompt with /no_think to suppress reasoning content
        # Model still emits <think>\n\n</think>\n\n (empty) then answers
        use_raw = True
        num_predict = 10  # enough for empty think tags + answer
        print("Detected thinking model — using /no_think with raw prompt")
        print()

    # Determine which prompt variants to run
    if args.variants:
        if use_raw:
            templates = [(f"v{vi}", t) for vi, t in enumerate(QWEN3_PROMPT_VARIANTS)]
        else:
            templates = [(f"v{vi}", t) for vi, t in enumerate(PROMPT_VARIANTS)]
        print(f"Running {len(templates)} prompt variants per item "
              f"({len(item_list) * len(templates)} total calls)")
        print()
    else:
        if use_raw:
            templates = [("v0", QWEN3_PROMPT_TEMPLATE)]
        else:
            templates = [("v0", PROMPT_TEMPLATE)]

    # Run each item (x each variant)
    # item_results stores the v0 result for backward compatibility
    # variant_evs[item_id] = [ev_v0, ev_v1, ...] for reliability analysis
    item_results = {}
    variant_evs = {}
    failed = []
    t0 = time.time()

    for i, (item_id, item_text) in enumerate(item_list):
        item_variant_evs = []
        for vi, (vname, template) in enumerate(templates):
            prompt = template.format(item_text=item_text)
            try:
                response = ollama_generate(args.model, prompt, args.top_logprobs,
                                           num_predict=num_predict, raw=use_raw)
            except Exception as e:
                if vi == 0:
                    print(f"  [{i+1}/{len(item_list)}] {item_id}: ERROR - {e}")
                    failed.append(item_id)
                item_variant_evs.append(None)
                continue

            dist, generated = extract_likert_distribution(response)

            if dist:
                argmax = max(dist, key=lambda k: dist[k])
                ev = sum(int(k) * v for k, v in dist.items())
                h = entropy(dist)
                item_variant_evs.append(ev)

                if vi == 0:
                    item_results[item_id] = {
                        "item_text": item_text,
                        "generated": generated,
                        "argmax": argmax,
                        "expected_value": round(ev, 4),
                        "entropy": round(h, 4),
                        "entropy_normalized": round(h / MAX_ENTROPY, 4),
                        "distribution": {k: round(v, 6) for k, v in dist.items()},
                    }
            else:
                item_variant_evs.append(None)
                if vi == 0:
                    item_results[item_id] = {
                        "item_text": item_text,
                        "generated": generated,
                        "argmax": None,
                        "expected_value": None,
                        "distribution": None,
                    }

        variant_evs[item_id] = item_variant_evs

        # Print progress for v0
        r = item_results.get(item_id, {})
        if r.get("distribution"):
            dist = r["distribution"]
            bar = " ".join(f"{int(float(v)*100):2d}%" for v in dist.values())
            ev_strs = ""
            if args.variants:
                valid = [x for x in item_variant_evs if x is not None]
                if len(valid) > 1:
                    spread = max(valid) - min(valid)
                    ev_strs = f" spread={spread:.2f}"
            print(f"  [{i+1}/{len(item_list)}] {item_id}: {r['item_text'][:45]:45s}  "
                  f"argmax={r['argmax']} ev={r['expected_value']:.2f} "
                  f"H={r['entropy']:.2f}{ev_strs}  [{bar}]")
        else:
            gen = r.get("generated", "?")
            print(f"  [{i+1}/{len(item_list)}] {item_id}: {item_text[:45]:45s}  "
                  f"NO LIKERT TOKENS (got: {gen!r})")

    elapsed = time.time() - t0
    total_calls = len(item_list) * len(templates)
    print(f"\nCompleted {len(item_results)}/{len(item_list)} items in {elapsed:.1f}s "
          f"({elapsed/total_calls:.2f}s/call)")

    if failed:
        print(f"Failed: {failed}")

    # Reliability analysis (if variants were run)
    if args.variants and len(templates) > 1:
        print(f"\n=== Prompt Variant Reliability ({len(templates)} variants) ===")

        # Compute ICC(2,1) — each item measured by k variants
        # ICC = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
        all_valid_evs = []
        for item_id in item_list:
            iid = item_id[0] if isinstance(item_id, tuple) else item_id
            evs_list = variant_evs.get(iid, variant_evs.get(item_id, []))
            valid = [x for x in evs_list if x is not None]
            if len(valid) == len(templates):
                all_valid_evs.append(valid)

        if all_valid_evs:
            n = len(all_valid_evs)
            k = len(templates)
            # Grand mean
            grand_mean = sum(sum(row) for row in all_valid_evs) / (n * k)
            # MS_between (between items)
            item_means = [sum(row) / k for row in all_valid_evs]
            ss_between = k * sum((m - grand_mean) ** 2 for m in item_means)
            ms_between = ss_between / (n - 1) if n > 1 else 0
            # MS_within (within items, across variants)
            ss_within = sum(
                sum((x - item_means[i]) ** 2 for x in row)
                for i, row in enumerate(all_valid_evs)
            )
            ms_within = ss_within / (n * (k - 1)) if n * (k - 1) > 0 else 0
            # ICC
            icc = ((ms_between - ms_within) /
                   (ms_between + (k - 1) * ms_within)) if (ms_between + (k - 1) * ms_within) > 0 else 0

            # Per-item spread stats
            spreads = [max(row) - min(row) for row in all_valid_evs]
            mean_spread = sum(spreads) / len(spreads)
            median_spread = sorted(spreads)[len(spreads) // 2]

            print(f"  Items with all variants valid: {n}/{len(item_list)}")
            print(f"  ICC(2,1): {icc:.3f}")
            print(f"  MS_between (item variance):  {ms_between:.4f}")
            print(f"  MS_within (variant noise):   {ms_within:.4f}")
            print(f"  Signal / (Signal+Noise):     {icc:.1%}")
            print(f"  Mean EV spread per item:     {mean_spread:.3f}")
            print(f"  Median EV spread per item:   {median_spread:.3f}")

            # Per-scale ICC
            with open(ADMIN_SESSION) as f:
                session_data = json.load(f)
            scales_def = session_data["measures"]["IPIP300"]["scales"]

            print(f"\n  Per-scale ICC:")
            for scale_id, scale_def in scales_def.items():
                scale_item_ids = set(scale_def["item_ids"])
                scale_evs = []
                for item_id in item_list:
                    iid = item_id[0] if isinstance(item_id, tuple) else item_id
                    if iid in scale_item_ids:
                        evs_list = variant_evs.get(iid, [])
                        valid = [x for x in evs_list if x is not None]
                        if len(valid) == k:
                            scale_evs.append(valid)
                if len(scale_evs) > 1:
                    s_n = len(scale_evs)
                    s_gm = sum(sum(r) for r in scale_evs) / (s_n * k)
                    s_im = [sum(r) / k for r in scale_evs]
                    s_ssb = k * sum((m - s_gm) ** 2 for m in s_im)
                    s_msb = s_ssb / (s_n - 1)
                    s_ssw = sum(sum((x - s_im[i]) ** 2 for x in r)
                                for i, r in enumerate(scale_evs))
                    s_msw = s_ssw / (s_n * (k - 1)) if s_n * (k - 1) > 0 else 0
                    s_icc = ((s_msb - s_msw) /
                             (s_msb + (k - 1) * s_msw)) if (s_msb + (k - 1) * s_msw) > 0 else 0
                    print(f"    {scale_def['user_readable_name']:35s}: ICC={s_icc:.3f} "
                          f"(MS_b={s_msb:.3f}, MS_w={s_msw:.3f}, n={s_n})")
        else:
            print("  No items had valid EVs across all variants.")

    # Score scales
    scale_scores = score_scales(item_results, scales)
    print("\n=== Big Five Scale Scores ===")
    print(f"{'Scale':<25s} {'Argmax':>8s} {'EV':>8s} {'Entropy':>8s}  (1-5, H: 0=certain {MAX_ENTROPY:.2f}=uniform)")
    print("-" * 70)
    for scores in scale_scores.values():
        am = f"{scores['argmax_mean']:.2f}" if scores['argmax_mean'] else "N/A"
        ev = f"{scores['ev_mean']:.2f}" if scores['ev_mean'] else "N/A"
        h = f"{scores['entropy_mean']:.3f}" if scores['entropy_mean'] else "N/A"
        print(f"{scores['name']:<25s} {am:>8s} {ev:>8s} {h:>8s}  "
              f"({scores['n_scored']}/{scores['n_items']} items)")

    # Save results
    output_path = args.output
    if not output_path:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{safe_model}_ipip300.json"

    output = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_items": len(item_list),
        "n_variants": len(templates),
        "elapsed_seconds": round(elapsed, 1),
        "scale_scores": scale_scores,
        "item_results": item_results,
    }
    if args.variants:
        output["variant_evs"] = {
            k: [round(x, 4) if x is not None else None for x in v]
            for k, v in variant_evs.items()
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
