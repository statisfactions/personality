#!/usr/bin/env python3
"""Run IPIP-300 personality inventory via HuggingFace with logprob collection.

Collects the full probability distribution over Likert scale responses,
not just the argmax. Scores Big Five scales from the admin session definition.

Usage:
    .venv/bin/python scripts/run_ipip300.py --model Gemma
    .venv/bin/python scripts/run_ipip300.py --model google/gemma-3-4b-it
    .venv/bin/python scripts/run_ipip300.py --model Gemma --items 10  # quick test
    .venv/bin/python scripts/run_ipip300.py --model Gemma --variants  # 4 prompt phrasings
"""

import argparse
import json
import math
import time
from pathlib import Path

# Local import; scripts/ is expected on PYTHONPATH.
from hf_logprobs import MODELS, resolve, load_model, likert_distribution


ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
LIKERT_TOKENS = ("1", "2", "3", "4", "5")
MAX_ENTROPY = math.log(len(LIKERT_TOKENS))  # ln(5) ≈ 1.609


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


def load_ipip300(session_path):
    with open(session_path) as f:
        session = json.load(f)
    ipip = session["measures"]["IPIP300"]
    return ipip["items"], ipip["scales"]


def score_scales(item_results, scales):
    scale_scores = {}
    for scale_id, scale_def in scales.items():
        item_ids = scale_def["item_ids"]
        reverse_keyed = set(scale_def["reverse_keyed_item_ids"])
        argmax_values, ev_values, entropy_values = [], [], []
        for item_id in item_ids:
            if item_id not in item_results:
                continue
            result = item_results[item_id]
            dist = result.get("distribution")
            am = result.get("argmax")
            if am is not None:
                val = int(am)
                if item_id in reverse_keyed:
                    val = 6 - val
                argmax_values.append(val)
            if dist:
                ev = sum(int(k) * v for k, v in dist.items())
                if item_id in reverse_keyed:
                    ev = 6.0 - ev
                ev_values.append(ev)
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


def compute_icc(per_item_evs, k):
    """ICC(2,1) over n items × k variants. per_item_evs is a list of length-k lists."""
    rows = [r for r in per_item_evs if len(r) == k and all(x is not None for x in r)]
    n = len(rows)
    if n < 2:
        return None, 0, 0, 0
    grand = sum(sum(r) for r in rows) / (n * k)
    item_means = [sum(r) / k for r in rows]
    ms_between = k * sum((m - grand) ** 2 for m in item_means) / (n - 1)
    ms_within = sum(
        sum((x - item_means[i]) ** 2 for x in r)
        for i, r in enumerate(rows)
    ) / (n * (k - 1)) if n * (k - 1) > 0 else 0
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0
    return icc, n, ms_between, ms_within


def main():
    parser = argparse.ArgumentParser(description="Run IPIP-300 via HuggingFace logprobs")
    parser.add_argument("--model", required=True,
                        help="Short name (Gemma/Llama/Phi4/Qwen/...) or HF repo ID")
    parser.add_argument("--items", type=int, default=0,
                        help="Limit to first N items (0 = all 300)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/<model>_ipip300.json)")
    parser.add_argument("--variants", action="store_true",
                        help="Run all prompt variants and compute reliability")
    args = parser.parse_args()

    items, scales = load_ipip300(ADMIN_SESSION)
    item_list = list(items.items())
    if args.items > 0:
        item_list = item_list[:args.items]

    print(f"Model: {args.model}  (HF: {resolve(args.model)})")
    print(f"Items: {len(item_list)} / {len(items)}")
    print()

    model, tok, device = load_model(args.model)
    print(f"Loaded on device={device}, dtype={model.dtype}")
    print()

    if args.variants:
        templates = [(f"v{i}", t) for i, t in enumerate(PROMPT_VARIANTS)]
        print(f"Running {len(templates)} prompt variants per item "
              f"({len(item_list) * len(templates)} total forward passes)")
        print()
    else:
        templates = [("v0", PROMPT_VARIANTS[0])]

    item_results = {}
    variant_evs = {}
    failed = []
    t0 = time.time()

    for i, (item_id, item_text) in enumerate(item_list):
        per_variant = []
        for vi, (_, template) in enumerate(templates):
            prompt = template.format(item_text=item_text)
            try:
                dist, argmax, h = likert_distribution(model, tok, prompt, device,
                                                     digits=LIKERT_TOKENS)
            except Exception as e:
                if vi == 0:
                    print(f"  [{i+1}/{len(item_list)}] {item_id}: ERROR - {e}")
                    failed.append(item_id)
                per_variant.append(None)
                continue
            ev = sum(int(k) * v for k, v in dist.items())
            per_variant.append(ev)
            if vi == 0:
                item_results[item_id] = {
                    "item_text": item_text,
                    "argmax": argmax,
                    "expected_value": round(ev, 4),
                    "entropy": round(h, 4),
                    "entropy_normalized": round(h / MAX_ENTROPY, 4),
                    "distribution": {k: round(v, 6) for k, v in dist.items()},
                }

        variant_evs[item_id] = per_variant
        r = item_results.get(item_id)
        if r:
            bar = " ".join(f"{int(v*100):2d}%" for v in r["distribution"].values())
            spread_str = ""
            if args.variants:
                valid = [x for x in per_variant if x is not None]
                if len(valid) > 1:
                    spread_str = f" spread={max(valid) - min(valid):.2f}"
            print(f"  [{i+1}/{len(item_list)}] {item_id}: {r['item_text'][:45]:45s}  "
                  f"argmax={r['argmax']} ev={r['expected_value']:.2f} "
                  f"H={r['entropy']:.2f}{spread_str}  [{bar}]")

    elapsed = time.time() - t0
    total_calls = len(item_list) * len(templates)
    print(f"\nCompleted {len(item_results)}/{len(item_list)} items in {elapsed:.1f}s "
          f"({elapsed/total_calls:.2f}s/call)")
    if failed:
        print(f"Failed: {failed}")

    if args.variants and len(templates) > 1:
        k = len(templates)
        print(f"\n=== Prompt Variant Reliability ({k} variants) ===")
        icc, n, msb, msw = compute_icc(list(variant_evs.values()), k)
        if icc is not None:
            print(f"  Items with all variants valid: {n}/{len(item_list)}")
            print(f"  ICC(2,1): {icc:.3f}")
            print(f"  MS_between: {msb:.4f}   MS_within: {msw:.4f}")

            print(f"\n  Per-scale ICC:")
            for scale_id, scale_def in scales.items():
                scale_item_ids = set(scale_def["item_ids"])
                scale_rows = [variant_evs[iid] for iid in variant_evs
                              if iid in scale_item_ids]
                s_icc, s_n, s_msb, s_msw = compute_icc(scale_rows, k)
                if s_icc is not None:
                    print(f"    {scale_def['user_readable_name']:35s}: ICC={s_icc:.3f} "
                          f"(MS_b={s_msb:.3f}, MS_w={s_msw:.3f}, n={s_n})")

    scale_scores = score_scales(item_results, scales)
    print("\n=== Big Five Scale Scores ===")
    print(f"{'Scale':<25s} {'Argmax':>8s} {'EV':>8s} {'Entropy':>8s}  "
          f"(1-5, H: 0=certain {MAX_ENTROPY:.2f}=uniform)")
    print("-" * 70)
    for scores in scale_scores.values():
        am = f"{scores['argmax_mean']:.2f}" if scores['argmax_mean'] else "N/A"
        ev = f"{scores['ev_mean']:.2f}" if scores['ev_mean'] else "N/A"
        hh = f"{scores['entropy_mean']:.3f}" if scores['entropy_mean'] else "N/A"
        print(f"{scores['name']:<25s} {am:>8s} {ev:>8s} {hh:>8s}  "
              f"({scores['n_scored']}/{scores['n_items']} items)")

    output_path = args.output
    if not output_path:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{safe_model}_ipip300.json"

    output = {
        "model": args.model,
        "hf_repo": resolve(args.model),
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
