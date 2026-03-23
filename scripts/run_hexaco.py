#!/usr/bin/env python3
"""Run HEXACO-100 personality inventory via Ollama with logprob collection.

Usage:
    python scripts/run_hexaco.py --model gemma3:4b
    python scripts/run_hexaco.py --model gemma3:4b --items 10
    python scripts/run_hexaco.py --model gemma3:4b --variants
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# Reuse core functions from the IPIP script
from run_ollama_logprobs import (
    OLLAMA_URL, LIKERT_TOKENS, MAX_ENTROPY, PROMPT_VARIANTS, QWEN3_PROMPT_VARIANTS,
    entropy, ollama_generate, is_thinking_model, extract_likert_distribution,
)

HEXACO_FILE = "instruments/hexaco100.json"


def load_hexaco(path):
    with open(path) as f:
        data = json.load(f)
    items = {k: v["text"] for k, v in data["items"].items()}
    scales = data["scales"]
    return items, scales


def score_scales(item_results, scales):
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

            argmax_val = result.get("argmax")
            if argmax_val is not None:
                val = int(argmax_val)
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
            "name": scale_def["name"],
            "n_items": len(item_ids),
            "n_scored": len(argmax_values),
            "argmax_mean": sum(argmax_values) / len(argmax_values) if argmax_values else None,
            "ev_mean": sum(ev_values) / len(ev_values) if ev_values else None,
            "entropy_mean": sum(entropy_values) / len(entropy_values) if entropy_values else None,
        }
    return scale_scores


def main():
    parser = argparse.ArgumentParser(description="Run HEXACO-100 via Ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--variants", action="store_true")
    args = parser.parse_args()

    items, scales = load_hexaco(HEXACO_FILE)
    item_list = list(items.items())
    if args.items > 0:
        item_list = item_list[:args.items]

    print(f"Model: {args.model}")
    print(f"HEXACO-100 items: {len(item_list)} / {len(items)}")
    print()

    try:
        urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except URLError as e:
        print(f"Cannot reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    probe = ollama_generate(args.model, "Say hello.", args.top_logprobs, num_predict=2)
    thinking = is_thinking_model(probe)
    use_raw = False
    num_predict = 1
    if thinking:
        use_raw = True
        num_predict = 10
        print("Detected thinking model — using /no_think with raw prompt")
        print()

    if args.variants:
        if use_raw:
            templates = [(f"v{vi}", t) for vi, t in enumerate(QWEN3_PROMPT_VARIANTS)]
        else:
            templates = [(f"v{vi}", t) for vi, t in enumerate(PROMPT_VARIANTS)]
        print(f"Running {len(templates)} prompt variants per item")
        print()
    else:
        if use_raw:
            templates = [("v0", QWEN3_PROMPT_VARIANTS[0])]
        else:
            templates = [("v0", PROMPT_VARIANTS[0])]

    item_results = {}
    variant_evs = {}
    failed = []
    t0 = time.time()

    for i, (item_id, item_text) in enumerate(item_list):
        item_variant_evs = []
        for vi, (_, template) in enumerate(templates):
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
                        "item_text": item_text, "generated": generated,
                        "argmax": None, "expected_value": None, "distribution": None,
                    }

        variant_evs[item_id] = item_variant_evs

        r = item_results.get(item_id, {})
        if r.get("distribution"):
            dist = r["distribution"]
            bar = " ".join(f"{int(float(v)*100):2d}%" for v in dist.values())
            print(f"  [{i+1}/{len(item_list)}] {item_id}: {r['item_text'][:45]:45s}  "
                  f"argmax={r['argmax']} ev={r['expected_value']:.2f} "
                  f"H={r['entropy']:.2f}  [{bar}]")
        else:
            print(f"  [{i+1}/{len(item_list)}] {item_id}: {item_text[:45]:45s}  "
                  f"NO LIKERT TOKENS")

    elapsed = time.time() - t0
    total_calls = len(item_list) * len(templates)
    print(f"\nCompleted {len(item_results)}/{len(item_list)} items in {elapsed:.1f}s "
          f"({elapsed/total_calls:.2f}s/call)")

    # Score
    scale_scores = score_scales(item_results, scales)
    print(f"\n=== HEXACO Scale Scores ===")
    print(f"{'Scale':<25s} {'Argmax':>8s} {'EV':>8s} {'Entropy':>8s}")
    print("-" * 55)
    for scores in scale_scores.values():
        am = f"{scores['argmax_mean']:.2f}" if scores['argmax_mean'] else "N/A"
        ev = f"{scores['ev_mean']:.2f}" if scores['ev_mean'] else "N/A"
        h = f"{scores['entropy_mean']:.3f}" if scores['entropy_mean'] else "N/A"
        print(f"{scores['name']:<25s} {am:>8s} {ev:>8s} {h:>8s}  "
              f"({scores['n_scored']}/{scores['n_items']} items)")

    # Save
    output_path = args.output
    if not output_path:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{safe_model}_hexaco100.json"

    output = {
        "model": args.model,
        "instrument": "HEXACO-100",
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
