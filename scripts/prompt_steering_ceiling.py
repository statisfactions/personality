#!/usr/bin/env python3
"""Prompt-steering ceiling for all four models × HEXACO-H on the holdout set.

Companion to rgb_reports/report_week5_meandiff.md §9. The §9 numbers were
Llama-only; this extends to Gemma, Phi4, Qwen.

Each model × condition is evaluated on 24 holdout pairs × 2 orderings
(position debiasing), reporting the fraction that pick the high-trait
response. Conditions per model:

  bare             — raw text prompt (how our residual-stream work was
                     evaluated until now)
  chat default     — model's default chat template with no explicit
                     system message (for Llama: date metadata injected;
                     for Qwen: "You are Qwen..." injected; for Gemma/Phi:
                     no system content)
  chat +H persona  — chat template with system prompt pushing toward high
                     Honesty-Humility
  chat -H persona  — chat template with system prompt pushing toward low
                     Honesty-Humility

See §9 for Llama-specific finding: the chat template alone bumps
baseline from 62.5% to 93.8%; persona steering covers 0.02-0.98.
Open question answered here: does this generalize across the other
three instruct-tuned models, given they ship very different templates?

Usage:
    python scripts/prompt_steering_ceiling.py
    python scripts/prompt_steering_ceiling.py --trait H --output results/prompt_ceiling.csv
"""

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODELS = [
    ("Llama", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Gemma", "google/gemma-3-4b-it"),
    ("Phi4",  "microsoft/Phi-4-mini-instruct"),
    ("Qwen",  "Qwen/Qwen2.5-3B-Instruct"),
]

CONTRAST_PAIRS_HOLDOUT = Path("instruments/contrast_pairs_holdout.json")
CONTRAST_PAIRS_TRAIN   = Path("instruments/contrast_pairs.json")


def build_bc_prompt(p, opt_a, opt_b):
    return (
        "Consider what a person most like you would do in the following situation: "
        f"{p['situation']}\n\n"
        f"Which would you do?\nA) {opt_a}\nB) {opt_b}\n\n"
        "Respond with just A or B.\nAnswer:"
    )


def load_model(repo, device, dtype):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=dtype, device_map=device,
    )
    model.eval()
    return model, tok


def pick_one(model, tok, text, a_id, b_id, device):
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].float()
    del out
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return logits[a_id].item() > logits[b_id].item()


def eval_condition(model, tok, pairs, system_content, use_chat_template, device):
    """Position-debiased high-trait pick rate on `pairs`.

    system_content: None → no system message. "" → empty system message
        (different from None on Llama/Qwen, same on Gemma/Phi).
    use_chat_template: if False, feed plain text; if True, wrap via
        tokenizer.apply_chat_template.
    """
    a_id = tok.encode("A", add_special_tokens=False)[-1]
    b_id = tok.encode("B", add_special_tokens=False)[-1]

    high_picks = total = 0
    for p in pairs:
        for opt_a, opt_b, a_is_high in [
            (p["high"], p["low"], True),
            (p["low"], p["high"], False),
        ]:
            user_content = build_bc_prompt(p, opt_a, opt_b)
            if use_chat_template:
                msgs = []
                if system_content is not None:
                    msgs.append({"role": "system", "content": system_content})
                msgs.append({"role": "user", "content": user_content})
                try:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception as e:
                    # Some templates reject system role (e.g. older Gemma). Fall
                    # back to folding system into user content.
                    folded = (system_content + "\n\n" + user_content) if system_content else user_content
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": folded}],
                        tokenize=False, add_generation_prompt=True,
                    )
            else:
                text = user_content
            pick_a = pick_one(model, tok, text, a_id, b_id, device)
            if pick_a == a_is_high:
                high_picks += 1
            total += 1
    return high_picks / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", default="H")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/prompt_ceiling.csv")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Load descriptors + holdout pairs
    with open(CONTRAST_PAIRS_TRAIN) as f:
        cp = json.load(f)
    high_desc = cp["traits"][args.trait]["high_descriptor"]
    low_desc = cp["traits"][args.trait]["low_descriptor"]

    with open(CONTRAST_PAIRS_HOLDOUT) as f:
        hold = json.load(f)
    pairs = hold["traits"][args.trait]["pairs"]

    conditions = [
        # (label, system_content, use_chat_template)
        ("bare",                   None,                                            False),
        ("chat default",           None,                                            True),
        ("chat empty-sys",         "",                                              True),
        ("chat +H persona",        f"You are a person who is {high_desc}. Respond as such a person would.", True),
        ("chat -H persona",        f"You are a person who is {low_desc}. Respond as such a person would.",  True),
    ]

    rows = []
    for short, repo in MODELS:
        print(f"\n=== Loading {short} ({repo}) ===")
        t0 = time.time()
        try:
            model, tok = load_model(repo, args.device, dtype)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue
        print(f"  loaded in {time.time()-t0:.1f}s")

        for label, system, use_chat in conditions:
            t0 = time.time()
            rate = eval_condition(model, tok, pairs, system, use_chat, args.device)
            dt = time.time() - t0
            print(f"  {label:>24s}  high-pick={rate:.3f}  ({dt:.1f}s)")
            rows.append({
                "model": short,
                "repo": repo,
                "trait": args.trait,
                "condition": label,
                "use_chat_template": use_chat,
                "system_content": system if system is not None else "<none>",
                "high_pick_rate": rate,
            })

        del model, tok
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "repo", "trait", "condition", "use_chat_template", "system_content", "high_pick_rate"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'model':>8s}  {'condition':>22s}  {'high-pick rate':>16s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:>8s}  {r['condition']:>22s}  {r['high_pick_rate']:>16.3f}")


if __name__ == "__main__":
    main()
