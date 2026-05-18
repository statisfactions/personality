#!/usr/bin/env python3
"""Administer Okada GFC-30 Big Five via HuggingFace (FP16 chat-template).

HF / FP16 counterpart to scripts/run_gfc_ollama.py. Built for cross-line
comparability with rgb's W8 Rep/Likert tracks (`scripts/persona_repr_mapping.py`,
`scripts/persona_instrument_response.py`), which use chat-template with persona
as system message and bf16 weights via `hf_logprobs.load_model`.

Output JSON schema is identical to run_gfc_ollama.py so the existing
TIRT fitting drivers (psychometrics/gfc_tirt/fit_tirt_*.R) consume it
without modification — modulo `model` field which holds the HF repo ID.

Usage:
    # Goldberg-marker persona (W7 baseline equivalent):
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
        --model Gemma --max-personas 50

    # IPIP-raw persona (W8 §3/§4/§5 raw equivalent):
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
        --model Gemma --max-personas 50 \
        --synthetic-personas instruments/synthetic_personas_ipip.json \
        --persona-field ipip_raw

    # IPIP-reflowed persona (W8 §3/§5 reflow equivalent):
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
        --model Gemma --max-personas 50 \
        --synthetic-personas instruments/synthetic_personas_ipip.json \
        --persona-field ipip_reflowed

    # Smoke test (2 pairs × 1 persona):
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
        --model Gemma --pairs 2 --max-personas 1
"""

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from hf_logprobs import MODELS, load_model, likert_distribution, resolve


INSTRUMENT_PATH = "instruments/okada_gfc30.json"
DEFAULT_PERSONAS = "instruments/synthetic_personas.json"
GFC_TOKENS = ("1", "2", "3", "4", "5", "6", "7")
MAX_ENTROPY = math.log(len(GFC_TOKENS))


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


# Fake-good / impression-management instruction, appended to the persona
# system message when --condition fake_good. Phrasing matches the
# faking-research convention (Viswesvaran & Ones 1999; Stark & Drasgow
# 2009): imagine a high-stakes evaluation context and answer to maximize
# perceived favorability.
FAKE_GOOD_SUFFIX = (
    "\n\nIMPORTANT INSTRUCTION: Imagine you are completing this "
    "questionnaire as part of an application for a position you very "
    "much want. Your responses will be evaluated and used to decide "
    "whether you are selected. Respond in a way that makes the most "
    "favorable possible impression."
)


def build_user_message(pair, randomize_lr, seed):
    """User-message body for one GFC pair (no persona; persona goes in system)."""
    rng = random.Random(seed)
    left_text = pair["left"]["text"]
    right_text = pair["right"]["text"]
    swapped = False
    if randomize_lr and rng.random() < 0.5:
        left_text, right_text = right_text, left_text
        swapped = True
    body = GFC_INSTRUCTION + GFC_ITEM_TEMPLATE.format(
        left_text=left_text, right_text=right_text
    )
    return body, swapped


def administer_one(pair, model, tok, device, persona_desc, randomize_lr, seed):
    """Single (persona × pair) call. Returns the per-pair record dict."""
    user_body, swapped = build_user_message(pair, randomize_lr, seed)
    dist, argmax, h = likert_distribution(
        model, tok, user_body, device,
        digits=GFC_TOKENS,
        use_chat_template=True,
        system_content=persona_desc or "",
    )
    ev = sum(int(k) * v for k, v in dist.items())

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
        "response_ev": round(ev, 4),
        "response_entropy": round(h, 4),
        "distribution": {k: round(v, 6) for k, v in dist.items()},
        "raw_logprobs": None,  # softmax over the 7 tokens; raw not retained
        "generated_text": None,  # logprob path; no text generated
    }


def load_personas(path, field):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for p in data["personas"]:
        text = p.get(field)
        if text is None:
            continue  # e.g. ipip_reflowed only present for first 50
        out[p["persona_id"]] = text
    if not out:
        raise ValueError(
            f"No personas with field {field!r} in {path}. "
            f"Did you mean a different --persona-field?"
        )
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def _save_output(path, args, pairs, persona_list, results, hf_repo, n_completed):
    instrument_id = Path(args.instrument).stem
    out = {
        "model": args.model,
        "hf_repo": hf_repo,
        "instrument": instrument_id,
        "instrument_path": args.instrument,
        "backend": "hf",
        "persona_field": args.persona_field,
        "personas_path": args.synthetic_personas,
        "condition": args.condition,
        "fg_position": args.fg_position,
        "n_pairs": len(pairs),
        "n_personas": len(persona_list),
        "n_completed": n_completed,
        "randomize_lr": not args.no_randomize,
        "seed": args.seed,
        "results": results,
    }
    if args.condition == "fake_good":
        out["fake_good_suffix"] = FAKE_GOOD_SUFFIX
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Administer Okada GFC-30 via HF chat-template (FP16/bf16)")
    parser.add_argument("--model", required=True,
                        help="Short name (Gemma/Llama/Phi4/Qwen/Gemma12/Llama8/Qwen7) "
                             "or HF repo ID")
    parser.add_argument("--synthetic-personas", type=str, default=DEFAULT_PERSONAS,
                        help="Path to synthetic persona JSON "
                             "(default: instruments/synthetic_personas.json)")
    parser.add_argument("--persona-field", type=str, default="description",
                        choices=["description", "ipip_raw", "ipip_reflowed"],
                        help="Field within each persona to use as system message. "
                             "description = Goldberg markers (W7 baseline); "
                             "ipip_raw = behavioral IPIP composition (W8 raw); "
                             "ipip_reflowed = Sonnet-smoothed IPIP (W8 reflow).")
    parser.add_argument("--max-personas", type=int, default=0,
                        help="Limit to first N personas (0 = all available)")
    parser.add_argument("--pairs", type=int, default=0,
                        help="Limit to first N GFC pairs (0 = all 30)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Don't randomize left/right assignment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for L/R randomization")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Save checkpoint every N completed prompts")
    parser.add_argument("--instrument", type=str, default=INSTRUMENT_PATH,
                        help=f"Instrument JSON path (default: {INSTRUMENT_PATH}). "
                             "Must match the okada_gfc30.json schema "
                             "(top-level 'pairs' list with left/right items).")
    parser.add_argument("--condition", type=str, default="honest",
                        choices=["honest", "fake_good"],
                        help="Response-style condition. honest = no extra "
                             "instruction (W7-W11 default). fake_good = "
                             "combine impression-management instruction with "
                             "persona system message (W12 SDR test).")
    parser.add_argument("--fg-position", type=str, default="suffix",
                        choices=["suffix", "prefix"],
                        help="Where FAKE_GOOD_SUFFIX sits relative to persona: "
                             "suffix (W12 §5b default) or prefix (W12 §5d "
                             "ordering test). No effect when condition=honest.")
    args = parser.parse_args()

    # Instrument
    with open(args.instrument) as f:
        instrument = json.load(f)
    pairs = instrument["pairs"]
    if args.pairs > 0:
        pairs = pairs[:args.pairs]

    # Personas
    personas = load_personas(args.synthetic_personas, args.persona_field)
    persona_list = list(personas.items())
    if args.max_personas > 0:
        persona_list = persona_list[:args.max_personas]

    total_prompts = len(persona_list) * len(pairs)
    hf_repo = resolve(args.model)
    print(f"Model:         {args.model}  (HF: {hf_repo})")
    print(f"Persona file:  {args.synthetic_personas}")
    print(f"Persona field: {args.persona_field}")
    print(f"Personas:      {len(persona_list)}")
    print(f"Pairs:         {len(pairs)} / {instrument['n_pairs']}")
    print(f"Total prompts: {total_prompts}")
    print()

    # Output path
    output_path = args.output
    if output_path is None:
        os.makedirs("psychometrics/gfc_tirt", exist_ok=True)
        # Use HF-anchored slug so rgb's cohort short names land on a
        # consistent filename across persona forms. Append _fake_good
        # only for the fake-good condition, so honest filenames remain
        # backward-compatible with W7-W11 artifacts.
        model_slug = args.model.replace("/", "_").replace(":", "-")
        cond_suffix = "" if args.condition == "honest" else f"_{args.condition}"
        if args.condition == "fake_good" and args.fg_position == "prefix":
            cond_suffix += "_fgpfx"
        output_path = (
            f"psychometrics/gfc_tirt/"
            f"{model_slug}_gfc30_hf_{args.persona_field}{cond_suffix}.json"
        )
    print(f"Output:        {output_path}")
    print()

    # Resume from checkpoint if present
    completed_keys = set()
    prior_results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            prior = json.load(f)
        prior_results = prior.get("results", [])
        for r in prior_results:
            completed_keys.add((r.get("persona_id", "none"), r["block"]))
        print(f"Resuming from checkpoint: {len(prior_results)}/{total_prompts} done")
        print()

    # Load model
    t_load = time.time()
    model, tok, device = load_model(args.model)
    print(f"Loaded {hf_repo} on {device}, dtype={model.dtype} "
          f"({time.time() - t_load:.1f}s)")
    print()

    # Run
    results = list(prior_results)
    n_completed = len(prior_results)
    n_new = 0
    t0 = time.time()

    for persona_id, persona_desc in persona_list:
        persona_done = sum(1 for k in completed_keys if k[0] == persona_id)
        if persona_done == len(pairs):
            continue

        # Apply fake-good instruction to persona system message. Position
        # per --fg-position: suffix (W12 §5b default, FG nearest the user
        # message and most recent in attention at decision time) or prefix
        # (W12 §5d, FG further from the response position).
        effective_persona = persona_desc
        if args.condition == "fake_good":
            if args.fg_position == "prefix":
                effective_persona = FAKE_GOOD_SUFFIX.lstrip("\n") + "\n\n" + persona_desc
            else:
                effective_persona = persona_desc + FAKE_GOOD_SUFFIX

        head = effective_persona.replace("\n", " ")[:60]
        print(f"--- {persona_id} ({head}...) ---")

        for pair in pairs:
            block = pair["block"]
            if (persona_id, block) in completed_keys:
                continue

            lr_seed = args.seed * 10000 + (hash(persona_id) % 10000) + block
            rec = administer_one(
                pair, model, tok, device, effective_persona,
                randomize_lr=not args.no_randomize, seed=lr_seed,
            )
            rec["persona_id"] = persona_id
            results.append(rec)
            n_completed += 1
            n_new += 1

            argmax = rec["response_argmax"]
            ev = rec["response_ev"]
            print(f"  [{n_completed}/{total_prompts}] B{block:02d} → "
                  f"{argmax}  EV={ev:.2f}  H={rec['response_entropy']:.2f}")

            if args.checkpoint_every > 0 and n_new % args.checkpoint_every == 0:
                _save_output(output_path, args, pairs, persona_list,
                             results, hf_repo, n_completed)
                rate = n_new / (time.time() - t0)
                print(f"  [checkpoint @ {n_completed}/{total_prompts}, "
                      f"{rate:.2f} prompts/s]")

    _save_output(output_path, args, pairs, persona_list, results, hf_repo,
                 n_completed)
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    if n_new > 0:
        print(f"Completed {n_new} new prompts in {elapsed:.0f}s "
              f"({n_new / elapsed:.2f}/s)")
    print(f"Saved {output_path}")

    # Free GPU
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
