#!/usr/bin/env python3
"""Extract trait direction vectors using Anthropic-style mean-diff extraction.

Differences from scripts/extract_trait_vectors.py (LDA pipeline):
  1. Hidden states averaged across RESPONSE tokens (not just the last token)
  2. Direction is the mean-of-high minus mean-of-low (not LDA or PCA)
  3. Optional confound removal by projecting out top PCs of a neutral corpus

Hyperparameters exposed:
  --prefix-mode {high,low,absent,generic}: what "Consider a person who is X"
     prefix gets prepended to each scenario. Keeping the high/low descriptor
     in the prefix replicates our current practice; removing it tests how
     much of the signal is in the response vs in the prefix framing.
  --neutral-variant {scenario_setups,shaggy_dog,factual,none}: which neutral
     corpus (from instruments/neutral_texts.json) to use for PC projection.
     "none" skips projection entirely.
  --layer-strategy {all,best-cv,two-thirds}: whether to save per-layer
     directions or a selected single layer.

Output: results/repe/meandiff/<model>_<trait>_<prefix>_<neutral>.pt
Content:
  - raw_direction: (n_layers, hidden_dim) mean_high - mean_low per layer
  - projected_direction: (n_layers, hidden_dim) after removing neutral PCs,
    or None if --neutral-variant=none
  - mean_high, mean_low: (n_layers, hidden_dim) per-class mean activations
  - per_pair_high, per_pair_low: (n_pairs, n_layers, hidden_dim) individual
    per-pair response-averaged activations, bfloat16 for storage
  - neutral_activations: (n_neutral, n_layers, hidden_dim) for inspection
  - metadata (prefix_mode, neutral_variant, model, trait, n_pairs, etc.)

Usage:
  python scripts/extract_meandiff_vectors.py --model google/gemma-3-4b-it \\
      --prefix-mode high --neutral-variant scenario_setups
  python scripts/extract_meandiff_vectors.py --model meta-llama/Llama-3.2-3B-Instruct \\
      --trait H --prefix-mode absent --neutral-variant factual
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CONTRAST_PAIRS = Path("instruments/contrast_pairs.json")
NEUTRAL_TEXTS = Path("instruments/neutral_texts.json")


PREFIX_TEMPLATES = {
    "high":    "Consider a person who is {high_descriptor}. ",
    "low":     "Consider a person who is {low_descriptor}. ",
    "absent":  "",
    "generic": "Consider a person. ",
}


def load_model(model_name, device="mps", dtype="bfloat16"):
    print(f"Loading {model_name} (dtype={dtype})...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, device_map=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded on {device}, {n_params:.1f}B params")
    return model, tokenizer


def _tokenize(tokenizer, text, chat_template):
    """Tokenize with or without chat-template wrapping.

    When chat_template=True the text is wrapped as a user turn with empty
    system message and no generation prompt, so hidden states reflect the
    model reading the text as user input inside its deployed chat context.
    """
    if chat_template:
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=False,
        )
        return tokenizer(wrapped, return_tensors="pt")
    return tokenizer(text, return_tensors="pt")


def hidden_states_for_text(model, tokenizer, text, device, split_prefix=None, chat_template=False):
    """Run text through model; return hidden states averaged over response tokens.

    If split_prefix is provided, the response-token span is everything after
    the tokens corresponding to split_prefix in the full tokenization.
    Otherwise, all tokens are averaged (with position 0 skipped).

    chat_template=True wraps the text as a user turn before tokenizing.

    Returns: tensor of shape (n_layers+1, hidden_dim), dtype float32 on CPU.
    """
    inputs = _tokenize(tokenizer, text, chat_template).to(device)
    n_total = inputs["input_ids"].shape[1]

    if split_prefix is not None and split_prefix != "":
        prefix_inputs = _tokenize(tokenizer, split_prefix, chat_template)
        if chat_template:
            # Chat-template wrapping adds an end-of-user-message marker at the
            # end of the truncated content. Find the point where the truncated
            # and full tokenizations diverge — that's where the response starts.
            p_ids = prefix_inputs["input_ids"][0].tolist()
            f_ids = inputs["input_ids"][0].tolist()
            diverge = min(len(p_ids), len(f_ids))
            for i in range(diverge):
                if p_ids[i] != f_ids[i]:
                    diverge = i
                    break
            start = min(diverge, n_total - 1)
        else:
            n_prefix = prefix_inputs["input_ids"].shape[1]
            start = min(n_prefix, n_total - 1)
    else:
        start = 1 if n_total > 1 else 0

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    avg_states = torch.stack([
        hs[0, start:n_total, :].mean(dim=0) for hs in outputs.hidden_states
    ])  # (n_layers+1, hidden_dim)

    return avg_states.cpu().float()


def extract_trait_activations(model, tokenizer, trait_data, prefix_mode, device, verbose=True, chat_template=False):
    """For a single trait, run all contrast pairs through the model.

    Returns:
      per_pair_high: (n_pairs, n_layers+1, hidden_dim)
      per_pair_low:  (n_pairs, n_layers+1, hidden_dim)
    """
    pairs = trait_data["pairs"]
    high_desc = trait_data["high_descriptor"]
    low_desc = trait_data["low_descriptor"]

    prefix = PREFIX_TEMPLATES[prefix_mode].format(
        high_descriptor=high_desc, low_descriptor=low_desc,
    )

    high_acts, low_acts = [], []
    for i, pair in enumerate(pairs):
        # The split_prefix we use to find response tokens is the prefix + situation.
        # Response tokens are whatever comes after.
        split = prefix + pair["situation"] + " "

        high_text = prefix + pair["situation"] + " " + pair["high"]
        low_text  = prefix + pair["situation"] + " " + pair["low"]

        h = hidden_states_for_text(model, tokenizer, high_text, device,
                                   split_prefix=split, chat_template=chat_template)
        l = hidden_states_for_text(model, tokenizer, low_text,  device,
                                   split_prefix=split, chat_template=chat_template)

        high_acts.append(h)
        low_acts.append(l)

        if verbose and (i + 1) % 10 == 0:
            print(f"    Pair {i+1}/{len(pairs)} done")

    return torch.stack(high_acts), torch.stack(low_acts)


def extract_neutral_activations(model, tokenizer, texts, device, verbose=True, chat_template=False):
    """Run neutral texts through model, return per-layer token-averaged activations.

    For neutral texts we have no 'response' — average all content tokens (skip BOS).
    Returns (n_texts, n_layers+1, hidden_dim).
    """
    acts = []
    for i, t in enumerate(texts):
        a = hidden_states_for_text(model, tokenizer, t, device,
                                   split_prefix=None, chat_template=chat_template)
        acts.append(a)
        if verbose and (i + 1) % 20 == 0:
            print(f"    Neutral {i+1}/{len(texts)} done")
    return torch.stack(acts)


def compute_pc_projection(neutral_activations_layer, variance_threshold=0.5):
    """Compute top PCs that together explain `variance_threshold` of variance.

    neutral_activations_layer: (n_texts, hidden_dim)
    Returns: (k, hidden_dim) matrix of top-k PC directions (unit-norm).
    """
    X = neutral_activations_layer.numpy().astype(np.float64)
    X_c = X - X.mean(axis=0, keepdims=True)

    # SVD instead of full covariance — faster and numerically stabler for high-d data
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    var = S ** 2
    var_frac = var / var.sum() if var.sum() > 0 else var
    cumvar = np.cumsum(var_frac)
    # Number of PCs to reach threshold (at least 1)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = max(1, min(k, len(var)))
    return Vt[:k], var_frac[:k], k


def project_out_pcs(direction, pcs):
    """Remove the span of pcs from direction.

    direction: (hidden_dim,) or (n_layers, hidden_dim)
    pcs: (k, hidden_dim) rows are unit-norm PC directions
    Returns direction with pc components subtracted.
    """
    if direction.ndim == 1:
        for pc in pcs:
            pc_u = pc / (np.linalg.norm(pc) + 1e-12)
            direction = direction - np.dot(direction, pc_u) * pc_u
        return direction
    else:
        out = direction.copy()
        for li in range(out.shape[0]):
            out[li] = project_out_pcs(out[li], pcs)
        return out


def load_neutral_texts(variant):
    if variant == "none":
        return None
    with open(NEUTRAL_TEXTS) as f:
        data = json.load(f)
    if variant == "scenario_setups":
        # Populate from contrast_pairs.json at runtime
        with open(CONTRAST_PAIRS) as cp_f:
            cp = json.load(cp_f)
        texts = []
        for td in cp["traits"].values():
            for p in td["pairs"]:
                texts.append(p["situation"])
        return texts
    if variant not in data["variants"]:
        raise ValueError(f"Unknown neutral variant: {variant}")
    texts = data["variants"][variant]["texts"]
    if len(texts) == 0:
        raise ValueError(f"Neutral variant '{variant}' has no texts")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Anthropic-style mean-diff trait extraction")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trait", type=str, default=None, help="H/E/X/A/C/O or None for all")
    parser.add_argument("--prefix-mode", choices=list(PREFIX_TEMPLATES.keys()), default="high")
    parser.add_argument("--neutral-variant",
                        choices=["scenario_setups", "shaggy_dog", "factual", "medium_trait", "none"],
                        default="scenario_setups")
    parser.add_argument("--variance-threshold", type=float, default=0.5,
                        help="Cumulative variance fraction for PC projection (default: 0.5)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dir", default="results/repe/meandiff")
    parser.add_argument("--input-file", default=str(CONTRAST_PAIRS),
                        help="Contrast-pairs JSON to extract from. Output filename includes "
                             "the input file's stem so results don't collide.")
    parser.add_argument("--skip-neutral", action="store_true",
                        help="Skip neutral-corpus extraction (e.g., when running on holdout, "
                             "neutral activations from the training run can be reused via PC "
                             "projection done downstream)")
    parser.add_argument("--chat-template", action="store_true",
                        help="Wrap each input as a user turn in the model's chat template "
                             "before extracting activations. Default is bare text. Instruct-"
                             "tuned models deploy under a chat template, so this is the "
                             "measurement-faithful setting.")
    args = parser.parse_args()

    # Load inputs
    with open(args.input_file) as f:
        cp = json.load(f)
    input_stem = Path(args.input_file).stem  # e.g., "contrast_pairs" or "contrast_pairs_holdout"
    traits = cp["traits"]
    if args.trait:
        traits = {args.trait: traits[args.trait]}

    neutral_texts = load_neutral_texts(args.neutral_variant)
    if neutral_texts is not None:
        print(f"Using neutral variant '{args.neutral_variant}' with {len(neutral_texts)} texts")
        if len(neutral_texts) < 100:
            print(f"  WARNING: only {len(neutral_texts)} neutral texts; top-PC estimation may be noisy in high-d space")

    # Load model
    model, tokenizer = load_model(args.model, args.device, args.dtype)

    # Extract neutral activations once (shared across traits for this model)
    neutral_acts = None
    if neutral_texts is not None and not args.skip_neutral:
        print(f"\nExtracting neutral corpus activations...")
        t0 = time.time()
        neutral_acts = extract_neutral_activations(model, tokenizer, neutral_texts, args.device,
                                                   chat_template=args.chat_template)
        print(f"  Done in {time.time()-t0:.1f}s  shape={tuple(neutral_acts.shape)}")
    elif args.skip_neutral:
        print(f"\nSkipping neutral corpus (--skip-neutral). Per-pair PC projection will not be applied.")

    # Output setup
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    for trait_id, trait_data in traits.items():
        print(f"\n=== {trait_data['name']} ({trait_id})  prefix={args.prefix_mode}  neutral={args.neutral_variant} ===")
        t0 = time.time()

        per_pair_high, per_pair_low = extract_trait_activations(
            model, tokenizer, trait_data, args.prefix_mode, args.device,
            chat_template=args.chat_template,
        )
        print(f"  Extracted activations in {time.time()-t0:.1f}s  shape={tuple(per_pair_high.shape)}")

        # Mean-diff direction per layer
        mean_high = per_pair_high.mean(dim=0)  # (n_layers+1, hidden_dim)
        mean_low  = per_pair_low.mean(dim=0)
        raw_direction = (mean_high - mean_low).numpy()  # (n_layers+1, hidden_dim)

        # PC projection, layer by layer
        projected_direction = None
        pc_info = None
        if neutral_acts is not None:
            n_layers = raw_direction.shape[0]
            projected_direction = np.zeros_like(raw_direction)
            ks = []
            for li in range(n_layers):
                # Get neutral activations at this layer
                n_layer = neutral_acts[:, li, :]  # (n_neutral, hidden_dim)
                try:
                    pcs, _, k = compute_pc_projection(n_layer, args.variance_threshold)
                    projected_direction[li] = project_out_pcs(raw_direction[li], pcs)
                    ks.append(k)
                except Exception as e:
                    print(f"    layer {li}: PC projection failed ({e}); using raw direction")
                    projected_direction[li] = raw_direction[li]
                    ks.append(0)
            pc_info = {"ks_per_layer": ks, "variance_threshold": args.variance_threshold}
            print(f"  PC projection: k across layers min={min(ks)} max={max(ks)} median={int(np.median(ks))}")

        # Two-thirds-depth layer (canonical Anthropic choice)
        n_layers_total = raw_direction.shape[0]
        two_thirds_layer = int(round(n_layers_total * 2 / 3))

        # Suffix file with input stem when not running on the canonical training set
        # so holdout extraction doesn't collide with training extraction.
        stem_suffix = "" if input_stem == "contrast_pairs" else f"_{input_stem}"
        format_tag = "chat" if args.chat_template else "bare"
        out_file = outdir / (
            f"{safe_model}_{trait_id}"
            f"_fmt-{format_tag}"
            f"_prefix-{args.prefix_mode}"
            f"_neutral-{args.neutral_variant}"
            f"{stem_suffix}.pt"
        )
        torch.save({
            "trait": trait_id,
            "trait_name": trait_data["name"],
            "model": args.model,
            "n_pairs": len(trait_data["pairs"]),
            "format": format_tag,
            "prefix_mode": args.prefix_mode,
            "neutral_variant": args.neutral_variant,
            "raw_direction": torch.from_numpy(raw_direction),
            "projected_direction": (torch.from_numpy(projected_direction)
                                    if projected_direction is not None else None),
            "mean_high": mean_high,
            "mean_low": mean_low,
            "per_pair_high": per_pair_high.to(torch.bfloat16),
            "per_pair_low":  per_pair_low.to(torch.bfloat16),
            "two_thirds_layer": two_thirds_layer,
            "n_layers_total": n_layers_total,
            "pc_info": pc_info,
        }, out_file)
        print(f"  Saved to {out_file}")
        print(f"  Two-thirds depth layer = {two_thirds_layer} (of {n_layers_total})")

    print("\nDone.")


if __name__ == "__main__":
    main()
