#!/usr/bin/env python3
"""Extract residual stream activations on HEXACO contrast pairs.

For each HEXACO trait, runs the high/low contrast pairs through the model
and saves per-pair, per-layer activation differences (raw_diffs) to disk.

Trait directions themselves are fit downstream via LDA on the saved raw_diffs
(see validate_protocol.py, cross_method_matrix.py, optimize_steering.py).
This script does not produce directions; it just collects the activations.

Activations are taken at the last token of each prompt. The contrast-pair
prompt templates are constructed so the last token is the sentence-ending
period — this is the "period-token protocol" referenced in the reports.

Usage:
    python scripts/extract_trait_vectors.py --model google/gemma-3-4b-it
    python scripts/extract_trait_vectors.py --model google/gemma-3-4b-it --trait H
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CONTRAST_PAIRS = "instruments/contrast_pairs.json"

# Template for wrapping contrast pairs into prompts.
# The {response} field is expected to end in sentence-terminal punctuation,
# so the last token (which we take as the measurement site) is the period.
CONTRAST_TEMPLATE_HIGH = (
    'Consider a person who is {descriptor}. '
    '{situation} {response}'
)
CONTRAST_TEMPLATE_LOW = (
    'Consider a person who is {descriptor}. '
    '{situation} {response}'
)


def load_model(model_name, device="mps", dtype="bfloat16"):
    """Load model and tokenizer, optimized for Apple Silicon."""
    print(f"Loading {model_name} (dtype={dtype})...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    print(f"  Loaded on {device}, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer


def get_hidden_states(model, tokenizer, text, device="mps"):
    """Run text through model and return hidden states at all layers.

    Returns tensor of shape (n_layers+1, hidden_dim) — the last token's
    hidden state at each layer. Prompts are constructed to end in a period,
    so the last token is the period-token measurement site.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers+1,) tensors, each (batch, seq_len, hidden_dim)
    # [0] is embeddings, [1:] are layer outputs
    hidden_states = outputs.hidden_states
    last_token_states = torch.stack([
        hs[0, -1, :] for hs in hidden_states
    ])  # (n_layers+1, hidden_dim)

    return last_token_states.cpu().float()


def extract_raw_diffs(model, tokenizer, trait_data, device="mps", verbose=True):
    """Extract per-pair high-vs-low activation differences at each layer.

    Returns:
        raw_diffs: tensor (n_pairs, n_layers+1, hidden_dim) — per-pair diffs
                   (high activation minus low activation) at each layer.
                   Downstream code fits LDA on these.
    """
    pairs = trait_data["pairs"]
    high_desc = trait_data["high_descriptor"]
    low_desc = trait_data["low_descriptor"]

    diffs = []  # list of (n_layers+1, hidden_dim) tensors

    for i, pair in enumerate(pairs):
        high_text = CONTRAST_TEMPLATE_HIGH.format(
            descriptor=high_desc,
            situation=pair["situation"],
            response=pair["high"],
        )
        low_text = CONTRAST_TEMPLATE_LOW.format(
            descriptor=low_desc,
            situation=pair["situation"],
            response=pair["low"],
        )

        high_states = get_hidden_states(model, tokenizer, high_text, device)
        low_states = get_hidden_states(model, tokenizer, low_text, device)

        diff = high_states - low_states  # (n_layers+1, hidden_dim)
        diffs.append(diff)

        if verbose and (i + 1) % 5 == 0:
            print(f"    Pair {i+1}/{len(pairs)} done")

    # Stack: (n_pairs, n_layers+1, hidden_dim)
    return torch.stack(diffs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract residual stream activations on HEXACO contrast pairs"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--trait", type=str, default=None,
                        help="Single trait to extract (H/E/X/A/C/O). Default: all")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype (bfloat16/float16/float32)")
    parser.add_argument("--output-dir", type=str, default="results/repe")
    parser.add_argument("--input-file", type=str, default=CONTRAST_PAIRS,
                        help="Contrast-pairs JSON to extract from. Output filename is "
                             "suffixed with the input file's stem when not the canonical "
                             "training file.")
    args = parser.parse_args()

    # Load contrast pairs
    with open(args.input_file) as f:
        contrast_data = json.load(f)
    traits = contrast_data["traits"]
    input_stem = Path(args.input_file).stem

    if args.trait:
        traits = {args.trait: traits[args.trait]}

    # Load model
    model, tokenizer = load_model(args.model, args.device, args.dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.model.replace("/", "_")
    stem_suffix = "" if input_stem == "contrast_pairs" else f"_{input_stem}"

    for trait_id, trait_data in traits.items():
        print(f"\n=== Extracting {trait_data['name']} ({trait_id}) ===")
        t0 = time.time()

        raw_diffs = extract_raw_diffs(model, tokenizer, trait_data, args.device)

        elapsed = time.time() - t0
        print(f"  Extracted in {elapsed:.1f}s  shape={tuple(raw_diffs.shape)}")

        # Save raw diffs. Filename kept as "_directions.pt" for back-compat
        # with downstream loaders, but the content is per-pair activation
        # differences, not a single direction vector.
        out_file = output_dir / f"{safe_model}_{trait_id}_directions{stem_suffix}.pt"
        torch.save({
            "trait": trait_id,
            "trait_name": trait_data["name"],
            "model": args.model,
            "n_pairs": len(trait_data["pairs"]),
            "raw_diffs": raw_diffs,
        }, out_file)
        print(f"  Saved to {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
