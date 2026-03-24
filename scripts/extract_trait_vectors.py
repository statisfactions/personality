#!/usr/bin/env python3
"""Extract personality trait direction vectors via representation engineering.

For each HEXACO trait, constructs contrast pairs (high vs low trait descriptions),
runs them through a model, extracts residual stream activations at each layer,
and computes the trait direction via PCA on the activation differences.

Then tests whether those directions activate on HEXACO/IPIP survey items.

Usage:
    python scripts/extract_trait_vectors.py --model google/gemma-3-4b-it
    python scripts/extract_trait_vectors.py --model google/gemma-3-4b-it --trait H --layers 15,20,25
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


CONTRAST_PAIRS = "instruments/contrast_pairs.json"
HEXACO_FILE = "instruments/hexaco100.json"

# Template for wrapping contrast pairs into prompts
CONTRAST_TEMPLATE_HIGH = (
    'Consider a person who is {descriptor}. '
    '{situation} {response}'
)
CONTRAST_TEMPLATE_LOW = (
    'Consider a person who is {descriptor}. '
    '{situation} {response}'
)

# Template for survey items (to test if trait vectors activate)
SURVEY_TEMPLATE = (
    'Rate how accurately each statement describes you.\n'
    '1 = very inaccurate, 2 = moderately inaccurate, 3 = neither, '
    '4 = moderately accurate, 5 = very accurate\n'
    'Respond with only a number.\n\n'
    'Statement: "{item_text}"\nRating: '
)


def load_model(model_name, device="mps"):
    """Load model and tokenizer, optimized for Apple Silicon."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    print(f"  Loaded on {device}, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer


def get_hidden_states(model, tokenizer, text, device="mps"):
    """Run text through model and return hidden states at all layers.

    Returns tensor of shape (n_layers, hidden_dim) — the last token's
    hidden state at each layer.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers+1,) tensors, each (batch, seq_len, hidden_dim)
    # [0] is embeddings, [1:] are layer outputs
    # Take the last token position from each layer
    hidden_states = outputs.hidden_states
    last_token_states = torch.stack([
        hs[0, -1, :] for hs in hidden_states
    ])  # (n_layers+1, hidden_dim)

    return last_token_states.cpu().float()


def extract_trait_direction(model, tokenizer, trait_data, device="mps", verbose=True):
    """Extract a trait direction vector via PCA on contrast pair activation differences.

    Returns:
        directions: tensor (n_layers+1, hidden_dim) — the trait direction at each layer
        explained_variance: tensor (n_layers+1,) — fraction of variance explained by PC1
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
    diffs = torch.stack(diffs)

    n_pairs, n_layers, hidden_dim = diffs.shape

    # For each layer, compute PCA-1 on the diffs
    directions = torch.zeros(n_layers, hidden_dim)
    explained_variance = torch.zeros(n_layers)

    for layer in range(n_layers):
        layer_diffs = diffs[:, layer, :]  # (n_pairs, hidden_dim)

        # Center
        mean_diff = layer_diffs.mean(dim=0)
        centered = layer_diffs - mean_diff

        # PCA via SVD (use float64 for numerical stability)
        try:
            _, S, Vt = torch.linalg.svd(centered.double(), full_matrices=False)
            directions[layer] = Vt[0].float()
            total_var = (S ** 2).sum()
            explained_variance[layer] = (S[0] ** 2) / total_var if total_var > 0 else 0
        except torch._C._LinAlgError:
            # SVD failed (ill-conditioned) — use mean diff as fallback direction
            fallback = layer_diffs.mean(dim=0)
            norm = fallback.norm()
            directions[layer] = fallback / norm if norm > 0 else fallback
            explained_variance[layer] = 0.0

    return directions, explained_variance, diffs.mean(dim=0)  # also return mean diff


def project_survey_items(model, tokenizer, directions, mean_diffs, items, device="mps",
                         layers=None, verbose=True):
    """Project survey item activations onto trait direction vectors.

    Returns dict of item_id -> {layer -> projection_value}
    """
    if layers is None:
        n_layers = directions.shape[0]
        # Default to middle and late layers (where behavioral properties tend to live)
        layers = list(range(n_layers // 3, n_layers))

    projections = {}

    for i, (item_id, item_text) in enumerate(items):
        prompt = SURVEY_TEMPLATE.format(item_text=item_text)
        states = get_hidden_states(model, tokenizer, prompt, device)

        item_proj = {}
        for layer in layers:
            # Project activation onto trait direction
            direction = directions[layer]
            direction_norm = direction / direction.norm()
            proj = torch.dot(states[layer], direction_norm).item()
            item_proj[layer] = proj

        projections[item_id] = item_proj

        if verbose and (i + 1) % 20 == 0:
            print(f"    Item {i+1}/{len(items)} projected")

    return projections


def main():
    parser = argparse.ArgumentParser(description="Extract personality trait vectors via RepE")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--trait", type=str, default=None,
                        help="Single trait to extract (H/E/X/A/C/O). Default: all")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices for projection (default: middle-to-end)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="results/repe")
    parser.add_argument("--skip-survey", action="store_true",
                        help="Skip survey item projection (just extract directions)")
    args = parser.parse_args()

    # Load contrast pairs
    with open(CONTRAST_PAIRS) as f:
        contrast_data = json.load(f)
    traits = contrast_data["traits"]

    if args.trait:
        traits = {args.trait: traits[args.trait]}

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Parse layers
    proj_layers = None
    if args.layers:
        proj_layers = [int(x) for x in args.layers.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.model.replace("/", "_")

    for trait_id, trait_data in traits.items():
        print(f"\n=== Extracting {trait_data['name']} ({trait_id}) ===")
        t0 = time.time()

        # Extract direction
        directions, explained_var, mean_diffs = extract_trait_direction(
            model, tokenizer, trait_data, args.device
        )

        elapsed = time.time() - t0
        print(f"  Extracted in {elapsed:.1f}s")

        # Report best layers by explained variance
        n_layers = directions.shape[0]
        print(f"\n  Explained variance by PC1 (top 10 layers):")
        sorted_layers = torch.argsort(explained_var, descending=True)
        for rank, layer_idx in enumerate(sorted_layers[:10]):
            print(f"    Layer {layer_idx:3d}: {explained_var[layer_idx]:.3f}")

        # Save direction vectors
        direction_file = output_dir / f"{safe_model}_{trait_id}_directions.pt"
        torch.save({
            "trait": trait_id,
            "trait_name": trait_data["name"],
            "model": args.model,
            "directions": directions,
            "explained_variance": explained_var,
            "mean_diffs": mean_diffs,
            "n_pairs": len(trait_data["pairs"]),
        }, direction_file)
        print(f"  Saved directions to {direction_file}")

        # Project survey items
        if not args.skip_survey:
            print(f"\n  Projecting HEXACO items onto {trait_data['name']} direction...")
            with open(HEXACO_FILE) as f:
                hexaco = json.load(f)

            # Get items for this trait's scale
            scale_def = hexaco["scales"].get(trait_id)
            if scale_def:
                scale_items = [
                    (iid, hexaco["items"][iid]["text"])
                    for iid in scale_def["item_ids"]
                ]
                reverse_keyed = set(scale_def["reverse_keyed_item_ids"])
            else:
                scale_items = []
                reverse_keyed = set()

            # Also get items from other scales for discriminant validity
            other_items = []
            for other_id, other_scale in hexaco["scales"].items():
                if other_id != trait_id and other_id != "ALT":
                    for iid in other_scale["item_ids"][:4]:  # first 4 from each other scale
                        other_items.append((iid, hexaco["items"][iid]["text"]))

            all_items = scale_items + other_items

            projections = project_survey_items(
                model, tokenizer, directions, mean_diffs, all_items, args.device,
                layers=proj_layers
            )

            # Analyze: do on-scale items project more strongly than off-scale items?
            if proj_layers is None:
                # Use top-5 layers by explained variance
                analysis_layers = sorted_layers[:5].tolist()
            else:
                analysis_layers = proj_layers

            print(f"\n  Convergent validity (top layers by explained variance):")
            for layer in analysis_layers:
                on_scale = []
                off_scale = []
                for iid, _ in scale_items:
                    proj = projections[iid].get(layer, 0)
                    # Reverse-keyed items should project negatively
                    if iid in reverse_keyed:
                        proj = -proj
                    on_scale.append(proj)
                for iid, _ in other_items:
                    off_scale.append(abs(projections[iid].get(layer, 0)))

                on_mean = sum(on_scale) / len(on_scale) if on_scale else 0
                off_mean = sum(off_scale) / len(off_scale) if off_scale else 0
                print(f"    Layer {layer:3d}: on-scale mean={on_mean:+.3f}, "
                      f"off-scale mean={off_mean:.3f}, "
                      f"ratio={on_mean/off_mean:.2f}" if off_mean != 0
                      else f"    Layer {layer:3d}: on-scale mean={on_mean:+.3f}, off-scale=0")

            # Save projections
            proj_file = output_dir / f"{safe_model}_{trait_id}_projections.json"
            with open(proj_file, "w") as f:
                json.dump({
                    "trait": trait_id,
                    "model": args.model,
                    "analysis_layers": analysis_layers,
                    "scale_item_ids": [iid for iid, _ in scale_items],
                    "reverse_keyed_item_ids": list(reverse_keyed),
                    "other_item_ids": [iid for iid, _ in other_items],
                    "projections": {
                        iid: {str(k): v for k, v in projs.items()}
                        for iid, projs in projections.items()
                    },
                }, f, indent=2)
            print(f"  Saved projections to {proj_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
