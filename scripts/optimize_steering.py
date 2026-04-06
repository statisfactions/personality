#!/usr/bin/env python3
"""Optimize a steering vector via backpropagation.

Instead of extracting a direction from contrast pairs (RepE/LDA), this
optimizes a perturbation vector δ added to the residual stream at a target
layer to maximize log-odds of the high-trait forced-choice response.

This tests whether the read/write dissociation is fundamental (no linear
perturbation at natural scale can steer) or just a property of the LDA
direction specifically.

Usage:
    python scripts/optimize_steering.py --model google/gemma-3-4b-it --trait H --layer 14
    python scripts/optimize_steering.py --model meta-llama/Llama-3.2-3B-Instruct --trait H --layer 12

Memory: runs one layer at a time with gradient checkpointing. Expects ~10-12GB
on a 4B model in bf16.
"""

import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

CONTRAST_PAIRS = "instruments/contrast_pairs.json"


def load_model(model_name, device="mps", dtype=torch.bfloat16):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    # Freeze all model parameters — we only optimize δ
    for p in model.parameters():
        p.requires_grad_(False)
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print(f"  Loaded, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params (frozen)")
    return model, tokenizer


def get_layer_module(model, model_name, layer_idx):
    """Get the transformer layer module for hook registration."""
    # Try common paths
    for path in ["model.layers", "model.language_model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj[layer_idx]
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Can't find layer {layer_idx} in {model_name}")


def build_fc_prompt(scenario, high_response, low_response):
    """Build a forced-choice prompt for a contrast pair."""
    return (
        f"Consider what a person most like you would do in the following situation: "
        f"{scenario}\n\n"
        f"Which would you do?\n"
        f"A) {high_response}\n"
        f"B) {low_response}\n\n"
        f"Respond with just A or B.\n"
        f"Answer:"
    )


def compute_fc_logodds(model, tokenizer, prompts, delta, layer_module, device):
    """Compute mean log-odds(A) - log-odds(B) with delta added at target layer.

    Returns scalar tensor with gradient attached to delta.
    """
    # Hook to add delta to residual stream output
    handle = None
    def hook_fn(module, input, output):
        # output is typically a tuple; first element is hidden states
        if isinstance(output, tuple):
            hidden = output[0]
            # Add delta to all token positions
            modified = hidden + delta.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            return output + delta.unsqueeze(0).unsqueeze(0)

    handle = layer_module.register_forward_hook(hook_fn)

    total_logodds = torch.tensor(0.0, device=device, dtype=torch.float32)
    n_valid = 0

    # Find token IDs for A and B
    # Try various encodings
    a_candidates = ["A", " A", "A)", " A)"]
    b_candidates = ["B", " B", "B)", " B)"]

    a_id = b_id = None
    for tok in a_candidates:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            a_id = ids[-1]  # take last token in case of multi-token
            break
    for tok in b_candidates:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            b_id = ids[-1]
            break

    if a_id is None or b_id is None:
        handle.remove()
        raise ValueError(f"Can't find A/B token IDs. a_id={a_id}, b_id={b_id}")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Forward pass (gradient flows through hook → delta)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :]  # last token logits

        # Log-odds of A vs B
        log_odds = logits[a_id].float() - logits[b_id].float()
        total_logodds = total_logodds + log_odds
        n_valid += 1

        del outputs, inputs
        # Don't empty cache here — we need the computation graph for backprop

    handle.remove()

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_logodds / n_valid


def optimize_delta(model, tokenizer, pairs, layer_module, hidden_size,
                   device="mps", n_steps=100, lr=0.1, batch_size=5,
                   norm_constraint=None):
    """Optimize steering vector delta via gradient ascent on FC log-odds.

    Args:
        pairs: list of (scenario, high_response, low_response)
        layer_module: the transformer layer to perturb
        hidden_size: dimension of the residual stream
        n_steps: optimization steps
        lr: learning rate
        batch_size: scenarios per step (random sample)
        norm_constraint: if set, project delta to this L2 norm after each step.
                        If None, use unconstrained optimization and report norm.
    """
    # Initialize delta as small random vector
    delta = torch.randn(hidden_size, device=device, dtype=torch.float32) * 0.01
    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    # Build all FC prompts
    all_prompts = [build_fc_prompt(s, h, l) for s, h, l in pairs]

    print(f"\n  Optimizing δ (dim={hidden_size}, lr={lr}, batch={batch_size}, "
          f"norm_constraint={norm_constraint})")
    print(f"  {'step':>5s}  {'loss':>8s}  {'|δ|':>8s}  {'log-odds':>10s}")
    print(f"  {'-' * 40}")

    best_logodds = -float('inf')
    best_delta = None

    for step in range(n_steps):
        optimizer.zero_grad()

        # Random batch of scenarios
        idx = np.random.choice(len(all_prompts), size=min(batch_size, len(all_prompts)),
                               replace=False)
        batch_prompts = [all_prompts[i] for i in idx]

        # Forward + compute log-odds (gradient attached to delta)
        mean_logodds = compute_fc_logodds(
            model, tokenizer, batch_prompts, delta, layer_module, device
        )

        # Maximize log-odds → minimize negative log-odds
        loss = -mean_logodds
        loss.backward()

        optimizer.step()

        # Project delta to norm constraint if specified
        if norm_constraint is not None:
            with torch.no_grad():
                current_norm = delta.norm().item()
                if current_norm > norm_constraint:
                    delta.data = delta.data * (norm_constraint / current_norm)

        # Logging
        delta_norm = delta.norm().item()
        lo_val = mean_logodds.item()

        if lo_val > best_logodds:
            best_logodds = lo_val
            best_delta = delta.detach().clone()

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  {step:5d}  {loss.item():8.3f}  {delta_norm:8.3f}  {lo_val:+10.3f}")

        # Memory cleanup
        del loss, mean_logodds
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    return best_delta, best_logodds


def evaluate_delta(model, tokenizer, pairs, delta, layer_module, device="mps"):
    """Evaluate a delta on all pairs, reporting per-scenario results."""
    all_prompts = [build_fc_prompt(s, h, l) for s, h, l in pairs]

    print(f"\n  Evaluating δ (|δ|={delta.norm().item():.3f}) on {len(pairs)} scenarios")

    # Evaluate with delta
    with torch.no_grad():
        # We need the hook but no gradients
        delta_eval = delta.clone().requires_grad_(False)

    # Can't use no_grad with hooks that modify the computation,
    # so we do eval without grad on the delta
    results_with = []
    results_without = []

    for label, d, results in [("no steer", None, results_without),
                               ("+ δ", delta, results_with)]:
        handle = None
        if d is not None:
            def make_hook(dv):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        return (hidden + dv.unsqueeze(0).unsqueeze(0),) + output[1:]
                    return output + dv.unsqueeze(0).unsqueeze(0)
                return hook_fn
            handle = layer_module.register_forward_hook(make_hook(d.detach()))

        a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
        b_id = tokenizer.encode("B", add_special_tokens=False)[-1]

        for prompt in all_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]
            lo = (logits[a_id] - logits[b_id]).float().item()
            results.append(lo)
            del outputs, inputs

        if handle:
            handle.remove()

        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # Report
    print(f"\n  {'#':>3s}  {'baseline':>10s}  {'steered':>10s}  {'shift':>8s}  scenario")
    print(f"  {'-' * 70}")

    shifts = []
    for i, (s, h, l) in enumerate(pairs):
        base = results_without[i]
        steered = results_with[i]
        shift = steered - base
        shifts.append(shift)
        base_pick = "A(high)" if base > 0 else "B(low)"
        steer_pick = "A(high)" if steered > 0 else "B(low)"
        flip = " FLIP" if (base > 0) != (steered > 0) else ""
        print(f"  {i:3d}  {base:+10.2f}  {steered:+10.2f}  {shift:+8.2f}  "
              f"{s[:45]}{flip}")

    n_flips = sum(1 for i in range(len(pairs))
                  if (results_without[i] > 0) != (results_with[i] > 0))
    mean_shift = np.mean(shifts)
    print(f"\n  Mean shift: {mean_shift:+.3f}")
    print(f"  Flips: {n_flips}/{len(pairs)}")
    print(f"  Baseline: {sum(1 for x in results_without if x > 0)}/{len(pairs)} high-trait")
    print(f"  Steered:  {sum(1 for x in results_with if x > 0)}/{len(pairs)} high-trait")

    return results_without, results_with


def main():
    parser = argparse.ArgumentParser(description="Optimize steering vector via backprop")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--trait", default="H", help="HEXACO trait (H/E/X/A/C/O)")
    parser.add_argument("--layer", type=int, required=True, help="Layer to perturb")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=5, help="Scenarios per step")
    parser.add_argument("--norm", type=float, default=None,
                        help="L2 norm constraint on δ (default: unconstrained)")
    parser.add_argument("--n-train", type=int, default=30,
                        help="Number of contrast pairs for training (rest for eval)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--save", default=None, help="Save optimized δ to this path")
    args = parser.parse_args()

    # Load contrast pairs
    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    trait_data = cp["traits"][args.trait]
    all_pairs = [(p["situation"], p["high"], p["low"]) for p in trait_data["pairs"]]

    # Train/eval split
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]
    print(f"Trait: {trait_data['name']} ({args.trait})")
    print(f"Pairs: {len(train_pairs)} train, {len(eval_pairs)} eval")

    # Load model
    model, tokenizer = load_model(args.model, args.device)
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}, target layer: {args.layer}")

    # Get layer module
    layer_module = get_layer_module(model, args.model, args.layer)
    print(f"Layer module: {type(layer_module).__name__}")

    # Optimize
    t0 = time.time()
    best_delta, best_logodds = optimize_delta(
        model, tokenizer, train_pairs, layer_module, hidden_size,
        device=args.device, n_steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, norm_constraint=args.norm,
    )
    elapsed = time.time() - t0
    print(f"\n  Optimization done in {elapsed:.0f}s")
    print(f"  Best mean log-odds: {best_logodds:+.3f}")
    print(f"  |δ|: {best_delta.norm().item():.3f}")

    # Compare with LDA direction
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        safe = args.model.replace("/", "_")
        data = torch.load(f"results/repe/{safe}_{args.trait}_directions.pt",
                         weights_only=False)
        diffs = data["raw_diffs"]
        d = diffs[:, args.layer, :].numpy()
        n_pairs_lda = d.shape[0]
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_pairs_lda + [0] * n_pairs_lda)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        lda_d = lda.coef_[0]
        lda_d = lda_d / np.linalg.norm(lda_d)

        delta_np = best_delta.cpu().float().numpy()
        delta_unit = delta_np / np.linalg.norm(delta_np)
        cosine = np.dot(delta_unit, lda_d)
        print(f"\n  Cosine(δ_opt, LDA direction): {cosine:.4f}")
    except Exception as e:
        print(f"\n  Could not compare with LDA: {e}")

    # Evaluate on held-out pairs
    if eval_pairs:
        print(f"\n  === Evaluation on {len(eval_pairs)} held-out pairs ===")
        evaluate_delta(model, tokenizer, eval_pairs, best_delta, layer_module, args.device)

    # Also evaluate on train pairs for comparison
    print(f"\n  === Evaluation on {len(train_pairs)} training pairs ===")
    evaluate_delta(model, tokenizer, train_pairs, best_delta, layer_module, args.device)

    # Save
    if args.save:
        torch.save({
            "delta": best_delta.cpu(),
            "model": args.model,
            "trait": args.trait,
            "layer": args.layer,
            "norm": best_delta.norm().item(),
            "best_logodds": best_logodds,
            "n_train": len(train_pairs),
            "steps": args.steps,
            "lr": args.lr,
        }, args.save)
        print(f"\n  Saved to {args.save}")


if __name__ == "__main__":
    main()
