#!/usr/bin/env python3
"""Score trait direction vectors on multiple evaluation dimensions.

Accepts direction files from either pipeline:
  * Legacy LDA: results/repe/<model>_<trait>_directions.pt
    — has `raw_diffs`; LDA is fit here at best-CV layer.
  * Mean-diff: results/repe/meandiff/<model>_<trait>_prefix-<p>_neutral-<n>.pt
    — has `raw_direction` and `projected_direction`.

Scoring dimensions:
  1. Held-out classification accuracy (per-pair signed projection sign)
  2. Convergent validity (sign of projection on matched HEXACO survey items)
  3. BC steering shift at 5% residual norm (requires model)
  4. Free-text steering judged by an LLM (requires model + judge; optional)

Dimensions 1 and 2 are activation-only (cheap). Dimensions 3 and 4 require
running the model with hooks — gated by --run-steering. Dimension 4 is
further gated by --run-freetext and requires a judge model configured via
--judge-ollama-model.

Usage:
    python scripts/score_directions.py --direction-file results/repe/meta-llama_Llama-3.2-3B-Instruct_H_directions.pt \
        --method lda
    python scripts/score_directions.py --direction-file results/repe/meandiff/...H_prefix-absent_neutral-factual.pt \
        --method meandiff --projection projected
    python scripts/score_directions.py --direction-file ... --method lda --run-steering
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

CONTRAST_PAIRS = Path("instruments/contrast_pairs.json")
HEXACO_FILE = Path("instruments/hexaco100.json")


# =============================================================================
# Direction loaders
# =============================================================================

def load_lda_direction(path, layer=None, return_per_pair_diffs=True):
    """Load an LDA direction (fit from raw_diffs in a legacy _directions.pt file).

    Returns dict with:
      direction: (hidden_dim,) unit-norm
      layer: int
      per_pair_diffs: (n_pairs, hidden_dim) at that layer (for classification scoring)
      source: "lda"
    """
    data = torch.load(path, weights_only=False)
    diffs = data["raw_diffs"]  # (n_pairs, n_layers, hidden_dim)
    n_pairs, n_layers, _ = diffs.shape

    if layer is None:
        # Select best layer by 5-fold CV accuracy
        best_acc, best_layer = 0.0, 0
        for l in range(n_layers):
            d = diffs[:, l, :].numpy()
            if np.any(np.isnan(d)) or np.all(d == 0):
                continue
            X = np.vstack([d / 2, -d / 2])
            y = np.array([1] * n_pairs + [0] * n_pairs)
            try:
                acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
                if acc > best_acc:
                    best_acc, best_layer = acc, l
            except Exception:
                pass
        layer = best_layer

    d = diffs[:, layer, :].numpy()
    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    direction = lda.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    return {
        "direction": direction,
        "layer": layer,
        "per_pair_diffs": d if return_per_pair_diffs else None,
        "source": "lda",
        "trait": data.get("trait"),
        "model": data.get("model"),
        "n_pairs": n_pairs,
    }


def load_meandiff_direction(path, projection="projected", layer=None, layer_strategy="best-signal"):
    """Load a mean-diff direction.

    projection: "raw" (no PC removal) or "projected" (PCs removed)
    layer_strategy:
      - "best-snr": layer with max signal-to-noise ratio of per-pair signed
        projections (mean / std). Norm-invariant — does not get fooled by
        residual-stream norm growth across layers.
      - "best-cv": layer with highest 5-fold CV LDA classification accuracy
        on per-pair diffs. Fully norm-invariant, equivalent to LDA pipeline.
      - "best-signal": [DEPRECATED — kept for back-compat] layer with max raw
        mean signed projection. Picks high-norm layers because signal scales
        with activation norm; not a meaningful signal quality measure.
      - "two-thirds": the two_thirds_layer recorded at extraction time
        (Anthropic emotion paper convention).
      - explicit int: use that layer
    """
    data = torch.load(path, weights_only=False)

    if projection == "projected":
        dir_tensor = data.get("projected_direction")
        if dir_tensor is None:
            dir_tensor = data["raw_direction"]
            print(f"  (projection='projected' requested but file has none; falling back to raw)")
    else:
        dir_tensor = data["raw_direction"]

    all_dirs = dir_tensor.numpy()  # (n_layers, hidden_dim)
    n_layers = all_dirs.shape[0]

    # Per-pair diffs: reconstruct from stored per_pair_high - per_pair_low
    pp_high = data["per_pair_high"].float()  # bf16 → fp32
    pp_low = data["per_pair_low"].float()
    diffs_all_layers = (pp_high - pp_low).numpy()  # (n_pairs, n_layers, hidden_dim)
    n_pairs = diffs_all_layers.shape[0]

    if isinstance(layer_strategy, int):
        layer = layer_strategy
    elif layer_strategy == "two-thirds":
        layer = data.get("two_thirds_layer", n_layers * 2 // 3)
    elif layer_strategy == "best-snr":
        # Norm-invariant: signal-to-noise ratio of per-pair signed projection
        best_snr, best_layer = -float("inf"), 0
        for l in range(n_layers):
            d_unit = all_dirs[l] / (np.linalg.norm(all_dirs[l]) + 1e-12)
            sig_per = diffs_all_layers[:, l, :] @ d_unit
            snr = sig_per.mean() / (sig_per.std() + 1e-12)
            if snr > best_snr:
                best_snr, best_layer = snr, l
        layer = best_layer
    elif layer_strategy == "best-cv":
        # 5-fold CV LDA accuracy on per-pair diffs (matches LDA-pipeline best-layer)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        best_acc, best_layer = 0.0, 0
        for l in range(n_layers):
            d = diffs_all_layers[:, l, :]
            if np.any(np.isnan(d)) or np.all(d == 0):
                continue
            X = np.vstack([d / 2, -d / 2])
            y = np.array([1] * n_pairs + [0] * n_pairs)
            try:
                acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
                if acc > best_acc:
                    best_acc, best_layer = acc, l
            except Exception:
                pass
        layer = best_layer
    elif layer_strategy == "best-signal":
        # DEPRECATED: kept for back-compat. Picks high-norm layers.
        best_sig, best_layer = -float("inf"), 0
        for l in range(n_layers):
            d_unit = all_dirs[l] / (np.linalg.norm(all_dirs[l]) + 1e-12)
            sig = (diffs_all_layers[:, l, :] @ d_unit).mean()
            if sig > best_sig:
                best_sig, best_layer = sig, l
        layer = best_layer
    else:
        raise ValueError(f"Unknown layer_strategy: {layer_strategy}")

    direction = all_dirs[layer]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    return {
        "direction": direction,
        "layer": layer,
        "per_pair_diffs": diffs_all_layers[:, layer, :],
        "source": f"meandiff-{projection}",
        "trait": data.get("trait"),
        "model": data.get("model"),
        "n_pairs": n_pairs,
        "prefix_mode": data.get("prefix_mode"),
        "neutral_variant": data.get("neutral_variant"),
    }


# =============================================================================
# Scoring dimension 1: classification accuracy
# =============================================================================

def score_classification(direction, per_pair_diffs):
    """Project per-pair diffs onto direction; return fraction with positive sign.

    Also returns mean signed projection (higher = stronger separation).
    """
    d_unit = direction / (np.linalg.norm(direction) + 1e-12)
    projs = per_pair_diffs @ d_unit
    acc = float((projs > 0).mean())
    return {
        "accuracy": acc,
        "mean_signed_proj": float(projs.mean()),
        "median_signed_proj": float(np.median(projs)),
        "frac_positive_signal": acc,
    }


# =============================================================================
# Scoring dimension 2: convergent validity on HEXACO survey items
# =============================================================================

def score_hexaco_convergent(direction, layer, model, tokenizer, trait_id, device="cpu"):
    """Project hidden state at survey-item prompts onto direction.

    Expected: items on-scale should project with correct sign (reverse-keyed negative).
    Off-scale items should project weaker (noise baseline).

    Returns: {'on_scale_correct_sign', 'on_scale_mean_proj', 'off_scale_mean_abs_proj'}
    """
    with open(HEXACO_FILE) as f:
        hexaco = json.load(f)

    scale_def = hexaco["scales"].get(trait_id)
    if scale_def is None:
        return {"error": f"no scale for {trait_id}"}

    on_scale_items = scale_def["item_ids"]
    reverse_keyed = set(scale_def["reverse_keyed_item_ids"])

    # Off-scale sample: 4 items from each other trait scale (same as validate_protocol)
    off_scale_items = []
    for other_id, other_scale in hexaco["scales"].items():
        if other_id != trait_id and other_id != "ALT":
            off_scale_items.extend(other_scale["item_ids"][:4])

    PROMPT = (
        'Rate how accurately each statement describes you.\n'
        '1 = very inaccurate, 2 = moderately inaccurate, 3 = neither, '
        '4 = moderately accurate, 5 = very accurate\n'
        'Respond with only a number.\n\n'
        'Statement: "{item_text}"\nRating: '
    )

    d_unit = direction / (np.linalg.norm(direction) + 1e-12)

    def project_item(iid):
        text = PROMPT.format(item_text=hexaco["items"][iid]["text"])
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        act = outputs.hidden_states[layer][0, -1, :].float().cpu().numpy()
        del outputs
        return float(act @ d_unit)

    on_projs, off_projs = [], []
    correct_signs = 0
    for iid in on_scale_items:
        p = project_item(iid)
        if iid in reverse_keyed:
            p = -p  # expect negative for reverse-keyed
        on_projs.append(p)
        if p > 0:
            correct_signs += 1

    for iid in off_scale_items:
        p = project_item(iid)
        off_projs.append(abs(p))

    return {
        "n_on_scale": len(on_projs),
        "on_scale_correct_sign": correct_signs,
        "on_scale_correct_frac": correct_signs / len(on_projs) if on_projs else 0,
        "on_scale_mean_proj": float(np.mean(on_projs)),
        "off_scale_mean_abs_proj": float(np.mean(off_projs)) if off_projs else 0,
        "signal_ratio": (float(np.mean(on_projs)) / (float(np.mean(off_projs)) + 1e-9)) if off_projs else 0,
    }


# =============================================================================
# Scoring dimension 3: BC steering at 5% residual norm
# =============================================================================

def score_bc_steering(direction, layer, model, tokenizer, trait_id, norm_frac=0.05,
                      n_scenarios=25, device="cpu"):
    """Add α·direction to the residual at `layer` and measure BC shift.

    α is chosen so that ||α·direction|| = norm_frac × typical ||residual at layer||
    (estimated from the unsteered activations on the BC prompts).
    """
    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"][trait_id]["pairs"][:n_scenarios]

    BC_TEMPLATE = (
        "Consider what a person most like you would do in the following situation: "
        "{situation}\n\nWhich would you do?\nA) {high}\nB) {low}\n\n"
        "Respond with just A or B.\nAnswer:"
    )

    # Locate transformer layer module for hook registration
    layer_module = None
    for path in ["model.layers", "model.language_model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            layer_module = obj[layer]
            break
        except (AttributeError, IndexError):
            continue
    if layer_module is None:
        return {"error": f"cannot find layer {layer}"}

    a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    # Pre-compute typical residual norm at this layer on these prompts
    prompts = [BC_TEMPLATE.format(situation=p["situation"], high=p["high"], low=p["low"])
               for p in pairs]
    norms = []
    with torch.no_grad():
        for prompt in prompts[:10]:  # sample
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            norms.append(outputs.hidden_states[layer][0, -1, :].float().norm().item())
            del outputs
    typical_norm = float(np.mean(norms))

    d_t = torch.tensor(direction, dtype=torch.float32, device=device)
    d_unit = d_t / (d_t.norm() + 1e-12)
    alpha = norm_frac * typical_norm
    delta = d_unit * alpha

    def eval_condition(dv):
        """Run BC eval with or without steering. Returns frac of A-picks (A = high trait)."""
        handle = None
        if dv is not None:
            def hook_fn(module, inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    return (hidden + dv.unsqueeze(0).unsqueeze(0).to(hidden.dtype),) + output[1:]
                return output + dv.unsqueeze(0).unsqueeze(0).to(output.dtype)
            handle = layer_module.register_forward_hook(hook_fn)

        n_a = n_b = 0
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits[0, -1, :].float()
                if logits[a_id] > logits[b_id]:
                    n_a += 1
                else:
                    n_b += 1
                del outputs
                gc.collect()

        if handle is not None:
            handle.remove()
        return n_a / (n_a + n_b) if (n_a + n_b) > 0 else 0

    baseline_frac_high = eval_condition(None)
    steered_frac_high = eval_condition(delta)

    return {
        "baseline_frac_high": baseline_frac_high,
        "steered_frac_high": steered_frac_high,
        "shift": steered_frac_high - baseline_frac_high,
        "alpha": alpha,
        "typical_residual_norm": typical_norm,
        "norm_frac": norm_frac,
        "n_scenarios": len(pairs),
    }


# =============================================================================
# Main scoring entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score trait directions on multiple dimensions")
    parser.add_argument("--direction-file", required=True)
    parser.add_argument("--method", choices=["lda", "meandiff"], required=True)
    parser.add_argument("--projection", choices=["raw", "projected"], default="projected",
                        help="For meandiff: use raw or PC-projected direction (default: projected)")
    parser.add_argument("--layer-strategy", default="best-signal",
                        help="best-signal, two-thirds, or int (meandiff only)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override layer selection (both methods)")
    parser.add_argument("--trait", default=None, help="Override trait (otherwise read from file)")
    parser.add_argument("--run-hexaco", action="store_true",
                        help="Run HEXACO convergent validity (requires model)")
    parser.add_argument("--run-steering", action="store_true",
                        help="Run BC steering (requires model)")
    parser.add_argument("--model-name", default=None,
                        help="Override HF model name for loading (default: from direction file)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default=None, help="Write results JSON here")
    args = parser.parse_args()

    # Load direction
    if args.method == "lda":
        dd = load_lda_direction(args.direction_file, layer=args.layer)
    else:
        ls = args.layer if args.layer is not None else args.layer_strategy
        dd = load_meandiff_direction(args.direction_file, projection=args.projection,
                                     layer_strategy=ls)

    print(f"\nLoaded {dd['source']} direction for trait {dd['trait']} at layer {dd['layer']}")
    print(f"  direction norm: {np.linalg.norm(dd['direction']):.4f}")
    print(f"  n_pairs: {dd['n_pairs']}")

    results = {
        "direction_file": str(args.direction_file),
        "method": args.method,
        "source": dd["source"],
        "trait": dd["trait"],
        "layer": dd["layer"],
        "model": dd["model"],
    }
    if "prefix_mode" in dd:
        results["prefix_mode"] = dd["prefix_mode"]
        results["neutral_variant"] = dd["neutral_variant"]

    # Dimension 1: classification
    cls = score_classification(dd["direction"], dd["per_pair_diffs"])
    results["classification"] = cls
    print(f"\nClassification: accuracy={cls['accuracy']:.3f}  "
          f"mean_signed_proj={cls['mean_signed_proj']:+.3f}")

    # Dimensions 2 and 3 need a loaded model
    if args.run_hexaco or args.run_steering:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = args.model_name or dd["model"]
        print(f"\nLoading model {model_name}...")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype_map.get(args.dtype, torch.bfloat16),
            device_map=args.device,
        )
        model.eval()
        print(f"  Loaded")

        trait_id = args.trait or dd["trait"]

        if args.run_hexaco:
            print(f"\nRunning HEXACO convergent validity...")
            hex_res = score_hexaco_convergent(
                dd["direction"], dd["layer"], model, tokenizer, trait_id, args.device,
            )
            results["hexaco_convergent"] = hex_res
            print(f"  on-scale correct-sign: {hex_res.get('on_scale_correct_sign')}/"
                  f"{hex_res.get('n_on_scale')}  "
                  f"signal ratio: {hex_res.get('signal_ratio'):+.2f}")

        if args.run_steering:
            print(f"\nRunning BC steering...")
            st = score_bc_steering(
                dd["direction"], dd["layer"], model, tokenizer, trait_id, device=args.device,
            )
            results["bc_steering"] = st
            print(f"  baseline: {st['baseline_frac_high']:.2f}  steered: {st['steered_frac_high']:.2f}  "
                  f"shift: {st['shift']:+.2f}  (α={st['alpha']:.2f})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
