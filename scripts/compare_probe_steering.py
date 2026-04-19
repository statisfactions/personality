#!/usr/bin/env python3
"""Compare LDA, LR, MD-projected directions as steering vectors.

For each (model, trait) cell:
  - Fit LDA, LR, MD-proj at LDA-CV-best layer on cached activations
  - Normalize each direction to unit, scale to 5% of residual norm at that layer
  - Apply as δ (and −δ) at the target layer on each of 24 held-out BC prompts
  - Measure log-odds(high) − log-odds(low) vs baseline; report shift, flips, pick rate

Uses cached activations in results/phase_b_cache/ — no re-extraction.

Usage:
    python scripts/compare_probe_steering.py
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM

import extract_meandiff_vectors as mdx


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
}
TRAITS = ["O", "E"]
FORMAT = "chat"
NORM_FRAC = 0.05
CACHE_DIR = Path("results/phase_b_cache")
HOLDOUT_FILE = Path("instruments/contrast_pairs_holdout.json")


def safe(s): return s.replace("/", "_")


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cv_best_layer(train_diffs, n_train):
    n_layers = train_diffs.shape[1]
    best_acc, best_layer = 0, 0
    for L in range(n_layers):
        d = train_diffs[:, L, :]
        if np.any(np.isnan(d)) or np.all(d == 0):
            continue
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_train + [0] * n_train)
        try:
            acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
            if acc > best_acc:
                best_acc, best_layer = acc, L
        except Exception:
            pass
    return best_layer, best_acc


def get_layer_module(model, layer_idx):
    for path in ["model.layers", "model.language_model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj[layer_idx]
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Can't find layer {layer_idx}")


def build_bc_prompt(tokenizer, scenario, high, low):
    msg = (
        f"Consider what a person most like you would do in the following situation: "
        f"{scenario}\n\n"
        f"Which would you do?\n"
        f"A) {high}\n"
        f"B) {low}\n\n"
        f"Respond with just A or B.\n"
        f"Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=True,
    )


def get_ab_ids(tokenizer):
    a_id = b_id = None
    for tok in ["A", " A"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            a_id = ids[-1]
            break
    for tok in ["B", " B"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            b_id = ids[-1]
            break
    return a_id, b_id


def bc_logodds(model, tokenizer, prompts, delta, layer_module, device, a_id, b_id):
    """Return list of log-odds(A) − log-odds(B) per prompt, with optional δ added at layer."""
    handle = None
    if delta is not None:
        d = delta.to(device)
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                return (out[0] + d.unsqueeze(0).unsqueeze(0),) + out[1:]
            return out + d.unsqueeze(0).unsqueeze(0)
        handle = layer_module.register_forward_hook(hook_fn)

    results = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :]
        results.append((logits[a_id] - logits[b_id]).float().item())
        del out, inputs

    if handle:
        handle.remove()
    return results


def run_cell(model, tokenizer, device, repo_tag, trait, hold_td):
    blob = torch.load(CACHE_DIR / f"{repo_tag}_{trait}_{FORMAT}_pairs.pt", weights_only=False)
    ph_tr, pl_tr = blob["ph_tr"], blob["pl_tr"]

    train_diffs = (ph_tr - pl_tr).numpy()
    n_train = train_diffs.shape[0]

    L, _ = cv_best_layer(train_diffs, n_train)
    d_at = train_diffs[:, L, :]
    X = np.vstack([d_at / 2, -d_at / 2])
    y = np.array([1] * n_train + [0] * n_train)

    lda_dir = unit(LinearDiscriminantAnalysis().fit(X, y).coef_[0])
    lr_dir = unit(LogisticRegression(C=1.0, max_iter=2000).fit(X, y).coef_[0])

    mean_high = ph_tr.mean(dim=0).numpy()
    mean_low = pl_tr.mean(dim=0).numpy()
    raw = mean_high[L] - mean_low[L]

    # MD-projected using cached neutral at layer L
    neutral = torch.load(CACHE_DIR / f"{repo_tag}_neutral_{FORMAT}.pt", weights_only=False)
    if isinstance(neutral, torch.Tensor):
        neutral = neutral.numpy()
    try:
        pcs, _, _ = mdx.compute_pc_projection(torch.from_numpy(neutral[:, L, :]), 0.5)
        md_proj = unit(mdx.project_out_pcs(raw, pcs))
    except Exception:
        md_proj = unit(raw)

    # Residual norm at layer L from pair activations (mean ||h||)
    all_acts = torch.cat([ph_tr[:, L, :], pl_tr[:, L, :]], dim=0).float()
    resid_norm = float(all_acts.norm(dim=-1).mean().item())
    scale = NORM_FRAC * resid_norm

    # Build BC prompts from held-out scenarios
    pairs = hold_td["pairs"]
    prompts = [build_bc_prompt(tokenizer, p["situation"], p["high"], p["low"]) for p in pairs]
    n = len(pairs)

    layer_module = get_layer_module(model, L)
    a_id, b_id = get_ab_ids(tokenizer)

    # Baseline (no δ)
    baseline = bc_logodds(model, tokenizer, prompts, None, layer_module, device, a_id, b_id)

    conditions = {
        "+LDA": scale * lda_dir, "-LDA": -scale * lda_dir,
        "+LR":  scale * lr_dir,  "-LR":  -scale * lr_dir,
        "+MDp": scale * md_proj, "-MDp": -scale * md_proj,
    }
    results = {"baseline": baseline}
    for name, dvec in conditions.items():
        dt = torch.tensor(dvec, dtype=torch.float32)
        results[name] = bc_logodds(model, tokenizer, prompts, dt, layer_module, device, a_id, b_id)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    def summarize(lo):
        lo = np.array(lo)
        base = np.array(baseline)
        return {
            "mean": float(lo.mean()),
            "shift": float((lo - base).mean()),
            "n_high": int((lo > 0).sum()),
            "flips_to_high": int(((base <= 0) & (lo > 0)).sum()),
            "flips_to_low":  int(((base >  0) & (lo <= 0)).sum()),
        }

    cell = {
        "model": None, "trait": trait, "layer": int(L),
        "resid_norm": resid_norm, "scale": scale, "n_hold": n,
        "stats": {k: summarize(v) for k, v in results.items()},
    }
    return cell


def main():
    device = "mps"
    dtype = torch.bfloat16

    with open(HOLDOUT_FILE) as f:
        hold = json.load(f)

    all_cells = []
    for short, repo in MODELS.items():
        print(f"\n{'=' * 70}\nModel: {short} ({repo})\n{'=' * 70}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, device_map=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"Loaded in {time.time() - t0:.1f}s")

        repo_tag = safe(repo)

        for trait in TRAITS:
            print(f"\n--- {short} / {trait} ---")
            hold_td = {
                "name": hold["traits"][trait]["name"],
                "pairs": hold["traits"][trait]["pairs"],
            }
            cell = run_cell(model, tokenizer, device, repo_tag, trait, hold_td)
            cell["model"] = short

            print(f"  layer={cell['layer']}  |h|={cell['resid_norm']:.1f}  |δ|={cell['scale']:.2f}")
            print(f"  {'cond':>8s}  {'mean_lo':>9s}  {'shift':>8s}  {'n_high':>6s}  {'+flip':>6s}  {'-flip':>6s}")
            for name, s in cell["stats"].items():
                print(f"  {name:>8s}  {s['mean']:>+9.3f}  {s['shift']:>+8.3f}  "
                      f"{s['n_high']:>3d}/{cell['n_hold']:<2d}  {s['flips_to_high']:>6d}  {s['flips_to_low']:>6d}")
            all_cells.append(cell)

        del model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    out = Path("results/probe_steering_compare.json")
    out.write_text(json.dumps(all_cells, indent=2))
    print(f"\nWrote {out}")

    # Summary
    print(f"\n{'=' * 70}\nSUMMARY: mean shift (+) - shift (-) = effect size\n{'=' * 70}")
    print(f"{'model':>6s} {'trait':>2s}  {'LDA':>14s}  {'LR':>14s}  {'MD-proj':>14s}")
    for c in all_cells:
        def fx(m): return c['stats'][f'+{m}']['shift'] - c['stats'][f'-{m}']['shift']
        print(f"{c['model']:>6s} {c['trait']:>2s}  "
              f"{c['stats']['+LDA']['shift']:>+6.2f}/{c['stats']['-LDA']['shift']:>+6.2f}  "
              f"{c['stats']['+LR']['shift']:>+6.2f}/{c['stats']['-LR']['shift']:>+6.2f}  "
              f"{c['stats']['+MDp']['shift']:>+6.2f}/{c['stats']['-MDp']['shift']:>+6.2f}")


if __name__ == "__main__":
    main()
