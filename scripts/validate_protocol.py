#!/usr/bin/env python3
"""Validate the format-invariant personality measurement protocol.

Runs five validation tests:
1. Layer sensitivity — does projection change across layers?
2. Framing sensitivity — does preamble text matter?
3. Cross-model transfer — does model A's LDA direction work on model B?
4. RepE vs Likert — period-token projections vs self-report EVs
5. Forced-choice vs free-text (Röttger test) — do models choose the same
   in forced-choice as they'd freely generate?

Usage:
    .venv/bin/python scripts/validate_protocol.py --model google/gemma-3-4b-it
    .venv/bin/python scripts/validate_protocol.py --model google/gemma-3-4b-it --test layer
    .venv/bin/python scripts/validate_protocol.py --all-models
"""

import argparse
import json
import gc
import math
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoTokenizer, AutoModelForCausalLM

CONTRAST_PAIRS = "instruments/contrast_pairs.json"
HEXACO_FILE = "instruments/hexaco100.json"
OLLAMA_URL = "http://localhost:11434"

MODELS = {
    "gemma3": "google/gemma-3-4b-it",
    "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
    "phi4": "microsoft/Phi-4-mini-instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
}

OLLAMA_MODELS = {
    "gemma3": "gemma3:4b",
    "qwen2.5": "qwen2.5:7b",
    "phi4": "phi4-mini",
    "llama3.2": "llama3.2:3b",
}

# Layer attribute paths differ by architecture
LAYER_PATHS = {
    "google/gemma-3-4b-it": "model.language_model.layers",
    "Qwen/Qwen2.5-3B-Instruct": "model.layers",
    "microsoft/Phi-4-mini-instruct": "model.layers",
    "meta-llama/Llama-3.2-3B-Instruct": "model.layers",
}


def get_layers(model_obj, model_name):
    path = LAYER_PATHS.get(model_name, "model.layers")
    obj = model_obj
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def load_lda_direction(model_prefix, trait="H", layer=None):
    """Load pre-extracted LDA direction for a model/trait."""
    safe = model_prefix.replace("/", "_")
    fpath = f"results/repe/{safe}_{trait}_directions.pt"
    data = torch.load(fpath, weights_only=False)
    diffs = data["raw_diffs"]
    n_pairs = diffs.shape[0]
    n_layers = diffs.shape[1]

    if layer is None:
        # Find best layer by LDA accuracy
        from sklearn.model_selection import cross_val_score
        best_acc, best_layer = 0, 0
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
            except:
                pass
        layer = best_layer

    d = diffs[:, layer, :].numpy()
    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_d = lda.coef_[0]
    lda_d = lda_d / np.linalg.norm(lda_d)

    return lda_d, layer, diffs


def get_activation(model, tokenizer, text, layer_idx, device="mps"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    act = outputs.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
    n_tokens = inputs["input_ids"].shape[1]
    del outputs
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return act, n_tokens


def get_activation_at_position(model, tokenizer, text, layer_idx, token_pos, device="mps"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    act = outputs.hidden_states[layer_idx][0, token_pos, :].float().cpu().numpy()
    del outputs
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return act


def ollama_generate(ollama_model, prompt, raw=False):
    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "raw": raw,
        "stream": False,
        "options": {"num_predict": 60, "temperature": 0},
    }).encode()
    req = Request(f"{OLLAMA_URL}/api/generate", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def ollama_likert(ollama_model, item_text, raw=False):
    """Get Likert EV for an item via Ollama logprobs."""
    prompt = (
        'Rate how accurately each statement describes you.\n'
        '1 = very inaccurate, 2 = moderately inaccurate, 3 = neither, '
        '4 = moderately accurate, 5 = very accurate\n'
        'Respond with only a number.\n\n'
        f'Statement: "{item_text}"\nRating: '
    )
    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "raw": raw,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "logprobs": True,
        "top_logprobs": 10,
    }).encode()
    req = Request(f"{OLLAMA_URL}/api/generate", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    logprobs_list = data.get("logprobs", [])
    if not logprobs_list:
        return None

    top = logprobs_list[0].get("top_logprobs", [])
    raw_lp = {}
    for entry in top:
        tok = entry.get("token", "").strip()
        if tok in ("1", "2", "3", "4", "5"):
            raw_lp[tok] = entry["logprob"]

    if not raw_lp:
        return None

    max_lp = max(raw_lp.values())
    exp_sum = sum(math.exp(lp - max_lp) for lp in raw_lp.values())
    dist = {}
    for tok in ("1", "2", "3", "4", "5"):
        dist[tok] = math.exp(raw_lp[tok] - max_lp) / exp_sum if tok in raw_lp else 0.0

    ev = sum(int(k) * v for k, v in dist.items())
    return ev


# =============================================================================
# Test 1: Layer sensitivity
# =============================================================================
def test_layer_sensitivity(model, tokenizer, model_name, lda_d, best_layer, device="mps"):
    print("\n" + "=" * 60)
    print("  TEST 1: Layer Sensitivity")
    print("=" * 60)

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"]["H"]["pairs"][:15]

    PREFIX = "Consider what a person most like you would do in the following situation: "

    layers_to_test = list(range(max(0, best_layer - 4), min(best_layer + 5, 40)))

    # Get LDA direction at each layer
    safe = model_name.replace("/", "_")
    data = torch.load(f"results/repe/{safe}_H_directions.pt", weights_only=False)
    diffs = data["raw_diffs"]
    n_pairs = diffs.shape[0]

    lda_dirs = {}
    for layer in layers_to_test:
        if layer >= diffs.shape[1]:
            continue
        d = diffs[:, layer, :].numpy()
        if np.any(np.isnan(d)) or np.all(d == 0):
            continue
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_pairs + [0] * n_pairs)
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            ld = lda.coef_[0]
            lda_dirs[layer] = ld / np.linalg.norm(ld)
        except:
            pass

    # For each scenario, project at each layer
    all_projs = {l: [] for l in lda_dirs}
    for p in pairs:
        text = PREFIX + p["situation"] + "."
        for layer, ld in lda_dirs.items():
            act, _ = get_activation(model, tokenizer, text, layer, device)
            proj = np.dot(act, ld)
            all_projs[layer].append(proj)

    # Cross-layer correlations relative to best layer
    print(f"\n  Correlations with best layer ({best_layer}):")
    print(f"  {'Layer':>5s}  {'corr':>6s}  {'mean':>8s}  {'std':>8s}")
    print(f"  {'-' * 32}")
    best_vals = np.array(all_projs.get(best_layer, []))
    for layer in sorted(lda_dirs.keys()):
        vals = np.array(all_projs[layer])
        if len(vals) == len(best_vals) and len(vals) > 2:
            r = np.corrcoef(vals, best_vals)[0, 1]
            print(f"  {layer:5d}  {r:6.3f}  {vals.mean():8.1f}  {vals.std():8.1f}"
                  f"{'  <-- best' if layer == best_layer else ''}")


# =============================================================================
# Test 2: Framing sensitivity
# =============================================================================
def test_framing_sensitivity(model, tokenizer, lda_d, best_layer, device="mps"):
    print("\n" + "=" * 60)
    print("  TEST 2: Framing Sensitivity")
    print("=" * 60)

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"]["H"]["pairs"][:15]

    framings = {
        "most_like_you": "Consider what a person most like you would do in the following situation: {scenario}.",
        "imagine_someone": "Imagine someone similar to you encountering the following situation: {scenario}.",
        "bare_scenario": "{scenario}.",
        "what_would_you": "What would you do in this situation? {scenario}.",
        "third_person": "A person encounters the following situation: {scenario}.",
    }

    framing_projs = {name: [] for name in framings}

    for p in pairs:
        for name, template in framings.items():
            text = template.format(scenario=p["situation"])
            act, _ = get_activation(model, tokenizer, text, best_layer, device)
            proj = np.dot(act, lda_d)
            framing_projs[name].append(proj)

    # Correlations between framings
    names = list(framings.keys())
    print(f"\n  Pairwise correlations between framings:")
    print(f"  {'':>18s}", "  ".join(f"{n[:8]:>8s}" for n in names))
    for n1 in names:
        row = f"  {n1:>18s}"
        for n2 in names:
            r = np.corrcoef(framing_projs[n1], framing_projs[n2])[0, 1]
            row += f"  {r:8.3f}"
        print(row)

    # Mean and std per framing
    print(f"\n  {'Framing':>18s}  {'mean':>8s}  {'std':>8s}")
    print(f"  {'-' * 38}")
    for name in names:
        vals = np.array(framing_projs[name])
        print(f"  {name:>18s}  {vals.mean():8.1f}  {vals.std():8.1f}")


# =============================================================================
# Test 3: Cross-model transfer
# =============================================================================
def test_cross_model_transfer(device="mps"):
    print("\n" + "=" * 60)
    print("  TEST 3: Cross-Model Transfer")
    print("=" * 60)

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"]["H"]["pairs"][:10]

    PREFIX = "Consider what a person most like you would do in the following situation: "

    # Load all available LDA directions
    available = {}
    for label, model_name in MODELS.items():
        try:
            lda_d, best_layer, _ = load_lda_direction(model_name, "H")
            available[label] = {"lda_d": lda_d, "layer": best_layer, "model_name": model_name}
        except FileNotFoundError:
            pass

    if len(available) < 2:
        print("  Need at least 2 models with extracted directions. Skipping.")
        return

    print(f"  Models with directions: {list(available.keys())}")

    # For each model, load it, project scenarios onto ALL models' directions
    results = {}  # (source_model, target_direction) -> [projections]

    for target_label, target_info in available.items():
        target_dir = target_info["lda_d"]
        target_layer = target_info["layer"]

        for source_label, source_info in available.items():
            print(f"  Loading {source_label} to project onto {target_label}'s direction...", flush=True)
            source_model_name = source_info["model_name"]

            tok = AutoTokenizer.from_pretrained(source_model_name)
            mdl = AutoModelForCausalLM.from_pretrained(
                source_model_name, dtype=torch.bfloat16, device_map=device)
            mdl.eval()

            # Use the TARGET's best layer for projection
            projs = []
            for p in pairs:
                text = PREFIX + p["situation"] + "."
                try:
                    act, _ = get_activation(mdl, tok, text, target_layer, device)
                    # Only project if dimensions match
                    if act.shape[0] == target_dir.shape[0]:
                        projs.append(np.dot(act, target_dir))
                    else:
                        projs.append(float('nan'))
                except:
                    projs.append(float('nan'))

            results[(source_label, target_label)] = projs

            del mdl, tok
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    # Report: for each pair, correlate projections
    labels = list(available.keys())
    print(f"\n  Cross-model projection correlations (same direction, different model's activations):")
    for dir_label in labels:
        same_model = results.get((dir_label, dir_label), [])
        for other_label in labels:
            if other_label == dir_label:
                continue
            other_model = results.get((other_label, dir_label), [])
            if same_model and other_model:
                valid = [(a, b) for a, b in zip(same_model, other_model)
                         if not (np.isnan(a) or np.isnan(b))]
                if len(valid) > 2:
                    a, b = zip(*valid)
                    r = np.corrcoef(a, b)[0, 1]
                    print(f"    {dir_label}'s direction: {dir_label} acts ↔ {other_label} acts: r={r:.3f}"
                          f"  (dims match: {available[dir_label]['lda_d'].shape[0]} vs "
                          f"{available[other_label]['lda_d'].shape[0]})")


# =============================================================================
# Test 4: RepE vs Likert
# =============================================================================
def test_repe_vs_likert(model, tokenizer, model_name, lda_d, best_layer,
                        short_name, device="mps"):
    print("\n" + "=" * 60)
    print("  TEST 4: RepE vs Likert Self-Report")
    print("=" * 60)

    with open(HEXACO_FILE) as f:
        hexaco = json.load(f)

    h_scale = hexaco["scales"]["H"]
    reverse_keyed = set(h_scale["reverse_keyed_item_ids"])

    PREFIX = "Consider what a person most like you would do in the following situation: "
    SUFFIX = "."

    ollama_model = OLLAMA_MODELS.get(short_name)
    if not ollama_model:
        print(f"  No Ollama model mapping for {short_name}. Skipping Likert comparison.")
        return

    # Check Ollama is reachable
    try:
        urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except URLError:
        print("  Ollama not reachable. Skipping Likert comparison.")
        return

    repe_projs = []
    likert_evs = []
    items_info = []

    print(f"\n  {'item':>6s} {'R':>1s}  {'RepE':>7s}  {'Likert':>6s}  text")
    print(f"  {'-' * 65}")

    for iid in h_scale["item_ids"]:
        item = hexaco["items"][iid]
        text = item["text"]
        is_rev = iid in reverse_keyed

        # RepE: project scenario onto LDA direction
        prompt = PREFIX + text + SUFFIX
        act, _ = get_activation(model, tokenizer, prompt, best_layer, device)
        proj = np.dot(act, lda_d)
        if is_rev:
            proj = -proj
        repe_projs.append(proj)

        # Likert: get EV from Ollama
        ev = ollama_likert(ollama_model, text)
        if ev is not None and is_rev:
            ev = 6.0 - ev
        likert_evs.append(ev)

        items_info.append({"iid": iid, "text": text, "rev": is_rev})

        ev_str = f"{ev:.2f}" if ev is not None else "N/A"
        r_mark = "R" if is_rev else " "
        print(f"  {iid:>6s} {r_mark}  {proj:7.1f}  {ev_str:>6s}  {text[:40]}")

    # Correlation
    valid = [(r, l) for r, l in zip(repe_projs, likert_evs) if l is not None]
    if len(valid) > 2:
        reps, liks = zip(*valid)
        r = np.corrcoef(reps, liks)[0, 1]
        print(f"\n  RepE ↔ Likert correlation: r={r:.3f} (n={len(valid)})")
    else:
        print(f"\n  Not enough valid pairs for correlation.")


# =============================================================================
# Test 5: Forced-choice vs free-text (Röttger test)
# =============================================================================
def test_rottger(model, tokenizer, model_name, lda_d, best_layer,
                 short_name, device="mps"):
    print("\n" + "=" * 60)
    print("  TEST 5: Forced-Choice vs Free-Text (Röttger Test)")
    print("=" * 60)

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"]["H"]["pairs"][:15]

    ollama_model = OLLAMA_MODELS.get(short_name)
    if not ollama_model:
        print(f"  No Ollama model mapping for {short_name}. Skipping.")
        return

    try:
        urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except URLError:
        print("  Ollama not reachable. Skipping.")
        return

    PREFIX = "Consider what a person most like you would do in the following situation: "

    results = []

    for i, p in enumerate(pairs):
        scenario_prompt = PREFIX + p["situation"] + "."

        # Period-token RepE projection
        act, n_toks = get_activation(model, tokenizer, scenario_prompt, best_layer, device)
        repe_proj = np.dot(act, lda_d)

        # Forced-choice via Ollama: which option does the model pick?
        fc_prompt = (f"{p['situation']}\n\n"
                     f"Which would you do?\n"
                     f"A) {p['high']}\nB) {p['low']}\n\n"
                     f"Respond with just the letter.")

        payload = json.dumps({
            "model": ollama_model,
            "messages": [{"role": "user", "content": fc_prompt}],
            "stream": False,
            "options": {"num_predict": 5, "temperature": 0},
            "logprobs": True,
            "top_logprobs": 10,
        }).encode()
        req = Request(f"{OLLAMA_URL}/api/chat", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as resp:
            fc_data = json.loads(resp.read())

        fc_logprobs = fc_data.get("logprobs", [])
        a_lp = b_lp = None
        for entry in fc_logprobs:
            for alt in entry.get("top_logprobs", []):
                tok = alt["token"].strip().upper()
                if tok == "A" and a_lp is None:
                    a_lp = alt["logprob"]
                elif tok == "B" and b_lp is None:
                    b_lp = alt["logprob"]
            if a_lp is not None and b_lp is not None:
                break

        fc_diff = (a_lp - b_lp) if (a_lp is not None and b_lp is not None) else None
        fc_choice = "HIGH" if (fc_diff and fc_diff > 0) else ("LOW" if fc_diff else "?")

        # Free-text via Ollama
        free_prompt = f"{p['situation']}\n\nWhat would you do?"
        free_response = ollama_generate(ollama_model, free_prompt)

        # Classify free text: project the free response through HF model
        full_free = PREFIX + p["situation"] + ". " + free_response
        free_act, _ = get_activation(model, tokenizer, full_free, best_layer, device)
        free_proj = np.dot(free_act, lda_d)

        # Simple classification: does free_proj > median?
        free_choice = "HIGH" if free_proj > repe_proj else "LOW"

        results.append({
            "pair": i,
            "repe_proj": repe_proj,
            "fc_diff": fc_diff,
            "fc_choice": fc_choice,
            "free_proj": free_proj,
            "free_choice": free_choice,
            "agree": fc_choice == free_choice if fc_choice != "?" else None,
            "free_text": free_response[:60],
        })

    # Report
    print(f"\n  {'pair':>4s}  {'RepE':>6s}  {'FC':>5s}  {'FC_diff':>7s}  {'Free':>6s}  {'agree':>5s}  free text")
    print(f"  {'-' * 75}")
    for r in results:
        fc_str = f"{r['fc_diff']:+7.2f}" if r['fc_diff'] else "     ?"
        agree_str = "YES" if r['agree'] else ("NO" if r['agree'] is False else "?")
        print(f"  {r['pair']:4d}  {r['repe_proj']:6.1f}  {r['fc_choice']:>5s}  {fc_str}  "
              f"{r['free_choice']:>6s}  {agree_str:>5s}  {r['free_text']}")

    n_agree = sum(1 for r in results if r['agree'] is True)
    n_valid = sum(1 for r in results if r['agree'] is not None)
    print(f"\n  Agreement: {n_agree}/{n_valid} ({n_agree / n_valid:.0%})" if n_valid else "")

    # Correlation between FC logodds and RepE projection
    valid_fc = [(r['repe_proj'], r['fc_diff']) for r in results if r['fc_diff'] is not None]
    if len(valid_fc) > 2:
        reps, fcs = zip(*valid_fc)
        r = np.corrcoef(reps, fcs)[0, 1]
        print(f"  RepE projection ↔ FC log-odds: r={r:.3f}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Validate format-invariant protocol")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name (e.g. google/gemma-3-4b-it)")
    parser.add_argument("--short-name", type=str, default=None,
                        help="Short name for Ollama mapping (gemma3/qwen2.5/phi4/llama3.2)")
    parser.add_argument("--test", type=str, default="all",
                        help="Which test: layer/framing/transfer/likert/rottger/all")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--all-models", action="store_true",
                        help="Run on all available models")
    args = parser.parse_args()

    tests = args.test.split(",") if args.test != "all" else [
        "layer", "framing", "transfer", "likert", "rottger"
    ]

    if args.all_models:
        model_list = [(label, name) for label, name in MODELS.items()]
    elif args.model:
        short = args.short_name or next(
            (k for k, v in MODELS.items() if v == args.model), "unknown")
        model_list = [(short, args.model)]
    else:
        print("Specify --model or --all-models")
        sys.exit(1)

    # Transfer test loads its own models
    if "transfer" in tests:
        test_cross_model_transfer(args.device)
        tests = [t for t in tests if t != "transfer"]

    for short_name, model_name in model_list:
        print(f"\n{'#' * 60}")
        print(f"  Model: {model_name} ({short_name})")
        print(f"{'#' * 60}")

        # Load model
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=args.device)
        model.eval()

        # Load LDA direction
        try:
            lda_d, best_layer, _ = load_lda_direction(model_name, "H")
            print(f"  LDA direction loaded (best layer: {best_layer})")
        except FileNotFoundError:
            print(f"  No extracted directions found. Run extract_trait_vectors.py first.")
            continue

        if "layer" in tests:
            test_layer_sensitivity(model, tokenizer, model_name, lda_d, best_layer, args.device)

        if "framing" in tests:
            test_framing_sensitivity(model, tokenizer, lda_d, best_layer, args.device)

        if "likert" in tests:
            test_repe_vs_likert(model, tokenizer, model_name, lda_d, best_layer,
                                short_name, args.device)

        if "rottger" in tests:
            test_rottger(model, tokenizer, model_name, lda_d, best_layer,
                         short_name, args.device)

        # Free model memory before next
        del model, tokenizer
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
