#!/usr/bin/env python3
"""Compute the full cross-method correlation matrix for HEXACO measurements.

Compares five measurement approaches across 4 models × 6 HEXACO traits:
  1. Likert-argmax:  HEXACO-100 declarative statements, argmax score (1-5)
  2. Likert-EV:      Same items, expected value over logprob distribution
  3. RepE-probe:     Scenario contrast pairs projected onto LDA direction (mean projection)
  4. BC-proportion:  Scenario binary-choice, proportion high-trait picks
  5. BC-logodds:     Same scenarios, mean log-odds favoring high-trait

Optionally runs the Rottger test (BC vs free-text agreement) for all traits
if --rottger is specified and Ollama + HuggingFace model are available.

Usage:
    # Just compute from existing data files:
    python scripts/cross_method_matrix.py

    # Also compute RepE projections (needs HuggingFace model):
    python scripts/cross_method_matrix.py --repe --model google/gemma-3-4b-it --short-name gemma3

    # Full matrix for one model including Rottger:
    python scripts/cross_method_matrix.py --repe --rottger --model google/gemma-3-4b-it --short-name gemma3
"""

import argparse
import json
import gc
import sys
from pathlib import Path

import numpy as np

TRAITS = ["H", "E", "X", "A", "C", "O"]
TRAIT_NAMES = {
    "H": "Honesty-Humility", "E": "Emotionality", "X": "Extraversion",
    "A": "Agreeableness", "C": "Conscientiousness", "O": "Openness",
}

HEXACO_FILE = "instruments/hexaco100.json"
CONTRAST_PAIRS = "instruments/contrast_pairs.json"

MODELS = {
    "gemma3":  {"hf": "google/gemma-3-4b-it",               "ollama": "gemma3:4b",    "likert": "results/gemma3_4b_hexaco100.json"},
    "llama":   {"hf": "meta-llama/Llama-3.2-3B-Instruct",   "ollama": "llama3.2:3b",  "likert": "results/llama3.2_3b_hexaco100.json"},
    "phi4":    {"hf": "microsoft/Phi-4-mini-instruct",       "ollama": "phi4-mini",    "likert": "results/phi4-mini_hexaco100.json"},
    "qwen":    {"hf": "Qwen/Qwen2.5-3B-Instruct",           "ollama": "qwen3:8b",     "likert": "results/qwen3_8b_hexaco100.json"},
}


# =============================================================================
# Data loaders
# =============================================================================

def load_likert_scores():
    """Load HEXACO-100 Likert argmax and EV scores for all models."""
    scores = {}  # scores[model][trait] = {"argmax": float, "ev": float}
    for mname, minfo in MODELS.items():
        path = minfo["likert"]
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping {mname}")
            continue
        scores[mname] = {}
        for t in TRAITS:
            s = data["scale_scores"].get(t, {})
            scores[mname][t] = {
                "argmax": s.get("argmax_mean"),
                "ev": s.get("ev_mean"),
            }
    return scores


def load_bc_scores():
    """Load binary-choice scenario scores (proportion + mean log-odds)."""
    try:
        with open("results/binary_choice_6trait.json") as f:
            bc_data = json.load(f)
    except FileNotFoundError:
        print("  WARNING: results/binary_choice_6trait.json not found")
        return {}

    scores = {}
    for mname, minfo in MODELS.items():
        ollama_name = minfo["ollama"]
        scores[mname] = {}
        for t in TRAITS:
            key = f"{t}_{ollama_name}"
            if key not in bc_data:
                scores[mname][t] = {"proportion": None, "logodds": None}
                continue
            vals = bc_data[key]
            n_high = sum(1 for v in vals if v > 0)
            scores[mname][t] = {
                "proportion": n_high / len(vals) if vals else None,
                "logodds": float(np.mean(vals)) if vals else None,
            }
    return scores


def load_repe_scores(trait, model_name):
    """Load LDA direction and compute mean projection on contrast pairs.

    Returns the mean signed projection of the contrast-pair diffs onto the
    LDA direction at the best layer. This gives a "how strongly does this
    model represent trait T" score comparable across models.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    safe = model_name.replace("/", "_")
    fpath = f"results/repe/{safe}_{trait}_directions.pt"

    try:
        import torch
        data = torch.load(fpath, weights_only=False)
    except FileNotFoundError:
        return None

    if "raw_diffs" in data:
        diffs = data["raw_diffs"]  # (n_pairs, n_layers, hidden_dim)
    else:
        # Older format: only has mean_diffs (n_layers, hidden_dim), no per-pair diffs.
        # Can't do LDA without per-pair data. Use mean_diff norm as a rough score.
        mean_diffs = data["mean_diffs"].numpy()
        # Use layer with highest explained variance as best
        ev = data["explained_variance"].numpy()
        best_layer = int(np.argmax(ev))
        mean_proj = float(np.linalg.norm(mean_diffs[best_layer]))
        return {"mean_proj": mean_proj, "best_layer": best_layer, "lda_acc": None,
                "note": "old format, mean_diff norm only"}

    n_pairs, n_layers, _ = diffs.shape

    # Find best layer by LDA accuracy
    best_acc, best_layer = 0, 0
    for l in range(n_layers):
        d = diffs[:, l, :].numpy()
        if np.any(np.isnan(d)) or np.all(d == 0):
            continue
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_pairs + [0] * n_pairs)
        try:
            from sklearn.model_selection import cross_val_score
            acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
            if acc > best_acc:
                best_acc, best_layer = acc, l
        except Exception:
            pass

    # Fit LDA at best layer
    d = diffs[:, best_layer, :].numpy()
    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_d = lda.coef_[0]
    lda_d = lda_d / np.linalg.norm(lda_d)

    # Mean projection of the diffs onto LDA direction
    # Positive = high-trait, negative = low-trait
    projs = d @ lda_d  # (n_pairs,)
    mean_proj = float(np.mean(projs))

    return {"mean_proj": mean_proj, "best_layer": best_layer, "lda_acc": best_acc}


def load_all_repe_scores(normalize=True):
    """Load RepE LDA projection scores for all available model×trait combos.

    If normalize=True, z-score within each model across its 6 trait scores.
    This removes cross-model activation scale differences (Gemma's residual
    stream norms are 10-100x larger) while preserving within-model profile shape.
    Note: makes scores ipsative (sum to 0 within model), same tradeoff as
    binary-choice scoring.
    """
    raw_scores = {}
    for mname, minfo in MODELS.items():
        raw_scores[mname] = {}
        for t in TRAITS:
            result = load_repe_scores(t, minfo["hf"])
            raw_scores[mname][t] = result["mean_proj"] if result else None

    if not normalize:
        return raw_scores

    # Z-score within each model across traits
    normed = {}
    for mname in raw_scores:
        vals = [raw_scores[mname][t] for t in TRAITS]
        if any(v is None for v in vals):
            normed[mname] = raw_scores[mname]  # can't normalize incomplete
            continue
        arr = np.array(vals, dtype=float)
        mu, sd = arr.mean(), arr.std()
        if sd > 0:
            normed[mname] = {t: float((v - mu) / sd) for t, v in zip(TRAITS, arr)}
        else:
            normed[mname] = {t: 0.0 for t in TRAITS}
    return normed


# =============================================================================
# Rottger test (BC vs free-text) — needs live HF model + Ollama
# =============================================================================

def run_rottger_for_trait(model, tokenizer, lda_d, best_layer, ollama_model,
                          trait, n_scenarios=15, device="mps"):
    """Run Rottger test for a single trait. Returns agreement fraction."""
    import torch
    from urllib.request import urlopen, Request

    OLLAMA_URL = "http://localhost:11434"

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    pairs = cp["traits"][trait]["pairs"][:n_scenarios]

    PREFIX = "Consider what a person most like you would do in the following situation: "
    agree_count = 0
    valid_count = 0

    for p in pairs:
        scenario_prompt = PREFIX + p["situation"] + "."

        # RepE projection at period token
        inputs = tokenizer(scenario_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        act = outputs.hidden_states[best_layer][0, -1, :].float().cpu().numpy()
        repe_proj = np.dot(act, lda_d)
        del outputs
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        # Binary-choice via Ollama
        bc_prompt = (f"{p['situation']}\n\nWhich would you do?\n"
                     f"A) {p['high']}\nB) {p['low']}\n\nRespond with just the letter.")
        payload = json.dumps({
            "model": ollama_model,
            "messages": [{"role": "user", "content": bc_prompt}],
            "stream": False,
            "options": {"num_predict": 5, "temperature": 0},
            "logprobs": True, "top_logprobs": 10,
        }).encode()
        from urllib.request import Request
        req = Request(f"{OLLAMA_URL}/api/chat", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as resp:
            bc_data = json.loads(resp.read())

        a_lp = b_lp = None
        for entry in bc_data.get("logprobs", []):
            for alt in entry.get("top_logprobs", []):
                tok = alt["token"].strip().upper()
                if tok == "A" and a_lp is None:
                    a_lp = alt["logprob"]
                elif tok == "B" and b_lp is None:
                    b_lp = alt["logprob"]
            if a_lp is not None and b_lp is not None:
                break

        if a_lp is None or b_lp is None:
            continue
        bc_choice = "HIGH" if a_lp > b_lp else "LOW"

        # Free-text via Ollama, classified by RepE
        free_prompt = f"{p['situation']}\n\nWhat would you do?"
        free_payload = json.dumps({
            "model": ollama_model, "prompt": free_prompt,
            "raw": False, "stream": False,
            "options": {"num_predict": 60, "temperature": 0},
        }).encode()
        req2 = Request(f"{OLLAMA_URL}/api/generate", data=free_payload,
                       headers={"Content-Type": "application/json"})
        with urlopen(req2, timeout=120) as resp:
            free_response = json.loads(resp.read()).get("response", "").strip()

        full_free = PREFIX + p["situation"] + ". " + free_response
        inputs2 = tokenizer(full_free, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs2 = model(**inputs2, output_hidden_states=True)
        free_act = outputs2.hidden_states[best_layer][0, -1, :].float().cpu().numpy()
        free_proj = np.dot(free_act, lda_d)
        del outputs2
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        free_choice = "HIGH" if free_proj > repe_proj else "LOW"

        if bc_choice == free_choice:
            agree_count += 1
        valid_count += 1

    return agree_count / valid_count if valid_count > 0 else None


# =============================================================================
# Correlation matrix computation
# =============================================================================

def compute_matrix(likert, bc, repe):
    """Build and print the cross-method correlation matrix."""
    model_order = [m for m in MODELS if m in likert]

    # Build flat vectors for each method (model × trait order)
    vectors = {}
    for method_name, extractor in [
        ("Likert-argmax", lambda m, t: likert.get(m, {}).get(t, {}).get("argmax")),
        ("Likert-EV",     lambda m, t: likert.get(m, {}).get(t, {}).get("ev")),
        ("BC-proportion", lambda m, t: bc.get(m, {}).get(t, {}).get("proportion")),
        ("BC-logodds",    lambda m, t: bc.get(m, {}).get(t, {}).get("logodds")),
        ("RepE-probe",    lambda m, t: repe.get(m, {}).get(t)),
    ]:
        vals = []
        for m in model_order:
            for t in TRAITS:
                vals.append(extractor(m, t))
        vectors[method_name] = vals

    # Filter to methods that have data
    active = {k: v for k, v in vectors.items() if any(x is not None for x in v)}

    names = list(active.keys())
    n_cells = len(model_order) * len(TRAITS)

    print(f"\n{'=' * 70}")
    print(f"  CROSS-METHOD CORRELATION MATRIX")
    print(f"  {len(model_order)} models × {len(TRAITS)} traits = {n_cells} cells")
    print(f"{'=' * 70}")

    # Overall correlation matrix
    print(f"\n--- Overall (all {n_cells} model×trait cells) ---\n")
    header = f"{'':16s}" + "".join(f"{n:>16s}" for n in names)
    print(header)
    for n1 in names:
        row = f"{n1:16s}"
        for n2 in names:
            # Pairwise complete: only use cells where both have data
            pairs = [(a, b) for a, b in zip(active[n1], active[n2])
                     if a is not None and b is not None]
            if len(pairs) > 2:
                a_vals, b_vals = zip(*pairs)
                r = np.corrcoef(a_vals, b_vals)[0, 1]
                row += f"{r:16.3f}"
            else:
                row += f"{'n/a':>16s}"
        print(row)

    # Per-model
    print(f"\n--- Per-model (across 6 traits) ---\n")
    for m in model_order:
        print(f"  {m}:")
        m_vecs = {}
        for name in names:
            idx_start = model_order.index(m) * len(TRAITS)
            m_vecs[name] = active[name][idx_start:idx_start + len(TRAITS)]

        # Compact: just show Likert-EV vs BC-prop, BC-lo, RepE
        cross_pairs = []
        for n1 in names:
            for n2 in names:
                if n1 >= n2:
                    continue
                pairs = [(a, b) for a, b in zip(m_vecs[n1], m_vecs[n2])
                         if a is not None and b is not None]
                if len(pairs) > 2:
                    a_v, b_v = zip(*pairs)
                    r = np.corrcoef(a_v, b_v)[0, 1]
                    cross_pairs.append((n1, n2, r, len(pairs)))
        for n1, n2, r, n in cross_pairs:
            print(f"    {n1:16s} ↔ {n2:16s}  r={r:+.3f} (n={n})")
        print()

    # Per-trait
    print(f"--- Per-trait (across {len(model_order)} models) ---\n")
    for t in TRAITS:
        print(f"  {t} ({TRAIT_NAMES[t]}):")
        t_vecs = {}
        for name in names:
            t_idx = TRAITS.index(t)
            t_vecs[name] = [active[name][i * len(TRAITS) + t_idx] for i in range(len(model_order))]

        cross_pairs = []
        for n1 in names:
            for n2 in names:
                if n1 >= n2:
                    continue
                pairs = [(a, b) for a, b in zip(t_vecs[n1], t_vecs[n2])
                         if a is not None and b is not None]
                if len(pairs) > 2:
                    a_v, b_v = zip(*pairs)
                    r = np.corrcoef(a_v, b_v)[0, 1]
                    cross_pairs.append((n1, n2, r, len(pairs)))
        for n1, n2, r, n in cross_pairs:
            print(f"    {n1:16s} ↔ {n2:16s}  r={r:+.3f} (n={n})")
        print()

    # Raw data table
    print(f"--- Raw data ---\n")
    print(f"{'Model':8s} {'Trait':5s}", end="")
    for n in names:
        print(f" {n:>16s}", end="")
    print()
    for m in model_order:
        for ti, t in enumerate(TRAITS):
            print(f"{m:8s} {t:5s}", end="")
            for name in names:
                idx = model_order.index(m) * len(TRAITS) + ti
                v = active[name][idx]
                if v is not None:
                    print(f" {v:16.3f}", end="")
                else:
                    print(f" {'—':>16s}", end="")
            print()

    return active, model_order


def main():
    parser = argparse.ArgumentParser(description="Cross-method correlation matrix")
    parser.add_argument("--repe", action="store_true",
                        help="Compute RepE projections (needs HF model loaded)")
    parser.add_argument("--rottger", action="store_true",
                        help="Run Rottger BC-vs-free-text test (needs Ollama + HF)")
    parser.add_argument("--model", type=str, default=None,
                        help="HF model for --repe/--rottger")
    parser.add_argument("--short-name", type=str, default=None,
                        help="Model short name (gemma3/llama/phi4/qwen)")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    print("Loading existing data...")
    likert = load_likert_scores()
    bc = load_bc_scores()
    repe = load_all_repe_scores()

    # Report what we have
    for mname in MODELS:
        repe_avail = sum(1 for t in TRAITS if repe.get(mname, {}).get(t) is not None)
        likert_avail = sum(1 for t in TRAITS if likert.get(mname, {}).get(t, {}).get("ev") is not None)
        bc_avail = sum(1 for t in TRAITS if bc.get(mname, {}).get(t, {}).get("proportion") is not None)
        print(f"  {mname:8s}: Likert={likert_avail}/6  BC={bc_avail}/6  RepE={repe_avail}/6")

    compute_matrix(likert, bc, repe)


if __name__ == "__main__":
    main()
