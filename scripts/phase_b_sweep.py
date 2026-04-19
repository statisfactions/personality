#!/usr/bin/env python3
"""Phase B sweep: 4 models × 6 traits with the Phase A winning configuration.

Per rgb_reports/report_week5_meandiff.md §10.8:
  - format primary = chat template, diagnostic = bare text
  - prefix = generic ("Consider a person.")
  - neutral = scenario_setups (300 items)
  - methods = LDA, MD-raw, MD-projected

Per §9.2 (chat-template bump is Llama-specific), we expect format to have
different effects across models: Llama is the one most affected by format
change, Gemma/Phi4/Qwen minimal. Reporting both formats lets us see
whether the method-vs-method conclusion is stable across formats.

Metrics per cell (24-pair holdout, facet-stratified 6 pairs × 4 facets):
  - layer selected (best-CV for LDA, best-SNR for MD)
  - sign_correct_train / 50
  - sign_correct_hold  / 24  (overall)
  - sign_correct_hold per facet (4 × 6/facet typically)
  - snr_train, snr_hold
  - k_pcs at best layer (MD-projected only)
  - cos_to_lda (MD methods only)

Output: results/phase_b_sweep.csv (one row per model × trait × format × method),
        results/phase_b_sweep.json (same plus full facet breakdown),
        results/phase_b_summary.txt (readable table).

Usage:
    python scripts/phase_b_sweep.py
    python scripts/phase_b_sweep.py --models Llama Gemma  # subset
    python scripts/phase_b_sweep.py --traits H E          # subset
"""

import argparse
import csv
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import extract_meandiff_vectors as mdx


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
    "Qwen":  "Qwen/Qwen2.5-3B-Instruct",
}

TRAITS = ["H", "E", "X", "A", "C", "O"]
TRAINING_FILE = Path("instruments/contrast_pairs.json")
HOLDOUT_FILE = Path("instruments/contrast_pairs_holdout.json")
CACHE_DIR = Path("results/phase_b_cache")


def safe_name(s):
    return s.replace("/", "_")


def cv_best_layer(train_diffs, n_train):
    """LDA 5-fold CV best layer."""
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


def snr_best_layer(train_diffs, dir_per_layer):
    """Best-SNR layer using training signal only (no holdout leakage)."""
    n_layers = train_diffs.shape[1]
    best, best_layer = -float("inf"), 0
    for L in range(n_layers):
        u = dir_per_layer[L] / (np.linalg.norm(dir_per_layer[L]) + 1e-12)
        sig = train_diffs[:, L, :] @ u
        snr = sig.mean() / (sig.std() + 1e-12)
        if snr > best:
            best, best_layer = snr, L
    return best_layer, best


def score_per_facet(hold_diffs, direction, layer, pairs):
    """Return {facet -> (sign_correct, total)} for this direction at this layer."""
    u = direction / (np.linalg.norm(direction) + 1e-12)
    sig = hold_diffs[:, layer, :] @ u
    by_facet = defaultdict(lambda: [0, 0])
    for i, p in enumerate(pairs):
        facet = p.get("facet", "?")
        by_facet[facet][1] += 1
        if sig[i] > 0:
            by_facet[facet][0] += 1
    return dict(by_facet)


def score_direction(direction, layer, train_diffs, hold_diffs, hold_pairs):
    u = direction / (np.linalg.norm(direction) + 1e-12)
    t_sig = train_diffs[:, layer, :] @ u
    h_sig = hold_diffs[:, layer, :] @ u
    return {
        "layer": int(layer),
        "sign_correct_train": int((t_sig > 0).sum()),
        "n_train": int(len(t_sig)),
        "sign_correct_hold": int((h_sig > 0).sum()),
        "n_hold": int(len(h_sig)),
        "snr_train": float(t_sig.mean() / (t_sig.std() + 1e-12)),
        "snr_hold": float(h_sig.mean() / (h_sig.std() + 1e-12)),
        "mean_signed_hold": float(h_sig.mean()),
        "per_facet_hold": score_per_facet(hold_diffs, direction, layer, hold_pairs),
    }


def sweep_one_model(model_short, repo, traits_to_run, formats, device, dtype):
    """Run all traits × formats for one model. Returns list of rows."""
    print(f"\n{'=' * 70}\nModel: {model_short} ({repo})\n{'=' * 70}")
    t_load = time.time()
    model, tokenizer = mdx.load_model(repo, device, dtype)
    print(f"Loaded in {time.time() - t_load:.1f}s")

    with open(TRAINING_FILE) as f:
        cp = json.load(f)
    with open(HOLDOUT_FILE) as f:
        hold = json.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = safe_name(repo)

    # Caches — keyed by (format, prefix) for pair activations, (format, variant) for neutral
    neutral_cache = {}  # (fmt,) -> neutral acts (one variant: scenario_setups)

    # Extract scenario_setups neutral activations once per format
    def get_neutral(fmt):
        if fmt not in neutral_cache:
            neutral_cache_path = CACHE_DIR / f"{model_tag}_neutral_{fmt}.pt"
            if neutral_cache_path.exists():
                neutral_cache[fmt] = torch.load(neutral_cache_path, weights_only=False).numpy()
                return neutral_cache[fmt]
            texts = mdx.load_neutral_texts("scenario_setups")
            print(f"\n  extract neutral ({fmt}, scenario_setups, {len(texts)} texts)...")
            t0 = time.time()
            acts = mdx.extract_neutral_activations(
                model, tokenizer, texts, device,
                chat_template=(fmt == "chat"), verbose=False,
            )
            print(f"    done in {time.time() - t0:.1f}s")
            torch.save(acts, neutral_cache_path)
            neutral_cache[fmt] = acts.numpy()
        return neutral_cache[fmt]

    rows = []
    for trait in traits_to_run:
        print(f"\n--- trait {trait} ({cp['traits'][trait]['name']}) ---")
        train_td = cp["traits"][trait]
        hold_td = {
            "name": hold["traits"][trait]["name"],
            "pairs": hold["traits"][trait]["pairs"],
            "high_descriptor": train_td["high_descriptor"],
            "low_descriptor": train_td["low_descriptor"],
        }
        hold_pairs = hold_td["pairs"]

        for fmt in formats:
            pair_cache_path = CACHE_DIR / f"{model_tag}_{trait}_{fmt}_pairs.pt"
            if pair_cache_path.exists():
                blob = torch.load(pair_cache_path, weights_only=False)
                ph_tr, pl_tr = blob["ph_tr"], blob["pl_tr"]
                ph_h, pl_h = blob["ph_h"], blob["pl_h"]
                print(f"  loaded cached pairs ({fmt})")
            else:
                print(f"  extract training ({fmt}, generic)...", end=" ", flush=True)
                t0 = time.time()
                ph_tr, pl_tr = mdx.extract_trait_activations(
                    model, tokenizer, train_td, "generic", device,
                    chat_template=(fmt == "chat"), verbose=False,
                )
                print(f"{time.time() - t0:.1f}s")

                print(f"  extract holdout  ({fmt}, generic)...", end=" ", flush=True)
                t0 = time.time()
                ph_h, pl_h = mdx.extract_trait_activations(
                    model, tokenizer, hold_td, "generic", device,
                    chat_template=(fmt == "chat"), verbose=False,
                )
                print(f"{time.time() - t0:.1f}s")

                torch.save({"ph_tr": ph_tr, "pl_tr": pl_tr, "ph_h": ph_h, "pl_h": pl_h,
                            "hold_pairs": hold_pairs}, pair_cache_path)

            train_diffs = (ph_tr - pl_tr).numpy()
            hold_diffs = (ph_h - pl_h).numpy()
            n_train = train_diffs.shape[0]
            n_layers = train_diffs.shape[1]

            # LDA at CV-best layer
            lda_layer, lda_cv = cv_best_layer(train_diffs, n_train)
            d_at = train_diffs[:, lda_layer, :]
            X = np.vstack([d_at / 2, -d_at / 2])
            y = np.array([1] * n_train + [0] * n_train)
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            lda_dir = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)

            lda_scores = score_direction(lda_dir, lda_layer, train_diffs, hold_diffs, hold_pairs)
            lda_scores["cv_acc"] = float(lda_cv)

            # Logistic regression at the same layer (same X, y)
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(X, y)
            lr_dir = lr.coef_[0] / (np.linalg.norm(lr.coef_[0]) + 1e-12)
            lr_scores = score_direction(lr_dir, lda_layer, train_diffs, hold_diffs, hold_pairs)

            # MD raw direction per layer
            mean_high = ph_tr.mean(dim=0).numpy()
            mean_low = pl_tr.mean(dim=0).numpy()
            raw_per_layer = mean_high - mean_low  # (n_layers, hidden_dim)

            # PC projection using this format's scenario_setups neutral
            neutral_acts = get_neutral(fmt)  # (n_texts, n_layers, hidden_dim)
            proj_per_layer = np.zeros_like(raw_per_layer)
            k_per_layer = []
            for L in range(n_layers):
                try:
                    pcs, _, k = mdx.compute_pc_projection(
                        torch.from_numpy(neutral_acts[:, L, :]), 0.5,
                    )
                    proj_per_layer[L] = mdx.project_out_pcs(raw_per_layer[L], pcs)
                    k_per_layer.append(k)
                except Exception:
                    proj_per_layer[L] = raw_per_layer[L]
                    k_per_layer.append(0)

            # MD layers (best-SNR on training)
            md_raw_layer, _ = snr_best_layer(train_diffs, raw_per_layer)
            md_proj_layer, _ = snr_best_layer(train_diffs, proj_per_layer)

            md_raw_dir = raw_per_layer[md_raw_layer]
            md_raw_dir = md_raw_dir / (np.linalg.norm(md_raw_dir) + 1e-12)
            md_proj_dir = proj_per_layer[md_proj_layer]
            md_proj_dir = md_proj_dir / (np.linalg.norm(md_proj_dir) + 1e-12)

            md_raw_scores = score_direction(md_raw_dir, md_raw_layer, train_diffs, hold_diffs, hold_pairs)
            md_proj_scores = score_direction(md_proj_dir, md_proj_layer, train_diffs, hold_diffs, hold_pairs)
            md_proj_scores["k_pcs_at_layer"] = int(k_per_layer[md_proj_layer])

            base = dict(model=model_short, trait=trait, format=fmt, prefix="generic",
                        neutral="scenario_setups")
            rows.append({**base, "method": "LDA", **lda_scores, "cos_to_lda": None})
            rows.append({**base, "method": "LR", **lr_scores,
                         "cos_to_lda": float(np.dot(lda_dir, lr_dir))})
            rows.append({**base, "method": "MD-raw", **md_raw_scores,
                         "cos_to_lda": float(np.dot(lda_dir, md_raw_dir))})
            rows.append({**base, "method": "MD-projected", **md_proj_scores,
                         "cos_to_lda": float(np.dot(lda_dir, md_proj_dir))})

            print(f"    LDA        layer={lda_layer:>2}  hold={lda_scores['sign_correct_hold']}/{lda_scores['n_hold']}  snr_h={lda_scores['snr_hold']:+.2f}")
            print(f"    LR         layer={lda_layer:>2}  hold={lr_scores['sign_correct_hold']}/{lr_scores['n_hold']}  snr_h={lr_scores['snr_hold']:+.2f}  cos_lda={np.dot(lda_dir, lr_dir):+.3f}")
            print(f"    MD-raw     layer={md_raw_layer:>2}  hold={md_raw_scores['sign_correct_hold']}/{md_raw_scores['n_hold']}  snr_h={md_raw_scores['snr_hold']:+.2f}")
            print(f"    MD-projected layer={md_proj_layer:>2}  hold={md_proj_scores['sign_correct_hold']}/{md_proj_scores['n_hold']}  snr_h={md_proj_scores['snr_hold']:+.2f}  k_pcs={k_per_layer[md_proj_layer]}")

            del ph_tr, pl_tr, ph_h, pl_h
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    del model, tokenizer, neutral_cache
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--traits", nargs="+", default=TRAITS)
    parser.add_argument("--formats", nargs="+", default=["chat", "bare"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-prefix", default="phase_b_sweep")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    all_rows = []
    t_start = time.time()
    for short in args.models:
        if short not in MODELS:
            print(f"Unknown model: {short} (known: {list(MODELS.keys())})")
            continue
        repo = MODELS[short]
        try:
            rows = sweep_one_model(short, repo, args.traits, args.formats, args.device, dtype)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FAILED on {short}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    total_time = time.time() - t_start
    print(f"\n\nTotal sweep time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Outputs
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.output_prefix}.csv"
    json_path = out_dir / f"{args.output_prefix}.json"
    txt_path = out_dir / f"{args.output_prefix}_summary.txt"

    # CSV (flat fields only)
    flat_fields = ["model", "trait", "format", "prefix", "neutral", "method",
                   "layer", "n_train", "sign_correct_train", "n_hold", "sign_correct_hold",
                   "snr_train", "snr_hold", "mean_signed_hold",
                   "cv_acc", "k_pcs_at_layer", "cos_to_lda"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    with open(json_path, "w") as f:
        json.dump({"rows": all_rows, "config": vars(args)}, f, indent=2, default=str)

    # Human-readable summary: table per format, one row per model × trait × method
    def pct(n, total):
        return f"{n:>2}/{total:<2} ({100 * n / total:>3.0f}%)"

    with open(txt_path, "w") as out:
        for fmt in args.formats:
            out.write(f"\n\n{'=' * 120}\n")
            out.write(f"FORMAT = {fmt}\n")
            out.write(f"{'=' * 120}\n")
            out.write(f"{'model':>6s}  {'trait':>5s}  {'method':>14s}  "
                      f"{'layer':>5s}  {'train':>12s}  {'holdout':>12s}  "
                      f"{'snr_t':>6s}  {'snr_h':>6s}  "
                      f"{'k_pcs':>5s}  {'cos_lda':>7s}\n")
            out.write("-" * 120 + "\n")
            for r in all_rows:
                if r["format"] != fmt:
                    continue
                tr = pct(r["sign_correct_train"], r["n_train"])
                ho = pct(r["sign_correct_hold"], r["n_hold"])
                kpc = str(r.get("k_pcs_at_layer", "")) if r["method"] == "MD-projected" else ""
                cos = r.get("cos_to_lda")
                cos_s = f"{cos:+.3f}" if cos is not None else ""
                out.write(f"{r['model']:>6s}  {r['trait']:>5s}  {r['method']:>14s}  "
                          f"{r['layer']:>5d}  {tr:>12s}  {ho:>12s}  "
                          f"{r['snr_train']:>6.2f}  {r['snr_hold']:>+6.2f}  "
                          f"{kpc:>5s}  {cos_s:>7s}\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")

    # Also print summary to stdout
    print(open(txt_path).read())


if __name__ == "__main__":
    main()
