#!/usr/bin/env python3
"""Phase A sweep: mean-diff extraction across format × prefix × neutral on one model × trait.

Target: the best apples-to-apples comparison of LDA vs MD we can get on Llama × H
before committing to Phase B's 4 models × 6 traits.

Grid:
  - format      ∈ {bare text, chat template}            (2)
  - prefix_mode ∈ {high, low, absent, generic}          (4)
  - neutral     ∈ {scenario_setups, shaggy_dog, factual}(3)
  = 24 cells

Metrics per cell, evaluated on the 24-pair holdout:
  - sign_correct_train  / 50  (sanity, always high)
  - sign_correct_hold   / 24  (the headline)
  - snr_train           (mean signed proj / std on training pairs)
  - snr_hold            (same on holdout)
  - layer               (best-snr layer)
  - k_pcs_at_layer      (how many neutral PCs were projected at best-snr layer)

LDA baseline (also on the same training/holdout activations) is reported
alongside each cell for direct method comparison.

Prompt-steering ceiling (from report_week5_meandiff.md §9) is reported as a
reference row, not computed in this script.

Output: CSV at results/phase_a_sweep_<model>_<trait>.csv, plus a rich JSON
at results/phase_a_sweep_<model>_<trait>.json with per-cell details.

Usage:
    python scripts/phase_a_sweep.py \
      --model meta-llama/Llama-3.2-3B-Instruct --trait H
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import extract_meandiff_vectors as mdx  # reuse its helpers


TRAINING_FILE = Path("instruments/contrast_pairs.json")
HOLDOUT_FILE  = Path("instruments/contrast_pairs_holdout.json")


def best_snr_layer(diffs, direction_per_layer):
    """Select layer with max mean-signed-projection / std on per-pair diffs."""
    n_layers = diffs.shape[1]
    best_snr, best_layer = -float("inf"), 0
    for L in range(n_layers):
        u = direction_per_layer[L] / (np.linalg.norm(direction_per_layer[L]) + 1e-12)
        sig = diffs[:, L, :] @ u
        snr = sig.mean() / (sig.std() + 1e-12)
        if snr > best_snr:
            best_snr, best_layer = snr, L
    return best_layer, best_snr


def score_at_layer(train_diffs, hold_diffs, direction, layer):
    """Return classification + SNR metrics at a fixed layer."""
    u = direction / (np.linalg.norm(direction) + 1e-12)
    tsig = train_diffs[:, layer, :] @ u
    hsig = hold_diffs[:, layer, :] @ u
    return {
        "sign_correct_train": int((tsig > 0).sum()),
        "sign_correct_hold":  int((hsig > 0).sum()),
        "snr_train": float(tsig.mean() / (tsig.std() + 1e-12)),
        "snr_hold":  float(hsig.mean() / (hsig.std() + 1e-12)),
        "mean_signed_train": float(tsig.mean()),
        "mean_signed_hold":  float(hsig.mean()),
    }


def fit_lda(train_diffs, layer):
    n_pairs = train_diffs.shape[0]
    d = train_diffs[:, layer, :]
    X = np.vstack([d / 2, -d / 2])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)


def load_neutral(variant):
    texts = mdx.load_neutral_texts(variant)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trait", default="H")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--formats",  nargs="+", default=["bare", "chat"])
    parser.add_argument("--prefixes", nargs="+",
                        default=["high", "low", "absent", "generic"])
    parser.add_argument("--neutrals", nargs="+",
                        default=["scenario_setups", "shaggy_dog", "factual"])
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    # Load data
    with open(TRAINING_FILE) as f:
        cp = json.load(f)
    with open(HOLDOUT_FILE) as f:
        hold = json.load(f)
    trait_data_train = cp["traits"][args.trait]
    trait_data_hold  = hold["traits"][args.trait]
    # Ensure holdout has same descriptors as training (they should)
    trait_data_hold["high_descriptor"] = trait_data_train["high_descriptor"]
    trait_data_hold["low_descriptor"]  = trait_data_train["low_descriptor"]

    # Load model
    model, tokenizer = mdx.load_model(args.model, args.device, args.dtype)

    # Cache activations per (format, prefix) and neutral per (format, variant)
    train_cache = {}  # (format, prefix) -> (per_pair_high, per_pair_low)
    hold_cache  = {}  # (format, prefix) -> (per_pair_high, per_pair_low)
    neutral_cache = {}  # (format, variant) -> (n_texts, n_layers, hidden_dim)

    def get_train(fmt, prefix):
        key = (fmt, prefix)
        if key not in train_cache:
            print(f"\n  extract training act   format={fmt}  prefix={prefix}")
            t0 = time.time()
            h, l = mdx.extract_trait_activations(
                model, tokenizer, trait_data_train, prefix, args.device,
                chat_template=(fmt == "chat"), verbose=False,
            )
            print(f"    done in {time.time()-t0:.1f}s  shape={tuple(h.shape)}")
            train_cache[key] = (h, l)
        return train_cache[key]

    def get_hold(fmt, prefix):
        key = (fmt, prefix)
        if key not in hold_cache:
            print(f"\n  extract holdout act    format={fmt}  prefix={prefix}")
            t0 = time.time()
            h, l = mdx.extract_trait_activations(
                model, tokenizer, trait_data_hold, prefix, args.device,
                chat_template=(fmt == "chat"), verbose=False,
            )
            print(f"    done in {time.time()-t0:.1f}s  shape={tuple(h.shape)}")
            hold_cache[key] = (h, l)
        return hold_cache[key]

    def get_neutral(fmt, variant):
        key = (fmt, variant)
        if key not in neutral_cache:
            texts = load_neutral(variant)
            print(f"\n  extract neutral act    format={fmt}  variant={variant}  ({len(texts)} texts)")
            t0 = time.time()
            acts = mdx.extract_neutral_activations(
                model, tokenizer, texts, args.device,
                chat_template=(fmt == "chat"), verbose=False,
            )
            print(f"    done in {time.time()-t0:.1f}s  shape={tuple(acts.shape)}")
            neutral_cache[key] = acts
        return neutral_cache[key]

    # Sweep
    rows = []
    for fmt in args.formats:
        for prefix in args.prefixes:
            ph_tr, pl_tr = get_train(fmt, prefix)
            ph_h,  pl_h  = get_hold(fmt, prefix)
            train_diffs = (ph_tr - pl_tr).numpy()  # (n_train, n_layers, d)
            hold_diffs  = (ph_h  - pl_h ).numpy()

            # LDA baseline once per (format, prefix) using training CV to pick layer
            n_train = train_diffs.shape[0]
            n_layers = train_diffs.shape[1]
            # CV-best layer for LDA
            from sklearn.model_selection import cross_val_score
            best_acc, lda_layer = 0, 0
            for L in range(n_layers):
                d = train_diffs[:, L, :]
                if np.any(np.isnan(d)) or np.all(d == 0):
                    continue
                X = np.vstack([d / 2, -d / 2])
                y = np.array([1] * n_train + [0] * n_train)
                try:
                    acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
                    if acc > best_acc:
                        best_acc, lda_layer = acc, L
                except Exception:
                    pass
            lda_dir = fit_lda(train_diffs, lda_layer)
            lda_scores = score_at_layer(train_diffs, hold_diffs, lda_dir, lda_layer)
            lda_scores["layer"] = lda_layer
            lda_scores["cv_acc"] = best_acc

            # Mean-diff raw direction (per layer)
            mean_high = ph_tr.mean(dim=0).numpy()
            mean_low  = pl_tr.mean(dim=0).numpy()
            raw_dir_per_layer = mean_high - mean_low  # (n_layers, d)

            for variant in args.neutrals:
                neutral_acts = get_neutral(fmt, variant).numpy()
                # PC-project per layer
                proj_dir_per_layer = np.zeros_like(raw_dir_per_layer)
                k_per_layer = []
                for L in range(n_layers):
                    try:
                        pcs, _, k = mdx.compute_pc_projection(
                            torch.from_numpy(neutral_acts[:, L, :]), 0.5,
                        )
                        proj_dir_per_layer[L] = mdx.project_out_pcs(raw_dir_per_layer[L], pcs)
                        k_per_layer.append(k)
                    except Exception:
                        proj_dir_per_layer[L] = raw_dir_per_layer[L]
                        k_per_layer.append(0)

                # Select layer by best-snr on TRAINING signal (no holdout leakage)
                md_raw_layer, md_raw_snr = best_snr_layer(train_diffs, raw_dir_per_layer)
                md_proj_layer, md_proj_snr = best_snr_layer(train_diffs, proj_dir_per_layer)

                md_raw_dir_best = raw_dir_per_layer[md_raw_layer]
                md_raw_dir_best = md_raw_dir_best / (np.linalg.norm(md_raw_dir_best) + 1e-12)
                md_proj_dir_best = proj_dir_per_layer[md_proj_layer]
                md_proj_dir_best = md_proj_dir_best / (np.linalg.norm(md_proj_dir_best) + 1e-12)

                md_raw_scores = score_at_layer(train_diffs, hold_diffs, md_raw_dir_best, md_raw_layer)
                md_raw_scores["layer"] = md_raw_layer

                md_proj_scores = score_at_layer(train_diffs, hold_diffs, md_proj_dir_best, md_proj_layer)
                md_proj_scores["layer"] = md_proj_layer
                md_proj_scores["k_pcs_at_layer"] = k_per_layer[md_proj_layer]

                # Record rows for each method
                base = dict(format=fmt, prefix=prefix, neutral=variant)
                rows.append({**base, "method": "LDA",         **lda_scores})
                rows.append({**base, "method": "MD-raw",      **md_raw_scores})
                rows.append({**base, "method": "MD-projected", **md_proj_scores})

                # Cosines are handy for comparison
                cos_lda_md_raw = float(np.dot(lda_dir, md_raw_dir_best))
                cos_lda_md_proj = float(np.dot(lda_dir, md_proj_dir_best))
                cos_md_raw_proj = float(np.dot(md_raw_dir_best, md_proj_dir_best))
                for r, cos in [
                    (rows[-3], None), (rows[-2], cos_lda_md_raw), (rows[-1], cos_lda_md_proj),
                ]:
                    r["cos_to_lda"] = cos
                rows[-2]["cos_raw_proj"] = cos_md_raw_proj
                rows[-1]["cos_raw_proj"] = cos_md_raw_proj

    # Output
    prefix_out = args.output_prefix or f"phase_a_sweep_{args.model.replace('/', '_')}_{args.trait}"
    out_csv = Path("results") / f"{prefix_out}.csv"
    out_json = Path("results") / f"{prefix_out}.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_json, "w") as f:
        json.dump({"rows": rows, "config": vars(args)}, f, indent=2, default=str)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")

    # Print compact summary table
    print("\n" + "=" * 120)
    print(f"{'fmt':>5s}  {'prefix':>8s}  {'neutral':>16s}  {'method':>14s}  "
          f"{'layer':>5s}  {'train_sc':>8s}  {'hold_sc':>8s}  {'snr_tr':>6s}  {'snr_h':>6s}  "
          f"{'k_pc':>5s}  {'cos_LDA':>7s}")
    print("-" * 120)
    for r in rows:
        cos = r.get("cos_to_lda")
        cos_str = "    -  " if cos is None else f"{cos:+7.3f}"
        k_pc = r.get("k_pcs_at_layer", 0)
        print(f"{r['format']:>5s}  {r['prefix']:>8s}  {r['neutral']:>16s}  {r['method']:>14s}  "
              f"{r['layer']:>5d}  "
              f"{r['sign_correct_train']:>3d}/50  "
              f"{r['sign_correct_hold']:>3d}/24  "
              f"{r['snr_train']:>6.2f}  {r['snr_hold']:>6.2f}  "
              f"{k_pc:>5}  {cos_str}")


if __name__ == "__main__":
    main()
