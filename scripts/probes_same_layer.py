#!/usr/bin/env python3
"""Re-fit LDA / LR / MD-raw / MD-projected all at the LDA-chosen layer.

Uses cached pair + neutral activations from results/phase_b_cache/.
Outputs pairwise cosines and holdout SNR/accuracy at the fixed layer
so we can separate "probes disagree" from "layers differ."

Usage:
    python scripts/probes_same_layer.py
"""

import csv
import json
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
FORMATS = ["chat", "bare"]
CACHE_DIR = Path("results/phase_b_cache")


def safe(s): return s.replace("/", "_")


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


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def snr(sig):
    return float(sig.mean() / (sig.std() + 1e-12))


def main():
    rows = []
    for model_short, repo in MODELS.items():
        tag = safe(repo)
        for fmt in FORMATS:
            neutral_path = CACHE_DIR / f"{tag}_neutral_{fmt}.pt"
            if not neutral_path.exists():
                print(f"skip {model_short} {fmt}: no neutral cache")
                continue
            neutral = torch.load(neutral_path, weights_only=False)
            neutral_np = neutral.numpy() if isinstance(neutral, torch.Tensor) else neutral

            for trait in TRAITS:
                pair_path = CACHE_DIR / f"{tag}_{trait}_{fmt}_pairs.pt"
                if not pair_path.exists():
                    print(f"skip {model_short} {trait} {fmt}: no pair cache")
                    continue
                blob = torch.load(pair_path, weights_only=False)
                ph_tr, pl_tr = blob["ph_tr"], blob["pl_tr"]
                ph_h, pl_h = blob["ph_h"], blob["pl_h"]

                train_diffs = (ph_tr - pl_tr).numpy()
                hold_diffs = (ph_h - pl_h).numpy()
                n_train = train_diffs.shape[0]

                L, cv_acc = cv_best_layer(train_diffs, n_train)
                d_at = train_diffs[:, L, :]
                X = np.vstack([d_at / 2, -d_at / 2])
                y = np.array([1] * n_train + [0] * n_train)

                lda = LinearDiscriminantAnalysis().fit(X, y)
                lda_dir = unit(lda.coef_[0])

                lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
                lr_dir = unit(lr.coef_[0])

                mean_high = ph_tr.mean(dim=0).numpy()
                mean_low = pl_tr.mean(dim=0).numpy()
                md_raw = unit(mean_high[L] - mean_low[L])

                try:
                    pcs, _, k = mdx.compute_pc_projection(
                        torch.from_numpy(neutral_np[:, L, :]), 0.5
                    )
                    md_proj = unit(mdx.project_out_pcs(mean_high[L] - mean_low[L], pcs))
                except Exception:
                    md_proj = md_raw
                    k = 0

                dirs = {"LDA": lda_dir, "LR": lr_dir, "MDraw": md_raw, "MDproj": md_proj}
                cos = {}
                for a in dirs:
                    for b in dirs:
                        if a < b:
                            cos[f"{a}_{b}"] = float(np.dot(dirs[a], dirs[b]))

                hold_snr = {m: snr(hold_diffs[:, L, :] @ d) for m, d in dirs.items()}
                hold_acc = {m: int(((hold_diffs[:, L, :] @ d) > 0).sum()) for m, d in dirs.items()}
                n_hold = hold_diffs.shape[0]

                rows.append({
                    "model": model_short, "trait": trait, "format": fmt,
                    "layer": L, "cv_acc": cv_acc, "k_pcs": k, "n_hold": n_hold,
                    **{f"cos_{k2}": v for k2, v in cos.items()},
                    **{f"snr_{m}": v for m, v in hold_snr.items()},
                    **{f"acc_{m}": v for m, v in hold_acc.items()},
                })

    out = Path("results/probes_same_layer.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}  ({len(rows)} rows)\n")

    # Summary
    import statistics as st
    def mean(xs): return st.mean(xs) if xs else float("nan")
    def pstd(xs): return st.pstdev(xs) if len(xs) > 1 else 0.0

    print(f"{'pair':>12s}  {'mean':>6s}  {'median':>7s}  {'min':>6s}  {'max':>6s}  {'sd':>5s}")
    for p in ["LDA_LR", "LDA_MDraw", "LDA_MDproj", "LR_MDproj", "LR_MDraw", "MDproj_MDraw"]:
        vs = [r[f"cos_{p}"] for r in rows]
        print(f"{p:>12s}  {mean(vs):>+6.3f}  {st.median(vs):>+7.3f}  {min(vs):>+6.3f}  {max(vs):>+6.3f}  {pstd(vs):>5.3f}")

    print(f"\nHoldout SNR at LDA layer:")
    for m in ["LDA", "LR", "MDraw", "MDproj"]:
        vs = [r[f"snr_{m}"] for r in rows]
        print(f"  {m:>7s}: mean={mean(vs):+.3f}  median={st.median(vs):+.3f}")

    print(f"\nHoldout accuracy at LDA layer (out of {rows[0]['n_hold']}):")
    for m in ["LDA", "LR", "MDraw", "MDproj"]:
        vs = [r[f"acc_{m}"] for r in rows]
        print(f"  {m:>7s}: mean={mean(vs):.2f}")

    print(f"\nPer-model LDA_MDproj cosine (at shared LDA layer):")
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["model"]].append(r["cos_LDA_MDproj"])
    for m, vs in by_m.items():
        print(f"  {m:>6s}: mean={mean(vs):+.3f}  range=[{min(vs):+.3f}, {max(vs):+.3f}]")


if __name__ == "__main__":
    main()
