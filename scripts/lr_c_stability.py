#!/usr/bin/env python3
"""Check LR direction stability across regularization strengths.

For each cached (model, trait, format) cell, fit LR at C ∈ {0.1, 1, 10, 100}
at the LDA-CV-best layer, normalize, compute pairwise cosines.

Usage:
    python scripts/lr_c_stability.py
"""

from pathlib import Path
from collections import defaultdict
import statistics as st

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


MODELS = {
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma": "google/gemma-3-4b-it",
    "Phi4":  "microsoft/Phi-4-mini-instruct",
    "Qwen":  "Qwen/Qwen2.5-3B-Instruct",
}
TRAITS = ["H", "E", "X", "A", "C", "O"]
FORMATS = ["chat", "bare"]
C_VALUES = [0.1, 1.0, 10.0, 100.0]
CACHE_DIR = Path("results/phase_b_cache")


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


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
    return best_layer


def main():
    all_cos = defaultdict(list)
    n_cells = 0
    for short, repo in MODELS.items():
        tag = safe(repo)
        for trait in TRAITS:
            for fmt in FORMATS:
                p = CACHE_DIR / f"{tag}_{trait}_{fmt}_pairs.pt"
                if not p.exists():
                    continue
                blob = torch.load(p, weights_only=False)
                ph, pl = blob["ph_tr"], blob["pl_tr"]
                td = (ph - pl).numpy()
                n_train = td.shape[0]

                L = cv_best_layer(td, n_train)
                d_at = td[:, L, :]
                X = np.vstack([d_at / 2, -d_at / 2])
                y = np.array([1] * n_train + [0] * n_train)

                dirs = {}
                for C in C_VALUES:
                    lr = LogisticRegression(C=C, max_iter=5000).fit(X, y)
                    dirs[C] = unit(lr.coef_[0])

                for i, Ci in enumerate(C_VALUES):
                    for Cj in C_VALUES[i + 1:]:
                        key = f"{Ci}_{Cj}"
                        all_cos[key].append(float(np.dot(dirs[Ci], dirs[Cj])))
                n_cells += 1

    print(f"N cells: {n_cells}\n")
    print(f"{'C_a':>5s} vs {'C_b':>5s}   {'mean':>6s}  {'median':>7s}  {'min':>6s}  {'max':>6s}  {'sd':>5s}")
    for i, Ci in enumerate(C_VALUES):
        for Cj in C_VALUES[i + 1:]:
            vs = all_cos[f"{Ci}_{Cj}"]
            print(f"{Ci:>5} vs {Cj:>5}   {st.mean(vs):>+6.3f}  {st.median(vs):>+7.3f}  "
                  f"{min(vs):>+6.3f}  {max(vs):>+6.3f}  {st.pstdev(vs):>5.3f}")


if __name__ == "__main__":
    main()
