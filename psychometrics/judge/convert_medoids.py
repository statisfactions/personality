#!/usr/bin/env python3
"""Tidy the 35-medoid raw JUDGE distributions (rgb `judge_dists_medoids` sample).

Source: data/medoids/<Model>_tom_likely_dists.npz  (gitignored)
  dists (35,35,7) f32  P(rating k+1) for "very {row}: how likely also {col}?"
  ev, entropy (35,35)  recomputed from dists (internally consistent)
  adjectives (35,)     medoid labels; prompt (str); reproduction_mean_abs_dev

Output: data/medoids_dists.csv.gz  — long: model, i, j, adj_i, adj_j, p1..p7, ev, entropy
        data/medoids_meta.csv      — model, display, reproduction_mean_abs_dev, n_adj
The full 523-matrix release keys the models with slightly different stems; map to
the canonical stems used by convert_judge.py / model_meta.csv.
"""
import glob, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "data", "medoids")
OUT = os.path.join(HERE, "data")

# medoid file stem -> canonical stem (matches model_meta.csv)
CANON = {"gemma3": "Gemma", "llama3.2": "Llama", "phi4": "Phi4", "qwen2.5": "Qwen"}

def main():
    files = sorted(glob.glob(os.path.join(SRC, "*_tom_likely_dists.npz")))
    assert files, f"no medoid npz in {SRC}"
    rows, meta = [], []
    for f in files:
        stem = os.path.basename(f).replace("_tom_likely_dists.npz", "")
        canon = CANON.get(stem, stem)
        z = np.load(f, allow_pickle=True)
        D, ev, ent = z["dists"], z["ev"], z["entropy"]
        adj = [str(a) for a in z["adjectives"]]
        n = len(adj)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = D[i, j]
                if np.isnan(p).any():
                    continue
                rows.append((canon, i, j, adj[i], adj[j], *[round(float(x), 6) for x in p],
                             round(float(ev[i, j]), 5), round(float(ent[i, j]), 5)))
        meta.append((canon, stem, float(z["reproduction_mean_abs_dev"]), n))
    cols = ["model", "i", "j", "adj_i", "adj_j"] + [f"p{k}" for k in range(1, 8)] + ["ev", "entropy"]
    pd.DataFrame(rows, columns=cols).to_csv(
        os.path.join(OUT, "medoids_dists.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(meta, columns=["model", "src_stem", "reproduction_mean_abs_dev", "n_adj"]).to_csv(
        os.path.join(OUT, "medoids_meta.csv"), index=False)
    print(f"{len(files)} models, {len(rows)} judgments -> medoids_dists.csv.gz")
    # capture the exact prompt once
    z0 = np.load(files[0], allow_pickle=True)
    with open(os.path.join(OUT, "medoids_prompt.txt"), "w") as fh:
        fh.write(str(z0["prompt"]))

if __name__ == "__main__":
    main()
