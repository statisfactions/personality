#!/usr/bin/env python3
"""Convert the JUDGE (tom_likely) npz matrices into tidy tables for R psychometrics.

Source: results/adjectives/introspect_full/<Model>_tom_likely_dir.npz  (rgb, gitignored)
  Each npz: B (523,523) float32 expected Likert 1-7, B[i,j]="given a person is
  adjectives[i], how strongly also adjectives[j]" (row-conditioned, NOT symmetric,
  diagonal NaN); Hent (523,523) Shannon entropy of the rating distribution;
  adjectives (523,) labels.

We treat the 12 models as *raters* of adjective-pair judgments. Outputs (data/):
  adjectives.csv        idx, adjective
  model_meta.csv        model, display, family, params_b, released_order
  dir_ev.csv.gz         i, j, <model...>   directional EV B[i,j], all i!=j (273006 rows)
  dir_hent.csv.gz       i, j, <model...>   directional entropy Hent[i,j]
  sym_ev.csv.gz         i, j (i<j), <model...>   symmetric EV (B[i,j]+B[j,i])/2  (136503 rows)
  asym_ev.csv.gz        i, j (i<j), <model...>   directed diff B[i,j]-B[j,i]
  sym_hent.csv.gz       i, j (i<j), <model...>   symmetric mean entropy
  marginals.csv         model, idx, adjective, row_mean(anchor-ascriptiveness),
                        col_mean(target-prevalence), row_hent, col_hent
  model_summary.csv     per-model global descriptives + response-style indices

All adjective order is the per-file `adjectives` array (authoritative). We assert
the order is identical across the 12 models so a single index space is valid.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "results", "adjectives", "introspect_full")
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# stem -> (display, family, params_b).  Stems per judge_likert/README.md.
META = {
    "Llama":       ("Llama3.2-3B",     "Llama",   3.2),
    "Llama8":      ("Llama3.1-8B",     "Llama",   8.0),
    "Qwen":        ("Qwen2.5-3B",      "Qwen",    3.0),
    "Qwen7":       ("Qwen2.5-7B",      "Qwen",    7.0),
    "Qwen32":      ("Qwen2.5-32B",     "Qwen",   32.0),
    "Gemma":       ("Gemma3-4B",       "Gemma",   4.0),
    "Gemma12":     ("Gemma3-12B",      "Gemma",  12.0),
    "Gemma27":     ("Gemma3-27B",      "Gemma",  27.0),
    "Gemma4":      ("Gemma-4-31B",     "Gemma",  31.0),
    "Phi4":        ("Phi4-mini-3.8B",  "Phi",     3.8),
    "Aya":         ("Aya-expanse-8B",  "Aya",     8.0),
    "FalconMamba": ("FalconMamba-7B",  "Falcon",  7.0),
}

def stem_of(path):
    return os.path.basename(path).replace("_tom_likely_dir.npz", "")

def main():
    paths = sorted(glob.glob(os.path.join(SRC, "*_tom_likely_dir.npz")))
    models = [p for p in paths if stem_of(p) in META]
    stems = [stem_of(p) for p in models]
    assert set(stems) == set(META), f"stem mismatch: {set(stems) ^ set(META)}"
    print(f"{len(stems)} models: {stems}")

    # Load all, verify adjective alignment.
    Bs, Hs, adj0 = {}, {}, None
    for p in models:
        s = stem_of(p)
        z = np.load(p, allow_pickle=True)
        B = np.asarray(z["B"], dtype=np.float64)
        H = np.asarray(z["Hent"], dtype=np.float64)
        adj = list(z["adjectives"])
        if adj0 is None:
            adj0 = adj
        assert adj == adj0, f"{s}: adjective order differs"
        Bs[s], Hs[s] = B, H
    n = len(adj0)
    print(f"n_adjectives = {n}")

    # canonical model order: by family then size
    order = sorted(stems, key=lambda s: (META[s][1], META[s][2]))

    # ---- adjectives.csv, model_meta.csv ----
    pd.DataFrame({"idx": range(n), "adjective": adj0}).to_csv(
        os.path.join(OUT, "adjectives.csv"), index=False)
    pd.DataFrame(
        [{"model": s, "display": META[s][0], "family": META[s][1],
          "params_b": META[s][2], "order": k} for k, s in enumerate(order)]
    ).to_csv(os.path.join(OUT, "model_meta.csv"), index=False)

    # ---- index spaces ----
    eye = np.eye(n, dtype=bool)
    di, dj = np.where(~eye)                      # all ordered i!=j  (273006)
    ti, tj = np.triu_indices(n, k=1)             # i<j               (136503)

    def wide(vals_by_model, ii, jj, fname):
        df = pd.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32)})
        for s in order:
            df[s] = np.round(vals_by_model[s], 5)
        df.to_csv(os.path.join(OUT, fname), index=False,
                  compression="gzip", float_format="%.5g")
        print(f"  wrote {fname}: {df.shape}")

    # directional EV / entropy (i!=j)
    wide({s: Bs[s][di, dj] for s in order}, di, dj, "dir_ev.csv.gz")
    wide({s: Hs[s][di, dj] for s in order}, di, dj, "dir_hent.csv.gz")

    # symmetric EV, directed difference, symmetric entropy (i<j)
    sym = {s: 0.5 * (Bs[s][ti, tj] + Bs[s][tj, ti]) for s in order}
    asy = {s: (Bs[s][ti, tj] - Bs[s][tj, ti]) for s in order}          # B[i,j]-B[j,i]
    syh = {s: 0.5 * (Hs[s][ti, tj] + Hs[s][tj, ti]) for s in order}
    wide(sym, ti, tj, "sym_ev.csv.gz")
    wide(asy, ti, tj, "asym_ev.csv.gz")
    wide(syh, ti, tj, "sym_hent.csv.gz")

    # ---- marginals (diagonal is NaN -> nanmean excludes self) ----
    mrows = []
    for s in order:
        B, H = Bs[s], Hs[s]
        rmean = np.nanmean(B, axis=1)     # anchor i ascribes others (row)
        cmean = np.nanmean(B, axis=0)     # target j ascribed by others (col) = prevalence
        rhent = np.nanmean(H, axis=1)
        chent = np.nanmean(H, axis=0)
        for k in range(n):
            mrows.append((s, k, adj0[k], rmean[k], cmean[k], rhent[k], chent[k]))
    pd.DataFrame(mrows, columns=["model", "idx", "adjective", "row_mean",
                                 "col_mean", "row_hent", "col_hent"]).to_csv(
        os.path.join(OUT, "marginals.csv"), index=False)

    # ---- per-model summary + response-style indices ----
    from scipy import stats as sps
    srows = []
    for s in order:
        B, H = Bs[s], Hs[s]
        off = B[~eye]                     # 273006 directional EVs
        ho = H[~eye]
        # asymmetry: variance of antisymmetric part / total (off-diagonal)
        S = 0.5 * (B + B.T); A = 0.5 * (B - B.T)
        so = S[~eye]; ao = A[~eye]
        asym_frac = np.sum(ao**2) / np.sum(off**2)
        # effective category usage: round EV to nearest integer 1..7
        rnd = np.clip(np.rint(off), 1, 7).astype(int)
        cat_frac = {f"cat{c}": np.mean(rnd == c) for c in range(1, 8)}
        srow = {
            "model": s, "display": META[s][0], "family": META[s][1],
            "params_b": META[s][2], "n_pairs": off.size,
            "ev_mean": off.mean(), "ev_sd": off.std(), "ev_median": np.median(off),
            "ev_min": off.min(), "ev_max": off.max(),
            "ev_skew": sps.skew(off), "ev_kurtosis": sps.kurtosis(off),  # excess
            "frac_floor_le1p5": np.mean(off <= 1.5),
            "frac_le2": np.mean(off <= 2.0),
            "frac_mid_3to5": np.mean((off >= 3) & (off <= 5)),
            "frac_ge6": np.mean(off >= 6.0),
            "frac_ceil_ge6p5": np.mean(off >= 6.5),
            "hent_mean": ho.mean(), "hent_median": np.median(ho),
            "frac_hent_lt0p1": np.mean(ho < 0.1),   # near-deterministic judgments
            "frac_hent_lt0p5": np.mean(ho < 0.5),
            "asym_frac": asym_frac,
            "asym_rms": np.sqrt(np.mean(ao**2)),
        }
        srow.update(cat_frac)
        srows.append(srow)
    pd.DataFrame(srows).to_csv(os.path.join(OUT, "model_summary.csv"), index=False)
    print("  wrote marginals.csv, model_summary.csv")

    # tiny manifest
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump({"n_adjectives": n, "n_models": len(order),
                   "models": order, "n_dir_pairs": int(di.size),
                   "n_sym_pairs": int(ti.size)}, f, indent=2)
    print("done.")

if __name__ == "__main__":
    main()
