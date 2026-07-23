#!/usr/bin/env python3
"""Tidy the FULL 523-set raw JUDGE distributions (rgb `judge_dists_full`).

Source: data/dists_full/judge_dists/<stem>_tom_likely_dists_full.npz  (gitignored)
  dists (523,523,7) float16  P(rating k+1) for "very {row}: how likely also {col}?"
                             row-conditioned, NOT symmetric, diagonal NaN.
  ev, entropy (523,523) f32  recomputed from dists by rgb (we re-recompute in f64).
  adjectives (523,) lowercase; prompt (str); reproduction_mean_abs_dev.

These SUPERSEDE results/adjectives/introspect_full/*_tom_likely_dir.npz (the June
EV-only release; bf16 numerics drifted, bimodal cells' EV flipped up to ~4). We
recompute B (=EV) and Hent (=entropy) directly from `dists` in float64 so the
entropy envelope (spread index) is internally consistent, and treat these as
canonical. The full 12-model cohort completed 2026-07-20 (batch 1 = 9 models
2026-07-12; batch 2 = FalconMamba/Gemma4/Qwen32 2026-07-22). This script processes
whatever npz are present under data/dists_full/judge_dists/ (currently all 12).

Adjective order in the npz is ALPHABETICAL and differs from the old
adjectives.csv; we regenerate adjectives.csv from the npz so the whole 9-model
dataset shares one internally-consistent index space.

Outputs mirror convert_judge.py (so Reports II–V rerun with minimal change) plus
NEW full-distribution products Report I needs (true category mass, floor/ceiling,
rgb's off-mode-mass bimodality metric, and a shape_class census over all 523²):

  adjectives.csv          idx, adjective (npz order, lowercase)
  model_meta.csv          model, display, family, params_b, order   (9 rows)
  dir_ev.csv.gz           i, j, <model...>   directional EV, all i!=j (273006 rows)
  dir_hent.csv.gz         i, j, <model...>   directional entropy
  sym_ev.csv.gz           i, j (i<j), <model...>   symmetric EV
  asym_ev.csv.gz          i, j (i<j), <model...>   directed diff B[i,j]-B[j,i]
  sym_hent.csv.gz         i, j (i<j), <model...>   symmetric mean entropy
  marginals.csv           model, idx, adjective, row_mean, col_mean, row_hent, col_hent
  model_summary.csv       per-model descriptives + response-style indices
                          (adds TRUE category mass massC1..7, mass-based floor/ceil,
                           off-mode-mass frac, and shape_class fractions)
  full_catmass.csv        model, cat(1-7), mean_mass                  (tidy, for plots)
  full_shape.csv          model, shape fractions + offmode_frac       (Report I §6)
  full_examples.csv       curated "same EV, opposite shape" example distributions
  MANIFEST_full.json, prompt_full.txt

All full-distribution helpers (shape_class, off-mode mass) are Python ports of the
R definitions in reports/_common.R and kept bit-for-bit consistent with them.
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "data", "dists_full", "judge_dists")
OUT = os.path.join(HERE, "data")

# npz file stem -> canonical stem (matches convert_judge.py META / model_meta.csv)
CANON = {"gemma3": "Gemma", "llama3.2": "Llama", "phi4": "Phi4", "qwen2.5": "Qwen"}
META = {  # canonical stem -> (display, family, params_b)
    "Llama":       ("Llama3.2-3B",    "Llama",  3.2),
    "Llama8":      ("Llama3.1-8B",    "Llama",  8.0),
    "Qwen":        ("Qwen2.5-3B",     "Qwen",   3.0),
    "Qwen7":       ("Qwen2.5-7B",     "Qwen",   7.0),
    "Qwen32":      ("Qwen2.5-32B",    "Qwen",  32.0),
    "Gemma":       ("Gemma3-4B",      "Gemma",  4.0),
    "Gemma12":     ("Gemma3-12B",     "Gemma", 12.0),
    "Gemma27":     ("Gemma3-27B",     "Gemma", 27.0),
    "Gemma4":      ("Gemma-4-31B",    "Gemma", 31.0),
    "Phi4":        ("Phi4-mini-3.8B", "Phi",    3.8),
    "Aya":         ("Aya-expanse-8B", "Aya",    8.0),
    "FalconMamba": ("FalconMamba-7B", "Falcon", 7.0),
}
K = np.arange(1, 8, dtype=np.float64)


def ev_ent(D):
    """EV and Shannon entropy (nats) of a (...,7) pmf tensor, in float64."""
    P = D.astype(np.float64)
    ev = (P * K).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(P > 0, P * np.log(P), 0.0).sum(-1)
    return ev, ent


def shape_class_row(p):
    """Python port of reports/_common.R shape_class() for one 7-vector pmf.
    mode = local max with p>=0.15; a real valley (< 0.6*smaller peak) between two
    consecutive modes => semantic (>=3 apart) / digit-notch (2 apart, valley at
    digit 2 or 6) / adjacent."""
    lm = [k for k in range(1, 8)
          if (k == 1 or p[k - 1] > p[k - 2]) and (k == 7 or p[k - 1] > p[k])
          and p[k - 1] >= 0.15]                       # 1-based category indices
    if len(lm) < 2:
        return "unimodal"
    for t in range(len(lm) - 1):
        a, b = lm[t], lm[t + 1]
        seg = p[a:b - 1]                              # p[(a+1):(b-1)] 1-based -> 0-based
        if seg.size and seg.min() < 0.6 * min(p[a - 1], p[b - 1]):
            if b - a >= 3:
                return "semantic"
            if b - a == 2 and (a + 1) in (2, 6):
                return "digit-notch"
            return "adjacent"
    return "unimodal"


def offmode_mass(P):
    """rgb's bimodality metric: total mass >=2 categories from the argmax mode."""
    m = P.argmax(-1)                                  # 0-based mode index
    idx = np.arange(7)
    far = np.abs(idx[None, :] - m[:, None]) >= 2      # (ncells,7)
    return (P * far).sum(-1)


def main():
    paths = sorted(glob.glob(os.path.join(SRC, "*_tom_likely_dists_full.npz")))
    paths = [p for p in paths if not os.path.basename(p).startswith("._")]
    assert paths, f"no full npz in {SRC}"

    Bs, Hs, Ds, adj0 = {}, {}, {}, None
    repro = {}
    for p in paths:
        stem = os.path.basename(p).replace("_tom_likely_dists_full.npz", "")
        canon = CANON.get(stem, stem)
        assert canon in META, f"unknown stem {stem}->{canon}"
        z = np.load(p, allow_pickle=True)
        D = np.asarray(z["dists"], dtype=np.float64)   # (523,523,7), diag NaN
        adj = [str(a) for a in z["adjectives"]]        # lowercase
        if adj0 is None:
            adj0 = adj
        assert adj == adj0, f"{canon}: adjective order differs from {paths[0]}"
        ev, ent = ev_ent(D)                            # NaN where diag NaN
        Bs[canon], Hs[canon], Ds[canon] = ev, ent, D
        repro[canon] = float(z["reproduction_mean_abs_dev"])
    n = len(adj0)
    order = sorted(Bs, key=lambda s: (META[s][1], META[s][2]))
    print(f"{len(order)} models: {order}\nn_adjectives = {n}")

    # ---- adjectives.csv, model_meta.csv ----
    pd.DataFrame({"idx": range(n), "adjective": adj0}).to_csv(
        os.path.join(OUT, "adjectives.csv"), index=False)
    pd.DataFrame(
        [{"model": s, "display": META[s][0], "family": META[s][1],
          "params_b": META[s][2], "order": k} for k, s in enumerate(order)]
    ).to_csv(os.path.join(OUT, "model_meta.csv"), index=False)

    # ---- index spaces ----
    eye = np.eye(n, dtype=bool)
    di, dj = np.where(~eye)                             # ordered i!=j
    ti, tj = np.triu_indices(n, k=1)                   # i<j

    def wide(vals_by_model, ii, jj, fname):
        df = pd.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32)})
        for s in order:
            df[s] = np.round(vals_by_model[s], 5)
        df.to_csv(os.path.join(OUT, fname), index=False,
                  compression="gzip", float_format="%.5g")
        print(f"  wrote {fname}: {df.shape}")

    wide({s: Bs[s][di, dj] for s in order}, di, dj, "dir_ev.csv.gz")
    wide({s: Hs[s][di, dj] for s in order}, di, dj, "dir_hent.csv.gz")
    wide({s: 0.5 * (Bs[s][ti, tj] + Bs[s][tj, ti]) for s in order}, ti, tj, "sym_ev.csv.gz")
    wide({s: (Bs[s][ti, tj] - Bs[s][tj, ti]) for s in order}, ti, tj, "asym_ev.csv.gz")
    wide({s: 0.5 * (Hs[s][ti, tj] + Hs[s][tj, ti]) for s in order}, ti, tj, "sym_hent.csv.gz")

    # ---- marginals ----
    mrows = []
    for s in order:
        B, H = Bs[s], Hs[s]
        rmean, cmean = np.nanmean(B, 1), np.nanmean(B, 0)
        rhent, chent = np.nanmean(H, 1), np.nanmean(H, 0)
        for k in range(n):
            mrows.append((s, k, adj0[k], rmean[k], cmean[k], rhent[k], chent[k]))
    pd.DataFrame(mrows, columns=["model", "idx", "adjective", "row_mean",
                                 "col_mean", "row_hent", "col_hent"]).to_csv(
        os.path.join(OUT, "marginals.csv"), index=False)

    # ---- per-model summary + response-style indices + FULL-distribution products ----
    srows, catrows, shprows = [], [], []
    shapes_by_model = {}
    for s in order:
        B, H, D = Bs[s], Hs[s], Ds[s]
        off = B[~eye]; ho = H[~eye]
        Poff = D[~eye]                                  # (273006, 7) raw pmfs
        S = 0.5 * (B + B.T); A = 0.5 * (B - B.T)
        ao = A[~eye]
        asym_frac = np.sum(ao ** 2) / np.sum(off ** 2)
        rnd = np.clip(np.rint(off), 1, 7).astype(int)   # legacy rounded-EV categories
        cat_frac = {f"cat{c}": np.mean(rnd == c) for c in range(1, 8)}
        # TRUE category mass (mean probability per category over all off-diag cells)
        massC = Poff.mean(0)                            # length-7
        # true floor/ceiling by mass on the end categories
        # shape census + off-mode mass over all off-diagonal cells
        omm = offmode_mass(Poff)
        shapes = np.array([shape_class_row(p) for p in Poff])
        shapes_by_model[s] = shapes
        shp_frac = {sh: float(np.mean(shapes == sh))
                    for sh in ("unimodal", "semantic", "digit-notch", "adjacent")}
        srow = {
            "model": s, "display": META[s][0], "family": META[s][1],
            "params_b": META[s][2], "n_pairs": off.size,
            "reproduction_mad": repro[s],
            "ev_mean": off.mean(), "ev_sd": off.std(), "ev_median": np.median(off),
            "ev_min": off.min(), "ev_max": off.max(),
            "ev_skew": sps.skew(off), "ev_kurtosis": sps.kurtosis(off),
            "frac_floor_le1p5": np.mean(off <= 1.5),
            "frac_le2": np.mean(off <= 2.0),
            "frac_mid_3to5": np.mean((off >= 3) & (off <= 5)),
            "frac_ge6": np.mean(off >= 6.0),
            "frac_ceil_ge6p5": np.mean(off >= 6.5),
            "hent_mean": ho.mean(), "hent_median": np.median(ho),
            "frac_hent_lt0p1": np.mean(ho < 0.1),
            "frac_hent_lt0p5": np.mean(ho < 0.5),
            "asym_frac": asym_frac, "asym_rms": np.sqrt(np.mean(ao ** 2)),
            # TRUE-distribution products:
            "mass_floor_c1": massC[0], "mass_ceil_c7": massC[6],
            "offmode_frac": float(np.mean(omm >= 0.25)),   # rgb bimodality metric
            "offmode_mass_mean": float(omm.mean()),
        }
        srow.update(cat_frac)
        srow.update({f"massC{c}": float(massC[c - 1]) for c in range(1, 8)})
        srow.update({f"shape_{k.replace('-', '_')}": v for k, v in shp_frac.items()})
        srows.append(srow)
        for c in range(1, 8):
            catrows.append((s, META[s][0], c, float(massC[c - 1])))
        shprows.append({"model": s, "display": META[s][0], "family": META[s][1],
                        "offmode_frac": float(np.mean(omm >= 0.25)), **shp_frac})
    pd.DataFrame(srows).to_csv(os.path.join(OUT, "model_summary.csv"), index=False)
    pd.DataFrame(catrows, columns=["model", "display", "cat", "mean_mass"]).to_csv(
        os.path.join(OUT, "full_catmass.csv"), index=False)
    pd.DataFrame(shprows).to_csv(os.path.join(OUT, "full_shape.csv"), index=False)
    print("  wrote marginals.csv, model_summary.csv, full_catmass.csv, full_shape.csv")

    # ---- curated "same EV, opposite shape" examples for Report I §6 ----
    # pick the most extreme semantic mixture per bimodal model, plus a matched-EV
    # unimodal partner from the sharpest (Gemma) rater.
    ex = []
    sharp = "Gemma27"  # ~0% off-mode
    Dsh, Bsh = Ds[sharp], Bs[sharp]
    sh_shapes = shapes_by_model[sharp]
    sh_i, sh_j = di[sh_shapes == "unimodal"], dj[sh_shapes == "unimodal"]
    sh_uni_ev = Bsh[sh_i, sh_j]; sh_uni_P = Dsh[sh_i, sh_j]
    for s in ("Phi4", "Aya"):
        D, B = Ds[s], Bs[s]
        Poff = D[~eye]
        shapes = shapes_by_model[s]
        sem = np.where(shapes == "semantic")[0]
        if sem.size == 0:
            continue
        # most "empty-middle" mixture: maximize p1+p7
        best = sem[np.argmax(Poff[sem, 0] + Poff[sem, 6])]
        bi, bj = di[best], dj[best]
        ev0 = B[bi, bj]
        ex.append({"model": s, "display": META[s][0], "adj_i": adj0[bi],
                   "adj_j": adj0[bj], "ev": float(ev0), "shape": "semantic",
                   **{f"p{c}": float(D[bi, bj, c - 1]) for c in range(1, 8)}})
        # matched-EV unimodal partner from the sharp rater
        m = np.argmin(np.abs(sh_uni_ev - ev0))
        ui, uj = sh_i[m], sh_j[m]
        ex.append({"model": sharp, "display": META[sharp][0], "adj_i": adj0[ui],
                   "adj_j": adj0[uj], "ev": float(sh_uni_ev[m]), "shape": "unimodal",
                   **{f"p{c}": float(sh_uni_P[m, c - 1]) for c in range(1, 8)}})
    pd.DataFrame(ex).to_csv(os.path.join(OUT, "full_examples.csv"), index=False)
    print(f"  wrote full_examples.csv: {len(ex)} rows")

    with open(os.path.join(OUT, "MANIFEST_full.json"), "w") as f:
        json.dump({"n_adjectives": n, "n_models": len(order), "models": order,
                   "n_dir_pairs": int(di.size), "n_sym_pairs": int(ti.size),
                   "source": "judge_dists_full", "canonical": True}, f, indent=2)
    z0 = np.load(paths[0], allow_pickle=True)
    with open(os.path.join(OUT, "prompt_full.txt"), "w") as fh:
        fh.write(str(z0["prompt"]))
    print("done.")


if __name__ == "__main__":
    main()
