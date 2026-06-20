"""Four-grid comparison: HUMAN / REPRESENT / JUDGE / ENACT (W17 spec).

Four 523^2 similarity structures over the 525-PDA adjectives:
  HUMAN     - raw self-report correlations (escs_525pda_corr_raw.json)
  REPRESENT - residual-stream cosine, pers framing, layer 2/3, adaptive denoise
  JUDGE     - tom_likely EV, both-directions symmetrized (introspect_full)
  ENACT     - persona-vector direction cosine at mid layer (persona_vectors)

Per the methods note (methods_persona_vectors.md, four-grid spec):
  - raw matrices, no ipsatization; Spearman primary / Pearson secondary on the
    common upper triangle
  - robustness rows applied SYMMETRICALLY to all four grids: own-PC1 removed
    (top eigencomponent of the zero-diagonal matrix) and double-centering
    (strip additive row/col prevalence marginals)
  - eval-antonym block (pos_eval x neg_eval, W15 groups) mean percentile per
    grid - the merge/split diagnostic
  - ENACT variants as robustness: raw vs massive-dim-ablated vs leak-filtered

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/four_grid_compare.py --model llama3.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from adjective_geom import model_matrix
from adjective_introspection import GROUPS
from hf_logprobs import MODELS

HUMAN_JSON = "results/adjectives/escs_525pda_corr_raw.json"
ACTS_DIR = Path("results/adjectives/acts")
JUDGE_DIR = Path("results/adjectives/introspect_full")
ENACT_DIR = Path("results/persona_vectors")

# cohort short name (persona_vectors naming) -> W14/W16 short name (acts/judge)
JUDGE_NAME = {"llama3.2": "Llama", "qwen2.5": "Qwen", "gemma3": "Gemma",
              "phi4": "Phi4"}


def load_human():
    h = json.load(open(HUMAN_JSON))
    return np.array(h["correlation_matrix"], float), list(h["labels"])


def load_represent(short, labels):
    repo = MODELS[short]
    path = ACTS_DIR / f"{repo.replace('/', '_')}__pers.pt"
    _, C, removed, ipr = model_matrix(path, labels)
    return C.astype(float), {"pc1_removed": bool(removed), "pc1_ipr": ipr}


def load_judge(short, labels):
    name = JUDGE_NAME.get(short, short)
    z = np.load(JUDGE_DIR / f"{name}_tom_likely_dir.npz", allow_pickle=True)
    adj = list(z["adjectives"])
    reorder = [adj.index(l) for l in labels]
    return z["B"].astype(float)[np.ix_(reorder, reorder)]


def load_enact(short, labels, ablate=False, max_leak=None):
    d = torch.load(ENACT_DIR / f"{short}_pda.pt", weights_only=False)
    mid = d["report"]["mid_layer"]
    hid = next(iter(d["directions"].values())).shape[1]
    mask = np.ones(hid)
    if ablate:
        mask[d["report"]["massive_dims"]] = 0.0
    leak = {a: v["leak_rate"] for a, v in d["report"]["adjectives"].items()}
    V = np.full((len(labels), hid), np.nan)
    for i, lab in enumerate(labels):
        k = lab.lower()
        if k in d["directions"] and (max_leak is None or leak[k] <= max_leak):
            v = d["directions"][k][mid].astype(np.float64) * mask
            V[i] = v / np.linalg.norm(v)
    return V @ V.T


def offdiag_mask(n):
    return ~np.eye(n, dtype=bool)


def remove_pc1(M):
    """Subtract the top eigencomponent of the zero-diagonal matrix."""
    A = M.copy()
    np.fill_diagonal(A, 0.0)
    w, v = np.linalg.eigh(A)
    k = np.argmax(np.abs(w))
    return A - w[k] * np.outer(v[:, k], v[:, k])


def double_center(M):
    """Strip additive row/col effects, computed on off-diagonal cells."""
    A = M.copy()
    np.fill_diagonal(A, np.nan)
    r = np.nanmean(A, axis=1, keepdims=True)
    c = np.nanmean(A, axis=0, keepdims=True)
    g = np.nanmean(A)
    return A - r - c + g


def block_percentile(M, labels):
    """Mean off-diagonal percentile of the pos_eval x neg_eval block."""
    idx = {l: i for i, l in enumerate(labels)}
    pos = [idx[w] for w in GROUPS["pos_eval"] if w in idx]
    neg = [idx[w] for w in GROUPS["neg_eval"] if w in idx]
    ut = M[np.triu_indices_from(M, k=1)]
    ut = ut[np.isfinite(ut)]
    cells = [M[i, j] for i in pos for j in neg if np.isfinite(M[i, j])]
    return float(np.mean([(ut < c).mean() for c in cells]))


def compare(grids, labels):
    """All pairwise Spearman/Pearson on the common upper triangle, for raw,
    PC1-removed, and double-centered variants (transforms applied to all
    grids symmetrically)."""
    names = list(grids)
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    valid = np.ones(len(iu[0]), bool)
    for M in grids.values():
        valid &= np.isfinite(M[iu])
    out = {"n_cells": int(valid.sum()), "pairs": {}}
    variants = {
        "raw": grids,
        "pc1_removed": {k: remove_pc1(np.nan_to_num(M)) for k, M in grids.items()},
        "double_centered": {k: double_center(M) for k, M in grids.items()},
    }
    for vname, gs in variants.items():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = gs[a][iu][valid], gs[b][iu][valid]
                ok = np.isfinite(x) & np.isfinite(y)
                sp = spearmanr(x[ok], y[ok]).statistic
                pe = pearsonr(x[ok], y[ok]).statistic
                out["pairs"].setdefault(f"{a}|{b}", {})[vname] = {
                    "spearman": float(sp), "pearson": float(pe)}
    out["antonym_block_pct"] = {k: block_percentile(M, labels)
                                for k, M in grids.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2",
                    help="cohort short name (persona_vectors naming)")
    ap.add_argument("--max-leak", type=float, default=0.30,
                    help="leak threshold for the ENACT leak-filtered variant")
    args = ap.parse_args()

    H, labels = load_human()
    R, rinfo = load_represent(args.model, labels)
    J = load_judge(args.model, labels)
    # Ablate ENACT massive dims in the MAIN grid: non-ablated is misleading for
    # the Gemma family (massive channels survive vs-mean and tank HUMAN|ENACT
    # spuriously — gemma3 0.30 raw -> 0.59 ablated). Negligible on clean models
    # (Llama 0.747->0.750, Qwen 0.617->0.622). Raw kept as a variant below.
    E = load_enact(args.model, labels, ablate=True)
    print(f"model={args.model}  REPRESENT: pc1_removed={rinfo['pc1_removed']} "
          f"(ipr={rinfo['pc1_ipr']:.1f})")

    grids = {"HUMAN": H, "REPRESENT": R, "JUDGE": J, "ENACT": E}
    result = {"model": args.model, "represent_info": rinfo,
              "main": compare(grids, labels)}

    print(f"\ncommon cells: {result['main']['n_cells']}")
    print(f"\n{'pair':>22s} {'raw S/P':>15s} {'pc1- S/P':>15s} {'dctr S/P':>15s}")
    for pair, v in result["main"]["pairs"].items():
        print(f"{pair:>22s} "
              f"{v['raw']['spearman']:+.3f}/{v['raw']['pearson']:+.3f}   "
              f"{v['pc1_removed']['spearman']:+.3f}/{v['pc1_removed']['pearson']:+.3f}   "
              f"{v['double_centered']['spearman']:+.3f}/{v['double_centered']['pearson']:+.3f}")
    print("\nantonym block mean percentile (pos_eval x neg_eval):")
    for k, p in result["main"]["antonym_block_pct"].items():
        print(f"  {k:>10s}: {100*p:.0f}th")

    # ENACT robustness: ablated and leak-filtered variants, ENACT edges only
    result["enact_variants"] = {}
    for tag, kw in [("raw_nonablated", dict(ablate=False)),
                    ("leak_filtered", dict(ablate=True, max_leak=args.max_leak))]:
        Ev = load_enact(args.model, labels, **kw)
        gv = {"HUMAN": H, "REPRESENT": R, "JUDGE": J, "ENACT": Ev}
        c = compare(gv, labels)
        result["enact_variants"][tag] = {
            "n_cells": c["n_cells"],
            **{p: c["pairs"][p]["raw"] for p in c["pairs"] if "ENACT" in p},
            "antonym_block_pct_enact": c["antonym_block_pct"]["ENACT"],
        }
        print(f"\nENACT variant [{tag}] (n={c['n_cells']}): " + "  ".join(
            f"{p}={c['pairs'][p]['raw']['spearman']:+.3f}"
            for p in c["pairs"] if "ENACT" in p)
            + f"  antonym_pct={100*c['antonym_block_pct']['ENACT']:.0f}th")

    out = ENACT_DIR / f"four_grid_{args.model}.json"
    json.dump(result, open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
