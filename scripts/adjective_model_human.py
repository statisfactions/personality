#!/usr/bin/env python3
"""Model adjective geometry vs human 525-PDA: framing sensitivity + affect-merge.

For each (model, framing) cache, build the 523x523 adjective cosine matrix at
2/3 depth with center + top-1-PC denoise (the single-adjective analog of the
facet pipeline's meandiff-itempc1; centering alone kills the anisotropy floor,
top-1 PC matches the facet denoise). Then:

  1. framing sensitivity  : within each model, pairwise matrix-r among framings
  2. model vs human       : upper-tri Pearson of each model matrix vs the human
                            525-PDA correlation matrix (matched 523 adjectives)
  3. affect-merge crux    : cohort-tracked Cheerful-Angry vs Cheerful-Joyful and
                            Sympathetic-Angry vs Sympathetic-Kind, compared to
                            human. Does the single-ADJECTIVE model geometry merge
                            affect (like the IPIP-ITEM facet geometry did) or
                            separate it (like humans)?

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_model_human.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

ACTS = "results/adjectives/acts"
HUMAN = "results/adjectives/escs_525pda_corr_raw.json"
OUT = Path("results/adjectives/model_human_summary.json")
LAYER_FRAC = 2 / 3

PAIRS = [("Cheerful", "Joyful"), ("Cheerful", "Angry"),
         ("Sympathetic", "Kind"), ("Sympathetic", "Angry"),
         ("Happy", "Angry"), ("Cheerful", "Sad")]


def denoise(X):
    """center on adjective mean + remove top-1 PC, then L2-normalize rows."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    D = Xc - (Xc @ Vt[:1].T) @ Vt[:1]
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def upper_r(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def main():
    human = json.load(open(HUMAN))
    H = np.array(human["correlation_matrix"], float)
    hlabels = human["labels"]
    hidx = {l: i for i, l in enumerate(hlabels)}
    n = len(hlabels)

    # model -> framing -> {matrix (aligned to human order), crux pairs}
    mats = {}
    for p in sorted(glob.glob(f"{ACTS}/*.pt")):
        b = torch.load(p, weights_only=False)
        model = b["model"]; framing = b["framing"]
        adj = b["adjectives"]
        acts = b["acts"].astype(np.float32)
        nl = acts.shape[1] - 1
        L = round(nl * LAYER_FRAC)
        D = denoise(acts[:, L, :])
        C = D @ D.T
        # align rows/cols to human label order
        reorder = [adj.index(l) for l in hlabels]
        C = C[np.ix_(reorder, reorder)]
        mats.setdefault(model, {})[framing] = C
        print(f"  built {model:<34} {framing:<5} L{L}/{nl}")

    framings = ["self", "pers", "desc", "bare"]
    summary = {"layer_frac": LAYER_FRAC, "models": {}}

    # human crux references
    def hp(a, b): return float(H[hidx[a], hidx[b]])
    human_pairs = {f"{a}-{b}": hp(a, b) for a, b in PAIRS}

    print("\n=== model vs human (upper-tri matrix r) + framing agreement ===")
    print(f"  {'model':<34}{'self':>7}{'pers':>7}{'desc':>7}{'bare':>7}  {'framing-r':>10}")
    rows_for_mean = {f: [] for f in framings}
    for model, fm in mats.items():
        rh = {f: upper_r(fm[f], H) for f in framings if f in fm}
        for f in rh:
            rows_for_mean[f].append(rh[f])
        # framing agreement: mean pairwise among self/pers/desc (exclude bare)
        ctx = [f for f in ("self", "pers", "desc") if f in fm]
        prs = [upper_r(fm[a], fm[b]) for i, a in enumerate(ctx) for b in ctx[i+1:]]
        fr = float(np.mean(prs)) if prs else float("nan")
        cells = "".join(f"{rh.get(f, float('nan')):>7.3f}" for f in framings)
        print(f"  {model:<34}{cells}  {fr:>10.3f}")
        summary["models"][model] = {"r_to_human": rh, "framing_agreement": fr,
            "pairs": {f: {f"{a}-{b}": float(fm[f][hidx[a], hidx[b]]) for a, b in PAIRS}
                      for f in fm}}

    print("\n  cohort-mean r-to-human:  " +
          "  ".join(f"{f}={np.nanmean(rows_for_mean[f]):+.3f}" for f in framings))

    # ---- affect-merge crux, cohort-mean per framing ----
    print("\n=== affect-merge crux: cohort-mean cosine (human ref in []) ===")
    print(f"  {'pair':<20}{'human':>8}" + "".join(f"{f:>9}" for f in framings))
    for a, b in PAIRS:
        key = f"{a}-{b}"
        cells = ""
        for f in framings:
            vals = [summary["models"][m]["pairs"][f][key]
                    for m in mats if f in summary["models"][m]["pairs"]]
            cells += f"{np.mean(vals):>9.3f}"
        print(f"  {key:<20}{human_pairs[key]:>8.3f}{cells}")

    summary["human_pairs"] = human_pairs
    summary["cohort_mean_r_to_human"] = {f: float(np.nanmean(rows_for_mean[f])) for f in framings}
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
