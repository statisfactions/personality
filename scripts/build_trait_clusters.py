"""Build pre-registered trait clusters for the steering capstone (W17).

Reproducible, run ONCE -> instruments/trait_clusters.json. The procedure (not
ad-hoc hand-picking):

  1. HUMAN 525-PDA correlation matrix (523 adjectives; 2 DENY_LABELS
     Inspirational/Insensitive already removed upstream). Human-space =
     model-independent -> one canonical cluster set across all models.
  2. Distance = 1 - r (POLE-RESPECTING: antonyms r<0 -> dist>1 -> land in
     separate clusters; the eval-antonym merge can't put wonderful with awful).
     NOT REPRESENT cosine (where the merge would).
  3. Ward linkage (balanced ~8-member clusters; average/complete chain badly:
     average gave median-3 + a 76-blob).
  4. Fine cut: maxclust = round(523/8) = 65. Harvest clusters with 5-12 members
     and within-cluster mean off-diagonal r > 0.25 -> candidate POOL (~35).
  5. Coarse cut: maxclust = N_SELECT (8) -> higher-level dendrogram BRANCHES
     for stratification (empirical, not an imposed Big5/HEXACO taxonomy).
  6. Assign each pool cluster to its majority branch; pick the most-coherent
     pool cluster per branch. Force-include the EVALUATION cluster (contains
     'wonderful') as a built-in valence control. -> LOCKED selection of 8.

Usage:  PYTHONPATH=scripts .venv/bin/python scripts/build_trait_clusters.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import four_grid_compare as fg

N_SELECT = 8
FINE_K = 65
COH_MIN = 0.25
SIZE_LO, SIZE_HI = 5, 12
EVAL_SEED = "wonderful"   # the cluster containing this = valence control
OUT = Path("instruments/trait_clusters.json")


def main():
    H, labels = fg.load_human()
    low = [l.lower() for l in labels]
    H = np.nan_to_num(H, nan=0.0); np.fill_diagonal(H, 1.0)
    D = 1.0 - H; D = (D + D.T) / 2; np.fill_diagonal(D, 0.0); D = np.clip(D, 0, None)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="ward")

    def coherence(idx):
        sub = H[np.ix_(idx, idx)]
        return float(sub[np.triu_indices_from(sub, 1)].mean())

    # --- fine cut -> candidate pool -------------------------------------
    fine = fcluster(Z, t=FINE_K, criterion="maxclust")
    pool = []
    for c in range(1, fine.max() + 1):
        idx = np.where(fine == c)[0]
        if SIZE_LO <= len(idx) <= SIZE_HI:
            coh = coherence(idx)
            if coh > COH_MIN:
                pool.append({"members": [low[i] for i in idx],
                             "idx": idx.tolist(), "coh": coh})
    pool.sort(key=lambda p: -p["coh"])

    # --- coarse cut -> branches for stratification ----------------------
    coarse = fcluster(Z, t=N_SELECT, criterion="maxclust")
    for p in pool:
        br = np.bincount(coarse[p["idx"]]).argmax()
        p["branch"] = int(br)

    # --- stratified selection: best-coh cluster per branch --------------
    eval_idx = next(i for i, p in enumerate(pool)
                    if EVAL_SEED in p["members"])
    selected, used_branches = [], set()
    # force the evaluation cluster first
    selected.append(eval_idx); used_branches.add(pool[eval_idx]["branch"])
    for i, p in enumerate(pool):       # pool already sorted by coherence desc
        if len(selected) >= N_SELECT:
            break
        if p["branch"] not in used_branches:
            selected.append(i); used_branches.add(p["branch"])
    # if branches < N_SELECT, top up with next-most-coherent unused clusters
    for i, p in enumerate(pool):
        if len(selected) >= N_SELECT:
            break
        if i not in selected:
            selected.append(i)
    sel = [pool[i] for i in selected]

    def label(members):  # crude human-readable tag = highest-coh member-ish
        return members[0]

    result = {
        "procedure": "Ward linkage on 1-r human-corr; fine cut k=65, harvest "
                     "5-12 members coh>0.25; coarse cut k=8 branches; "
                     "best-coh per branch + forced evaluation cluster.",
        "params": {"fine_k": FINE_K, "n_select": N_SELECT, "coh_min": COH_MIN,
                   "size_range": [SIZE_LO, SIZE_HI], "eval_seed": EVAL_SEED},
        "pool": [{"members": p["members"], "coh": round(p["coh"], 3),
                  "branch": p["branch"]} for p in pool],
        "selected": [{"tag": label(p["members"]), "members": p["members"],
                      "coh": round(p["coh"], 3), "branch": p["branch"],
                      "is_eval_control": EVAL_SEED in p["members"]}
                     for p in sel],
    }
    OUT.parent.mkdir(exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=1)

    print(f"pool: {len(pool)} candidate clusters across "
          f"{len(set(p['branch'] for p in pool))} branches")
    print(f"\nSELECTED {len(sel)} (stratified by branch; * = eval/valence control):")
    for p in sel:
        star = " *" if EVAL_SEED in p["members"] else "  "
        print(f" {star} br{p['branch']} coh={p['coh']:+.2f}: "
              f"{', '.join(p['members'][:9])}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
