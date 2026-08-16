"""Build the 44-block standing partition (v2 of the adjective facets).

Same procedure as build_trait_clusters.py's harvest with the size CEILING
LIFTED (rgb 2026-08-16): Ward on pole-respecting 1-r human distance,
fine cut maxclust=65, keep clusters with >=5 members and within-cluster
mean off-diagonal r > 0.25. The 9 blocks the old 5-12 window excluded are
the trait MAJORS (warmth .47, trait anger .43, anxiety .41, honesty .41,
pos-eval .44, attractiveness .41, likeability, worthlessness, moral
unreliability) — mostly more coherent than the kept mean. 44 blocks,
448/523 adjectives (86%). Geometry-channel congruences move <=.03 vs the
frozen 35; SELF rises .37->.43 (majors are its best material).

The 35-cluster instruments/trait_clusters.json stays FROZEN for
continuity with W17-W18 reports; new facet work uses this file.

Usage: PYTHONPATH=scripts python scripts/build_trait_blocks44.py
Out:   instruments/trait_blocks_44.json
"""
import json

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

SIZE_MIN, COH_MIN, FINE_K, N_BRANCH = 5, 0.25, 65, 8
OUT = "instruments/trait_blocks_44.json"


def main():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    H0 = np.array(h["correlation_matrix"], float)
    D = np.clip(1 - H0, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="ward")
    fine = fcluster(Z, t=FINE_K, criterion="maxclust")
    coarse = fcluster(Z, t=N_BRANCH, criterion="maxclust")
    Hoff = H0.copy()
    np.fill_diagonal(Hoff, np.nan)
    blocks = []
    for k in sorted(set(fine)):
        ix = np.where(fine == k)[0]
        if len(ix) < SIZE_MIN:
            continue
        coh = float(np.nanmean(Hoff[np.ix_(ix, ix)]))
        if not coh > COH_MIN:
            continue
        branch = int(np.bincount(coarse[ix]).argmax())
        tag = labels[ix[int(np.argmax(np.nansum(H0[np.ix_(ix, ix)], 1)))]]
        blocks.append({"tag": tag, "members": [labels[i] for i in ix],
                       "coh": round(coh, 3), "branch": branch,
                       "major": len(ix) > 12})
    out = {"procedure": ("Ward on pole-respecting 1-r (escs_525pda_corr_raw"
                         ".json, 523 adj), fine cut maxclust=65, keep "
                         "size>=5 & coh>0.25 — NO upper size cap (v2; the "
                         "5-12 window excluded the trait majors). Branch = "
                         "majority coarse-cut-8 label. Frozen "
                         "trait_clusters.json (35) kept for W17-W18 "
                         "continuity."),
           "params": {"fine_k": FINE_K, "size_min": SIZE_MIN,
                      "coh_min": COH_MIN, "n_branch": N_BRANCH},
           "pool": blocks}
    json.dump(out, open(OUT, "w"), indent=1)
    n = sum(len(b["members"]) for b in blocks)
    print(f"wrote {OUT}: {len(blocks)} blocks, {n} adjectives, "
          f"{sum(b['major'] for b in blocks)} majors")


if __name__ == "__main__":
    main()
