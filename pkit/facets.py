"""Adjective facet clustering — the visualization/aggregation layer.

Two standing partitions of the adjective space (instruments/):
  "blocks44"   trait_blocks_44.json — 44-block v2 partition (no size cap,
               majors included); the current default for dashboards/slides.
  "clusters35" trait_clusters.json — frozen 35-cluster file, kept for
               W17-W18 continuity.

Cluster display order is (branch, -coherence); each cluster is labeled by
its medoid under the HUMAN raw correlation matrix (the member with max
mean correlation to the others) — conventions from
adjective_facet_cohort.py / adjective_facet_dashboard.py, verbatim.
"""
import json

import numpy as np
import pandas as pd

from . import load as _load
from . import paths

FILES = {"blocks44": "trait_blocks_44.json",
         "clusters35": "trait_clusters.json"}


def clusters(which="blocks44", labels=None):
    """Ordered cluster list: dicts with label (human medoid), members, idx
    (positions in `labels`, default the canonical adjective order), branch,
    coh, plus tag/major where the file provides them."""
    labels = labels if labels is not None else _load.adjectives()
    tc = json.load(open(paths.ROOT / "instruments" / FILES[which]))
    H = _load.human_corr().values
    hl = list(_load.human_corr().index)
    out = []
    for c in sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"])):
        hidx = [hl.index(m) for m in c["members"]]
        sub = H[np.ix_(hidx, hidx)]
        entry = dict(c)
        entry["label"] = c["members"][int(np.argmax(sub.mean(1)))]
        entry["idx"] = [labels.index(m) for m in c["members"]]
        out.append(entry)
    return out


def block(S, cl):
    """Cluster-block aggregation of an adjective x adjective matrix:
    mean over each block, off-diagonal-only on the diagonal blocks.
    Returns a DataFrame indexed by medoid labels."""
    S = np.asarray(S, float)
    k = len(cl)
    B = np.zeros((k, k))
    for i, ci in enumerate(cl):
        for j, cj in enumerate(cl):
            sub = S[np.ix_(ci["idx"], cj["idx"])]
            if i == j:
                m = ~np.eye(len(ci["idx"]), dtype=bool)
                B[i, j] = sub[m].mean()
            else:
                B[i, j] = sub.mean()
    names = [c["label"] for c in cl]
    return pd.DataFrame(B, index=names, columns=names)
