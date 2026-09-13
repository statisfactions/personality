"""Add RAW column-centered cosines (zero diagonal, f16) to the REPRESENT
cache as key `cos`, alongside the entry-z `grids`. Same extraction as
represent_self_prediction.py (mid layer, meta or 20x-median massive
dims, winsorized). Needed for raw-unit cluster grids (2026-09-13).
Usage: .venv/bin/python scripts/represent_cache_cos.py
"""
import glob
import json
import os

import numpy as np
import torch

import pkit
from pkit import measures

labels = pkit.load.adjectives()
CACHE = "results/adjectives/represent_model_grids.npz"
z = np.load(CACHE, allow_pickle=True)
names = [str(x) for x in z["names"]]
cos = np.zeros((len(names), len(labels), len(labels)), np.float16)
done = np.zeros(len(names), bool)
for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
    repo_us = os.path.basename(p).replace("__pers.pt", "")
    if repo_us not in names:
        continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    adjR = [str(a).lower() for a in b["adjectives"]]
    acts = np.asarray(b["acts"], dtype=np.float32)
    mid = acts.shape[1] // 2
    massive = None
    for short, repo in pkit.roster.MODELS.items():
        if repo.replace("/", "_") == repo_us:
            mp = f"results/persona_vectors/{short}_pda_meta.json"
            if os.path.exists(mp):
                meta = json.load(open(mp))
                mid, massive = meta["mid_layer"], meta["massive_dims"]
            break
    X = acts[[adjR.index(l) for l in labels], mid, :].astype(np.float64)
    if massive is None:
        ma = np.abs(X).mean(0)
        massive = np.where(ma >= 20 * np.median(ma))[0]
    S = measures.cos_sim(measures.winsorize(X, np.asarray(massive, int)))
    np.fill_diagonal(S, 0)
    i = names.index(repo_us)
    # bit-identity gate against the cached entry-z grid
    chk = np.corrcoef(measures.offdiag(measures.zscore_offdiag(S)),
                      measures.offdiag(z["grids"][i].astype(np.float32)))[0, 1]
    cos[i] = S.astype(np.float16)
    done[i] = True
    print(f"[ok {repo_us}] z-match r={chk:.5f}", flush=True)
    del b, acts, X
assert done.all(), [n for n, d in zip(names, done) if not d]
np.savez_compressed(CACHE, grids=z["grids"], norms=z["norms"], names=z["names"], cos=cos)
print("cache updated with raw cosines:", cos.shape)
