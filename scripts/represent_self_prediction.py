"""REPRESENT <-> SELF, both framings (2026-09-12, rgb's gap b).

Stage 1: per-model REPRESENT RDMs from acts (mid layer, winsorized,
zscore_offdiag; deep models take mid/massive from pda_meta, wide models
compute mid=n_layers//2 and massive = mean|act| >= 20x median).
Cached to results/adjectives/represent_model_grids.npz (f16).
Stage 2: (i) second-order RSA vs SELF first-order distances (Mantel);
(ii) incremental: idio-represent grid vs idio-self pair products;
(iii) amplitude: idio act-norms vs idio-self levels.

Usage: .venv/bin/python scripts/represent_self_prediction.py
"""
import glob
import json
import os

import numpy as np
import torch

import pkit
from pkit import measures

labels = pkit.load.adjectives()
FR = pkit.load.FRAMINGS
CACHE = "results/adjectives/represent_model_grids.npz"
iu = np.triu_indices(len(labels), 1)

if not os.path.exists(CACHE):
    grids, norms, names = [], [], []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        repo_us = os.path.basename(p).replace("__pers.pt", "")
        try:
            b = torch.load(p, map_location="cpu", weights_only=False)
            adjR = [str(a).lower() for a in b["adjectives"]]
            if not set(labels) <= set(adjR):
                print(f"[skip {repo_us}] missing adjectives", flush=True)
                continue
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
            grids.append(measures.zscore_offdiag(S).astype(np.float16))
            norms.append(np.linalg.norm(X, axis=1))
            names.append(repo_us)
            print(f"[ok {repo_us}] mid={mid} massive={len(np.atleast_1d(massive))}",
                  flush=True)
            del b, acts, X
        except Exception as e:
            print(f"[fail {repo_us}] {type(e).__name__}: {e}", flush=True)
    np.savez_compressed(CACHE, grids=np.array(grids), norms=np.array(norms),
                        names=np.array(names))
    print(f"cached {len(names)} model grids", flush=True)

z = np.load(CACHE, allow_pickle=True)
names = [str(n) for n in z["names"]]
grids = z["grids"].astype(np.float32)
norms = z["norms"].astype(np.float64)

# SELF profiles for the same models
S = {}
for i, n in enumerate(names):
    try:
        d = json.load(open(pkit.load._self_file(n.replace("_", "/", 1))))["results"]
        X = np.array([[d[f][a]["ev"] for a in labels] for f in FR])
        S[n] = measures.ipsatize(X.mean(0)[None, :])[0]
    except Exception:
        pass
keep = [i for i, n in enumerate(names) if n in S]
names = [names[i] for i in keep]
grids = grids[keep]; norms = norms[keep]
n = len(names)
print(f"\nanalysis n = {n} models")

Sm = np.array([S[x] for x in names])
Sbar = Sm.mean(0)
gv = np.array([g[iu] for g in grids])

# (i) second-order RSA
ds, dr = [], []
for i in range(n):
    for j in range(i + 1, n):
        dr.append(1 - np.corrcoef(gv[i], gv[j])[0, 1])
        ds.append(np.linalg.norm(Sm[i] - Sm[j]))
r_rsa = np.corrcoef(ds, dr)[0, 1]
rng = np.random.default_rng(0)
Dm = np.zeros((n, n)); Dm[np.triu_indices(n, 1)] = ds; Dm += Dm.T
null = []
for _ in range(2000):
    p_ = rng.permutation(n)
    null.append(np.corrcoef(Dm[np.ix_(p_, p_)][np.triu_indices(n, 1)], dr)[0, 1])
pval = (np.sum(np.abs(null) >= abs(r_rsa)) + 1) / 2001
print(f"(i) Mantel r(SELF distance, REPRESENT-RDM distance) = {r_rsa:+.3f}, p = {pval:.4f}")

# (ii) incremental pairwise + (iii) amplitude
gbar = gv.mean(0)
nz = (norms - norms.mean(1, keepdims=True)) / norms.std(1, keepdims=True)
nbar = nz.mean(0)
r_pair, r_amp = [], []
for i in range(n):
    loo = (gv.sum(0) - gv[i]) / (n - 1)
    idio_g = gv[i] - loo
    idio_s = Sm[i] - Sbar
    r_pair.append(np.corrcoef(np.outer(idio_s, idio_s)[iu], idio_g)[0, 1])
    loo_n = (nz.sum(0) - nz[i]) / (n - 1)
    r_amp.append(np.corrcoef(idio_s, nz[i] - loo_n)[0, 1])
from scipy import stats
for nm, arr in (("(ii) pairwise", np.array(r_pair)), ("(iii) amplitude", np.array(r_amp))):
    t, p = stats.ttest_1samp(arr, 0)
    print(f"{nm}: mean r {np.mean(arr):+.3f}, positive {np.sum(np.array(arr)>0)}/{n}, "
          f"t({n-1})={t:+.2f}, p={p:.4f}")
