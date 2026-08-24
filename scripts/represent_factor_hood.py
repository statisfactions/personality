"""REPRESENT channel factor-hood via bootstrap-over-models (2026-08-23).

The channel analog of the human varimax certification: 63 wide-cohort
REPRESENT grids are the "respondents"; resample models, refactor the
consensus (mean) cosine grid, Tucker-match to the full-sample solution.
Certified-factor count per k is then comparable to human 5 raw / 7
ipsatized. Registered predictions (to_try 2026-08-23): consensus
certifies 2-3 (evaluative core + affect-presence); JUDGE > REPRESENT;
ENACT fewest (both too low-n to run: 12 / 10 models).

Per-model grid construction matches facet_slides_wide.collect_represent:
mid stored layer, per-model massive-dim winsorization, mean-centered,
row-unit cosine. Raw cosine kept (diag=1) — factoring semantics; the
display pipeline's zscore_offdiag is for cross-channel comparability
and is affine anyway.

Usage: PYTHONPATH=scripts python scripts/represent_factor_hood.py
Cache: results/adjectives/represent_permodel_S.npz  (63 x 523 x 523 fp32)
Out:   results/adjectives/represent_factor_hood.json + stdout
"""
import glob
import json
import os

import numpy as np
import torch

import facet_slides_wide as fw
from adjective_factor_bootstrap import kfactors, match_cong

CACHE = "results/adjectives/represent_permodel_S.npz"
ACTS_DIR = "results/adjectives/acts"


def build_cache(labels):
    mats, names = [], []
    for p in sorted(glob.glob(f"{ACTS_DIR}/*__pers.pt")):
        if fw.EXCLUDE.search(os.path.basename(p)):
            continue
        try:
            b = torch.load(p, map_location="cpu", weights_only=False)
            adj = [str(a).lower() for a in b["adjectives"]]
            idx = [adj.index(l) for l in labels]
            A = np.asarray(b["acts"])
            mid = (A.shape[1] - 1) // 2
            X = A[idx, mid, :].astype(np.float64)
            ma = np.abs(X).mean(0)
            med = np.median(ma)
            massive = np.where(ma >= 20 * med)[0]
            keep = np.setdiff1d(np.arange(X.shape[1]), massive)
            std = X.std(0)
            cap = std[keep].max() if len(keep) else std.max()
            for d_ in massive:
                if std[d_] > cap:
                    X[:, d_] *= cap / std[d_]
            Xc = X - X.mean(0)
            Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
            mats.append((Xn @ Xn.T).astype(np.float32))
            names.append(os.path.basename(p).replace("__pers.pt", ""))
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {type(e).__name__}")
    np.savez_compressed(CACHE, S=np.stack(mats), names=np.array(names))
    print(f"cached {len(names)} per-model grids -> {CACHE}")


def main():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    if not os.path.exists(CACHE):
        build_cache(labels)
    z = np.load(CACHE)
    S, names = z["S"].astype(np.float64), list(z["names"])
    n_m = len(names)
    print(f"{n_m} per-model REPRESENT grids loaded")

    Sbar = S.mean(0)
    rng = np.random.default_rng(0)
    B = 100
    out = {"n_models": n_m, "names": [str(n) for n in names], "ks": {}}
    for k in [2, 3, 4, 5, 6, 7]:
        L0, var0 = kfactors(Sbar, k)
        rec = np.zeros(k)
        for b in range(B):
            bs = rng.integers(0, n_m, n_m)
            Lb, _ = kfactors(S[bs].mean(0), k)
            rec += (match_cong(L0, Lb) >= 0.90)
        rec /= B
        stable = int((rec >= 0.5).sum())
        poles = []
        for j in range(k):
            o = np.argsort(-L0[:, j])
            u = np.argsort(L0[:, j])
            poles.append({"pos": [labels[i] for i in o[:8]],
                          "neg": [labels[i] for i in u[:8]],
                          "p_cong": float(rec[j]), "var": float(var0[j])})
        out["ks"][k] = {"stable": stable, "factors": poles}
        print(f"k={k}: P(cong>=.90) " + " ".join(f"{r:.2f}" for r in rec)
              + f"  -> {stable}/{k} certified")
        if k <= 4:
            for j, f in enumerate(poles):
                print(f"   F{j+1} (var {f['var']:.0f}, P {f['p_cong']:.2f})"
                      f"  +: {', '.join(f['pos'][:6])}")
                print(f"        -: {', '.join(f['neg'][:6])}")
    with open("results/adjectives/represent_factor_hood.json", "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote results/adjectives/represent_factor_hood.json")


if __name__ == "__main__":
    main()
