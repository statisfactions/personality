"""Vector-level REPRESENT vs ENACT geometry (W17).

The four-grid comparison relates the two channels' Gram matrices (rotation-
invariant). Here we ask about the actual vector sets: for each adjective,
REPRESENT (residual activation of the adjective in the pers carrier, centered
across adjectives) vs ENACT (persona-vector direction = role mean - role
centroid), both at the SAME layer. Decomposition:

  1. in-place overlap: direct per-adjective cosine; principal angles between
     the two top-k subspaces. (W4 predicts ~orthogonal.)
  2. rotation: orthogonal Procrustes after Frobenius normalization --
     "how well does SOME rotation superimpose the two configurations"
     (= shape match; equals 1 iff the grids agree perfectly).
  3. shear: within matched top-k PC coordinates, R^2 of a general linear map
     minus R^2 of the best orthogonal map. (Full-space linear fit is
     meaningless: n=523 < d, perfect fit guaranteed.)
  4. scale: correlation of per-adjective norms; normalized singular spectra.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/represent_enact_geometry.py \
      --model llama3.2 --layers 14 19
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hf_logprobs import MODELS

ACTS_DIR = Path("results/adjectives/acts")
ENACT_DIR = Path("results/persona_vectors")


def load_pair(short, layer):
    """Return (X=REPRESENT, Y=ENACT) as (n_adj, hidden), column-centered,
    aligned on common adjectives (lowercase)."""
    repo = MODELS[short]
    b = torch.load(ACTS_DIR / f"{repo.replace('/', '_')}__pers.pt",
                   weights_only=False)
    rep = {a.lower(): b["acts"][i, layer, :].astype(np.float64)
           for i, a in enumerate(b["adjectives"])}
    d = torch.load(ENACT_DIR / f"{short}_pda.pt", weights_only=False)
    ena = {a: v[layer].astype(np.float64) for a, v in d["directions"].items()}
    common = sorted(set(rep) & set(ena))
    X = np.stack([rep[a] for a in common])
    Y = np.stack([ena[a] for a in common])
    X -= X.mean(0)   # ENACT is centered by construction (vs role centroid)
    Y -= Y.mean(0)
    return X, Y, common


def principal_angles(X, Y, k):
    Vx = np.linalg.svd(X, full_matrices=False)[2][:k]
    Vy = np.linalg.svd(Y, full_matrices=False)[2][:k]
    s = np.linalg.svd(Vx @ Vy.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0, 1)))


def procrustes_stat(X, Y):
    """Procrustes correlation in a joint reduced basis: trace norm of
    X~^T Y~ with both Frobenius-normalized. 1 = same shape up to rotation."""
    Q = np.linalg.svd(np.vstack([X, Y]), full_matrices=False)[2]
    Xr, Yr = X @ Q.T, Y @ Q.T
    Xr /= np.linalg.norm(Xr)
    Yr /= np.linalg.norm(Yr)
    s = np.linalg.svd(Xr.T @ Yr, compute_uv=False)
    return float(s.sum())


def rotation_vs_linear(X, Y, k):
    """Within top-k PC coords of each set: R^2 of best orthogonal map vs
    general linear map. The gap is shear+anisotropic scale."""
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    A = Ux[:, :k] * Sx[:k]          # (n, k) coords
    B = Uy[:, :k] * Sy[:k]
    # orthogonal (+ global scale)
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    c = s.sum() / (A ** 2).sum()
    r2_rot = 1 - ((B - c * A @ R) ** 2).sum() / (B ** 2).sum()
    # general linear
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    r2_lin = 1 - ((B - A @ W) ** 2).sum() / (B ** 2).sum()
    return float(r2_rot), float(r2_lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--layers", type=int, nargs="+", default=[14, 19])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50])
    args = ap.parse_args()

    out = {"model": args.model, "layers": {}}
    for L in args.layers:
        X, Y, common = load_pair(args.model, L)
        n = len(common)
        cos = np.sum((X / np.linalg.norm(X, axis=1, keepdims=True)) *
                     (Y / np.linalg.norm(Y, axis=1, keepdims=True)), axis=1)
        norm_r = float(np.corrcoef(np.linalg.norm(X, axis=1),
                                   np.linalg.norm(Y, axis=1))[0, 1])
        pa = procrustes_stat(X, Y)
        res = {"n": n,
               "direct_cos_mean": float(cos.mean()),
               "direct_cos_mean_abs": float(np.abs(cos).mean()),
               "direct_cos_p95_abs": float(np.percentile(np.abs(cos), 95)),
               "norm_corr": norm_r,
               "procrustes_corr": pa}
        print(f"\n=== layer {L} (n={n}) ===")
        print(f"  direct cosine: mean={cos.mean():+.3f}  mean|cos|="
              f"{np.abs(cos).mean():.3f}  p95|cos|={np.percentile(np.abs(cos), 95):.3f}")
        print(f"  norm corr (||rep|| vs ||enact||): {norm_r:+.3f}")
        for k in args.k:
            ang = principal_angles(X, Y, k)
            r2r, r2l = rotation_vs_linear(X, Y, k)
            res[f"k{k}"] = {"princ_angles_deg_min_med_max":
                            [float(ang.min()), float(np.median(ang)),
                             float(ang.max())],
                            "r2_rotation": r2r, "r2_linear": r2l}
            print(f"  k={k:>3d}: principal angles min/med/max = "
                  f"{ang.min():.0f}/{np.median(ang):.0f}/{ang.max():.0f} deg"
                  f"   R2 rotation={r2r:.3f}  linear={r2l:.3f}  "
                  f"shear gap={r2l - r2r:+.3f}")
        # spectra (top-10, fraction of total variance)
        sx = np.linalg.svd(X, compute_uv=False)
        sy = np.linalg.svd(Y, compute_uv=False)
        res["spectrum_top10_frac"] = {
            "represent": (sx[:10] ** 2 / (sx ** 2).sum()).round(3).tolist(),
            "enact": (sy[:10] ** 2 / (sy ** 2).sum()).round(3).tolist()}
        print(f"  procrustes corr (shape match after best rotation): {pa:.3f}")
        print(f"  spectrum top-3 var frac: REPRESENT "
              f"{(sx[:3]**2/(sx**2).sum()).round(3)}  ENACT "
              f"{(sy[:3]**2/(sy**2).sum()).round(3)}")
        out["layers"][L] = res

    p = ENACT_DIR / f"rep_enact_geometry_{args.model}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
