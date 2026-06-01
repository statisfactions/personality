#!/usr/bin/env python3
"""Shared adjective-geometry denoise (W13 §3.11).

Building a model's adjective cosine matrix needs de-anisotropization, but the
right amount is REGIME-DEPENDENT (the §3.7/§3.10 two-regime story), as the
Gemma-3 check showed:

- Most models: centering removes the anisotropy/norm offset; the top PC of the
  CENTERED data is a real axis of variation (e.g. a settled/assertive contrast)
  and must be KEPT. PC1 is distributed across many dims (high IPR).
- Gemma-3 family: massive activations (~1e5 in 1-2 dims) survive centering and
  dominate the variance; centered-PC1 IS that rogue dimension (IPR = 1-2). Here
  the top PC must be REMOVED — it is the artifact, not signal.

`adaptive_denoise` routes automatically on PC1's inverse participation ratio:
remove top-1 only when PC1 is concentrated on < IPR_THRESHOLD effective dims.
"""
import numpy as np
import torch

IPR_THRESHOLD = 10.0
LAYER_FRAC = 2 / 3


def adaptive_denoise(X):
    """Center; remove top-1 PC iff PC1 is a concentrated rogue dim. Unit rows.

    Returns (D, removed_top1: bool, pc1_ipr: float)."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    ipr = float(1.0 / np.sum(v ** 4))
    removed = ipr < IPR_THRESHOLD
    if removed:
        Xc = Xc - (Xc @ Vt[:1].T) @ Vt[:1]
    D = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    return D, removed, ipr


def model_matrix(path, labels):
    """Load a cache, build the adaptive-denoised cosine matrix aligned to `labels`.

    Returns (short_name, C, removed_top1, pc1_ipr)."""
    b = torch.load(path, weights_only=False)
    A = b["acts"].astype(np.float32)
    L = round((A.shape[1] - 1) * LAYER_FRAC)
    adj = b["adjectives"]
    reorder = [adj.index(l) for l in labels]
    D, removed, ipr = adaptive_denoise(A[:, L, :][reorder])
    return b["model"].split("/")[-1], D @ D.T, removed, ipr
