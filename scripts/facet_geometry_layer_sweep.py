#!/usr/bin/env python3
"""Depth profile of facet-geometry recovery (W13 §3.10; logit-lens test).

§3.9 showed the facet covariance geometry our models recover is item semantics,
recoverable by any encoder. rgb's logit-lens prediction: if the geometry lives
"so close to the textual level," it should already be present in the LLM's own
*token embeddings* — i.e. at layer 0 (HF hidden_states[0] = embedding-layer
output, before any transformer block) — not built up with depth.

This rebuilds the meandiff-itempc1 facet matrix at EVERY layer from the cached
IPIP activations (results/phase_b_cache_ipip/, mean-all-skip0 token aggregation)
and correlates each layer's 30x30 cosine matrix against the human facet
correlation matrix. If layer 0 is already near the peak, the geometry is in the
static token-embedding table (the strongest form of the embedding-view result).

Free: no model loads, reads existing cache only.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/facet_geometry_layer_sweep.py
"""

import glob
import json
import os

import numpy as np
import torch

from ipip_facet_cluster import FACETS, TRAIT_ORDER, unit

HUMAN_PATH = "instruments/ipip300_human_facet_correlations.json"
CACHE_GLOB = "results/phase_b_cache_ipip/*_ipip_chat.pt"
FACET_ORDER = [f"{t}:{f}" for t in TRAIT_ORDER for f in FACETS[t]]

# repo-tag -> short cohort name (match hf_logprobs / ipip_facet_cluster labels)
TAG_TO_NAME = {
    "google_gemma-3-4b-it": "Gemma", "google_gemma-3-12b-it": "Gemma12",
    "google_gemma-3-27b-it": "Gemma27", "google_gemma-4-31B-it": "Gemma4",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama8",
    "microsoft_Phi-4-mini-instruct": "Phi4",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen", "Qwen_Qwen2.5-7B-Instruct": "Qwen7",
    "Qwen_Qwen2.5-32B-Instruct": "Qwen32",
    "CohereLabs_aya-expanse-8b": "Aya",
    "tiiuae_falcon-mamba-7b-instruct": "FalconMamba",
}


def top1_pc(X):
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0]


def facet_matrix_at_layer(acts, meta, layer):
    """meandiff-itempc1 at a given layer. acts: (n_items, n_layers+1, hidden)."""
    L = acts[:, layer, :].astype(np.float64)
    pc1 = top1_pc(L)
    pc1u = pc1 / (np.linalg.norm(pc1) + 1e-12)
    by = {}
    for i, m in enumerate(meta):
        by.setdefault((m["trait"], m["facet"]), {"fwd": [], "rev": []})[m["pole"]].append(L[i])
    rows = []
    for t in TRAIT_ORDER:
        for f in FACETS[t]:
            pol = by[(t, f)]
            raw = np.mean(pol["fwd"], axis=0) - np.mean(pol["rev"], axis=0)
            d = raw - np.dot(raw, pc1u) * pc1u
            rows.append(unit(d))
    D = np.vstack(rows)
    return D @ D.T


def upper(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def main():
    H = np.array(json.load(open(HUMAN_PATH))["correlation_matrix"], dtype=float)
    hu = upper(H)

    print(f"{'model':<12}{'L0':>7}{'peak':>7}{'@lyr':>6}{'2/3':>7}"
          f"{'L0/peak':>9}{'L0/2/3':>8}")
    print("-" * 57)
    summary = {}
    for f in sorted(glob.glob(CACHE_GLOB)):
        tag = os.path.basename(f).replace("_ipip_chat.pt", "")
        name = TAG_TO_NAME.get(tag, tag)
        blob = torch.load(f, weights_only=False)
        acts = blob["acts"] if isinstance(blob["acts"], np.ndarray) else blob["acts"].numpy()
        meta = blob["meta"]
        n_layers = acts.shape[1] - 1
        mid = int(round(n_layers * 2 / 3))

        rs = []
        for L in range(acts.shape[1]):
            M = facet_matrix_at_layer(acts, meta, L)
            r = float(np.corrcoef(upper(M), hu)[0, 1])
            rs.append(r)
        rs = np.array(rs)
        peak_layer = int(np.argmax(rs))
        r0, rpeak, rmid = rs[0], rs[peak_layer], rs[mid]
        summary[name] = {"n_layers": n_layers, "r_by_layer": rs.tolist(),
                         "L0": r0, "peak": rpeak, "peak_layer": peak_layer,
                         "mid_2_3": rmid}
        print(f"{name:<12}{r0:>7.3f}{rpeak:>7.3f}{peak_layer:>6d}{rmid:>7.3f}"
              f"{r0/rpeak:>9.2f}{r0/rmid:>8.2f}")

    L0s = [v["L0"] for v in summary.values()]
    peaks = [v["peak"] for v in summary.values()]
    mids = [v["mid_2_3"] for v in summary.values()]
    print("-" * 57)
    print(f"{'COHORT':<12}{np.mean(L0s):>7.3f}{np.mean(peaks):>7.3f}{'':>6}"
          f"{np.mean(mids):>7.3f}{np.mean(L0s)/np.mean(peaks):>9.2f}"
          f"{np.mean(L0s)/np.mean(mids):>8.2f}")
    print(f"\nLayer 0 = token-embedding table (pre-block). If L0/peak ~ 1, the")
    print(f"facet geometry is already in the static token embeddings (logit-lens")
    print(f"prediction confirmed: it's lexical, not computed with depth).")

    out = "results/facets/facet_geometry_layer_sweep.json"
    json.dump({"facet_order": FACET_ORDER, "human_within": float(np.mean(hu[hu > 0])),
               "models": summary}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
