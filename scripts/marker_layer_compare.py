#!/usr/bin/env python3
"""Matched W_E-vs-mid-layer comparison on the SAME single-token markers (W13 §3.10).

The W_E marker test (marker_embedding_geometry.py) shows a weak trait-clustering
seed in the static token embeddings. The facet layer sweep
(facet_geometry_layer_sweep.py) shows item geometry is built up with depth. This
closes the loop with matched stimuli: run the *same* single-token Saucier/
Goldberg markers through each model and compute the identical clustering metrics
at (a) layer 0 = W_E and (b) the 2/3-depth middle layer, so the only thing that
changes is depth.

Markers are run bare (BOS + single marker token); we read the marker position's
hidden state at the middle layer — a contextless contextualization that isolates
what depth does to the token, with no chat-template dilution.

Loads each model (forward pass). Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/marker_layer_compare.py
"""

import gc
import json
import os
from collections import defaultdict

import numpy as np
import torch

import sys
sys.path.insert(0, "scripts")
import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS, load_model
from marker_embedding_geometry import (SAUCIER, GOLDBERG, single_token_id, unit,
                                       load_embedding_matrix, TRAITS)

CACHE_DIR = "results/phase_b_cache"


def safe_tag(repo):
    return repo.replace("/", "_")

OUT_PATH = "results/facets/marker_layer_compare.json"
MARKER_SETS = {"saucier": SAUCIER, "goldberg": GOLDBERG}
COHORT = ["Gemma", "Gemma12", "Gemma27", "Gemma4", "Llama", "Llama8",
          "Phi4", "Qwen", "Qwen7", "Qwen32", "Aya", "FalconMamba"]


def _pair_means(cos, labels):
    n = len(labels)
    within = [cos[i, j] for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j]]
    across = [cos[i, j] for i in range(n) for j in range(i + 1, n) if labels[i] != labels[j]]
    return float(np.mean(within)), float(np.mean(across))


def metrics(V, labels, poles, proj_vectors=None):
    """Clustering metrics with an anisotropy axis projected out. Bare single-
    token reps at depth are extremely anisotropic (within~across~1.0), so raw
    cosine is uninformative. proj_vectors: unit PC rows to remove (e.g. the
    neutral-corpus PCs, mirroring single-pcs); if None, remove the top-1 PC of
    the marker matrix itself. We also report raw within/across as the
    anisotropy indicator."""
    n = len(labels)
    raw_within, raw_across = _pair_means(V @ V.T, labels)

    # Mean-center FIRST. Gemma-class models carry massive activations (a few
    # near-constant dims ~1000x the median) that saturate raw cosine to ~1.0;
    # meandiff extraction cancels them by subtraction, so to match it we must
    # subtract the set mean here too. PC-projection alone does NOT remove a
    # near-constant offset (it has little variance, so it's not in the top PCs,
    # yet it dominates the norm). Center, then remove residual anisotropy PC.
    Vc = V - V.mean(0, keepdims=True)
    if proj_vectors is None:
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        pcs = Vt[:1]
    else:
        pcs = proj_vectors
    Vp = Vc
    for pc in pcs:
        pcu = pc / (np.linalg.norm(pc) + 1e-12)
        Vp = Vp - (Vp @ pcu)[:, None] * pcu
    Vp = Vp / (np.linalg.norm(Vp, axis=1, keepdims=True) + 1e-12)
    cos = Vp @ Vp.T
    within, across = _pair_means(cos, labels)

    nn = 0
    for i in range(n):
        s = cos[i].copy(); s[i] = -np.inf
        if labels[int(np.argmax(s))] == labels[i]:
            nn += 1
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = np.clip(1 - cos, 0, 2); np.fill_diagonal(dist, 0); dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    cl = fcluster(Z, t=5, criterion="maxclust")
    c2t = defaultdict(list)
    for c, t in zip(cl, labels):
        c2t[int(c)].append(t)
    purity = sum(max(g.count(t) for t in set(g)) for g in c2t.values()) / n
    # meandiff trait-direction off-diagonal (on projected vecs)
    dirs = []
    for tr in TRAITS:
        hi = [Vp[i] for i in range(n) if labels[i] == tr and poles[i] == 1]
        lo = [Vp[i] for i in range(n) if labels[i] == tr and poles[i] == -1]
        if hi and lo:
            dirs.append(unit(np.mean(hi, 0) - np.mean(lo, 0)))
    off = []
    if len(dirs) > 1:
        Dm = np.vstack(dirs); dc = Dm @ Dm.T
        off = [abs(dc[i, j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
    return {"n": n, "raw_within": raw_within, "raw_across": raw_across,
            "within": within, "across": across, "w_minus_a": within - across,
            "nn_purity": nn / n, "cluster5_purity": float(purity),
            "trait_dir_offdiag": float(np.mean(off)) if off else float("nan")}


@torch.no_grad()
def midlayer_marker_vecs(model, tok, device, markers, mid):
    bos = tok.bos_token_id
    ids, labels, poles = [], [], []
    for word, tr, pole in markers:
        tid = single_token_id(tok, word)
        if tid is None:
            continue
        ids.append([bos, tid] if bos is not None else [tid])
        labels.append(tr); poles.append(pole)
    inp = torch.tensor(ids, device=device)
    out = model(inp, output_hidden_states=True)
    mi = min(mid, len(out.hidden_states) - 1)
    h = out.hidden_states[mi][:, -1, :].float().cpu().numpy()  # marker position
    V = np.vstack([unit(h[i]) for i in range(len(h))])
    return V, labels, poles, mi


def neutral_pcs_at_mid(repo):
    """(neutral PC1 rows, neutral 50%-var PC rows, mid index) from the cached
    neutral activations, or (None, None, None) if uncached."""
    path = os.path.join(CACHE_DIR, safe_tag(repo) + "_neutral_chat.pt")
    if not os.path.exists(path):
        return None, None, None
    nb = torch.load(path, weights_only=False)
    neutral = nb.numpy() if isinstance(nb, torch.Tensor) else np.asarray(nb)
    nl = neutral.shape[1] - 1
    mid = int(round(nl * 2 / 3))
    nmid = torch.from_numpy(neutral[:, mid, :].astype(np.float64))
    pcs50, _, _ = mdx.compute_pc_projection(nmid, 0.5)
    return pcs50[:1], pcs50, mid


# loc keys: W_E (static); mid (self-PC1); mid_nPC1 (neutral top PC); mid_nPCs
# (neutral 50%-var basis = the single-pcs de-anisotropizer).
LOCS = ["W_E", "mid", "mid_nPC1", "mid_nPCs"]


def main():
    results = {}
    print("(within/across/w-a/NN/purity/diroff are AFTER projection; rawW/rawA "
          "= pre-projection anisotropy. mid=self-PC1, mid_nPC1/mid_nPCs=neutral PCs)")
    print(f"{'model':<12}{'set':<9}{'loc':<9}{'rawW':>7}{'rawA':>7}{'within':>8}"
          f"{'across':>8}{'w-a':>7}{'NNpur':>7}{'5clpur':>8}{'dir|off|':>9}")
    print("-" * 91)
    for name in COHORT:
        if name not in MODELS:
            continue
        repo = MODELS[name]
        we_loaded = load_embedding_matrix(repo)
        if we_loaded is None:
            print(f"{name}: no embedding tensor"); continue
        W_E_mat = we_loaded[0]
        npc1, npcs, n_mid = neutral_pcs_at_mid(repo)
        try:
            model, tok, device = load_model(name)
        except Exception as e:
            print(f"{name}: load failed ({e})"); continue
        results[name] = {"repo": repo, "sets": {}}
        for mset, markers in MARKER_SETS.items():
            we_vecs, we_lab, we_pol = [], [], []
            for word, tr, pole in markers:
                tid = single_token_id(tok, word)
                if tid is None or tid >= W_E_mat.shape[0]:
                    continue
                we_vecs.append(unit(W_E_mat[tid])); we_lab.append(tr); we_pol.append(pole)
            target_mid = n_mid if n_mid is not None else 10**9
            V, lab, pol, mi = midlayer_marker_vecs(model, tok, device, markers, target_mid)
            rec = {"mid_layer": mi,
                   "W_E": metrics(np.vstack(we_vecs), we_lab, we_pol),
                   "mid": metrics(V, lab, pol)}
            if npc1 is not None:
                rec["mid_nPC1"] = metrics(V, lab, pol, proj_vectors=npc1)
                rec["mid_nPCs"] = metrics(V, lab, pol, proj_vectors=npcs)
            results[name]["sets"][mset] = rec
            for loc in LOCS:
                if loc not in rec:
                    continue
                r = rec[loc]
                print(f"{name:<12}{mset:<9}{loc:<9}{r['raw_within']:>7.3f}{r['raw_across']:>7.3f}"
                      f"{r['within']:>8.3f}{r['across']:>8.3f}"
                      f"{r['w_minus_a']:>7.3f}{r['nn_purity']:>7.2f}{r['cluster5_purity']:>8.2f}"
                      f"{r['trait_dir_offdiag']:>9.3f}")
        del model, tok, we_loaded, W_E_mat
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # cohort means
    print("-" * 91)
    for mset in MARKER_SETS:
        for loc in LOCS:
            rows = [m["sets"][mset][loc] for m in results.values()
                    if mset in m["sets"] and loc in m["sets"][mset]]
            if not rows:
                continue
            agg = lambda k: np.nanmean([r[k] for r in rows])
            print(f"{'COHORT':<12}{mset:<9}{loc:<9}{agg('raw_within'):>7.3f}{agg('raw_across'):>7.3f}"
                  f"{agg('within'):>8.3f}{agg('across'):>8.3f}"
                  f"{agg('w_minus_a'):>7.3f}{agg('nn_purity'):>7.2f}{agg('cluster5_purity'):>8.2f}"
                  f"{agg('trait_dir_offdiag'):>9.3f}")

    os.makedirs("results/facets", exist_ok=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
