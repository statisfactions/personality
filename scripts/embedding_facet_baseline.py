#!/usr/bin/env python3
"""Embedding baseline for facet-geometry recovery (W13; to_try.md §19).

The W9 §7 result: our cohort's 30x30 IPIP-NEO facet cosine matrix (built with
meandiff-itempc1) recovers the human facet *correlation* matrix at cohort-mean
r~0.56. Wulff & Mata (2025, Nat. Hum. Behav.) and Milano et al. (2025, CRBS)
show the human factor/correlation structure is recoverable from item *text
alone* via sentence embeddings. So the open question this script answers:

    Is the model's facet geometry MORE than item semantics — i.e. does it
    exceed an embedding baseline built the same way?

We build a facet cosine matrix from sentence-embedding item vectors using the
SAME extraction as the model side (meandiff-itempc1: per-facet direction is
unit(project_out_top1PC(mean(fwd) - mean(rev))), top-1 PC computed from the
item embeddings themselves), then correlate the upper triangle against the
human matrix and against each cohort model's matrix.

The user's framing (important, baked into to_try §19): the encoder is *itself a
model*, just with a contrastive/masked objective rather than autoregressive +
alignment. A null ("embedding ~= model") therefore means the facet structure is
recoverable across objectives -> it lives in the items both consume (item
semantics), NOT in anything alignment-specific. It is the embedding-view
winning, but for a more precise reason than "embeddings are lexical."

Encoders (bracketing, per to_try §19):
  - all-mpnet-base-v2     : honest semantics-only baseline (out-of-the-box)
  - dwulff/mpnet-personality : CONTAMINATED reference -- fine-tuned ON unsigned
        empirical item correlations (CosineSimilarityLoss, direction-independent).
        Predicts empirical correlations at r~.6. Not a clean baseline; it has
        seen the target. Reported separately as an upper reference. Because it
        is trained to IGNORE direction/negation, keyed-diff (fwd-rev) is
        expected to collapse there -- so we also report a direction-agnostic
        mean-pool variant.
  - bge-large-en-v1.5     : different-lineage encoder (retrieval-contrastive).

Two pre-registered checks the user called ahead of the run -- does the
embedding reproduce the model's *divergences from human* structure?
  (a) E:Cheerf positively correlates with the N facets (the W9 §7.6 affect-axis
      merge; humans have it negative).
  (b) O:Liberal is independent (near-zero similarity to everything; the
      instrument-artifact outlier).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/embedding_facet_baseline.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ipip_facet_cluster import build_facet_pool, FACETS, TRAIT_ORDER, unit

HUMAN_PATH = "instruments/ipip300_human_facet_correlations.json"
MODEL_MATRIX_PATH = "results/facets/ipip_facet_cluster.json"
OUT_PATH = Path("results/facets/embedding_facet_baseline.json")

ENCODERS = {
    "mpnet-base":   "sentence-transformers/all-mpnet-base-v2",
    "dwulff":       "dwulff/mpnet-personality",
    "bge-large":    "BAAI/bge-large-en-v1.5",
}

# Facet order produced by iterating TRAIT_ORDER x FACETS -- this equals the
# human facet_order and the model facet_names (verified at runtime).
FACET_ORDER = [f"{t}:{f}" for t in TRAIT_ORDER for f in FACETS[t]]
N_FACET_PER_TRAIT = {f"{t}:{f}": t for t in TRAIT_ORDER for f in FACETS[t]}


def project_out_top1(direction, pc1):
    """Remove the single top PC (unit-normed) from direction. Mirrors
    extract_meandiff_vectors.project_out_pcs with a 1-row pcs matrix."""
    pc_u = pc1 / (np.linalg.norm(pc1) + 1e-12)
    return direction - np.dot(direction, pc_u) * pc_u


def top1_pc(X):
    """Top-1 PC of mean-centered X (n, d). Mirrors compute_pc_projection."""
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0]


def top1_pc_varfrac(X):
    """Variance fraction explained by PC1 -- a proxy for how artifact-like the
    dominant axis is. In our pre-norm transformers PC1 r=1.0 with activation
    norm (a content-free axis); if a sentence encoder's PC1 explains far less
    variance, projecting it out is more likely removing real signal."""
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    S = np.linalg.svd(Xc, compute_uv=False)
    var = S ** 2
    return float(var[0] / var.sum())


def build_facet_matrix(embeds, meta, mode, project):
    """embeds: (n_items, dim) item embeddings, row i described by meta[i].
    mode: 'keyed-diff' (mirror meandiff-itempc1) or 'mean-pool' (direction-
    agnostic; for dwulff which ignores negation).
    project: if True remove the top-1 item PC (mirrors meandiff-itempc1's
    anisotropy removal). For sentence encoders trained so raw cosine IS the
    semantic signal, PC1 may carry real content -- removing it deletes signal,
    not a norm artifact -- so we also run project=False (the user's caveat).
    Returns (facet_names, cos 30x30) in FACET_ORDER."""
    pc1 = top1_pc(embeds)
    by_facet = {}
    for i, m in enumerate(meta):
        key = (m["trait"], m["facet"])
        by_facet.setdefault(key, {"fwd": [], "rev": []})[m["pole"]].append(embeds[i])

    names, rows = [], []
    for t in TRAIT_ORDER:
        for fname in FACETS[t]:
            pol = by_facet[(t, fname)]
            fwd_mean = np.mean(pol["fwd"], axis=0)
            if mode == "keyed-diff":
                rev_mean = np.mean(pol["rev"], axis=0) if pol["rev"] else 0.0
                raw = fwd_mean - rev_mean
            elif mode == "mean-pool":
                allv = pol["fwd"] + pol["rev"]
                raw = np.mean(allv, axis=0)
            else:
                raise ValueError(mode)
            d = unit(project_out_top1(raw, pc1) if project else raw)
            names.append(f"{t}:{fname}")
            rows.append(d)
    return names, np.vstack(rows) @ np.vstack(rows).T


def upper_tri(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def matrix_r(A, B):
    """Pearson r between the upper triangles of two reordered 30x30 matrices."""
    a, b = upper_tri(A), upper_tri(B)
    return float(np.corrcoef(a, b)[0, 1])


def cheerf_vs_N(M, names):
    """Mean off-diagonal similarity of E:Cheerf to the 6 N facets."""
    idx = {n: i for i, n in enumerate(names)}
    c = idx["E:Cheerf"]
    nfac = [idx[f"N:{f}"] for f in FACETS["N"]]
    return float(np.mean([M[c, j] for j in nfac]))


def liberal_independence(M, names):
    """Mean |similarity| of O:Liberal to all other facets (low = independent)."""
    idx = {n: i for i, n in enumerate(names)}
    L = idx["O:Liberal"]
    others = [j for j in range(len(names)) if j != L]
    return float(np.mean([abs(M[L, j]) for j in others]))


def main():
    # ---- human matrix ----
    with open(HUMAN_PATH) as f:
        human = json.load(f)
    assert human["facet_order"] == FACET_ORDER, "human order drift"
    H = np.array(human["correlation_matrix"], dtype=float)

    # ---- model matrices ----
    with open(MODEL_MATRIX_PATH) as f:
        models = json.load(f)
    model_mats = {}
    for mname, rec in models.items():
        assert rec["facet_names"] == FACET_ORDER, f"{mname} order drift"
        model_mats[mname] = np.array(rec["cosine_matrix"], dtype=float)
    cohort_mean_mat = np.mean(list(model_mats.values()), axis=0)

    # ---- build item embeddings + facet matrices per encoder ----
    pool = build_facet_pool()
    items = []  # (trait, facet, pole, text)
    for (t, f), poles in pool.items():
        for pole in ("fwd", "rev"):
            for pair in poles[pole]:
                items.append({"trait": t, "facet": f, "pole": pole, "text": pair[1]})
    texts = [it["text"] for it in items]
    print(f"{len(texts)} IPIP items, {len(pool)} facets")

    # mode key -> (pooling, project). 'proj' mirrors meandiff-itempc1's PC1
    # removal; 'raw' leaves the dominant axis in (the user's caveat: for
    # cosine-trained encoders PC1 may be signal, not a norm artifact).
    MODES = {
        "keyed-diff/proj": ("keyed-diff", True),
        "keyed-diff/raw":  ("keyed-diff", False),
        "mean-pool/proj":  ("mean-pool", True),
        "mean-pool/raw":   ("mean-pool", False),
    }
    results = {"facet_order": FACET_ORDER, "encoders": {}}

    for tag, repo in ENCODERS.items():
        print(f"\n=== {tag}  ({repo}) ===")
        model = SentenceTransformer(repo)
        embeds = np.asarray(model.encode(texts, show_progress_bar=False,
                                         normalize_embeddings=False), dtype=np.float64)
        pc1_var = top1_pc_varfrac(embeds)
        print(f"  embeddings {embeds.shape}   PC1 var-frac {pc1_var:.3f}")
        enc_rec = {"repo": repo, "dim": int(embeds.shape[1]),
                   "pc1_var_frac": pc1_var, "modes": {}}
        for mkey, (pooling, project) in MODES.items():
            names, M = build_facet_matrix(embeds, items, pooling, project)
            assert names == FACET_ORDER
            within = [M[i, j] for i in range(30) for j in range(i + 1, 30)
                      if names[i].split(":")[0] == names[j].split(":")[0]]
            across = [M[i, j] for i in range(30) for j in range(i + 1, 30)
                      if names[i].split(":")[0] != names[j].split(":")[0]]
            enc_rec["modes"][mkey] = {
                "r_to_human": matrix_r(M, H),
                "r_to_cohort_mean_model": matrix_r(M, cohort_mean_mat),
                "r_to_each_model": {m: matrix_r(M, model_mats[m]) for m in model_mats},
                "within_trait_mean": float(np.mean(within)),
                "across_trait_mean": float(np.mean(across)),
                "cheerf_vs_N": cheerf_vs_N(M, names),
                "liberal_independence": liberal_independence(M, names),
                "cosine_matrix": M.tolist(),
            }
        results["encoders"][tag] = enc_rec

    # ---- reference rows for the two pre-registered checks ----
    results["reference_checks"] = {
        "cheerf_vs_N": {
            "human": cheerf_vs_N(H, FACET_ORDER),
            "cohort_mean_model": cheerf_vs_N(cohort_mean_mat, FACET_ORDER),
            **{m: cheerf_vs_N(model_mats[m], FACET_ORDER) for m in model_mats},
        },
        "liberal_independence": {
            "human": liberal_independence(H, FACET_ORDER),
            "cohort_mean_model": liberal_independence(cohort_mean_mat, FACET_ORDER),
            **{m: liberal_independence(model_mats[m], FACET_ORDER) for m in model_mats},
        },
    }
    results["model_r_to_human"] = {m: matrix_r(model_mats[m], H) for m in model_mats}
    results["cohort_mean_model_r_to_human"] = matrix_r(cohort_mean_mat, H)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ----
    print("\n" + "=" * 70)
    print("FACET-GEOMETRY RECOVERY: model vs embedding baseline")
    print("=" * 70)
    print(f"\nModel cosine-matrix r to HUMAN correlation matrix:")
    for m in sorted(model_mats, key=lambda k: results['model_r_to_human'][k]):
        print(f"  {m:<12} {results['model_r_to_human'][m]:+.3f}")
    print(f"  {'COHORT-MEAN':<12} {results['cohort_mean_model_r_to_human']:+.3f}")

    print(f"\nPC1 variance fraction per encoder (high ~ plausibly a norm/aniso")
    print(f"  artifact worth projecting out; low ~ PC1 is signal, keep it):")
    for tag in ENCODERS:
        print(f"  {tag:<12} {results['encoders'][tag]['pc1_var_frac']:.3f}")

    print(f"\nEmbedding baseline r to HUMAN, keyed-diff (proj | raw):")
    for tag in ENCODERS:
        kp = results["encoders"][tag]["modes"]["keyed-diff/proj"]["r_to_human"]
        kr = results["encoders"][tag]["modes"]["keyed-diff/raw"]["r_to_human"]
        note = "  <- contaminated (trained on the target)" if tag == "dwulff" else ""
        print(f"  {tag:<12} {kp:+.3f} | {kr:+.3f}{note}")

    print(f"\nEXCESS over honest baseline (cohort-mean model r-to-human minus")
    print(f"  mpnet-base keyed-diff r-to-human), proj vs raw:")
    for v in ("keyed-diff/proj", "keyed-diff/raw"):
        base = results["encoders"]["mpnet-base"]["modes"][v]["r_to_human"]
        d = results['cohort_mean_model_r_to_human'] - base
        print(f"  vs mpnet-base [{v:<15}] {base:+.3f}:  "
              f"{results['cohort_mean_model_r_to_human']:+.3f} - {base:+.3f} = {d:+.3f}")

    print(f"\nEmbedding r to COHORT-MEAN MODEL matrix (does the embedding look")
    print(f"  like our models specifically? keyed-diff proj | raw):")
    for tag in ENCODERS:
        rp = results["encoders"][tag]["modes"]["keyed-diff/proj"]["r_to_cohort_mean_model"]
        rr = results["encoders"][tag]["modes"]["keyed-diff/raw"]["r_to_cohort_mean_model"]
        print(f"  {tag:<12} {rp:+.3f} | {rr:+.3f}")

    print(f"\n--- PRE-REGISTERED CHECK (a): E:Cheerf <-> N facets (positive = "
          f"model-like affect merge; humans expected negative) ---")
    rc = results["reference_checks"]["cheerf_vs_N"]
    print(f"  human            {rc['human']:+.3f}")
    print(f"  cohort-mean model {rc['cohort_mean_model']:+.3f}")
    for tag in ENCODERS:
        for mkey in results["encoders"][tag]["modes"]:
            v = results["encoders"][tag]["modes"][mkey]["cheerf_vs_N"]
            print(f"  {tag:<12} [{mkey:<15}] {v:+.3f}")

    print(f"\n--- PRE-REGISTERED CHECK (b): O:Liberal independence "
          f"(mean |sim| to all others; LOW = independent) ---")
    rl = results["reference_checks"]["liberal_independence"]
    print(f"  human            {rl['human']:.3f}")
    print(f"  cohort-mean model {rl['cohort_mean_model']:.3f}")
    for tag in ENCODERS:
        for mkey in results["encoders"][tag]["modes"]:
            v = results["encoders"][tag]["modes"][mkey]["liberal_independence"]
            print(f"  {tag:<12} [{mkey:<15}] {v:.3f}")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
