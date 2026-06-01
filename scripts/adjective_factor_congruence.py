#!/usr/bin/env python3
"""Are the human Big Five the LLM's axes? Full-523 factor congruence (W13 §3.11).

If models weight "affect-presence" over valence, that should warp the WHOLE
adjective structure, not just the affect block — which cycles back to "are the
5 human factors even the LLM's organizing axes?". Test, per model:

  (a) recover the human Big Five? Tucker congruence between each model's 5-factor
      varimax loadings and the human 5-factor loadings (the C&C metric — they ran
      it on DeBERTa, an ENCODER; never on decoder LLMs).
  (b) what ARE the model's own dominant axes? Top-loading adjectives per factor +
      eigenspectrum (is the model more dominated by one top axis than humans?).

Human + LLM cohort (pers, center+top1) + bge/mpnet. Tucker thresholds (C&C):
>.95 identical, .85-.94 fair, lower = dissimilar.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_factor_congruence.py
"""
import glob
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from adjective_factor_heatmap import varimax
from adjective_geom import model_matrix as _model_matrix

HUMAN = "results/adjectives/escs_525pda_corr_raw.json"
ENCODERS = ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]
K = 5


def loadings(C, k=K):
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    Phi = V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None))
    L = varimax(Phi)
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return L, ev[o]


def tuck(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def match(Lh, Lm):
    """greedy: each human factor -> best-|congruence| unused model factor."""
    T = np.array([[tuck(Lh[:, i], Lm[:, j]) for j in range(K)] for i in range(K)])
    order = sorted(((abs(T[i, j]), i, j) for i in range(K) for j in range(K)),
                   reverse=True)
    hi, mj, out = set(), set(), {}
    for _, i, j in order:
        if i in hi or j in mj:
            continue
        out[i] = T[i, j]; hi.add(i); mj.add(j)
    return [out[i] for i in range(K)]


def top_words(L, j, labels, n=7):
    o = np.argsort(-L[:, j])
    return ", ".join(labels[i] for i in o[:n])


def model_matrix_from_cache(path, labels):
    name, C, removed, ipr = _model_matrix(path, labels)  # adaptive denoise
    return name, C


def main():
    human = json.load(open(HUMAN))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    n = len(labels)
    Lh, evh = loadings(H)

    # label human factors from top words
    print("=== HUMAN Big Five (5-factor varimax, top words) ===")
    hnames = []
    for j in range(K):
        w = top_words(Lh, j, labels, 6)
        guess = ("A/warmth" if "Kind" in w else "N/neg-aff" if any(x in w for x in ("Irritated","Angry","Troubled")) else
                 "C/order" if any(x in w for x in ("Organized","Thorough","Responsible")) else
                 "O/intellect" if any(x in w for x in ("Intelligent","Thinking","Smart")) else
                 "E/surgency")
        hnames.append(f"F{j+1}:{guess}"); print(f"  F{j+1} ({guess}): {w}")
    print(f"  human top-5 eigval var-frac: {', '.join(f'{v/n:.3f}' for v in evh[:5])}")

    # per-model congruence
    print(f"\n=== Tucker congruence to human factors (>.85 fair, >.95 identical) ===")
    print(f"  {'model':<24}" + "".join(f"{hn.split(':')[1][:7]:>9}" for hn in hnames) + f"{'mean':>8}{'PC1f':>7}")
    rows = []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        name, C = model_matrix_from_cache(p, labels)
        Lm, evm = loadings(C)
        cong = match(Lh, Lm)
        rows.append((name, cong, evm))
        print(f"  {name:<24}" + "".join(f"{c:>+9.2f}" for c in cong)
              + f"{np.mean(np.abs(cong)):>+8.2f}{evm[0]/n:>7.2f}")
    arr = np.array([r[1] for r in rows])
    print(f"  {'COHORT MEAN |cong|':<24}" + "".join(f"{np.mean(np.abs(arr[:, j])):>9.2f}" for j in range(K))
          + f"{np.mean(np.abs(arr)):>+8.2f}")

    # encoders
    texts = [f"someone who is {l.lower()}" for l in labels]
    for repo in ENCODERS:
        emb = np.asarray(SentenceTransformer(repo).encode(
            texts, show_progress_bar=False, normalize_embeddings=True), float)
        De = emb - emb.mean(0); De /= (np.linalg.norm(De, axis=1, keepdims=True) + 1e-9)
        Lm, evm = loadings(De @ De.T); cong = match(Lh, Lm)
        print(f"  {'ENC '+repo.split('/')[-1]:<24}" + "".join(f"{c:>+9.2f}" for c in cong)
              + f"{np.mean(np.abs(cong)):>+8.2f}{evm[0]/n:>7.2f}")

    # what ARE a representative model's factors? (cohort-mean matrix)
    print("\n=== cohort-mean LLM's OWN 5 factors (top words) — Big Five or not? ===")
    Cs = [model_matrix_from_cache(p, labels)[1]
          for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt"))]
    Lm, evm = loadings(np.mean(Cs, 0))
    # align model factors to human for readability
    for j in range(K):
        best = int(np.argmax([abs(tuck(Lm[:, j], Lh[:, i])) for i in range(K)]))
        print(f"  M{j+1} (~{hnames[best]}, cong {tuck(Lm[:,j],Lh[:,best]):+.2f}): {top_words(Lm, j, labels, 7)}")
    print(f"  cohort-mean model top-5 var-frac: {', '.join(f'{v/n:.3f}' for v in evm[:5])} "
          f"(human: {', '.join(f'{v/n:.3f}' for v in evh[:5])})")


if __name__ == "__main__":
    main()
