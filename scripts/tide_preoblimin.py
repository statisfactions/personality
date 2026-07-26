"""Pre-oblimin structure of Persona Cartography's TIDE response matrix (W19 §1).

Their shipped fit (persona-cartography/monorepo, run ...q_v7_fc_pair...pf3):
72-item FC-pair questionnaire appended to 2,500 Llama-3.1-8B persona-rollouts,
response = signed choice-mass; paper reports k=4 oblimin PAF (TIDE, 40.7%
cumulative). rgb's question: what does the matrix look like BEFORE rotation —
is there an unexamined general/desirability factor (the W14 worry)?

Findings (2026-07-26): NO desirability boulder — PC1 14.0% vs PC2 11.1%
(ratio 1.26; only 61% same-sign loadings). The FC-pair format (two defensible
conduct styles per item) desirability-matches by construction. Unrotated PC1
= persona-expressiveness vs assistant-compliance (persona-amplitude axis,
same leading axis as ENACT). Spectrum long: Kaiser 15, PR effdim 19.8, Horn
~13 (their fig: 11) — k=4 is retention parsimony, not discovered
dimensionality; two-level agreement with our ENACT bandwidth (retained 4-5,
probed ~9-13).

Usage: PYTHONPATH=scripts python scripts/tide_preoblimin.py
(downloads the matrix from persona-cartography/monorepo on first run)
"""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

RUN = ("unsupervised/runs/questionnaire-rollouts-llama318binstruct-t1.0-15t-"
       "2500p-seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-"
       "lp20-p2-pf3")
LOCAL = "dependencies/pc_monorepo"


def fetch(f):
    return hf_hub_download("persona-cartography/monorepo", f"{RUN}/{f}",
                           repo_type="dataset", local_dir=LOCAL)


def main():
    R = np.load(fetch("questionnaire/response_matrix.npy"))
    items = json.load(open(fetch("questionnaire/items.json")))
    dims = [it["dimension"] for it in items]
    n, k = R.shape
    print(f"response matrix: {n} rollout-respondents x {k} items")

    C = np.corrcoef(R.T)
    w, V = np.linalg.eigh(C)
    w = w[::-1]; V = V[:, ::-1]
    tot = w.sum()
    print("top 12 eigenvalues:", np.round(w[:12], 2))
    print("% variance:        ", np.round(100 * w[:12] / tot, 1))
    print(f"Kaiser(>1) {np.sum(w > 1)}   PR effdim {w.sum()**2/(w**2).sum():.1f}"
          f"   PC1 share {w[0]/tot:.1%}   PC1/PC2 {w[0]/w[1]:.2f}")
    print(f"first-4 cumulative {w[:4].sum()/tot:.1%} (their PAF k=4: 40.7%)")

    l1 = V[:, 0] * np.sqrt(w[0])
    sgn = np.sign(np.median(l1))
    print(f"\nunrotated PC1: {np.mean(np.sign(l1) == sgn):.0%} same-sign, "
          f"mean |loading| {np.abs(l1).mean():.2f}")
    o = np.argsort(-l1 * sgn)
    print("  expressive pole: " + ", ".join(dims[i] for i in o[:6]))
    print("  compliant pole:  " + ", ".join(dims[i] for i in o[-6:]))

    rng = np.random.default_rng(3)
    null = []
    for _ in range(100):
        Rp = np.column_stack([rng.permutation(R[:, j]) for j in range(k)])
        null.append(np.linalg.eigvalsh(np.corrcoef(Rp.T))[::-1])
    above = int(np.sum(w > np.percentile(null, 95, axis=0)))
    print(f"\nHorn (100 col-perms): {above} eigenvalues above 95th-pct null "
          "(their figure: 11)")


if __name__ == "__main__":
    main()
