#!/usr/bin/env python3
"""Audit the 525-PDA for mislabeled / sign-flipped columns (data != label).

The INSPIRAT column's data behaves like "Inconsiderate", not "Inspirational" —
a content/label mismatch in Saucier's deposit. To find others WITHOUT a circular
reference, we use an encoder as an external semantic oracle:

  For each adjective, compare its EMPIRICAL correlation profile (who it correlates
  with across the 525) against its SEMANTIC profile (encoder cosine to the other
  524 words). A healthy column's data-neighbors ARE its meaning-neighbors, so the
  two profiles correlate positively. A mislabeled/sign-flipped column's empirical
  neighbors are semantic opposites -> the agreement r goes to zero or negative.

Second, high-confidence signal: morphological antonym pairs (X / UnX, DisX, InX).
A word and its explicit negation MUST correlate negatively; a positive r flags one
of them as bad.

Outputs a ranked suspect list. Nothing is dropped here — see DENY in
adjective_corr_cluster.py for exclusions.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_audit.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ENCODER = "BAAI/bge-large-en-v1.5"
SRC = Path("results/adjectives/escs_525pda_corr_raw.json")
OUT = Path("results/adjectives/audit_suspects.json")


def main():
    d = json.load(open(SRC))
    C = np.array(d["correlation_matrix"], float)
    labels = d["labels"]
    n = len(labels)
    np.fill_diagonal(C, np.nan)

    # Semantic oracle: encode each adjective as a personality descriptor.
    model = SentenceTransformer(ENCODER)
    texts = [f"someone who is {w.lower()}" for w in labels]
    E = np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
    S = E @ E.T
    np.fill_diagonal(S, np.nan)

    # Per-column agreement: corr(empirical profile, semantic profile).
    agree = np.full(n, np.nan)
    for i in range(n):
        ce, se = C[i], S[i]
        m = ~np.isnan(ce) & ~np.isnan(se)
        agree[i] = np.corrcoef(ce[m], se[m])[0, 1]

    mu, sd = np.nanmean(agree), np.nanstd(agree)
    print(f"profile-agreement r: mean={mu:.3f} sd={sd:.3f} "
          f"min={np.nanmin(agree):.3f}")
    print(f"\n=== worst 20 columns (empirical neighbors disagree with meaning) ===")
    print(f"  {'adjective':<16}{'agree_r':>8}  empirical top-4 neighbors")
    order = np.argsort(agree)
    suspects = []
    for i in order[:20]:
        nbr = np.argsort(-np.nan_to_num(C[i], nan=-9))[:4]
        z = (agree[i] - mu) / sd
        flag = "  <== FLAG" if z < -3 else ""
        print(f"  {labels[i]:<16}{agree[i]:>+8.3f}  "
              f"{', '.join(labels[j] for j in nbr)}{flag}")
        if z < -3:
            suspects.append({"adjective": labels[i], "agree_r": float(agree[i]),
                             "z": float(z),
                             "empirical_neighbors": [labels[j] for j in nbr]})

    # Antonym-pair consistency (independent, high-confidence).
    lab_set = {l.lower(): i for i, l in enumerate(labels)}
    print(f"\n=== morphological antonym pairs that DON'T anti-correlate ===")
    bad_pairs = []
    for l, i in lab_set.items():
        m = re.match(r'^(un|dis|in|im|non)(.+)', l)
        if not m:
            continue
        base = m.group(2)
        for cand in (base, "im" + base if base.startswith("p") else None):
            if cand and cand in lab_set:
                j = lab_set[cand]
                r = C[i, j]
                if not np.isnan(r) and r > -0.1:  # should be strongly negative
                    print(f"  {labels[i]:<16} vs {labels[j]:<16} r={r:+.3f}")
                    bad_pairs.append([labels[i], labels[j], float(r)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"encoder": ENCODER, "agreement_mean": float(mu),
               "agreement_sd": float(sd),
               "flagged_z_lt_-3": suspects,
               "antonym_pair_anomalies": bad_pairs}, open(OUT, "w"), indent=2)
    print(f"\nflagged (z<-3): {[s['adjective'] for s in suspects]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
