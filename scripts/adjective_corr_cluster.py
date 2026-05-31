#!/usr/bin/env python3
"""Human adjective correlation matrix + re-clustering from the ESCS 525-PDA.

Theory-neutral substrate for the affect-axis question. The Johnson IPIP-NEO
human matrix is Big-Five-scaffolded (30 facets pre-assigned to 5 factors), so it
cannot cleanly show an affect axis cutting across the Big Five. Saucier's 525
Person-Descriptive Adjectives (525-PDA, Harvard Dataverse doi:10.7910/DVN/GHYMEV)
are raw 1-7 self-ratings (700 respondents) with NO a-priori factor structure
imposed — we cluster the 525x525 correlation matrix from scratch.

Crucially these are HUMAN RATINGS (behavioral covariance), in the same single-
adjective units the LLM/encoder geometry operates on. So the sharp test is:
does Cheerful correlate NEGATIVELY with Angry/Sad here (the behavioral fact the
model inverts), confirming the model's positive Cheerf<->affect is a lexical
departure from human behavior?

Preprocessing is RAW by default (we control it, unlike C&C's pre-ipsatized
S&G-1996 matrix); --ipsatize applies C&C's within-person z (t(apply(set,1,z)))
to match their pipeline for cross-check.

Usage:
  .venv/bin/python scripts/adjective_corr_cluster.py
  .venv/bin/python scripts/adjective_corr_cluster.py --ipsatize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

POR = "data/escs_525pda/525_PDA-1.por"
OUT_DIR = Path("results/adjectives")

# Columns dropped as corrupted (label != data) per scripts/adjective_audit.py.
# Both have empirical neighborhoods that contradict their label:
#   Inspirational -> behaves like "Inconsiderate" (r +0.53 with Inconsiderate)
#   Insensitive   -> behaves like a surgency word (Exciting/Remarkable; and
#                    fails the antonym check, r +0.25 with Sensitive)
DENY_LABELS = {"Inspirational", "Insensitive"}

# Affect-axis probe set: words present in the 525 list whose human (behavioral)
# vs lexical (model) relations are the crux. Names must match .por columns
# (SPSS 8-char uppercased); we resolve them by case-insensitive prefix below.
AFFECT_PROBES = ["CHEERFUL", "HAPPY", "SAD", "UNHAPPY", "ANGRY", "MOODY",
                 "NERVOUS", "TENSE", "SYMPATHE", "UNSYMPAT", "WARM", "KIND",
                 "EMOTIONA", "EXCITED"]


def ipsatize(M):
    """C&C's within-person z: standardize each ROW (person) across items."""
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    return (M - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipsatize", action="store_true",
                    help="apply C&C within-person z before correlating")
    args = ap.parse_args()

    df, meta = pyreadstat.read_por(POR)
    # Drop ID + the corrupted columns (matched by their full adjective label).
    deny_cols = {c for c in df.columns
                 if (meta.column_names_to_labels.get(c) or c) in DENY_LABELS}
    if deny_cols:
        print(f"dropping {len(deny_cols)} corrupted column(s): "
              f"{sorted(meta.column_names_to_labels.get(c, c) for c in deny_cols)}")
    adj_cols = [c for c in df.columns if c.upper() != "ID" and c not in deny_cols]
    A = df[adj_cols].astype(float)
    print(f"loaded {POR}: {A.shape[0]} respondents x {A.shape[1]} adjectives")
    print(f"  NaN cells: {int(A.isna().sum().sum())} "
          f"({100*A.isna().sum().sum()/A.size:.2f}%); value range "
          f"{np.nanmin(A.values):.0f}-{np.nanmax(A.values):.0f}")

    mode = "ipsatized" if args.ipsatize else "raw"
    M = A.values
    if args.ipsatize:
        M = ipsatize(M)
    # Pairwise-complete Pearson (robust to scattered NaN).
    C = pd.DataFrame(M, columns=adj_cols).corr(method="pearson")

    # Pretty labels from the SPSS variable labels (full adjective spelling).
    label = {c: (meta.column_names_to_labels.get(c) or c) for c in adj_cols}

    # ---- affect-axis probe ----
    present = [c for c in AFFECT_PROBES if c in C.columns]
    print(f"\n=== affect-axis probe ({mode} human ratings, N={A.shape[0]}) ===")
    key_pairs = [("CHEERFUL", "ANGRY"), ("CHEERFUL", "SAD"), ("CHEERFUL", "MOODY"),
                 ("CHEERFUL", "NERVOUS"), ("SYMPATHE", "ANGRY"),
                 ("SYMPATHE", "KIND"), ("SYMPATHE", "WARM"), ("HAPPY", "ANGRY")]
    print("  key pairs (behavioral prediction: cheerful<->angry/sad NEGATIVE):")
    for a, b in key_pairs:
        if a in C.columns and b in C.columns:
            print(f"    {label[a]:<14} <-> {label[b]:<14} r = {C.loc[a,b]:+.3f}")
    # full affect submatrix mean off-diagonal
    sub = C.loc[present, present].values
    off = sub[np.triu_indices(len(present), 1)]
    print(f"  affect-block mean off-diagonal r: {off.mean():+.3f} "
          f"(n={len(present)} adjectives)")

    # ---- which adjectives does CHEERFUL actually cluster with? (top human neighbors) ----
    if "CHEERFUL" in C.columns:
        nbr = C["CHEERFUL"].drop("CHEERFUL").sort_values(ascending=False)
        print("\n  CHEERFUL's strongest human neighbors (+):")
        for c, v in nbr.head(8).items():
            print(f"    {label[c]:<16} {v:+.3f}")
        print("  CHEERFUL's strongest human anti-correlates (-):")
        for c, v in nbr.tail(6).items():
            print(f"    {label[c]:<16} {v:+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"escs_525pda_corr_{mode}.json"
    json.dump({
        "name": "525-PDA",
        "source": "Saucier, Gerard. 525 Person-descriptive Adjectives. "
                  "Harvard Dataverse doi:10.7910/DVN/GHYMEV (ESCS self-ratings).",
        "n_respondents": int(A.shape[0]),
        "preprocessing": mode,
        "adjectives": adj_cols,
        "labels": [label[c] for c in adj_cols],
        "correlation_matrix": C.values.round(5).tolist(),
    }, open(out, "w"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
