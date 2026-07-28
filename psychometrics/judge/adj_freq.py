#!/usr/bin/env python3
"""Lexical frequency for the 523 JUDGE adjectives -> data/adj_freq.csv.

Report VI (Tversky contrast model) uses this as a *rival* predictor of the recovered
prominence scale f: if f were lexical availability rather than a feature-salience
measure, corpus frequency should predict it. (It does not — DR2 = 0.000.)

Zipf scale = log10(occurrences per billion words); 0 means "not in the wordlist"
(one item, `unenvious`) and is stored as NaN. Requires `pip install wordfreq`.
Run after convert_full_dists.py (needs data/adjectives.csv).
"""
import os
import pandas as pd
import wordfreq

HERE = os.path.dirname(os.path.abspath(__file__))
adj = pd.read_csv(os.path.join(HERE, "data", "adjectives.csv"))
z = [wordfreq.zipf_frequency(w, "en") for w in adj.adjective]
adj["zipf"] = [v if v > 0 else float("nan") for v in z]
out = os.path.join(HERE, "data", "adj_freq.csv")
adj.to_csv(out, index=False)
print(f"wrote {out}: {adj.zipf.notna().sum()}/{len(adj)} with frequency, "
      f"range {adj.zipf.min():.2f}-{adj.zipf.max():.2f}")
