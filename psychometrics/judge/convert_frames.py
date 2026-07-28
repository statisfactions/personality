#!/usr/bin/env python3
"""Tidy the frame-experiment JSONL into data/frames_tidy.csv (Report VII).

One row per (model, pair, frame, order) with the EV (logprob-weighted mean over 1-7),
the argmax response, and the design columns joined back from frame_pairs.csv.

Polarity: `cond` and `sim` are alikeness scales; `diff` is a differentness scale. We keep
the raw value in `ev` and add `ev_alike` = 8 - ev for `diff` so all three frames share an
alikeness polarity. Analyses that test non-complementarity use the RAW values.
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

pairs = pd.read_csv(os.path.join(DATA, "frame_pairs.csv"))

rows = []
for path in sorted(glob.glob(os.path.join(HERE, "frames_raw", "*_frames.jsonl"))):
    model = os.path.basename(path).replace("_frames.jsonl", "")
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            rows.append({"model": model, "i": r["i"], "j": r["j"], "frame": r["frame"],
                         "order": r["order"], "subject": r["subject"],
                         "referent": r["referent"], "response": r.get("response"),
                         "ev": r.get("ev"), "valid_mass": r.get("valid_mass")})

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no frame records found under data/frames/")
# EV needs logprobs; fall back to the parsed argmax where they are missing
df["ev"] = df["ev"].fillna(df["response"])
df = df.dropna(subset=["ev"])
df["ev_alike"] = df.apply(lambda r: 8 - r.ev if r.frame == "diff" else r.ev, axis=1)
df = df.merge(pairs, on=["i", "j"], how="left", validate="many_to_one")

out = os.path.join(DATA, "frames_tidy.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows")
print(df.groupby(["model", "frame"]).agg(n=("ev", "size"), mean_ev=("ev", "mean"),
                                         sd_ev=("ev", "std"),
                                         mass=("valid_mass", "mean")).round(3))
