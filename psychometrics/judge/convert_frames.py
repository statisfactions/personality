#!/usr/bin/env python3
"""Tidy the frame-experiment JSONL into data/frames_tidy.csv (Report VII).

One row per (model, pair, frame, order) with the design columns joined from frame_pairs.csv.

PRIMARY MEASURE = `rating`, the first digit 1-7 in the EMITTED TEXT. Not the logprob EV.
At temperature 0 the emitted token is the model's actual answer, and this endpoint's
`top_logprobs` are demonstrably not always the distribution that produced it:
qwen2.5:3b emits "5" on cells where the reported distribution puts 0.80 on "7" (20% of
its cells disagree), phi4-mini on 61%. `ev` is retained for diagnosis, but no analysis
should lead with it until the misalignment is understood.

Polarity: `cond` and `sim` are alikeness scales, `diff` is a differentness scale, so
`rating_alike` = 8 - rating for `diff`. Non-complementarity tests use the RAW values.
"""
import re
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
            m = re.search(r"[1-7]", (r.get("text") or ""))
            rows.append({"model": model, "i": r["i"], "j": r["j"], "frame": r["frame"],
                         "order": r["order"], "subject": r["subject"],
                         "referent": r["referent"],
                         "rating": int(m.group()) if m else None,
                         "text_ok": bool(m), "argmax": r.get("response"),
                         "ev": r.get("ev"), "valid_mass": r.get("valid_mass")})

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no frame records found under data/frames/")
resp_rate = df.groupby(["model", "frame"])["text_ok"].mean().rename("digit_rate")
df = df.dropna(subset=["rating"])
df["rating_alike"] = df.apply(lambda r: 8 - r.rating if r.frame == "diff" else r.rating, axis=1)
df["lp_agree"] = df["argmax"] == df["rating"]
df = df.merge(pairs, on=["i", "j"], how="left", validate="many_to_one")

out = os.path.join(DATA, "frames_tidy.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows")
summ = df.groupby(["model", "frame"]).agg(
    n=("rating", "size"), mean=("rating", "mean"), sd=("rating", "std"),
    lp_agree=("lp_agree", "mean")).join(resp_rate).round(3)
print(summ)
print("\nlp_agree = logprob argmax matches the emitted digit; "
      "digit_rate = prompts yielding a parsable digit.")
