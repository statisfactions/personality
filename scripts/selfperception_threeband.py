"""Aggregate three-band discriminant statistic for the self-perception dose.

For every dose row, classify each read item as target / mate / anti (from the
selection file's item sets) and take its cold-EV shift vs the adjective-
independent K=0 baseline. Average within class, by K, across adjectives and
seeds. Turns the per-item "target up, neighbors up-less, antonyms anti-move"
story from anecdote (individual pairs) into a cohort statistic.

  target Δ  : self-report shift on the dosed trait word
  mate  Δ   : shift on its 4 human-cluster neighbors
  anti  Δ   : shift on its 4 desirability-partialled antonyms
  disc      : target − anti (full three-band spread)

Usage:
  PYTHONPATH=scripts python scripts/selfperception_threeband.py
  PYTHONPATH=scripts python scripts/selfperception_threeband.py --tag _long
  PYTHONPATH=scripts python scripts/selfperception_threeband.py \
      --models Llama8 --tags "" _dosedefault    # main vs null control
Output: prints markdown; writes results/selfperception/threeband_<scope>.json
"""
import argparse
import json
import os

import numpy as np

SRC = "results/selfperception"
COHORT = ["llama3.2", "Llama8", "gemma3", "Gemma12", "Gemma27",
          "qwen2.5", "Qwen7", "Qwen32", "phi4", "Aya"]
FAMILY = {"llama3.2": "llama", "Llama8": "llama", "gemma3": "gemma",
          "Gemma12": "gemma", "Gemma27": "gemma", "qwen2.5": "qwen",
          "Qwen7": "qwen", "Qwen32": "qwen", "phi4": "phi4", "Aya": "aya"}


def load(model, tag=""):
    p = f"{SRC}/{model}{tag}_part.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    k0row = next(r for r in rows if r["adj"] == "__k0__")
    k0 = {w: v["cold"]["ev"] for w, v in k0row["readings"].items()}
    items = json.load(open(f"{SRC}/{model}{tag}_selection.json"))["items"]
    return rows, k0, items


def threeband(rows, k0, items, arm="A"):
    """K -> {class: [per-item cold-EV shifts vs K0]}"""
    acc = {}
    for r in rows:
        if r["adj"] == "__k0__" or r["arm"] != arm or r["K"] == 0:
            continue
        adj = r["adj"]
        if adj not in items:
            continue
        mates = set(items[adj]["mates"])
        antis = set(items[adj]["anti"])
        for w, v in r["readings"].items():
            if w not in k0:
                continue
            sh = v["cold"]["ev"] - k0[w]
            if w == adj:
                cls = "target"
            elif w in mates:
                cls = "mate"
            elif w in antis:
                cls = "anti"
            else:
                continue
            acc.setdefault(r["K"], {}).setdefault(cls, []).append(sh)
    return acc


def cell(d, cls):
    xs = d.get(cls, [])
    return (np.mean(xs), len(xs)) if xs else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=COHORT)
    ap.add_argument("--tags", nargs="*", default=[""])
    ap.add_argument("--arm", default="A")
    ap.add_argument("--all-k", action="store_true",
                    help="print every K, not just the max")
    args = ap.parse_args()

    out = []
    print(f"| model | fam | tag | K | target | mate | anti | disc(t−a) | n_adj |")
    print(f"|---|---|---|--:|--:|--:|--:|--:|--:|")
    for m in args.models:
        for tag in args.tags:
            r = load(m, tag)
            if r is None:
                continue
            acc = threeband(*r, arm=args.arm)
            ks = sorted(acc) if args.all_k else [max(acc)]
            for K in ks:
                d = acc[K]
                t, nt = cell(d, "target")
                mt, _ = cell(d, "mate")
                an, _ = cell(d, "anti")
                tg = tag or "main"
                print(f"| {m} | {FAMILY.get(m,'?')} | {tg} | {K} "
                      f"| {t:+.2f} | {mt:+.2f} | {an:+.2f} | {t-an:+.2f} "
                      f"| {nt // 1} |")
                out.append(dict(model=m, family=FAMILY.get(m), tag=tag, K=K,
                                target=float(t), mate=float(mt),
                                anti=float(an), n_target=nt))
    scope = "_".join(args.models) if len(args.models) <= 3 else "cohort"
    dst = f"{SRC}/threeband_{scope}.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
