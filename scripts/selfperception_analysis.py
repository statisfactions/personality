"""Analyze self-perception dose-response stage-1 output (design doc §8).

Reads results/selfperception/<model>_part.jsonl (tolerates in-flight
partial data), prints the dose-response tables, and writes
results/selfperception/<model>_analysis.json.

Readouts:
  1. target cold EV by (K, arm): the core dose-response
  2. 13-item composite: mean(target+mates) - mean(anti)
  3. DIRECTED - COLD gap
  4. moderators of per-adjective arm-A slope: enactability, baseline EV,
     believability, selfclaim (leak-vs-conduct covariate)
  5. BE vectors: mean dosed-turn activation (K=8 arm A) - default acts;
     cos to ENACT direction; readout-state projection by K
Massive dims are zeroed before any geometry (house winsorize convention).

Usage: PYTHONPATH=scripts python scripts/selfperception_analysis.py --model Llama8
"""
import argparse
import json
import os

import numpy as np

OUT = "results/selfperception"


def load_rows(model, tag=""):
    rows = [json.loads(l) for l in open(f"{OUT}/{model}{tag}_part.jsonl")]
    sel = json.load(open(f"{OUT}/{model}{tag}_selection.json"))
    return rows, sel


def slope(ks, ys):
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    if len(ks) < 2:
        return float("nan")
    return float(np.polyfit(ks, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama8")
    ap.add_argument("--anchor", default="default")
    args = ap.parse_args()
    m = args.model
    tag = "" if args.anchor == "default" else f"_anchor-{args.anchor}"

    rows, sel = load_rows(m, tag)
    picked = {p["adj"]: p for p in sel["picked"]}
    items = sel["items"]
    ks_all = [k for k in sel["ks"] if k > 0]

    k0row = next((r for r in rows if r["adj"] == "__k0__"), None)
    k0 = {w: v["cold"]["ev"] for w, v in k0row["readings"].items()} \
        if k0row else {}

    # ---- assemble per (adj, K, arm): seed-mean readings
    cells = {}
    for r in rows:
        if r["adj"] == "__k0__":
            continue
        cells.setdefault((r["adj"], r["K"], r["arm"]), []).append(r)

    summary = {}
    print(f"\n=== {m}: self-perception dose-response "
          f"({len(rows)} context records) ===")
    hdr = "adj              K0    " + "  ".join(
        f"A@{k}   B@{k}" for k in ks_all)
    print("\n-- target cold EV --\n" + hdr)
    for adj in items:
        if adj not in picked:
            continue
        iset = items[adj]
        line, rec = f"{adj:15s} {k0.get(adj, float('nan')):5.2f}", {}
        for k in ks_all:
            for arm in "AB":
                rs = cells.get((adj, k, arm), [])
                evs = [r["readings"][adj]["cold"]["ev"] for r in rs
                       if adj in r["readings"]]
                mu = np.mean(evs) if evs else float("nan")
                rec[f"{arm}{k}"] = dict(
                    ev=float(mu), sd=float(np.std(evs)) if len(evs) > 1
                    else None, n=len(evs))
                line += f" {mu:5.2f}"
        print(line)

        def series(arm, field="cold", agg="target"):
            xs, ys = [], []
            for k in ks_all:
                rs = cells.get((adj, k, arm), [])
                for r in rs:
                    if agg == "target":
                        if adj not in r["readings"] or \
                                field not in r["readings"][adj]:
                            continue
                        ys.append(r["readings"][adj][field]["ev"])
                    else:  # composite
                        pos = [r["readings"][w][field]["ev"]
                               for w in [adj] + iset["mates"]
                               if w in r["readings"]
                               and field in r["readings"][w]]
                        neg = [r["readings"][w][field]["ev"]
                               for w in iset["anti"] if w in r["readings"]
                               and field in r["readings"][w]]
                        if not pos or not neg:
                            continue
                        ys.append(np.mean(pos) - np.mean(neg))
                    xs.append(k)
            return xs, ys

        xs, ys = series("A")
        xa, ya = ([0] * (1 if adj in k0 else 0) + xs,
                  ([k0[adj]] if adj in k0 else []) + ys)
        sA = slope(xa, ya)
        xs, ys = series("B")
        sB = slope(xs, ys)
        xc, yc = series("A", agg="composite")
        bel = [np.mean(r["believability"]) for k in ks_all
               for r in cells.get((adj, k, "A"), []) if r.get("believability")]
        scl = [np.sum(r.get("selfclaim", [])) / max(r["K"], 1)
               for k in ks_all for r in cells.get((adj, k, "A"), [])]
        gaps = []
        for k in ks_all:
            for r in cells.get((adj, k, "A"), []):
                rd = r["readings"].get(adj, {})
                if "directed" in rd and "cold" in rd:
                    gaps.append(rd["directed"]["ev"] - rd["cold"]["ev"])
        summary[adj] = dict(
            k0=k0.get(adj), cells=rec, slopeA=sA, slopeB=sB,
            slopeA_composite=slope(xc, yc),
            gap_directed_cold=float(np.mean(gaps)) if gaps else None,
            believability=float(np.mean(bel)) if bel else None,
            selfclaim_per_turn=float(np.mean(scl)) if scl else None,
            enactability=picked[adj]["enact"],
            base_ev=picked[adj]["base_ev"])

    # ---- anti-marker movement (should be ~flat)
    anti_sl = []
    for adj in summary:
        xs, ys = [], []
        for k in ks_all:
            for r in cells.get((adj, k, "A"), []):
                for w in items[adj]["anti"]:
                    if w in r["readings"]:
                        xs.append(k)
                        ys.append(r["readings"][w]["cold"]["ev"])
        if xs:
            anti_sl.append(slope(xs, ys))

    # ---- moderator correlations across adjectives
    adjs = [a for a in summary if np.isfinite(summary[a]["slopeA"])]
    mods = {}
    if len(adjs) >= 4:
        sl = np.array([summary[a]["slopeA"] for a in adjs])
        for name, key in [("enactability", "enactability"),
                          ("base_ev", "base_ev"),
                          ("believability", "believability"),
                          ("selfclaim", "selfclaim_per_turn")]:
            v = np.array([summary[a][key] if summary[a][key] is not None
                          else np.nan for a in adjs], float)
            ok = np.isfinite(v) & np.isfinite(sl)
            if ok.sum() >= 4:
                mods[name] = float(np.corrcoef(sl[ok], v[ok])[0, 1])
        print("\n-- arm-A slope moderators (r across adjectives) --")
        for k2, v2 in mods.items():
            print(f"  {k2:14s} {v2:+.2f}")
        ga = [summary[a]["gap_directed_cold"] for a in adjs
              if summary[a]["gap_directed_cold"] is not None]
        print(f"\nmean arm-A slope {np.mean([summary[a]['slopeA'] for a in adjs]):+.3f}"
              f"  arm-B slope {np.nanmean([summary[a]['slopeB'] for a in adjs]):+.3f}"
              f"  DIRECTED-COLD gap {np.mean(ga):+.2f}"
              f"  anti slope {np.mean(anti_sl):+.3f}")

    # ---- BE vectors vs ENACT
    be = {}
    actdir = f"{OUT}/acts/{m}{tag}"
    try:
        import torch
        pda = torch.load(f"results/persona_vectors/{m}_pda.pt",
                         map_location="cpu", weights_only=False)
        meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
        mid = meta["mid_layer"]
        massive = json.loads(meta["massive_dims"]) if isinstance(
            meta["massive_dims"], str) else meta["massive_dims"]
        li = pda["acts_layers"].index(mid) if isinstance(
            pda["acts_layers"], list) else list(pda["acts_layers"]).index(mid)
        default_mu = np.asarray(pda["acts"]["__default__"])[:, li, :].mean(0)

        def wins(v):
            v = v.astype(np.float32).copy()
            v[massive] = 0.0
            return v

        kmax = max(ks_all)
        cos_rows = []
        for adj in summary:
            fs = [f for f in os.listdir(actdir)
                  if f.startswith(f"{adj}_K{kmax}_A_")]
            if not fs:
                continue
            tm = np.mean([np.load(f"{actdir}/{f}")["turn_mean"]
                          for f in fs], axis=0)
            bev = wins(tm - default_mu)
            en = wins(np.asarray(pda["directions"][adj])[mid])
            c_en = float(np.dot(bev, en) /
                         (np.linalg.norm(bev) * np.linalg.norm(en) + 1e-9))
            # readout-state projection onto ENACT by K (arm A)
            proj = {}
            for k in ks_all:
                fs2 = [f for f in os.listdir(actdir)
                       if f.startswith(f"{adj}_K{k}_A_")]
                if fs2:
                    lh = np.mean([wins(np.load(f"{actdir}/{f}")["last_hidden"]
                                       .astype(np.float32))
                                  for f in fs2], axis=0)
                    proj[k] = float(np.dot(lh, en) / (np.linalg.norm(en)
                                                      + 1e-9))
            be[adj] = dict(cos_BE_ENACT=c_en, enact_proj_by_K=proj)
            cos_rows.append(c_en)
        if cos_rows:
            print(f"\n-- BE vs ENACT: mean cos {np.mean(cos_rows):+.3f} "
                  f"(n={len(cos_rows)}; massive dims zeroed) --")
    except FileNotFoundError as e:
        print(f"(BE analysis skipped: {e})")

    out = f"{OUT}/{m}{tag}_analysis.json"
    json.dump(dict(model=m, summary=summary, moderators=mods,
                   anti_slope=float(np.mean(anti_sl)) if anti_sl else None,
                   be=be), open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
