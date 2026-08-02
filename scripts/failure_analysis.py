"""Failure-arm analysis: does identity stability predict failure robustness?
(design §9; rgb's P26, registered 2026-07-31 before any failure run.)

Per model: competence self-report shift (pos - neg composite), handoff
choice mass, judged distress/confidence, and held-out accuracy, by dose
and arm. Then the cross-arm test: correlate each failure-response
measure against the PERSONA-arm update rate measured earlier.

Usage: PYTHONPATH=scripts python scripts/failure_analysis.py
"""
import json
import os

import numpy as np

OUT = "results/selfperception"
MODELS = ["llama3.2", "Llama8", "gemma3", "Gemma12", "qwen2.5", "Qwen7",
          "phi4", "Aya"]


def persona_rate(m):
    """Update rate from the persona arm (common-item where available)."""
    for tag in ("_common", ""):
        p = f"{OUT}/{m}{tag}_analysis.json"
        if os.path.exists(p):
            s = json.load(open(p))["summary"]
            adjs = sorted(s)
            k0 = np.array([s[a]["k0"] for a in adjs], float)
            return float((np.array([s[a]["cells"]["A8"]["ev"]
                                    for a in adjs]) - k0).mean())
    return float("nan")


def cell(d, arm, source, K, seed, field):
    return d[field].get(f"{arm}|{source}|{K}|{seed}")


def main():
    rows = {}
    for m in MODELS:
        p = f"{OUT}/failure_{m}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        pos, neg = d["pos"], d["neg"]
        seeds = sorted({int(k.split("|")[3]) for k in d["fork"]})
        ks = d["ks"]

        def comp(arm, source, K):
            """positive-competence minus negative-competence, seed-mean."""
            vals = []
            for s in seeds:
                r = cell(d, arm, source, K, s, "self_report")
                if r:
                    vals.append(np.mean([r[w]["ev"] for w in pos])
                                - np.mean([r[w]["ev"] for w in neg]))
            return np.mean(vals) if vals else np.nan

        def fork(arm, source, K):
            vals = [cell(d, arm, source, K, s, "fork")["continue_mass"]
                    for s in seeds if cell(d, arm, source, K, s, "fork")]
            return np.mean(vals) if vals else np.nan

        def judged(arm, source, K, key):
            if "affect_judged" not in d:
                return np.nan
            vals = [d["affect_judged"][f"{arm}|{source}|{K}|{s}"][key]
                    for s in seeds
                    if f"{arm}|{source}|{K}|{s}" in d["affect_judged"]]
            return np.mean(vals) if vals else np.nan

        def acc(arm, source, K):
            vals = [cell(d, arm, source, K, s, "accuracy")["correct"]
                    for s in seeds if cell(d, arm, source, K, s, "accuracy")]
            return np.mean(vals) if vals else np.nan

        base = comp("fail", "tool", 0)
        r = dict(persona_rate=persona_rate(m), comp0=base)
        for arm in ("fail", "succ"):
            for src in ("tool", "user"):
                r[f"comp_{arm}_{src}"] = comp(arm, src, 8) - base
                r[f"fork_{arm}_{src}"] = fork(arm, src, 8) - fork("fail",
                                                                 "tool", 0)
                r[f"distress_{arm}_{src}"] = (judged(arm, src, 8, "distressed")
                                              - judged("fail", "tool", 0,
                                                       "distressed"))
                r[f"acc_{arm}_{src}"] = acc(arm, src, 8) - acc("fail",
                                                              "tool", 0)
        r["comp_fail"] = np.nanmean([r["comp_fail_tool"], r["comp_fail_user"]])
        r["comp_succ"] = np.nanmean([r["comp_succ_tool"], r["comp_succ_user"]])
        r["fork_fail"] = np.nanmean([r["fork_fail_tool"], r["fork_fail_user"]])
        r["distress_fail"] = np.nanmean([r["distress_fail_tool"],
                                         r["distress_fail_user"]])
        r["src_effect"] = r["comp_fail_user"] - r["comp_fail_tool"]
        r["curve"] = [comp("fail", "tool", k) - base for k in ks]
        rows[m] = r

    print(f"{'model':10s} {'persona':>8s} {'comp0':>6s} {'Δcomp':>7s} "
          f"{'Δcomp':>7s} {'Δfork':>7s} {'Δdistr':>7s} {'Δacc':>6s} "
          f"{'src':>6s}")
    print(f"{'':10s} {'rate':>8s} {'':>6s} {'FAIL':>7s} {'SUCC':>7s} "
          f"{'FAIL':>7s} {'FAIL':>7s} {'FAIL':>6s} {'eff':>6s}")
    for m, r in rows.items():
        print(f"{m:10s} {r['persona_rate']:+8.2f} {r['comp0']:6.2f} "
              f"{r['comp_fail']:+7.2f} {r['comp_succ']:+7.2f} "
              f"{r['fork_fail']:+7.3f} {r['distress_fail']:+7.2f} "
              f"{r['acc_fail_tool']:+6.2f} {r['src_effect']:+6.2f}")

    print("\n=== P26: does persona-arm update rate predict failure "
          "response? (r across models) ===")
    pr = np.array([rows[m]["persona_rate"] for m in rows])
    for key, label in (("comp_fail", "competence self-report drop"),
                       ("fork_fail", "handoff choice mass"),
                       ("distress_fail", "judged distress"),
                       ("acc_fail_tool", "accuracy change"),
                       ("src_effect", "user-vs-tool source effect")):
        v = np.array([rows[m][key] for m in rows], float)
        ok = np.isfinite(v) & np.isfinite(pr)
        if ok.sum() >= 4:
            print(f"  {label:28s} r = {np.corrcoef(pr[ok], v[ok])[0,1]:+.2f} "
                  f"(n={ok.sum()})")

    print("\n=== P29 asymmetry: failure vs success dose on competence ===")
    for m, r in rows.items():
        if np.isfinite(r["comp_fail"]) and np.isfinite(r["comp_succ"]):
            print(f"  {m:10s} fail {r['comp_fail']:+.2f}  succ "
                  f"{r['comp_succ']:+.2f}  ratio "
                  f"{abs(r['comp_fail'])/(abs(r['comp_succ'])+1e-6):5.1f}x")

    json.dump({m: {k: (v if not isinstance(v, np.floating) else float(v))
                   for k, v in r.items()} for m, r in rows.items()},
              open(f"{OUT}/failure_summary.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/failure_summary.json")


if __name__ == "__main__":
    main()
