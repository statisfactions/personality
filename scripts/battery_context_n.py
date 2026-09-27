"""How many random contexts per adjective does a battery arm need? (rgb,
2026-09-27). For one model and k: reliability of the n-context mean
(disjoint n-vs-n context splits, 30 draws) against the Spearman-Brown
prediction from the single-context retest; and the level quantities of
the n-context mean (elevation, gain, idiosyncratic share, r with the k=1
arm) as n grows. Greedy readouts are deterministic, so context is the only
noise source: same-context repeats would be identical.
Usage: .venv/bin/python scripts/battery_context_n.py --k 20 FILE [FILE ...]
"""
import argparse, json, os
import numpy as np
import pkit
from pkit import channels as chn, roster

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=20); ap.add_argument("files", nargs="+")
args = ap.parse_args()
labels = pkit.load.adjectives(); n = len(labels); li = {a: i for i, a in enumerate(labels)}
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])

rows = []
for f in args.files:
    d = json.load(open(f)); C = max(r["context"] for r in d["records"] if r["k"] == args.k) + 1
    M = np.full((C, n), np.nan)
    for r in d["records"]:
        if r["k"] == args.k: M[r["context"], li[r["adjective"]]] = r["ev"]
    rows.append(M); model = d["model"]
M = np.vstack(rows); C = M.shape[0]
repo_us = model.replace("/", "_")
ev1 = None
for p in [f"results/adjectives/selfreport/{repo_us}_self_full.json"] + [f"results/adjectives/selfreport/{k}_self_full.json" for k, vv in roster.MODELS.items() if vv == model]:
    if os.path.exists(p):
        R = json.load(open(p))["results"]["pda"]; ev1 = np.array([R[a]["ev"] for a in labels], float); break


def level(ev):
    xc = ev - ev.mean(); r = np.corrcoef(xc, t)[0, 1]
    return ev.mean(), (xc @ t) / np.sqrt(n), 1 - r ** 2


rng = np.random.default_rng(0)
print(f"{model}: k={args.k}, {C} contexts")
r1 = None
print(f"{'n':>3s} {'split r':>8s} {'S-B pred':>9s} {'elev':>6s} {'gain':>6s} {'idio':>6s} {'r(k=1)':>7s}")
for nn in range(1, C // 2 + 1):
    rs = []
    for _ in range(30):
        p = rng.permutation(C); a, b = p[:nn], p[nn:2 * nn]
        ma, mb = np.nanmean(M[a], 0), np.nanmean(M[b], 0); ok = np.isfinite(ma) & np.isfinite(mb)
        rs.append(np.corrcoef(ma[ok], mb[ok])[0, 1])
    r = float(np.mean(rs)); r1 = r1 or r
    sb = nn * r1 / (1 + (nn - 1) * r1)
    # level quantities of an n-context mean (average over draws)
    lv = []
    for _ in range(30):
        m = np.nanmean(M[rng.permutation(C)[:nn]], 0); ok = np.isfinite(m); m = np.where(ok, m, np.nanmean(m))
        e, g, idio = level(m); rk = np.corrcoef(m[ok], ev1[ok])[0, 1] if ev1 is not None else np.nan
        lv.append((e, g, idio, rk))
    e, g, idio, rk = np.mean(lv, 0)
    print(f"{nn:3d} {r:8.3f} {sb:9.3f} {e:6.2f} {g:6.2f} {idio:6.2f} {rk:7.2f}")
m = np.nanmean(M, 0); e, g, idio = level(np.where(np.isfinite(m), m, np.nanmean(m)))
print(f"all {C:2d} contexts (no split): elev {e:.2f} gain {g:.2f} idio {idio:.2f}; reliability of the full mean by S-B from r1: {C*r1/(1+(C-1)*r1):.3f}")
