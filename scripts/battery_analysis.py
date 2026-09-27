"""Battery pilot analysis (P21). Per model and k: compliance (digit mass),
tracking (EV/entropy vs position, lag-1 digit autocorrelation), context
test-retest of adjective EVs, agreement with the k=1 single-item pda arm,
and the level quantities: elevation, gain along the human desirability
axis, idiosyncratic share (1 - r^2), plus the residual's human congruence.
Usage: .venv/bin/python scripts/battery_analysis.py [files...]
"""
import glob, json, os, sys
import numpy as np
import pkit
from pkit import channels as chn, measures

labels = pkit.load.adjectives(); n = len(labels); li = {a: i for i, a in enumerate(labels)}
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
files = sys.argv[1:] or sorted(glob.glob("results/adjectives/battery/*_battery.json"))


def level_stats(ev):
    """ev: 525 vector -> elevation, gain (SD along t), idiosyncratic share."""
    xc = ev - ev.mean(); r = np.corrcoef(xc, t)[0, 1]
    return ev.mean(), (xc @ t) / np.sqrt(n), 1 - r ** 2, r


def single_item_ev(repo_us):
    """pda-framing EV vector and the framing stability (mean pairwise r of the
    six single-item framings' EV vectors: the k=1 analog of context retest)."""
    p = f"results/adjectives/selfreport/{repo_us}_self_full.json"
    if not os.path.exists(p): return None, np.nan
    R = json.load(open(p))["results"]
    ev = np.array([R.get("pda", {}).get(a, {}).get("ev", np.nan) for a in labels], float)
    F = [f for f in R if len(R[f]) > 500]
    M = np.array([[R[f].get(a, {}).get("ev", np.nan) for a in labels] for f in F], float)
    rr = [np.corrcoef(M[i][np.isfinite(M[i]) & np.isfinite(M[j])], M[j][np.isfinite(M[i]) & np.isfinite(M[j])])[0, 1]
          for i in range(len(F)) for j in range(i + 1, len(F))]
    return (ev if np.isfinite(ev).sum() > 500 else None), (np.mean(rr) if rr else np.nan)


for f in files:
    d = json.load(open(f)); recs = d["records"]; repo_us = os.path.basename(f).split("_battery")[0]
    print(f"\n=== {d['model']}  ({len(recs)} reads)")
    ev1, fstab = single_item_ev(repo_us)
    hdr = f"{'k':>3s} {'mass':>5s} {'%<.1':>5s} {'lag1':>5s} {'EVslope':>8s} {'Hslope':>7s} {'retest':>6s} {'r(k=1)':>6s} {'elev':>5s} {'gain':>5s} {'idio':>5s} {'r(t)':>5s}"
    print(hdr)
    if ev1 is not None:
        e, g, idio, r = level_stats(ev1)
        print(f"{1:3d} {'':>5s} {'':>5s} {'':>5s} {'':>8s} {'':>7s} {fstab:6.2f} {'':>6s} {e:5.2f} {g:5.2f} {idio:5.2f} {r:5.2f}   (single-item pda arm; retest = framing stability)")
    for k in sorted({r["k"] for r in recs}):
        rk = [r for r in recs if r["k"] == k]
        mass = np.array([r["mass"] for r in rk]); pos = np.array([r["position"] for r in rk], float)
        ev = np.array([r["ev"] for r in rk]); ent = np.array([r["entropy"] for r in rk])
        # lag-1 autocorrelation of greedy digits within batteries
        ac = []
        for (c, b) in {(r["context"], r["battery"]) for r in rk}:
            s = [r["digit"] for r in rk if r["context"] == c and r["battery"] == b]
            if len(s) > 3 and np.std(s) > 0: ac.append(np.corrcoef(s[:-1], s[1:])[0, 1])
        evslope = np.polyfit(pos, ev, 1)[0] if k > 1 else 0; hslope = np.polyfit(pos, ent, 1)[0] if k > 1 else 0
        # per-adjective EV by context; test-retest = mean pairwise r across contexts
        ctxs = sorted({r["context"] for r in rk}); M = np.full((len(ctxs), n), np.nan)
        for r in rk: M[ctxs.index(r["context"]), li[r["adjective"]]] = r["ev"]
        rr = []
        for i in range(len(ctxs)):
            for j in range(i + 1, len(ctxs)):
                m = np.isfinite(M[i]) & np.isfinite(M[j]); rr.append(np.corrcoef(M[i][m], M[j][m])[0, 1])
        evk = np.nanmean(M, 0); ok = np.isfinite(evk)
        r1 = np.corrcoef(evk[ok], ev1[ok])[0, 1] if ev1 is not None else np.nan
        e, g, idio, r = level_stats(np.where(ok, evk, np.nanmean(evk)))
        print(f"{k:3d} {mass.mean():5.2f} {100*np.mean(mass<.1):5.1f} {np.nanmean(ac) if ac else np.nan:5.2f} {evslope:+8.4f} {hslope:+7.4f} {np.mean(rr) if rr else np.nan:6.2f} {r1:6.2f} {e:5.2f} {g:5.2f} {idio:5.2f} {r:5.2f}")
