"""Centered PCs of the single-item vs battery SELF cohorts (rgb, 2026-09-27):
strength (share of the positive spectrum), cross-cohort |cos| between the
top-K components, top-subspace overlap, and per component the high-loading
and HIGH-UNIQUE-loading words (unique = loading concentrated on this
component relative to the other top-K). Both cohorts: same gate-passed
models, row-centered design (adopted recipe), item correlation.
Usage: .venv/bin/python scripts/battery_pcs.py [--K 8] [--gate .5]
"""
import argparse, glob, json
import numpy as np
import pkit
from pkit import channels as chn

ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=8); ap.add_argument("--gate", type=float, default=0.5)
args = ap.parse_args(); K = args.K
labels = pkit.load.adjectives(); n = len(labels); li = {a: i for i, a in enumerate(labels)}
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
hmean = np.asarray(chn.human_members(labels), float).mean(0)

B, retest = {}, {}
for f in sorted(glob.glob("results/adjectives/battery/*_battery.json")):
    d = json.load(open(f)); recs = [r for r in d["records"] if r["k"] == 20]
    if not recs: continue
    C = max(r["context"] for r in recs) + 1; M = np.full((C, n), np.nan)
    for r in recs: M[r["context"], li[r["adjective"]]] = r["ev"]
    if np.isfinite(M).sum(1).min() < 500: continue
    rr = [np.corrcoef(M[i], M[j])[0, 1] for i in range(C) for j in range(i + 1, C)]
    nm = d["model"].split("/")[-1]; B[nm] = np.nanmean(M, 0); retest[nm] = np.mean(rr)
S_all, s_names = chn.self_members(labels); S_all = np.asarray(S_all, float)
common = [nm for nm in s_names if nm in B and retest[nm] >= args.gate]
Sm = np.array([S_all[s_names.index(nm)] for nm in common]); Bm = np.array([B[nm] for nm in common])
print(f"{len(common)} gate-passed models in both cohorts")


def pcs(X):
    R = chn.level_removed("SELF", "respondents", X); w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]
    share = w[o] / w[w > 0].sum(); V = V[:, o[:K]]
    for k in range(K):   # sign: positive pole = higher human item mean (more desirable/common) where it matters
        if V[:, k] @ t < 0: V[:, k] *= -1
    return share[:K], V


sh_s, V_s = pcs(Sm); sh_b, V_b = pcs(Bm)
C = np.abs(V_s.T @ V_b)
print("\nstrength (% of positive spectrum), single-item vs battery:")
print("  PC:        " + "".join(f"{k+1:6d}" for k in range(K)))
print("  single:    " + "".join(f"{100*s:6.1f}" for s in sh_s)); print("  battery:   " + "".join(f"{100*s:6.1f}" for s in sh_b))
print("\n|cos| single-item PC (rows) x battery PC (cols):")
print("        " + "".join(f"  b{k+1:<3d}" for k in range(K)))
for i in range(K): print(f"  s{i+1:<3d} " + "".join(f"{C[i,j]:6.2f}" for j in range(K)) + f"   best {C[i].max():.2f} (b{C[i].argmax()+1})")
for k in (2, 3, 5, K):
    P = V_s[:, :k].T @ V_b[:, :k]; print(f"top-{k} subspace overlap (mean sq cos, 1 = identical): {(P**2).sum()/k:.2f}")


def describe(tag, V, sh):
    print(f"\n--- {tag}: loadings ---")
    for k in range(K):
        u = np.abs(V[:, k]) / np.sqrt((V ** 2).sum(1) + 1e-12)          # share of the word's top-K loading mass on this PC
        o = np.argsort(V[:, k]); uo = np.argsort(-(u * (np.abs(V[:, k]) > np.percentile(np.abs(V[:, k]), 80))))
        print(f"PC{k+1} {100*sh[k]:4.1f}%  r(t) {np.corrcoef(V[:,k],t)[0,1]:+.2f}  r(human mean) {np.corrcoef(V[:,k],hmean)[0,1]:+.2f}")
        print("   +: " + ", ".join(labels[i] for i in o[-8:][::-1])); print("   -: " + ", ".join(labels[i] for i in o[:8]))
        print("   unique: " + ", ".join(f"{labels[i]}({'+' if V[i,k]>0 else '-'})" for i in uo[:8]))


describe("single-item", V_s, sh_s); describe("battery", V_b, sh_b)
