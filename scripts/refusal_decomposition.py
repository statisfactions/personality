"""Desirability vs refusal via the more/less pair (P18, 2026-09-14).
Reads results/adjectives/refgroup/*.json. d = (more-less)/2, s = (more+less)/2 - 4.
Usage: .venv/bin/python scripts/refusal_decomposition.py [--ref assistant|ai|lm]
"""
import argparse, glob, json
import numpy as np
import pkit

ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="assistant"); args = ap.parse_args()
NA = ["beautiful", "sickly", "romantic", "busy"]
labels = pkit.load.adjectives()
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w, v = np.linalg.eigh(H); ev = v[:, np.argmax(w)]; ev *= np.sign(ev[labels.index("kind")])
runs = {json.load(open(p))["model"]: json.load(open(p)) for p in sorted(glob.glob("results/adjectives/refgroup/*.json"))}
A = next(iter(runs.values()))["adjectives"]; des = np.array([ev[labels.index(a)] for a in A])
na = np.array([a in NA for a in A])
E = lambda m, vn: np.array([runs[m]["results"][vn][a]["ev"] for a in A])
Hn = lambda m, vn: np.array([runs[m]["results"][vn][a]["entropy"] for a in A])
D, S = {}, {}
print(f"ref = {args.ref}\n{'model':8s} {'s NA':>6s} {'s trait':>8s} {'r(s,des)':>9s} {'r(d,des)':>9s} {'r(dir,des)':>10s} | direct ~ d + s: {'b_d':>5s} {'b_s':>5s} {'R2':>4s} | {'H NA-dd':>7s} {'H trait':>7s}  most-refused (s)")
for m in runs:
    mo, le, di = E(m, f"more_{args.ref}"), E(m, f"less_{args.ref}"), E(m, "direct")
    d, s = (mo - le) / 2, (mo + le) / 2 - 4; D[m], S[m] = d, s
    X = np.c_[d, s, np.ones_like(d)]; b, *_ = np.linalg.lstsq(X, di, rcond=None)
    r2 = 1 - ((di - X @ b) ** 2).sum() / ((di - di.mean()) ** 2).sum()
    Hm, Hl = Hn(m, f"more_{args.ref}"), Hn(m, f"less_{args.ref}")
    dd = (mo < 3.5) & (le < 3.5)
    hna = np.mean((Hm + Hl)[dd] / 2) if dd.any() else np.nan; htr = np.mean((Hm + Hl)[~na] / 2)
    o = np.argsort(s)
    print(f"{m:8s} {s[na].mean():6.2f} {s[~na].mean():8.2f} {np.corrcoef(s, des)[0,1]:9.2f} {np.corrcoef(d, des)[0,1]:9.2f} {np.corrcoef(di, des)[0,1]:10.2f} | {'':16s}{b[0]:5.2f} {b[1]:5.2f} {r2:4.2f} | {hna:7.2f} {htr:7.2f}  " + ", ".join(f"{A[i]} {s[i]:+.1f}" for i in o[:5]))
ms = list(runs); k = len(ms)
def agree(Z):
    C = np.corrcoef(np.array([Z[m] for m in ms])); return C[~np.eye(k, dtype=bool)].mean()
print(f"\nbetween-model agreement (mean off-diag r): direction d {agree(D):.2f}   stance s {agree(S):.2f}   direct {agree({m: E(m,'direct') for m in ms}):.2f}")
# consensus refusal profile
sbar = np.mean([S[m] for m in ms], 0); o = np.argsort(sbar)
print("consensus stance (mean s over models), lowest:", ", ".join(f"{A[i]} {sbar[i]:+.2f}" for i in o[:8]))
print("                                       highest:", ", ".join(f"{A[i]} {sbar[i]:+.2f}" for i in o[-6:]))
