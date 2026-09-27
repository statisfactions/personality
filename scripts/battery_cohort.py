"""Battery-based SELF channel (P22). Builds each model's k=20 EV vector as
the mean over its 3 contexts, applies the retest gate (mean pairwise
context r >= --gate), and sets the battery cohort beside the single-item
cohort under the adopted recipe: gain/elevation spread, similarity and
residual congruence with HUMAN (44-block), residual split-half, and the
row-z PC1 (grandeur vs humble-assistant) replication. Also the per-model
retest, gain, and idiosyncratic share table.
Usage: .venv/bin/python scripts/battery_cohort.py [--gate .5] [--k 20]
"""
import argparse, glob, json, os
import numpy as np
import pkit
from pkit import channels as chn, measures, roster

ap = argparse.ArgumentParser(); ap.add_argument("--gate", type=float, default=0.5); ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()
labels = pkit.load.adjectives(); n = len(labels); li = {a: i for i, a in enumerate(labels)}
m44 = ~np.eye(44, dtype=bool)
Hm = np.asarray(chn.human_members(labels), float); H = chn.cohort_matrix("respondents", Hm)
Hres = chn.residual("HUMAN", "respondents", Hm)
w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])

# ---- battery cohort ----
rows, names, retest, keep = [], [], [], []
for f in sorted(glob.glob("results/adjectives/battery/*_battery.json")):
    d = json.load(open(f)); recs = [r for r in d["records"] if r["k"] == args.k]
    if not recs: continue
    C = max(r["context"] for r in recs) + 1; M = np.full((C, n), np.nan)
    for r in recs: M[r["context"], li[r["adjective"]]] = r["ev"]
    if np.isfinite(M).sum(1).min() < 500: continue
    rr = [np.corrcoef(M[i][np.isfinite(M[i]) & np.isfinite(M[j])], M[j][np.isfinite(M[i]) & np.isfinite(M[j])])[0, 1] for i in range(C) for j in range(i + 1, C)]
    ev = np.nanmean(M, 0); ev = np.where(np.isfinite(ev), ev, np.nanmean(ev))
    rows.append(ev); names.append(d["model"].split("/")[-1]); retest.append(np.mean(rr)); keep.append(np.mean(rr) >= args.gate)
B = np.array(rows); retest = np.array(retest); keep = np.array(keep)
print(f"battery models with k={args.k}: {len(B)}; pass retest gate {args.gate}: {keep.sum()}; failing:", [f"{nm} ({r:.2f})" for nm, r, k in zip(names, retest, keep) if not k])

# ---- single-item cohort, matched to the same models where possible ----
S_all, s_names = chn.self_members(labels); S_all = np.asarray(S_all, float)
common = [nm for nm in names if nm in s_names]
Sm = np.array([S_all[s_names.index(nm)] for nm in common]); Bm = np.array([B[names.index(nm)] for nm in common]); km = np.array([keep[names.index(nm)] for nm in common])
print(f"matched to single-item core: {len(common)} models ({km.sum()} pass gate)")


def level(X):
    Xc = X - X.mean(1, keepdims=True); gain = (Xc @ t) / np.sqrt(n)
    r = np.array([np.corrcoef(Xc[i], t)[0, 1] for i in range(len(X))])
    return X.mean(1), gain, 1 - r ** 2


def cohort_report(tag, X, rng):
    e, g, idio = level(X)
    S = chn.cohort_matrix("respondents", X); R = chn.residual("SELF", "respondents", X)
    rs = []
    for _ in range(20):
        p = rng.permutation(len(X)); a, b = p[: len(X) // 2], p[len(X) // 2:]
        rs.append(chn.congruence(chn.blockify(chn.residual("SELF", "respondents", X[a])), chn.blockify(chn.residual("SELF", "respondents", X[b]))))
    sh = np.mean(rs); sb = 2 * sh / (1 + sh)
    Z = measures.ipsatize(X); Rz = np.corrcoef(Z.T); wz, vz = np.linalg.eigh(Rz); v1 = vz[:, np.argmax(wz)]
    print(f"{tag:22s} n={len(X):2d} elev {e.mean():.2f}±{e.std():.2f}  gain {g.mean():.2f}±{g.std():.2f} (CV {g.std()/g.mean():.2f})  idio {idio.mean():.2f}  "
          f"sim r(H) {chn.congruence(chn.blockify(S), chn.blockify(H)):.3f}  resid r(H) {chn.congruence(chn.blockify(R), chn.blockify(Hres)):.3f}  resid split-half {sh:.3f} (S-B {sb:.2f})")
    return v1, R


rng = np.random.default_rng(0)
v1_s, R_s = cohort_report("single-item (matched)", Sm, rng)
v1_b, R_b = cohort_report("battery (matched)", Bm, rng)
v1_bg, R_bg = cohort_report("battery (gate-passed)", Bm[km], rng)
v1_sg, R_sg = cohort_report("single-item (same set)", Sm[km], rng)
print(f"\nrow-z PC1 replication (grandeur vs humble-assistant): |cos(single, battery)| = {abs(v1_s @ v1_b):.2f}; residual grids single vs battery r = {chn.congruence(chn.blockify(R_s), chn.blockify(R_b)):.3f}")
o = np.argsort(v1_b * np.sign(v1_b @ v1_s))
print("  battery row-z PC1 +:", ", ".join(labels[i] for i in o[-10:][::-1])); print("                    -:", ", ".join(labels[i] for i in o[:10]))
print("\nper-model (matched): retest, gain single -> battery, idio single -> battery")
es, gs, ids_ = level(Sm); eb, gb, idb = level(Bm)
for i in np.argsort(gs):
    print(f"  {common[i]:34s} retest {retest[names.index(common[i])]:.2f}  gain {gs[i]:.2f} -> {gb[i]:.2f}   idio {ids_[i]:.2f} -> {idb[i]:.2f}   elev {es[i]:.2f} -> {eb[i]:.2f}")
