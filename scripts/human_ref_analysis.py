"""Grade P19: human-referenced more-or-less (d_person) vs the person frame, the
AI-referenced d, and direct, on the 44 medoids across the core roster.
Reads results/adjectives/refgroup/<model>_pair.json + <model>_human.json and
the model's plain-arm person frame via pkit.load.load_self.
Usage: .venv/bin/python scripts/human_ref_analysis.py
"""
import glob, json, os
import numpy as np
import pkit

labels = pkit.load.adjectives()
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w, v = np.linalg.eigh(H); ev = v[:, np.argmax(w)]; ev *= np.sign(ev[labels.index("kind")])
HHH = ["polite", "professional", "organized", "competent", "smart", "trustworthy", "respected", "practical", "thinking"]
EMB = ["beautiful", "sickly", "romantic", "sad", "joyful", "enthusiastic", "busy", "relaxed"]
rows, seen = [], set()
for p in sorted(glob.glob("results/adjectives/refgroup/*_human.json")):
    h = json.load(open(p)); m = h["model"]; repo = pkit.roster.MODELS.get(m, m).split("/")[-1]
    if repo in seen: continue
    pp = p.replace("_human.json", "_pair.json")
    if not os.path.exists(pp): continue
    seen.add(repo); q = json.load(open(pp)); A = h["adjectives"]
    E = lambda R, vn: np.array([R["results"][vn][a]["ev"] for a in A])
    dP = (E(h, "more_person") - E(h, "less_person")) / 2
    dA = np.mean([(E(q, f"more_{k}") - E(q, f"less_{k}")) / 2 for k in ["assistant", "ai", "lm"]], 0)
    direct = E(q, "direct")
    try:
        sf = pkit.load.load_self(m); pf = sf[sf.framing == "person"].set_index("adjective")["ev"]; person = np.array([pf.get(a, np.nan) for a in A])
    except Exception:
        person = np.full(len(A), np.nan)
    mirP = np.mean(np.abs(E(h, "more_person") + E(h, "less_person") - 8) < 1); mirA = np.mean(np.abs(E(q, "more_assistant") + E(q, "less_assistant") - 8) < 1)
    rows.append(dict(model=repo, dP=dP, dA=dA, direct=direct, person=person, sdP=dP.std(), sdA=dA.std(), mirP=mirP, mirA=mirA,
                     acqP=(E(h, "more_person") + E(h, "less_person")).mean() - 8, A=A))
A = rows[0]["A"]; des = np.array([ev[labels.index(a)] for a in A]); n = len(rows)
flatP = sum(r["sdP"] < .5 for r in rows); flatA = sum(r["sdA"] < .5 for r in rows)
print(f"n = {n} models\nP19a flat (sd < .5): human-ref {flatP} vs AI-ref {flatA}")
ok = [r for r in rows if r["sdP"] >= .5 and r["sdA"] >= .5]
rPA = [np.corrcoef(r["dP"], r["dA"])[0, 1] for r in ok]
print(f"P19b r(d_person, d_AI) within model (both non-flat, n={len(ok)}): mean {np.mean(rPA):.2f}, median {np.median(rPA):.2f}, min {np.min(rPA):.2f}")
off = np.mean([r["dP"] - r["dA"] for r in ok], 0); o = np.argsort(-off)
hhh = np.mean([off[A.index(a)] for a in HHH if a in A]); emb = np.mean([off[A.index(a)] for a in EMB if a in A])
print(f"      offset d_person - d_AI: HHH/competence medoids {hhh:+.2f}, embodied/affect {emb:+.2f}; r(offset, desirability) {np.corrcoef(off, des)[0,1]:.2f}")
print("      largest +: " + ", ".join(f"{A[i]} {off[i]:+.2f}" for i in o[:7]) + "\n      largest -: " + ", ".join(f"{A[i]} {off[i]:+.2f}" for i in o[-7:]))
okp = [r for r in ok if not np.isnan(r["person"]).any() and r["person"].std() >= .5]
rD = [np.corrcoef(r["dP"], r["direct"])[0, 1] for r in okp]; rPer = [np.corrcoef(r["dP"], r["person"])[0, 1] for r in okp]
print(f"P19c r(d_person, direct) mean {np.mean(rD):.2f} vs r(d_person, person frame) mean {np.mean(rPer):.2f} (n={len(okp)}); d_person closer to direct in {sum(a > b for a, b in zip(rD, rPer))}/{len(okp)}")
print(f"P19d mirror rate: human-ref {np.mean([r['mirP'] for r in rows]):.2f} vs AI-ref (assistant) {np.mean([r['mirA'] for r in rows]):.2f}; higher under human-ref in {sum(r['mirP'] > r['mirA'] for r in rows)}/{n}")
print(f"      acquiescence index vs humans: mean {np.mean([r['acqP'] for r in rows]):+.2f}; models > +1: {sum(r['acqP'] > 1 for r in rows)}, < -1: {sum(r['acqP'] < -1 for r in rows)}")
print(f"\nhalo: mean r(d_person, desirability) {np.mean([np.corrcoef(r['dP'], des)[0,1] for r in ok]):.2f} vs d_AI {np.mean([np.corrcoef(r['dA'], des)[0,1] for r in ok]):.2f} vs direct {np.mean([np.corrcoef(r['direct'], des)[0,1] for r in ok]):.2f}")
print("level: mean d_person over medoids", f"{np.mean([r['dP'].mean() for r in ok]):+.2f}", "(0 = on average, no different from the average person); mean d_AI", f"{np.mean([r['dA'].mean() for r in ok]):+.2f}")
# grid-level: human-ref d grid vs HUMAN, same treatment as fig_moreless_grid_core
idx = [labels.index(a) for a in A]; m44 = ~np.eye(44, dtype=bool)
Hc = pkit.load.human_corr().values[np.ix_(idx, idx)].copy(); np.fill_diagonal(Hc, 0)
def grid(M): S = np.corrcoef(M.T); np.fill_diagonal(S, 0); return S
def center(M): B = M.copy(); B[m44] -= B[m44].mean(); return B
def r_(a, b): return np.corrcoef(a[m44], b[m44])[0, 1]
from pkit import measures
XP = np.array([r["dP"] for r in rows if r["sdP"] >= .5]); XA = np.array([r["dA"] for r in rows if r["sdA"] >= .5])
for nm, X in [("human-ref d", XP), ("AI-ref d", XA)]:
    G = grid(X); print(f"grid {nm:12s} n={len(X):2d}: r vs HUMAN raw {r_(G, Hc):.3f}, top-removed {r_(measures.remove_pc1(center(G)), measures.remove_pc1(center(Hc))):.3f}, offdiag mean {G[m44].mean():.3f}")
