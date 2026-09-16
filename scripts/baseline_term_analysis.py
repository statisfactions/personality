"""Grade P20: the baseline term of the comparative self. Reads refgroup/*_base.json
(+ _pair, _human) and the 525-PDA respondent means on the 44 medoids.
Usage: .venv/bin/python scripts/baseline_term_analysis.py
"""
import glob, json, os
import numpy as np
import pkit

labels = pkit.load.adjectives()
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
w, v = np.linalg.eigh(H); ev = v[:, np.argmax(w)]; ev *= np.sign(ev[labels.index("kind")])
Hm, hlab = pkit.load.load_human(); hlab = [l.lower() for l in hlab]
runs, seen = [], set()
for p in sorted(glob.glob("results/adjectives/refgroup/*_base.json")):
    b = json.load(open(p)); m = b["model"]; repo = pkit.roster.MODELS.get(m, m).split("/")[-1]
    if repo in seen: continue
    pp, ph = p.replace("_base.json", "_pair.json"), p.replace("_base.json", "_human.json")
    if not (os.path.exists(pp) and os.path.exists(ph)): continue
    seen.add(repo); q, h = json.load(open(pp)), json.load(open(ph)); A = b["adjectives"]
    E = lambda R, vn: np.array([R["results"][vn][a]["ev"] for a in A])
    r = dict(repo=repo, A=A, direct=E(q, "direct"), pda=E(q, "pda"))
    for k in ["assistant", "ai", "lm"]:
        r[f"base_{k}"] = E(b, f"base_{k}"); r[f"d_{k}"] = (E(q, f"more_{k}") - E(q, f"less_{k}")) / 2
    r["base_person"] = E(b, "base_person"); r["d_person"] = (E(h, "more_person") - E(h, "less_person")) / 2
    runs.append(r)
A = runs[0]["A"]; des = np.array([ev[labels.index(a)] for a in A]); hum_mean = np.array([Hm[:, hlab.index(a)].mean() for a in A])
n = len(runs); print(f"n = {n} models")
def R2(y, x):
    b = np.polyfit(x, y, 1); yh = np.polyval(b, x); return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b[0]
print("\nP20a additive reconstruction d_ref ~ a + b*(direct - base_ref), answering models only (sd d >= .5):")
for k in ["assistant", "ai", "lm", "person"]:
    ok = [r for r in runs if r[f"d_{k}"].std() >= .5]
    r2 = [R2(r[f"d_{k}"], r["direct"] - r[f"base_{k}"]) for r in ok]
    r2_self = [R2(r[f"d_{k}"], r["direct"])[0] for r in ok]; r2_base = [R2(r[f"d_{k}"], -r[f"base_{k}"])[0] for r in ok]
    print(f"  {k:9s} n={len(ok):2d}  median R2 {np.median([x[0] for x in r2]):.2f} (>.5 in {sum(x[0] > .5 for x in r2)}/{len(ok)}), slope b median {np.median([x[1] for x in r2]):.2f} | self-only R2 {np.median(r2_self):.2f}, baseline-only R2 {np.median(r2_base):.2f}")
print("\nP20b variance: sd(base_ref) >= sd(direct)?")
for k in ["assistant", "ai", "lm", "person"]:
    c = sum(r[f"base_{k}"].std() >= r["direct"].std() for r in runs)
    print(f"  {k:9s} {c}/{n} models; median sd base {np.median([r[f'base_{k}'].std() for r in runs]):.2f} vs direct {np.median([r['direct'].std() for r in runs]):.2f}")
print("\nP20c r(base_ref, direct): is the estimated average assistant the model's own self?")
for k in ["assistant", "ai", "lm", "person"]:
    rs = [np.corrcoef(r[f"base_{k}"], r["direct"])[0, 1] for r in runs if r["direct"].std() >= .5 and r[f"base_{k}"].std() >= .3]
    print(f"  {k:9s} median {np.median(rs):.2f}, > .8 in {sum(x > .8 for x in rs)}/{len(rs)}; mean level base {np.mean([r[f'base_{k}'].mean() for r in runs]):.2f} vs direct {np.mean([r['direct'].mean() for r in runs]):.2f}")
rs = [np.corrcoef(r["base_person"], hum_mean)[0, 1] for r in runs if r["base_person"].std() >= .3]
print(f"\nP20d r(base_person, actual 525-PDA respondent means on the medoids): median {np.median(rs):.2f}, > .5 in {sum(x > .5 for x in rs)}/{len(rs)}  (desirability vs human means r = {np.corrcoef(des, hum_mean)[0,1]:.2f})")
print("P20e halo of the baseline estimates, r(base_ref, desirability) median:", {k: round(np.median([np.corrcoef(r[f'base_{k}'], des)[0, 1] for r in runs if r[f'base_{k}'].std() >= .3]), 2) for k in ["assistant", "ai", "lm", "person"]})
print("     for reference r(direct, desirability) median:", round(np.median([np.corrcoef(r["direct"], des)[0, 1] for r in runs]), 2))
# what the average person / assistant is believed to be: consensus profiles
for k in ["assistant", "person"]:
    cons = np.mean([r[f"base_{k}"] for r in runs], 0); o = np.argsort(-cons)
    print(f"\nconsensus 'the average {k} is ...': top " + ", ".join(f"{A[i]} {cons[i]:.1f}" for i in o[:7]) + " | bottom " + ", ".join(f"{A[i]} {cons[i]:.1f}" for i in o[-6:]))
cons_p = np.mean([r["base_person"] for r in runs], 0); cons_a = np.mean([r["base_assistant"] for r in runs], 0); diff = cons_a - cons_p; o = np.argsort(-diff)
print("\nassistant minus person (consensus): " + ", ".join(f"{A[i]} {diff[i]:+.1f}" for i in o[:6]) + " | " + ", ".join(f"{A[i]} {diff[i]:+.1f}" for i in o[-6:]))
# the rejecters: is base_assistant == direct for them specifically?
flat = [r for r in runs if r["d_assistant"].std() < .5]
if flat: print(f"\nrejecters/don't-knows (n={len(flat)}): median r(base_assistant, direct) {np.median([np.corrcoef(r['base_assistant'], r['direct'])[0,1] for r in flat]):.2f}, mean |direct - base_assistant| {np.mean([np.abs(r['direct']-r['base_assistant']).mean() for r in flat]):.2f}; answering: {np.mean([np.abs(r['direct']-r['base_assistant']).mean() for r in runs if r['d_assistant'].std() >= .5]):.2f}")
