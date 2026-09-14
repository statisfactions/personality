"""Grade P17 (reference-group probe). Reads results/adjectives/refgroup/*.json.
Usage: .venv/bin/python scripts/reference_group_analysis.py
"""
import glob, json, os
import numpy as np
import pkit

REFS = ["assistant", "ai", "lm"]
H = pkit.load.human_corr().values.copy(); np.fill_diagonal(H, 0)
labels = pkit.load.adjectives()
w, v = np.linalg.eigh(H); ev_axis = v[:, np.argmax(w)]; ev_axis *= np.sign(ev_axis[labels.index("kind")])

runs = {}
for p in sorted(glob.glob("results/adjectives/refgroup/*.json")):
    d = json.load(open(p)); runs[d["model"]] = d
if not runs:
    raise SystemExit("no results yet")
adjs = next(iter(runs.values()))["adjectives"]
desir = np.array([ev_axis[labels.index(a)] for a in adjs])
def E(m, vn): return np.array([runs[m]["results"][vn][a]["ev"] for a in adjs])
def Hn(m, vn): return np.mean([runs[m]["results"][vn][a]["entropy"] for a in adjs])
models = list(runs)
print("models:", models, "| medoids:", len(adjs), "| r(medoid desirability) available\n")

print("P17a/c  elevation, spread, shape vs direct  (comparative-more, REF=AI assistant; less is reverse-coded 8-EV)")
print(f"{'model':8s} {'dir EV':>7s} {'dir sd':>6s} {'more EV':>8s} {'more sd':>7s} {'less EV':>8s} {'8-less sd':>9s} {'r(more,dir)':>11s} {'r(8-less,dir)':>13s} {'r(more,8-less)':>14s} {'H dir':>6s} {'H more':>7s}")
for m in models:
    d, mo, le = E(m, "direct"), E(m, "more_assistant"), E(m, "less_assistant")
    print(f"{m:8s} {d.mean():7.2f} {d.std():6.2f} {mo.mean():8.2f} {mo.std():7.2f} {le.mean():8.2f} {(8-le).std():9.2f} {np.corrcoef(mo, d)[0,1]:11.2f} {np.corrcoef(8-le, d)[0,1]:13.2f} {np.corrcoef(mo, 8-le)[0,1]:14.2f} {Hn(m,'direct'):6.2f} {Hn(m,'more_assistant'):7.2f}")

print("\nP17b  more/less mirror: mean(EV_more + EV_less) (8 = perfect mirror; >8 = agreement bias), per REF")
print(f"{'model':8s}" + "".join(f"{r:>12s}" for r in REFS))
for m in models:
    print(f"{m:8s}" + "".join(f"{(E(m,'more_'+r) + E(m,'less_'+r)).mean():12.2f}" for r in REFS))

print("\nP17d  REF label: comparative-more elevation / sd / r with desirability, per REF")
print(f"{'model':8s}" + "".join(f"{r+' EV':>10s}{r+' sd':>8s}{r+' rDes':>9s}" for r in REFS))
for m in models:
    print(f"{m:8s}" + "".join(f"{E(m,'more_'+r).mean():10.2f}{E(m,'more_'+r).std():8.2f}{np.corrcoef(E(m,'more_'+r), desir)[0,1]:9.2f}" for r in REFS))
print("  direct rDes per model:", {m: round(np.corrcoef(E(m, 'direct'), desir)[0, 1], 2) for m in models})

print("\nP17f  contextualized framings: |dEV| vs the uncontextualized original (mean over medoids), and r")
for f in ["direct", "observer", "outputs", "pda"]:
    line = f"  {f:9s}"
    for r in REFS:
        dd = [np.abs(E(m, f"{f}_ctx_{r}") - E(m, f)).mean() for m in models]
        rr = [np.corrcoef(E(m, f"{f}_ctx_{r}"), E(m, f))[0, 1] for m in models]
        line += f"  {r}: |dEV| {np.mean(dd):.2f} r {np.mean(rr):.2f}"
    print(line)
sh = [np.abs(E(m, "more_assistant") - E(m, "direct")).mean() for m in models]
print(f"  reference: direct -> comparative-more shift |dEV| mean {np.mean(sh):.2f}")

if len(models) < 2:
    raise SystemExit("P17e needs >= 2 models")
print("\nP17e  between-model structure (items centered): first-PC share of the models x 44 matrix")
def pc1_share(vn):
    X = np.array([E(m, vn) for m in models]); X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return s[0] / s.sum()
for vn in ["direct", "pda", "more_assistant", "more_ai", "more_lm", "direct_ctx_assistant"]:
    print(f"  {vn:22s} PC1 share {pc1_share(vn):.2f}")
print("\n  between-model mean |r| of medoid profiles (do models agree less when comparing?)")
for vn in ["direct", "more_assistant", "more_lm"]:
    X = np.array([E(m, vn) for m in models]); C = np.corrcoef(X)
    print(f"  {vn:16s} mean off-diag r {C[~np.eye(len(models), dtype=bool)].mean():.2f}")

# gain-model check: is the comparative profile a scaled version of the direct profile (rotation vs gain)?
print("\n  per-model regression more = a + b*direct (b = gain retained):")
for m in models:
    d, mo = E(m, "direct"), E(m, "more_assistant"); b = np.polyfit(d, mo, 1)
    print(f"  {m:8s} b={b[0]:.2f} a={b[1]:.2f}")
