"""Medoid ENACT arm, cohort analysis (P24 grading, 2026-10-05). For every
model with a medoids15 capture: displacement D(m) = ||mean_m - mean_default||
over the 44 medoids (mid layer, massive dims winsorized), S(m) = D in units
of the default cloud's spread along that direction, split-half reliability,
r with the model's own SELF EVs (and the partial beyond human desirability),
D-profile loadings on the row-centered SELF PCs, cross-model correlation of
those loadings with the models' SELF PC scores, family summaries, and the
overlap check against the full-recipe D (scripts/enact_default_profiles.py).
Usage: .venv/bin/python scripts/enact_medoid_cohort.py
"""
import glob, os, re
import numpy as np, torch
import pkit
from pkit import channels as chn, measures, load

K = 5
labels = load.adjectives(); n = len(labels)
medoids = open("tmp/medoids44.txt").read().split(); midx = np.array([labels.index(m) for m in medoids])
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
X, names = chn.self_members(labels); X = np.asarray(X, float)
Xc = X - X.mean(1, keepdims=True)
R = chn.level_removed("SELF", "respondents", X); w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
share = w[:K] / w[w > 0].sum()
Z = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12); scores = Z @ V[:, :K]
# sign convention: the pole named first in the P24 registration is positive
REF = {0: "kind", 1: "extraordinary", 2: "prosperous", 3: None}
for k in range(K):
    ref = REF.get(k)
    s = np.sign(V[labels.index(ref), k]) if ref else np.sign(V[np.argmax(np.abs(V[:, k])), k])
    V[:, k] *= s; scores[:, k] *= s
print("row-centered SELF PCs (49-core):")
for k in range(K):
    oo = np.argsort(V[:, k])
    print(f"  PC{k+1} {100*share[k]:4.1f}%  r(des) {np.corrcoef(V[:,k],t)[0,1]:+.2f}   +: " + ", ".join(labels[i] for i in oo[-7:][::-1]) + "   -: " + ", ".join(labels[i] for i in oo[:7]))

FAM = [("gemma", "gemma"), ("Llama|Tulu", "llama"), ("Qwen", "qwen"), ("Mistral|Ministral", "mistral"), ("Phi", "phi"), ("Yi-", "yi"),
       ("alcon", "falcon"), ("granite", "granite"), ("OLMo", "olmo"), ("aya|c4ai", "cohere"), ("glm", "glm"), ("zephyr", "zephyr"), ("EXAONE", "exaone")]
def family(tail): return next((f for pat, f in FAM if re.search(pat, tail)), "other")

rng = np.random.default_rng(0); rows = {}; prof = {"D": {}, "S": {}}
for p in sorted(glob.glob("results/persona_vectors/medoids15/*_medoids15.pt")):
    tail = os.path.basename(p).replace("_medoids15.pt", "").split("_", 1)[1]
    b = torch.load(p, map_location="cpu", weights_only=False); rep = b["report"]; acts = b["acts"]
    li = b["acts_layers"].index(rep["mid_layer"]); mass = np.asarray(rep["massive_dims"], int)
    if "__default__" not in acts or any(m not in acts for m in medoids): print(tail, "skipped (missing default or medoids)"); continue
    def mid(key): return measures.winsorize(np.asarray(acts[key][:, li, :], np.float32).astype(np.float64), mass)
    D0 = mid("__default__"); mu0 = D0.mean(0); A = {m: mid(m) for m in medoids}
    perm = {m: rng.permutation(len(A[m])) for m in medoids}; p0 = rng.permutation(len(D0))
    def disp(half):  # half: None = all rows; 0/1 = first/second half of each adjective's own rollouts
        sel = (lambda k: slice(None)) if half is None else (lambda k: (perm[k] if k else p0)[half * (len(perm[k] if k else p0) // 2):][: None if half else len(perm[k] if k else p0) // 2])
        m0 = D0[sel(None)].mean(0); return np.array([np.linalg.norm(A[m][sel(m)].mean(0) - m0) for m in medoids])
    D = disp(None); Da, Db = disp(0), disp(1)
    rh = np.corrcoef(Da, Db)[0, 1]; rel = 2 * rh / (1 + rh)
    dirs = np.array([A[m].mean(0) - mu0 for m in medoids]); U = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    sd_along = np.array([np.std((D0 - mu0) @ U[i]) for i in range(len(medoids))]); S = D / sd_along
    prof["D"][tail] = D; prof["S"][tail] = S
    tm = t[midx]
    def r(a, c): return np.corrcoef(a, c)[0, 1]
    def partial(a, c, ctrl):
        Q = np.c_[np.ones_like(ctrl), ctrl]; ra = a - Q @ np.linalg.lstsq(Q, a, rcond=None)[0]; rc = c - Q @ np.linalg.lstsq(Q, c, rcond=None)[0]
        return r(ra, rc)
    def resid(a, ctrl):
        Q = np.c_[np.ones_like(ctrl), ctrl]; return a - Q @ np.linalg.lstsq(Q, a, rcond=None)[0]
    d = dict(fam=family(tail), rel=rel, r_D_des=r(D, tm), r_S_des=r(S, tm), S_med=np.median(S), n_roll=int(np.median([len(A[m]) for m in medoids])), n_def=len(D0))
    if tail in names:
        ev = X[names.index(tail), midx]; d.update(r_D_self=r(D, ev), r_D_self_partial=partial(D, ev, tm), r_S_self=r(S, ev))
        d["loadD"] = [r(D, V[midx, k]) for k in range(K)]; d["loadS"] = [r(S, V[midx, k]) for k in range(K)]
        d["loadDp"] = [r(resid(D, tm), resid(V[midx, k], tm)) for k in range(K)]  # desirability partialled over the 44 medoids
        d["scores"] = scores[names.index(tail), :K]
    oS = np.argsort(S); d["cheap"] = ", ".join(medoids[i] for i in oS[:6]); d["expensive"] = ", ".join(medoids[i] for i in oS[::-1][:6])
    rows[tail] = d

print(f"\n{len(rows)} models with medoid captures; {sum('scores' in d for d in rows.values())} also in the SELF core")
print(f"{'model':32s} {'fam':8s} {'rel':>5s} {'r(D,self)':>9s} {'partial':>8s} {'r(D,des)':>8s} {'r(S,des)':>8s} {'S med':>6s}  loadD PC1..{K}")
for m, d in sorted(rows.items(), key=lambda kv: (kv[1]["fam"], kv[0])):
    ld = "".join(f"{x:+6.2f}" for x in d["loadD"]) if "loadD" in d else ""
    print(f"{m:32s} {d['fam']:8s} {d['rel']:5.2f} {d.get('r_D_self', np.nan):9.2f} {d.get('r_D_self_partial', np.nan):8.2f} {d['r_D_des']:8.2f} {d['r_S_des']:8.2f} {d['S_med']:6.2f}  {ld}")

NON_ENACTOR = {"Phi-3-medium-4k-instruct"}  # ignores the persona system prompt (verified in _texts.json 2026-10-05): rollouts are plain assistant answers
core = [m for m in rows if "scores" in rows[m] and m not in NON_ENACTOR]
print(f"excluded as non-enactors: {sorted(NON_ENACTOR & set(rows))}")
def summ(key):
    a = np.array([rows[m][key] for m in core]); return f"median {np.median(a):+.2f}, negative in {int((a < 0).sum())}/{len(a)}"
print("\nP24.2  r(D, SELF) on medoids:", summ("r_D_self"), "|  partial beyond desirability:", summ("r_D_self_partial"))
print(f"       reliability (split-half, Spearman-Brown): median {np.median([rows[m]['rel'] for m in rows]):.2f}, min {min(rows[m]['rel'] for m in rows):.2f}")
print("\ndesirability confound inside the 44-medoid subset: r(v_k[medoids], des[medoids]) = " + ", ".join(f"PC{k+1} {np.corrcoef(V[midx,k], t[midx])[0,1]:+.2f}" for k in range(K)))
print("\nP24.3-5  cross-model: SELF PC_k score vs D-profile loading on v_k (n = %d)" % len(core))
LD = np.array([rows[m]["loadD"] for m in core]); SC = np.array([rows[m]["scores"] for m in core]); LS = np.array([rows[m]["loadS"] for m in core])
for k in range(K):
    LP = np.array([rows[m]["loadDp"] for m in core])
    print(f"  PC{k+1}: r(score, loadD) {np.corrcoef(SC[:,k], LD[:,k])[0,1]:+.2f}   r(score, loadS) {np.corrcoef(SC[:,k], LS[:,k])[0,1]:+.2f}   r(score, loadD|des) {np.corrcoef(SC[:,k], LP[:,k])[0,1]:+.2f}   loadD mean {LD[:,k].mean():+.2f} (|des {LP[:,k].mean():+.2f}), negative in {int((LD[:,k]<0).sum())}/{len(core)} (|des {int((LP[:,k]<0).sum())}/{len(core)})")
print("\nP24.6  family summaries (|r(D,des)| mean; n):")
for f in sorted({rows[m]["fam"] for m in rows}):
    ms = [m for m in rows if rows[m]["fam"] == f]
    print(f"  {f:8s} n={len(ms):2d}  |r(D,des)| {np.mean([abs(rows[m]['r_D_des']) for m in ms]):.2f}   rel {np.mean([rows[m]['rel'] for m in ms]):.2f}")
for m in rows:
    if rows[m]["fam"] in ("llama", "gemma"): print(f"  {m:32s} cheap: {rows[m]['cheap']}\n  {'':32s} expensive: {rows[m]['expensive']}")
M = np.array([prof["D"][m] for m in rows if m not in NON_ENACTOR]); C = np.corrcoef(M); off = C[~np.eye(len(M), dtype=bool)]
meanD = ((M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)).mean(0)  # z within model first: raw D units are model-specific
print(f"\ncross-model agreement of medoid D profiles (enactors): mean pairwise r {off.mean():.2f} (min {off.min():.2f}); cohort-mean within-model-z D: r(des) {np.corrcoef(meanD, t[midx])[0,1]:+.2f}")
oo = np.argsort(meanD); print("  cohort cheapest:", ", ".join(medoids[i] for i in oo[:8]), "| most expensive:", ", ".join(medoids[i] for i in oo[::-1][:8]))

# P24.1 overlap against the full recipe
z = np.load("results/persona_vectors/enact_default_profiles.npz", allow_pickle=True)
full_labels = [str(a) for a in z["adjectives"]]
OVER = {"Qwen2.5-7B-Instruct": "Qwen7", "Llama-3.1-8B-Instruct": "Llama8", "gemma-3-4b-it": "gemma3", "gemma-3-12b-it": "Gemma12", "gemma-3-27b-it": "Gemma27", "Qwen2.5-32B-Instruct": "Qwen32", "aya-expanse-8b": "Aya"}
print("\nP24.1  medoid-15 D vs full-recipe D on the overlap models (44 medoids):")
for tail, short in OVER.items():
    if tail in prof["D"] and f"D__{short}" in z:
        fd = z[f"D__{short}"]; fi = [full_labels.index(m) for m in medoids]
        print(f"  {tail:24s} r(D) {np.corrcoef(prof['D'][tail], fd[fi])[0,1]:.3f}   r(S) {np.corrcoef(prof['S'][tail], z[f'S__{short}'][fi])[0,1]:.3f}")
np.savez_compressed("results/persona_vectors/enact_medoid_cohort.npz", medoids=np.array(medoids), models=np.array(list(rows)),
                    **{f"{k}__{m}": prof[k][m] for k in prof for m in rows})
print("wrote results/persona_vectors/enact_medoid_cohort.npz")
