"""The default ENACT rollout as data (P23, 2026-10-03). Per model: how far
the assistant must move to enact each adjective (D), how far in units of
its own question-to-question variation (S), and the Def-2 cosine profile
(c). Reliability across rollout halves, correlations with the model's own
SELF EVs and with human desirability, cross-model agreement.
Usage: .venv/bin/python scripts/enact_default_profiles.py
"""
import glob, json, os
import numpy as np, torch
import pkit
from pkit import channels as chn, measures, roster

labels = pkit.load.adjectives(); n = len(labels)
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")])
S_all, s_names = chn.self_members(labels); S_all = np.asarray(S_all, float)
rng = np.random.default_rng(0)
rows = {}; profiles = {}
for p in sorted(glob.glob("results/persona_vectors/*_pda.pt")):
    short = os.path.basename(p).replace("_pda.pt", "")
    if short.endswith("Base") or "backfill" in short: continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    rep = b["report"]; mid = rep["mid_layer"]; li = b["acts_layers"].index(mid); mass = np.asarray(rep["massive_dims"], int)
    acts = b["acts"]; adjs = [a for a in rep["adjectives"] if a in acts and a.lower() in labels]
    if "__default__" not in acts or len(adjs) < 500: print(short, "skipped: no default or few adjectives"); continue
    def mid_acts(key): return measures.winsorize(np.asarray(acts[key][:, li, :], np.float32).astype(np.float64), mass)
    D0 = mid_acts("__default__"); mu0 = D0.mean(0); R = len(D0); half = rng.permutation(R)
    A = {a: mid_acts(a) for a in adjs}
    def disp(sel):
        m0 = D0[sel].mean(0); return np.array([np.linalg.norm(A[a][sel].mean(0) - m0) for a in adjs])
    D = disp(np.arange(R)); Da, Db = disp(half[: R // 2]), disp(half[R // 2:])
    rel_D = np.corrcoef(Da, Db)[0, 1]
    dirs = np.array([A[a].mean(0) - mu0 for a in adjs]); U = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    sd_along = np.array([np.std((D0 - mu0) @ U[i]) for i in range(len(adjs))]); S = D / sd_along
    axis = mu0 - np.concatenate([D0] + [A[a] for a in adjs]).mean(0)
    c = U @ (axis / np.linalg.norm(axis))
    # align to canonical label order
    idx = [labels.index(a.lower()) for a in adjs]
    for name, vec in [("D", D), ("S", S), ("c", c)]:
        full = np.full(n, np.nan); full[idx] = vec; profiles.setdefault(name, {})[short] = full
    repo_tail = roster.MODELS.get(short, short).split("/")[-1]
    ev = S_all[s_names.index(repo_tail)] if repo_tail in s_names else None
    ok = np.isfinite(profiles["D"][short]) & (np.isfinite(ev) if ev is not None else True)
    def r(x, y): return np.corrcoef(x[ok], y[ok])[0, 1]
    tt = t; rows[short] = dict(n_adj=len(adjs), rel_D=rel_D, r_D_self=r(profiles["D"][short], ev) if ev is not None else np.nan,
        r_S_self=r(profiles["S"][short], ev) if ev is not None else np.nan, r_c_self=r(profiles["c"][short], ev) if ev is not None else np.nan,
        r_D_des=r(profiles["D"][short], tt), r_S_des=r(profiles["S"][short], tt), r_c_des=r(profiles["c"][short], tt),
        S_med=np.nanmedian(S), S_lt1=float(np.mean(S < 1)), default_pc1=float(np.linalg.eigvalsh(np.cov((D0 - mu0).T))[-1] / np.trace(np.cov((D0 - mu0).T))))
    sS = profiles["S"][short]; o = np.argsort(np.where(np.isfinite(sS), sS, np.inf))
    rows[short]["cheap"] = ", ".join(labels[i] for i in o[:8]); rows[short]["expensive"] = ", ".join(labels[i] for i in o[::-1][:8] if np.isfinite(sS[i]))
print(f"{'model':10s} {'rel_D':>6s} {'r(D,self)':>9s} {'r(S,self)':>9s} {'r(c,self)':>9s} {'r(D,des)':>8s} {'r(S,des)':>8s} {'r(c,des)':>8s} {'S med':>6s} {'%S<1':>5s} {'defPC1':>6s}")
for m, d in rows.items():
    print(f"{m:10s} {d['rel_D']:6.2f} {d['r_D_self']:9.2f} {d['r_S_self']:9.2f} {d['r_c_self']:9.2f} {d['r_D_des']:8.2f} {d['r_S_des']:8.2f} {d['r_c_des']:8.2f} {d['S_med']:6.2f} {100*d['S_lt1']:5.0f} {d['default_pc1']:6.2f}")
for m, d in rows.items(): print(f"  {m:10s} cheap: {d['cheap']}\n  {'':10s} expensive: {d['expensive']}")
np.savez_compressed("results/persona_vectors/enact_default_profiles.npz", adjectives=np.array(labels), models=np.array(list(rows)),
                    **{f"{name}__{m}": profiles[name][m] for name in ("D", "S", "c") for m in rows})
print("wrote results/persona_vectors/enact_default_profiles.npz")
for name in ("D", "S", "c"):
    M = np.array([profiles[name][m] for m in rows]); ok = np.all(np.isfinite(M), 0); C = np.corrcoef(M[:, ok]); off = C[~np.eye(len(M), dtype=bool)]
    print(f"cross-model agreement of {name} profiles: mean pairwise r {off.mean():.2f} (min {off.min():.2f})")
