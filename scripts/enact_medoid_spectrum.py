"""Spectral structure of the medoid displacement matrix (exploratory, rgb 2026-10-07): item-correlation spectrum of the within-model z-scored D (40 enactors x 44 medoids) and of its deviation from the cohort mean; within-model permutation null; split-half (rollout halves) component reliability; comparison with SELF-deviation PCs on the same words. Writes results/persona_vectors/enact_medoid_halves.npz. Usage: .venv/bin/python scripts/enact_medoid_spectrum.py"""
import glob, os, re, numpy as np, torch, pkit
from pkit import channels as chn, measures, load
labels = load.adjectives(); meds = open("tmp/medoids44.txt").read().split(); midx = np.array([labels.index(m) for m in meds])
H = chn.channel_matrices(labels)["HUMAN"]; w, v = np.linalg.eigh(H); t = v[:, np.argmax(w)]; t *= np.sign(t[labels.index("kind")]); tm = t[midx]
X, names = chn.self_members(labels); X = np.asarray(X, float)
R = chn.level_removed("SELF", "respondents", X); w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]; V = V[:, o]
FAM = [("gemma","gemma"),("Llama|Tulu","llama"),("Qwen","qwen"),("Mistral|Ministral|zephyr","mistral"),("Phi","phi"),("Yi-","yi"),("alcon","falcon"),("granite","granite"),("OLMo","olmo"),("aya|c4ai","cohere"),("glm","glm")]
fam = lambda s: next((f for p, f in FAM if re.search(p, s)), "other")
rng = np.random.default_rng(1); Dfull, Da, Db, used, Smed, nparams = [], [], [], [], [], []
for p in sorted(glob.glob("results/persona_vectors/medoids15/*_medoids15.pt")):
    tail = os.path.basename(p).replace("_medoids15.pt", "").split("_", 1)[1]
    if tail == "Phi-3-medium-4k-instruct" or tail not in names: continue
    b = torch.load(p, map_location="cpu", weights_only=False); rep = b["report"]; acts = b["acts"]; li = b["acts_layers"].index(rep["mid_layer"]); mass = np.asarray(rep["massive_dims"], int)
    mid = lambda k: measures.winsorize(np.asarray(acts[k][:, li, :], np.float32).astype(np.float64), mass)
    D0 = mid("__default__"); A = {m: mid(m) for m in meds}; p0 = rng.permutation(len(D0)); perm = {m: rng.permutation(len(A[m])) for m in meds}
    def disp(h):
        s0 = slice(None) if h is None else (p0[: len(p0)//2] if h == 0 else p0[len(p0)//2:]); m0 = D0[s0].mean(0)
        sel = lambda m: slice(None) if h is None else (perm[m][: len(perm[m])//2] if h == 0 else perm[m][len(perm[m])//2:])
        return np.array([np.linalg.norm(A[m][sel(m)].mean(0) - m0) for m in meds])
    D = disp(None); Dfull.append(D); Da.append(disp(0)); Db.append(disp(1)); used.append(tail)
    mu0 = D0.mean(0); U = np.array([(A[m].mean(0) - mu0) for m in meds]); U /= np.linalg.norm(U, axis=1, keepdims=True)
    Smed.append(np.median(D / np.array([np.std((D0 - mu0) @ U[i]) for i in range(len(meds))])))
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", tail); nparams.append(float(m.group(1)) if m else np.nan)
zz = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
Z, Za, Zb = zz(np.array(Dfull)), zz(np.array(Da)), zz(np.array(Db)); n_m, n_w = Z.shape
np.savez_compressed("results/persona_vectors/enact_medoid_halves.npz", models=np.array(used), medoids=np.array(meds), Z=Z, Za=Za, Zb=Zb)
def spectrum(M, label, k=6):
    C = np.corrcoef(M.T); C = np.nan_to_num(C); w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    print(f"\n{label}: item-correlation eigen shares " + " ".join(f"{100*x/w.sum():.1f}" for x in w[:k]) + f"   (PR {w.sum()**2/(w**2).sum():.1f})")
    return w, V
# 1. full z-D
w1, V1 = spectrum(Z, "FULL z-D (40 x 44)")
print("  PC1 r(des) %+.2f; poles +: %s | -: %s" % (np.corrcoef(V1[:,0], tm)[0,1], ", ".join(meds[i] for i in np.argsort(V1[:,0])[::-1][:6]), ", ".join(meds[i] for i in np.argsort(V1[:,0])[:6])))
# 2. deviation from the cohort mean (the individual-difference matrix)
Dev, Deva, Devb = Z - Z.mean(0), Za - Za.mean(0), Zb - Zb.mean(0)
w2, V2 = spectrum(Dev, "DEVIATION z-D (cohort mean removed)")
# null: permute each model's deviation across words independently -> no shared item structure
null = []
for _ in range(500):
    P = np.array([rng.permutation(row) for row in Dev]); C = np.corrcoef(P.T); wn = np.sort(np.linalg.eigvalsh(C))[::-1]; null.append(wn / wn.sum())
null = np.array(null); print("  permutation null top shares (95th pct): " + " ".join(f"{100*np.percentile(null[:,k],95):.1f}" for k in range(6)))
# component reliability: scores on half a vs half b using full-data loadings; also cross-half loading congruence
for k in range(6):
    sa, sb = Deva @ V2[:, k], Devb @ V2[:, k]; rr = np.corrcoef(sa, sb)[0, 1]
    Ca = np.corrcoef(Deva.T); Cb = np.corrcoef(Devb.T); wa, Va = np.linalg.eigh(Ca); wb, Vb = np.linalg.eigh(Cb)
    Va, Vb = Va[:, np.argsort(wa)[::-1]], Vb[:, np.argsort(wb)[::-1]]
    cong = np.abs(Va[:, :6].T @ Vb[:, k]).max()
    sc = Dev @ V2[:, k]; fams = np.array([fam(m) for m in used]); eta = 1 - sum(((sc[fams==f] - sc[fams==f].mean())**2).sum() for f in set(fams)) / ((sc - sc.mean())**2).sum()
    size_r = np.corrcoef(np.log(np.array(nparams)[~np.isnan(nparams)]), sc[~np.isnan(nparams)])[0,1]
    print(f"  dev-PC{k+1} {100*w2[k]/w2.sum():4.1f}%  split-half score r {rr:+.2f} (SB {2*rr/(1+rr):+.2f})  best cross-half loading |cos| {cong:.2f}  family eta2 {eta:.2f}  r(log size) {size_r:+.2f}  r(S med) {np.corrcoef(Smed, sc)[0,1]:+.2f}  r(des) {np.corrcoef(V2[:,k], tm)[0,1]:+.2f}")
    print(f"           +: {', '.join(meds[i] for i in np.argsort(V2[:,k])[::-1][:6])}\n           -: {', '.join(meds[i] for i in np.argsort(V2[:,k])[:6])}")
    o = np.argsort(sc); print(f"           models +: {', '.join(used[i] for i in o[::-1][:4])}\n           models -: {', '.join(used[i] for i in o[:4])}")
# 3. do the D-deviation components live in the SELF PC space at all?
Sd = X[[names.index(m) for m in used]][:, midx]; Sd = Sd - Sd.mean(0)
Cs = np.corrcoef(Sd.T); ws, Vs = np.linalg.eigh(Cs); Vs = Vs[:, np.argsort(ws)[::-1]]
print("\n|cos| D-deviation PCs (rows) x SELF-deviation PCs on the same 44 words (cols):")
for i in range(4): print(f"  dPC{i+1} " + " ".join(f"{abs(V2[:,i] @ Vs[:,j]):5.2f}" for j in range(4)))
print("  (SELF-deviation PC shares: " + " ".join(f"{100*x/ws.sum():.1f}" for x in np.sort(ws)[::-1][:4]) + ")")
