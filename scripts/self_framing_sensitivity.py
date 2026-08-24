"""SELF framing sensitivity at wide-n (2026-08-22, rgb's request).

The population-story results (unipolar elevation PC1, ipsatized PR 11.6,
channel ranking SELF .28) were computed on model-mean profiles with the six
framings averaged (the a priori design: framings are a measurement facet).
This script asks how much rides on that: variance decomposition of the full
model x framing x adjective tensor, then the population story re-run inside
each framing separately, then within/between agreement structure.

Usage: PYTHONPATH=scripts python scripts/self_framing_sensitivity.py
Out:   stdout table + results/adjectives/self_framing_sensitivity.json
"""
import glob
import json
import os

import numpy as np

import adjective_facet_cohort as afc
import facet_slides_wide as fw
import hf_logprobs as hf

SELF_DIR = "results/adjectives/selfreport"
FR = afc.FRAMINGS


def collect_tensor(labels):
    by_repo = {}
    for p in glob.glob(f"{SELF_DIR}/*_self_full.json"):
        name = os.path.basename(p).replace("_self_full.json", "")
        if fw.EXCLUDE.search(name):
            continue
        repo = hf.resolve(name) if name in hf.MODELS else name.replace("_", "/", 1)
        by_repo.setdefault(repo, p)
    T, models = [], []
    for repo, p in sorted(by_repo.items()):
        d = json.load(open(fw.selfreport_path(repo, p)))["results"]
        try:
            T.append([[d[f][a]["ev"] for a in labels] for f in FR])
            models.append(repo.split("/")[-1])
        except (KeyError, TypeError):
            print(f"  [skip] {repo}")
    return np.array(T), models          # (n_m, 6, n_a)


def eta2_twoway(X):
    """X (n_m, n_f): additive two-way decomposition, no replication.
    Returns eta2 shares (model, framing, interaction/residual)."""
    g = X.mean()
    a = X.mean(1) - g
    b = X.mean(0) - g
    resid = X - g - a[:, None] - b[None, :]
    ss = (X - g).__pow__(2).sum()
    return (len(X[0]) * (a ** 2).sum() / ss,
            len(X) * (b ** 2).sum() / ss,
            (resid ** 2).sum() / ss)


def pr(w):
    p = w[w > 0]
    return p.sum() ** 2 / (p ** 2).sum()


def horn_k(R, n_perm=20, seed=0):
    rng = np.random.default_rng(seed)
    w = np.sort(np.linalg.eigvalsh(np.corrcoef(R.T)))[::-1]
    null = []
    for _ in range(n_perm):
        P = np.array([rng.permutation(R[:, j]) for j in range(R.shape[1])]).T
        null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1])
    thr = np.percentile(null, 95, axis=0)
    k = 0
    while k < len(w) and w[k] > thr[k]:
        k += 1
    return k


def block_grid(S, idxs):
    kb = len(idxs)
    B = np.zeros((kb, kb))
    for i, ii in enumerate(idxs):
        for j, jj in enumerate(idxs):
            sub = S[np.ix_(ii, jj)]
            B[i, j] = (sub[~np.eye(len(ii), dtype=bool)].mean()
                       if i == j else sub.mean())
    return B


def story_row(R, idxs, Hg, Hg_p):
    """One framing's population story: PC1 character, PRs, Horn, human-match."""
    C = np.corrcoef(R.T)
    v = np.linalg.eigh(C)[1]
    v1 = v[:, -1] * np.sign(v[:, -1].sum())
    scores = (R - R.mean(0)) @ v1
    r_elev = np.corrcoef(scores, R.mean(1))[0, 1]
    Z = (R - R.mean(1, keepdims=True)) / np.maximum(
        R.std(1, keepdims=True), 1e-9)
    S = np.corrcoef(R.T)
    np.fill_diagonal(S, 0)
    # slide convention (facet_slides_wide): z-score entries FIRST, then
    # remove the top component of the z-scored matrix
    M = afc.zscore_offdiag(S)
    g = block_grid(M, idxs)
    g_p = block_grid(afc.zscore_offdiag(afc.remove_pc1(M)), idxs)
    m = ~np.eye(g.shape[0], dtype=bool)
    wi, vi = np.linalg.eigh(np.corrcoef(Z.T))
    oi = np.argsort(-wi)
    return {
        "pc1_pos_frac": float((v1 > 0).mean()),
        "pc1_elev_r": float(r_elev),
        "pr_raw": float(pr(np.sort(np.linalg.eigvalsh(C))[::-1])),
        "pr_ips": float(pr(np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1])),
        "horn_raw": horn_k(R),
        "human_raw": float(np.corrcoef(g[m], Hg[m])[0, 1]),
        "human_p": float(np.corrcoef(g_p[m], Hg_p[m])[0, 1]),
        "ips_axes": vi[:, oi[:3]],
    }


def main():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    tc = json.load(open("instruments/trait_blocks_44.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    idxs = [[labels.index(m) for m in c["members"]] for c in clusters]
    Hm = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(Hm, 0)
    HM = afc.zscore_offdiag(Hm)
    Hg = block_grid(HM, idxs)
    Hg_p = block_grid(afc.zscore_offdiag(afc.remove_pc1(HM)), idxs)

    T, models = collect_tensor(labels)
    n_m, n_f, n_a = T.shape
    print(f"tensor: {n_m} models x {n_f} framings x {n_a} adjectives\n")

    # ---- A. variance decomposition ----
    e_m, e_f, e_r = eta2_twoway(T.mean(-1))
    print(f"A. elevation eta2:  model {e_m:.2f}  framing {e_f:.2f}  "
          f"interaction {e_r:.2f}   (n=10 audit was .18/.55)")
    per_item = np.array([eta2_twoway(T[:, :, j]) for j in range(n_a)])
    print(f"   per-item raw EV (median): model {np.median(per_item[:, 0]):.2f}  "
          f"framing {np.median(per_item[:, 1]):.2f}  "
          f"interaction {np.median(per_item[:, 2]):.2f}")
    Zt = ((T - T.mean(-1, keepdims=True))
          / np.maximum(T.std(-1, keepdims=True), 1e-9))
    per_item_z = np.array([eta2_twoway(Zt[:, :, j]) for j in range(n_a)])
    print(f"   per-item ipsatized (median): model {np.median(per_item_z[:, 0]):.2f}  "
          f"framing {np.median(per_item_z[:, 1]):.2f}  "
          f"interaction {np.median(per_item_z[:, 2]):.2f}\n")

    # ---- B. per-framing population story ----
    rows = {}
    for i, f in enumerate(FR):
        rows[f] = story_row(T[:, i, :], idxs, Hg, Hg_p)
    rows["mean6"] = story_row(T.mean(1), idxs, Hg, Hg_p)
    ref_axes = rows["mean6"]["ips_axes"]
    print("B. population story per framing:")
    print(f"   {'framing':>9}  pc1+%  r(elev)  PRraw  PRips  Horn  "
          f"hum_raw  hum_pc1rm  ax1  ax2  ax3")
    for f in FR + ["mean6"]:
        r = rows[f]
        ax = [max(abs(np.corrcoef(r["ips_axes"][:, k], ref_axes[:, k2])[0, 1])
                  for k2 in range(3)) for k in range(3)]
        print(f"   {f:>9}  {100*r['pc1_pos_frac']:4.0f}   {r['pc1_elev_r']:.3f}  "
              f"{r['pr_raw']:5.1f}  {r['pr_ips']:5.1f}  {r['horn_raw']:4d}  "
              f"{r['human_raw']:.3f}    {r['human_p']:.3f}   "
              f"{ax[0]:.2f} {ax[1]:.2f} {ax[2]:.2f}")
    print("   (ax_k = best |r| of this framing's ipsatized PC_k against "
          "mean6's top-3)\n")

    # ---- B2. robustness of the framing gradient ----
    def hum_p_of(R):
        S = np.corrcoef(R.T)
        np.fill_diagonal(S, 0)
        M = afc.zscore_offdiag(S)
        g_p = block_grid(afc.zscore_offdiag(afc.remove_pc1(M)), idxs)
        m = ~np.eye(g_p.shape[0], dtype=bool)
        return np.corrcoef(g_p[m], Hg_p[m])[0, 1]

    print("B2. leave-one-framing-out means (hum_pc1rm):")
    for i, f in enumerate(FR):
        keep = [j for j in range(n_f) if j != i]
        print(f"   drop {f:>9}: {hum_p_of(T[:, keep, :].mean(1)):.3f}")
    rng = np.random.default_rng(1)
    boot = {f: [] for f in FR + ["mean6"]}
    for _ in range(200):
        bs = rng.integers(0, n_m, n_m)
        for i, f in enumerate(FR):
            boot[f].append(hum_p_of(T[bs][:, i, :]))
        boot["mean6"].append(hum_p_of(T[bs].mean(1)))
    print("   bootstrap (models, 200x) hum_pc1rm 95% CI:")
    for f in FR + ["mean6"]:
        lo, hi = np.percentile(boot[f], [2.5, 97.5])
        print(f"   {f:>9}: [{lo:.3f}, {hi:.3f}]")
    d = np.array(boot["observer"]) - np.array(boot["mean6"])
    print(f"   P(observer > mean6) = {(d > 0).mean():.2f}   "
          f"P(observer > REPRESENT .41) = "
          f"{(np.array(boot['observer']) > 0.41).mean():.2f}\n")

    # ---- B3. elevation eta2 on the standing-10 subset (inversion check) ----
    stand = {hf.resolve(m).split("/")[-1] for m in
             ["llama3.2", "Llama8", "qwen2.5", "Qwen7", "Qwen32",
              "gemma3", "Gemma12", "Gemma27", "phi4", "Aya"]}
    sub = [i for i, m in enumerate(models) if m in stand]
    if len(sub) >= 8:
        em, efr, er = eta2_twoway(T[sub].mean(-1))
        print(f"B3. elevation eta2, standing-{len(sub)} subset: model {em:.2f}  "
              f"framing {efr:.2f}  interaction {er:.2f}  "
              f"(wide-64: {e_m:.2f}/{e_f:.2f}/{e_r:.2f})\n")

    # ---- C. within/between agreement on ipsatized profiles ----
    zf = Zt.reshape(n_m * n_f, n_a)
    Rho = np.corrcoef(zf)
    wm, wf, neither = [], [], []
    for a in range(n_m * n_f):
        for b in range(a + 1, n_m * n_f):
            ma, fa, mb, fb = a // n_f, a % n_f, b // n_f, b % n_f
            (wm if ma == mb else wf if fa == fb else neither).append(Rho[a, b])
    print(f"C. ipsatized profile agreement: within-model cross-framing "
          f"{np.mean(wm):.3f}   within-framing cross-model {np.mean(wf):.3f}   "
          f"neither {np.mean(neither):.3f}")
    FF = np.zeros((n_f, n_f))
    for i in range(n_f):
        for j in range(n_f):
            FF[i, j] = np.mean([np.corrcoef(Zt[m, i], Zt[m, j])[0, 1]
                                for m in range(n_m)])
    print("   framing x framing (mean within-model r):")
    print("   " + "".join(f"{f[:6]:>8}" for f in FR))
    for i, f in enumerate(FR):
        print(f"   {f[:6]:>8} " + "".join(f"{FF[i, j]:8.2f}"
                                          for j in range(n_f)))
    print()

    # ---- D. per-model framing stability ----
    stab = np.array([np.mean([np.corrcoef(Zt[m, i], Zt[m, j])[0, 1]
                              for i in range(n_f) for j in range(i + 1, n_f)])
                     for m in range(n_m)])
    o = np.argsort(stab)
    print("D. per-model cross-framing shape stability (mean pairwise r):")
    print("   least: " + ", ".join(f"{models[i]} {stab[i]:.2f}"
                                   for i in o[:5]))
    print("   most:  " + ", ".join(f"{models[i]} {stab[i]:.2f}"
                                   for i in o[-5:]))
    elev = T.mean((1, 2))
    Zm = Zt.mean(1)
    sm = Zm.mean(0)
    conform = np.array([np.corrcoef(Zm[i], sm)[0, 1] for i in range(n_m)])
    print(f"   r(stability, conformity) = "
          f"{np.corrcoef(stab, conform)[0, 1]:.2f}   "
          f"r(stability, elevation) = {np.corrcoef(stab, elev)[0, 1]:.2f}\n")

    # ---- E. elevation by framing ----
    ef = T.mean(-1)                      # (n_m, n_f)
    ranks = np.argsort(np.argsort(ef, axis=1), axis=1) + 1.0
    Rj = ranks.sum(0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    W = 12 * S / (n_m ** 2 * (n_f ** 3 - n_f))
    print(f"E. elevation by framing (marginal mean EV; Kendall's W on "
          f"framing order across models = {W:.2f}):")
    for i, f in enumerate(FR):
        print(f"   {f:>9}: {ef[:, i].mean():.2f}  (sd across models "
              f"{ef[:, i].std():.2f})")

    out = {"models": models, "framings": FR,
           "eta2_elevation": [e_m, e_f, e_r],
           "story": {f: {k: v for k, v in r.items() if k != "ips_axes"}
                     for f, r in rows.items()},
           "framing_agreement": FF.tolist(),
           "stability": {models[i]: float(stab[i]) for i in range(n_m)}}
    with open("results/adjectives/self_framing_sensitivity.json", "w") as fp:
        json.dump(out, fp, indent=1)
    print("\nwrote results/adjectives/self_framing_sensitivity.json")


if __name__ == "__main__":
    main()
