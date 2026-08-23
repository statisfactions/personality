"""Human 525-PDA axis naming + bootstrap rank stability (2026-08-23).

The same treatment as the model population (self_population_slides /
iPC5-11 analysis): raw + C&C-ipsatized PCA over respondents, Horn
retention, and 200-rep bootstrap-over-respondents rank/identity
stability. Comparison target: models keep 4 nameable rank-stable axes
then mix at rank 5; where do humans mix?

Usage: PYTHONPATH=scripts python scripts/human_axis_stability.py
Out:   results/adjectives/human_axis_stability.json + stdout
"""
import json

import numpy as np
import pyreadstat

from adjective_corr_cluster import DENY_LABELS, POR, SWAPPED_LABEL_PAIRS


def load_human():
    df, meta = pyreadstat.read_por(POR)
    deny = {c for c in df.columns
            if (meta.column_names_to_labels.get(c) or c) in DENY_LABELS}
    cols = [c for c in df.columns if c.upper() != "ID" and c not in deny]
    A = df[cols].astype(float)
    lab = {c: (meta.column_names_to_labels.get(c) or c) for c in cols}
    for x, y in SWAPPED_LABEL_PAIRS:
        cx = [c for c in cols if lab[c] == x]
        cy = [c for c in cols if lab[c] == y]
        if cx and cy:
            A[[cx[0], cy[0]]] = A[[cy[0], cx[0]]].values
    M = A.values
    M = np.where(np.isnan(M), np.nanmean(M, axis=0, keepdims=True), M)
    return M, [lab[c] for c in cols]


def axes(X, kmax=60):
    w, v = np.linalg.eigh(np.corrcoef(X.T))
    o = np.argsort(-w)[:kmax]
    return w[o], v[:, o]


def pr(w):
    p = w[w > 0]
    return p.sum() ** 2 / (p ** 2).sum()


def main():
    M, labels = load_human()
    n_r, n_a = M.shape
    print(f"humans: {n_r} x {n_a} (NaN mean-imputed, <1% cells)")
    rng = np.random.default_rng(0)
    out = {"n_respondents": n_r, "n_adjectives": n_a}
    for mode, X in [("raw", M),
                    ("ipsatized", (M - M.mean(1, keepdims=True))
                     / np.maximum(M.std(1, keepdims=True), 1e-9))]:
        w0, V0 = axes(X)
        wf = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
        null = []
        for _ in range(10):
            P = np.array([rng.permutation(X[:, j]) for j in range(n_a)]).T
            null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1][:120])
        thr = np.percentile(null, 95, axis=0)
        k_horn = int(np.argmax(wf[:120] <= thr))
        B, KK = 200, 15
        mr = np.zeros((B, KK))
        rk = np.zeros((B, KK), int)
        for b in range(B):
            bs = rng.integers(0, n_r, n_r)
            _, Vb = axes(X[bs], kmax=40)
            C = np.abs(V0[:, :KK].T @ Vb)
            for kx in range(KK):
                j = np.argmax(C[kx])
                mr[b, kx] = C[kx, j]
                rk[b, kx] = j + 1
        axes_named = []
        for kk in range(8):
            vv = V0[:, kk] * np.sign(V0[:, kk][np.argmax(np.abs(V0[:, kk]))])
            axes_named.append({
                "rank": kk + 1, "pct_var": float(100 * w0[kk] / n_a),
                "pos": [labels[i] for i in np.argsort(-vv)[:8]],
                "neg": [labels[i] for i in np.argsort(vv)[:8]]})
        out[mode] = {
            "pr": float(pr(wf)), "horn_k": k_horn, "axes": axes_named,
            "stability": [{"rank": kx + 1,
                           "med_abs_r": float(np.median(mr[:, kx])),
                           "p_same_rank": float((rk[:, kx] == kx + 1).mean()),
                           "p_within_1": float((np.abs(rk[:, kx] - (kx + 1))
                                                <= 1).mean())}
                          for kx in range(KK)]}
        print(f"{mode}: PR {out[mode]['pr']:.1f}  Horn {k_horn}  "
              f"P(same rank) by axis: "
              + " ".join(f"{s['p_same_rank']:.2f}"
                         for s in out[mode]["stability"]))
    with open("results/adjectives/human_axis_stability.json", "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote results/adjectives/human_axis_stability.json")


if __name__ == "__main__":
    main()
