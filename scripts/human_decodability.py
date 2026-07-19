"""Is HUMAN structure a linear image of REPRESENT? (rgb's procrustes check)

The four-grid compares similarity geometries in place: r(REPRESENT, HUMAN)
drops hard once human PC1 is removed. But geometry mismatch != information
absence: the human structure could sit in the representation rotated away
from the similarity-dominant axes (exactly the W lesson for ENACT).

Test: 5-fold CV over adjectives. Embed HUMAN as top-30 eigencomponents of the
523x523 human correlation matrix (Y = V sqrt(w)). Map model mid-layer
activations (winsorized, centered, top-100 PCs) to Y by ridge (and orthogonal
procrustes as the rotation-only variant). On held-out adjectives, rebuild
predicted human similarities Y_hat Y_hat^T and score against the actual
human block — full, and with the human-PC1 column dropped from both sides.
Baselines: the model's own cosine similarity on the same held-out block
(raw / own-PC1-removed). Plus per-human-PC CV R^2 (how deep decodability goes).

Usage: PYTHONPATH=scripts python scripts/human_decodability.py
"""
import json

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

import hf_logprobs as hf

MODELS = ["gemma3", "Qwen7", "Llama8"]
K_H, K_X, FOLDS, SEED = 30, 100, 5, 17


def winsorize(X, massive):
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    std = X.std(0)
    cap = std[keep].max()
    for m_ in massive:
        if std[m_] > cap:
            X[:, m_] /= (std[m_] / cap)
    return X


def remove_pc1(M):
    A = M.copy()
    np.fill_diagonal(A, 0.0)
    w, v = np.linalg.eigh(A)
    k = np.argmax(np.abs(w))
    return A - w[k] * np.outer(v[:, k], v[:, k])


h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
labels = [l.lower() for l in h["labels"]]
H = np.array(h["correlation_matrix"], float)
w, V = np.linalg.eigh(H)
o = np.argsort(w)[::-1][:K_H]
Y = V[:, o] * np.sqrt(np.maximum(w[o], 0))          # (523, 30); col 0 = human PC1
n = len(labels)
rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
folds = np.array_split(perm, FOLDS)


def offdiag_pairs(S, T):
    m = ~np.eye(S.shape[0], dtype=bool)
    return S[m], T[m]


for mname in MODELS:
    meta = json.load(open(f"results/persona_vectors/{mname}_pda_meta.json"))
    mid, massive = meta["mid_layer"], np.array(meta["massive_dims"], int)
    repo = hf.resolve(mname).replace("/", "_")
    dr = torch.load(f"results/adjectives/acts/{repo}__pers.pt",
                    map_location="cpu", weights_only=False)
    adjR = [str(a).lower() for a in dr["adjectives"]]
    X = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
    X = winsorize(X[[adjR.index(l) for l in labels]], massive)
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = U[:, :K_X] * s[:K_X]                        # (523, 100)

    # baselines: in-place similarity match on the pooled held-out blocks
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    S_model = Xn @ Xn.T
    S_model_p = remove_pc1(S_model)
    H_p = remove_pc1(H)

    acc = {k: ([], []) for k in ("ridge", "ridge_p", "proc", "base", "base_p")}
    Yhat_all = np.zeros_like(Y)
    for f in range(FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(FOLDS) if g != f])
        A, B = Xp[tr], Y[tr]
        # ridge with small alpha grid on an inner split
        best = None
        itr, iva = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
        for al in (1e1, 1e2, 1e3, 1e4):
            Wr = np.linalg.solve(Xp[itr].T @ Xp[itr] + al * np.eye(K_X),
                                 Xp[itr].T @ Y[itr])
            err = ((Xp[iva] @ Wr - Y[iva]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, al)
        al = best[1]
        Wr = np.linalg.solve(A.T @ A + al * np.eye(K_X), A.T @ B)
        Yh = Xp[te] @ Wr
        Yhat_all[te] = Yh

        # procrustes: whiten train X-PCs to 30 dims, rotation-only
        A30 = Xp[tr][:, :K_H] / s[:K_H]
        R, sc = orthogonal_procrustes(A30, B)
        Yp = (Xp[te][:, :K_H] / s[:K_H]) @ R * sc / len(tr)

        for key, Yh_ in (("ridge", Yh), ("proc", Yp)):
            Sp = Yh_ @ Yh_.T
            St = Y[te] @ Y[te].T
            a, b = offdiag_pairs(Sp, St)
            acc[key][0].append(a); acc[key][1].append(b)
        # PC1-dropped variant (ridge only)
        Sp = Yh[:, 1:] @ Yh[:, 1:].T
        St = Y[te][:, 1:] @ Y[te][:, 1:].T
        a, b = offdiag_pairs(Sp, St)
        acc["ridge_p"][0].append(a); acc["ridge_p"][1].append(b)
        # baselines on same held-out blocks
        a, b = offdiag_pairs(S_model[np.ix_(te, te)], H[np.ix_(te, te)])
        acc["base"][0].append(a); acc["base"][1].append(b)
        a, b = offdiag_pairs(S_model_p[np.ix_(te, te)], H_p[np.ix_(te, te)])
        acc["base_p"][0].append(a); acc["base_p"][1].append(b)

    def rr(key):
        a = np.concatenate(acc[key][0]); b = np.concatenate(acc[key][1])
        return np.corrcoef(a, b)[0, 1]

    # per-human-PC CV R^2 from pooled held-out predictions
    ss_res = ((Yhat_all - Y) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    print(f"\n{mname}: held-out human-similarity match")
    print(f"  in-place cos-sim baseline   raw {rr('base'):+.3f}   "
          f"PC1-removed {rr('base_p'):+.3f}")
    print(f"  mapped (ridge)              raw {rr('ridge'):+.3f}   "
          f"PC1-dropped {rr('ridge_p'):+.3f}")
    print(f"  mapped (procrustes-only)    raw {rr('proc'):+.3f}")
    print("  CV R^2 by human PC: " +
          " ".join(f"{i+1}:{r2[i]:+.2f}" for i in range(10)) +
          f"   mean PC2-10 {r2[1:10].mean():+.2f}, PC11-30 {r2[10:].mean():+.2f}")
