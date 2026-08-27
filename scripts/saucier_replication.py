"""Saucier (1997) replication: Figure 1 scree + split-half-VARIABLES stability.

Saucier's within-sample stability index (JPSP 73:1296): split the
adjectives randomly into halves, PCA+varimax each half (k=2..10) on the
full N, compute factor SCORES from each half, correlate matched factors
across halves (maximize |r|, 1:1). Item-facet reliability — unlike
Everett's respondent-split (our bootstrap), it does not starve at small
respondent n, so it applies to the model population directly.

Runs on: HUMAN (700 x 525, C&C-ipsatized), MODEL population (n x 523
ipsatized SELF profiles, current deck roster), REPRESENT per model (523
adjectives x hidden dims, dims as observations, mid stored layer).

Usage: PYTHONPATH=scripts python scripts/saucier_replication.py
Out:   results/adjectives/saucier_replication.json + figs/saucier_fig1.png
"""
import glob
import json
import os

import numpy as np
import torch

import adjective_facet_cohort as afc
import facet_slides_wide as fw
import hf_logprobs as hf
from adjective_factor_bootstrap import kfactors
from human_axis_stability import load_human

KS = list(range(2, 11))
N_SPLITS = 20
DROP = {"internlm2_5-7b-chat", "falcon-7b-instruct"}
REPR_MODELS = ["google_gemma-3-12b-it", "Qwen_Qwen2.5-7B-Instruct",
               "meta-llama_Llama-3.1-8B-Instruct"]


def ips(M):
    return (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-9)


def scores(X, k):
    """Varimax factor scores (regression method) for observations X (n x p)."""
    Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    L, _ = kfactors(np.corrcoef(Xz.T), k)
    R = np.corrcoef(Xz.T)
    W = np.linalg.solve(R + 1e-6 * np.eye(len(R)), L)
    return Xz @ W


def split_half_stability(X, k, rng):
    p = X.shape[1]
    perm = rng.permutation(p)
    A, B = scores(X[:, perm[:p // 2]], k), scores(X[:, perm[p // 2:]], k)
    C = np.abs(np.corrcoef(A.T, B.T)[:k, k:])
    used, out = set(), []
    for i in np.argsort(-C.max(1)):
        j = max((jj for jj in range(k) if jj not in used), key=lambda jj: C[i, jj])
        used.add(j); out.append(C[i, j])
    return float(np.mean(out))


def stability_curve(X, tag, rng):
    curve = {}
    for k in KS:
        curve[k] = float(np.mean([split_half_stability(X, k, rng)
                                  for _ in range(N_SPLITS)]))
    print(f"  {tag:32s} " + " ".join(f"{curve[k]:.2f}" for k in KS))
    return curve


def load_model_population(labels):
    by_repo = {}
    for p in glob.glob("results/adjectives/selfreport/*_self_full.json"):
        name = os.path.basename(p).replace("_self_full.json", "")
        if fw.EXCLUDE.search(name):
            continue
        repo = hf.resolve(name) if name in hf.MODELS else name.replace("_", "/", 1)
        by_repo.setdefault(repo, p)
    for r in fw.THINK_PREFER:
        by_repo.setdefault(r, None)
    rows = []
    for repo, p in sorted(by_repo.items()):
        if repo.split("/")[-1] in DROP:
            continue
        path = fw.selfreport_path(repo, p)
        if path is None:
            continue
        d = json.load(open(path))["results"]
        try:
            M = np.array([[d[f][a]["ev"] for a in labels] for f in afc.FRAMINGS],
                         dtype=float)
            rows.append(np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M).mean(0))
        except (KeyError, TypeError):
            pass
    return np.array(rows)


def load_represent_X(name, labels):
    b = torch.load(f"results/adjectives/acts/{name}__pers.pt", map_location="cpu",
                   weights_only=False)
    adj = [str(a).lower() for a in b["adjectives"]]
    idx = [adj.index(l) for l in labels]
    A = np.asarray(b["acts"])
    X = A[idx, (A.shape[1] - 1) // 2, :].astype(np.float64)
    ma = np.abs(X).mean(0)
    massive = np.where(ma >= 20 * np.median(ma))[0]
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    std = X.std(0)
    cap = std[keep].max()
    for d_ in massive:
        if std[d_] > cap:
            X[:, d_] *= cap / std[d_]
    return (X - X.mean(0)).T          # dims x adjectives: dims are observations


def main():
    rng = np.random.default_rng(0)
    out = {"ks": KS, "saucier_table5_all500": [.94, .85, .84, .78, .73, .76, .64, .61, .58]}
    Mh, labels_h = load_human()
    Zh = ips(Mh)
    w_h = np.sort(np.linalg.eigvalsh(np.corrcoef(Zh.T)))[::-1][:25]
    out["fig1_human_ipsatized_eigs"] = w_h.tolist()
    print("Figure 1 (human ipsatized, first 10 eigs):", np.round(w_h[:10], 1))
    print("split-half-variables stability, k = 2..10 (Saucier Table 5 all-500: "
          + " ".join(f"{v:.2f}" for v in out["saucier_table5_all500"]) + ")")
    out["human_ipsatized"] = stability_curve(Zh, "HUMAN ipsatized (700x525)", rng)
    out["human_raw"] = stability_curve(Mh, "HUMAN raw", rng)

    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    R = load_model_population(labels)
    Zm = ips(R)
    w_m = np.sort(np.linalg.eigvalsh(np.corrcoef(Zm.T)))[::-1][:25]
    out["fig1_model_ipsatized_eigs"] = w_m.tolist()
    out["model_n"] = int(len(R))
    out["model_ipsatized"] = stability_curve(Zm, f"MODEL pop ipsatized (n={len(R)})", rng)
    out["model_raw"] = stability_curve(R, "MODEL pop raw", rng)

    out["represent"] = {}
    for name in REPR_MODELS:
        try:
            X = load_represent_X(name, labels)
            out["represent"][name] = stability_curve(X, f"REPRESENT {name[:24]} ({X.shape[0]} dims)", rng)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}")

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 26)), y=w_h, name="humans, ipsatized (Saucier Fig. 1 'All terms')",
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=list(range(1, 26)), y=w_m * (len(labels_h) / len(labels)),
                             name=f"model population, ipsatized (n={len(R)})", mode="lines+markers"))
    fig.update_layout(title="Saucier (1997) Figure 1 replication: first 25 eigenvalues",
                      xaxis_title="eigenvalue rank", yaxis_title="magnitude of eigenvalue",
                      width=1000, height=600)
    os.makedirs("results/adjectives/figs", exist_ok=True)
    fig.write_image("results/adjectives/figs/saucier_fig1.png", scale=2)
    with open("results/adjectives/saucier_replication.json", "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote results/adjectives/saucier_replication.json + figs/saucier_fig1.png")


if __name__ == "__main__":
    main()
