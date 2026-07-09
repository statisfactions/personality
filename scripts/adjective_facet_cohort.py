"""Compact cohort facet figure: HUMAN | SELF | REPRESENT | JUDGE | ENACT (W18).

Two rows of 35x35 cluster-block heatmaps (same clusters/order as
adjective_facet_dashboard.py): raw, and PC1-REMOVED (top eigencomponent of
each zero-diagonal 523x523 matrix subtracted, the four-grid convention) —
rgb's point that the raw row hinges on human PC1, which is very strong. The
three geometry channels are cohort MEANS over the 10 ENACT models. SELF gets
a real geometry via the human-PDA construction applied to models: 60
"respondents" (10 models x 6 framings), each a 523-vector of self-rating EVs.

Caches the cohort-mean 523x523 z-scored similarity per channel in
results/steer_map/facet_channel_sims.npz (delete to force recompute).

Usage: PYTHONPATH=scripts python scripts/adjective_facet_cohort.py
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import hf_logprobs as hf

FIG = Path("results/persona_vectors/figs")
MODELS = ["llama3.2", "Llama8", "qwen2.5", "Qwen7", "Qwen32",
          "gemma3", "Gemma12", "Gemma27", "phi4", "Aya"]
JUDGE_NAME = {"llama3.2": "Llama", "qwen2.5": "Qwen", "gemma3": "Gemma",
              "phi4": "Phi4"}
FRAMINGS = ["direct", "assistant", "person", "pda", "observer", "outputs"]
PANELS = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
DIV = [[0.0, "#1d5fb8"], [0.5, "#f0efe9"], [1.0, "#c93a3a"]]


def zscore_offdiag(S):
    off = S[~np.eye(S.shape[0], dtype=bool)]
    return (S - off.mean()) / off.std()


def winsorize(X, massive):
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    std = X.std(0)
    cap = std[keep].max()
    for m_ in massive:
        if std[m_] > cap:
            X[:, m_] /= (std[m_] / cap)
    return X


def cos_sim(X):
    Xc = X - X.mean(0)
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xn @ Xn.T


def remove_pc1(M):
    """Subtract the top eigencomponent of the zero-diagonal matrix
    (four_grid_compare convention)."""
    A = M.copy()
    np.fill_diagonal(A, 0.0)
    w, v = np.linalg.eigh(A)
    k = np.argmax(np.abs(w))
    return A - w[k] * np.outer(v[:, k], v[:, k])

CACHE = "results/steer_map/facet_channel_sims.npz"


def main():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    tc = json.load(open("instruments/trait_clusters.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    H0 = np.array(h["correlation_matrix"], float)

    def block(S):
        k = len(clusters)
        B = np.zeros((k, k))
        for i, ci in enumerate(clusters):
            ii = [labels.index(m) for m in ci["members"]]
            for j, cj in enumerate(clusters):
                jj = [labels.index(m) for m in cj["members"]]
                sub = S[np.ix_(ii, jj)]
                B[i, j] = (sub[~np.eye(len(ii), dtype=bool)].mean()
                           if i == j else sub.mean())
        return B

    names = []
    for c in clusters:
        hidx = [labels.index(m) for m in c["members"]]
        sub = H0[np.ix_(hidx, hidx)]
        names.append(c["members"][int(np.argmax(sub.mean(1)))])

    grids = {}
    Hm = H0.copy()
    np.fill_diagonal(Hm, 0)
    grids["HUMAN"] = block(zscore_offdiag(Hm))

    # SELF: 60 model-respondents -> adjective correlation matrix
    resp = []
    for m in MODELS:
        d = json.load(open(f"results/adjectives/selfreport/{m}_self_full.json"))
        for f in FRAMINGS:
            resp.append([d["results"][f][a]["ev"] for a in labels])
    R = np.array(resp)                       # (60, 523)
    S = np.corrcoef(R.T)                     # adjective x adjective
    np.fill_diagonal(S, 0)
    grids["SELF"] = block(zscore_offdiag(S))

    # geometry channels: cohort means (cached — 80GB of acts reads otherwise)
    import os
    if os.path.exists(CACHE):
        cz = np.load(CACHE)
        sims = {ch: cz[ch] for ch in ("REPRESENT", "JUDGE", "ENACT")}
        print("loaded channel sims from cache")
    else:
        sims = compute_channel_sims(labels)
        np.savez_compressed(CACHE, **sims)
    Hm2 = zscore_offdiag(Hm)
    S2 = zscore_offdiag(S)
    mats = {"HUMAN": Hm2, "SELF": S2, **sims}
    grids = {ch: block(M) for ch, M in mats.items()}
    grids_p = {ch: block(zscore_offdiag(remove_pc1(M))) for ch, M in mats.items()}
    render(grids, grids_p, names, clusters)


def compute_channel_sims(labels):
    acc = {"REPRESENT": [], "JUDGE": [], "ENACT": []}
    for m in MODELS:
        meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
        mid, massive = meta["mid_layer"], meta["massive_dims"]
        repo = hf.resolve(m).replace("/", "_")
        dr = torch.load(f"results/adjectives/acts/{repo}__pers.pt",
                        map_location="cpu", weights_only=False)
        adjR = [str(a).lower() for a in dr["adjectives"]]
        X = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
        X = X[[adjR.index(l) for l in labels]]
        Sm = cos_sim(winsorize(X, massive))
        np.fill_diagonal(Sm, 0)
        acc["REPRESENT"].append(zscore_offdiag(Sm))

        jn = JUDGE_NAME.get(m, m)
        z = np.load(f"results/adjectives/introspect_full/{jn}_tom_likely_dir.npz",
                    allow_pickle=True)
        ja = [str(a).lower() for a in z["adjectives"]]
        B = z["B"].astype(float)
        B = 0.5 * (B + B.T)
        B = B[np.ix_([ja.index(l) for l in labels],
                     [ja.index(l) for l in labels])]
        np.fill_diagonal(B, 0)
        acc["JUDGE"].append(zscore_offdiag(B))

        ze = np.load(f"results/persona_vectors/enact_mid/{m}.npz",
                     allow_pickle=True)
        ea = [str(a).lower() for a in ze["adjectives"]]
        E = ze["dir"].astype(np.float64)[[ea.index(l) for l in labels]]
        Xe = cos_sim(winsorize(E, massive))
        np.fill_diagonal(Xe, 0)
        acc["ENACT"].append(zscore_offdiag(Xe))
        print(f"  {m} done", flush=True)
    return {ch: np.mean(ms, axis=0) for ch, ms in acc.items()}


def render(grids, grids_p, names, clusters):
    fig = make_subplots(rows=2, cols=5, column_titles=PANELS,
                        row_titles=["raw", "PC1 removed"],
                        horizontal_spacing=0.018, vertical_spacing=0.10)
    hmask = ~np.eye(len(clusters), dtype=bool)
    for ri, G in enumerate((grids, grids_p)):
        for ci, ch in enumerate(PANELS):
            Bm = G[ch]
            fig.add_trace(go.Heatmap(z=Bm[::-1], x=names, y=names[::-1],
                                     colorscale=DIV, zmin=-2, zmax=2,
                                     showscale=False), row=ri + 1, col=ci + 1)
            if ch != "HUMAN":
                r = np.corrcoef(Bm[hmask], G["HUMAN"][hmask])[0, 1]
                fig.add_annotation(text=f"r(HUMAN)={r:.2f}",
                                   xref="x domain", yref="y domain",
                                   x=0.99, y=1.05, xanchor="right",
                                   showarrow=False,
                                   font=dict(size=10, color=INK2),
                                   row=ri + 1, col=ci + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    for ci in range(5):
        fig.update_xaxes(showticklabels=True, tickfont=dict(size=5),
                         tickangle=45, row=2, col=ci + 1)
    for ri in range(2):
        fig.update_yaxes(showticklabels=True, tickfont=dict(size=5),
                         row=ri + 1, col=1)
    fig.update_layout(
        title=dict(text="Cohort facet summary — 35 human trait clusters; "
                        "geometry channels are 10-model means; SELF = "
                        "human-PDA construction on 60 model-respondents; "
                        "bottom row: top eigencomponent removed from each "
                        "523×523 matrix before blocking",
                   font=dict(size=13, color=INK)),
        width=1500, height=760, margin=dict(l=80, r=40, t=90, b=90),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    out = FIG / "facet_cohort_summary"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
