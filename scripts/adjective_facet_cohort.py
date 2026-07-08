"""Compact cohort facet figure: HUMAN | SELF | REPRESENT | JUDGE | ENACT (W18).

One row of 35x35 cluster-block heatmaps (same clusters/order as
adjective_facet_dashboard.py). The three geometry channels are cohort MEANS
over the 10 ENACT models. SELF gets a real geometry via the human-PDA
construction applied to models: 60 "respondents" (10 models x 6 framings),
each a 523-vector of self-rating EVs; adjective x adjective correlation across
respondents = the model self-report covariance, directly comparable to
HUMAN's inter-respondent matrix sitting next to it.

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

    # geometry channels: cohort means
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
        acc["REPRESENT"].append(block(zscore_offdiag(Sm)))

        jn = JUDGE_NAME.get(m, m)
        z = np.load(f"results/adjectives/introspect_full/{jn}_tom_likely_dir.npz",
                    allow_pickle=True)
        ja = [str(a).lower() for a in z["adjectives"]]
        B = z["B"].astype(float)
        B = 0.5 * (B + B.T)
        B = B[np.ix_([ja.index(l) for l in labels],
                     [ja.index(l) for l in labels])]
        np.fill_diagonal(B, 0)
        acc["JUDGE"].append(block(zscore_offdiag(B)))

        ze = np.load(f"results/persona_vectors/enact_mid/{m}.npz",
                     allow_pickle=True)
        ea = [str(a).lower() for a in ze["adjectives"]]
        E = ze["dir"].astype(np.float64)[[ea.index(l) for l in labels]]
        Sm = cos_sim(winsorize(E, massive))
        np.fill_diagonal(Sm, 0)
        acc["ENACT"].append(block(zscore_offdiag(Sm)))
        print(f"  {m} done", flush=True)
    for ch, mats in acc.items():
        grids[ch] = np.mean(mats, axis=0)

    fig = make_subplots(rows=1, cols=5, column_titles=PANELS,
                        horizontal_spacing=0.018)
    hmask = ~np.eye(len(clusters), dtype=bool)
    for ci, ch in enumerate(PANELS):
        Bm = grids[ch]
        fig.add_trace(go.Heatmap(z=Bm[::-1], x=names, y=names[::-1],
                                 colorscale=DIV, zmin=-2, zmax=2,
                                 showscale=False), row=1, col=ci + 1)
        if ch != "HUMAN":
            r = np.corrcoef(Bm[hmask], grids["HUMAN"][hmask])[0, 1]
            fig.add_annotation(text=f"r(HUMAN)={r:.2f}",
                               xref="x domain", yref="y domain",
                               x=0.99, y=1.06, xanchor="right",
                               showarrow=False,
                               font=dict(size=10, color=INK2),
                               row=1, col=ci + 1)
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=5), tickangle=45)
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=5), row=1, col=1)
    fig.update_layout(
        title=dict(text="Cohort facet summary — 35 human trait clusters; "
                        "geometry channels are 10-model means; SELF is the "
                        "human-PDA construction on 60 model-respondents "
                        "(10 models × 6 framings); z-scored blocks, "
                        "blue − / red +",
                   font=dict(size=13, color=INK)),
        width=1500, height=400, margin=dict(l=80, r=20, t=90, b=90),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    out = FIG / "facet_cohort_summary"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
