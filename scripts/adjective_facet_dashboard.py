"""Adjective-facet dashboard: the 523-set made visualizable (W17/18).

The 523 adjectives are too many to eyeball, so we aggregate to the 35
pre-registered HUMAN-derived trait clusters (instruments/trait_clusters.json,
pole-respecting Ward pool; build_trait_clusters.py) — the adjective analog of
the 30 IPIP facets in results/facets/ipip_facet_vs_human_dashboard.html.

Grid: rows = HUMAN reference + the 10 ENACT-cohort models; cols = the three
model geometry channels (REPRESENT / JUDGE / ENACT). Each panel is the 35x35
cluster-block mean of the channel's z-scored 523x523 similarity, in shared
branch order, diverging blue(-)/red(+). Panel annotation: correlation of its
block values with HUMAN's (the facet-congruence number).

Usage: PYTHONPATH=scripts python scripts/adjective_facet_dashboard.py
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
CHANNELS = ["REPRESENT", "JUDGE", "ENACT"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
DIV = [[0.0, "#1d5fb8"], [0.5, "#f0efe9"], [1.0, "#c93a3a"]]


def zscore_offdiag(S):
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    return (S - off.mean()) / off.std()


def load_clusters(labels_lower):
    tc = json.load(open("instruments/trait_clusters.json"))
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(h["correlation_matrix"])
    hl = [l.lower() for l in h["labels"]]
    clusters = []
    for c in sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"])):
        idx = [labels_lower.index(m) for m in c["members"]]
        hidx = [hl.index(m) for m in c["members"]]
        sub = H[np.ix_(hidx, hidx)]
        medoid = c["members"][int(np.argmax(sub.mean(1)))]
        clusters.append({"label": medoid, "idx": idx,
                         "branch": c["branch"]})
    return clusters


def block(S, clusters):
    k = len(clusters)
    B = np.zeros((k, k))
    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            sub = S[np.ix_(ci["idx"], cj["idx"])]
            if i == j:
                m = ~np.eye(len(ci["idx"]), dtype=bool)
                B[i, j] = sub[m].mean()
            else:
                B[i, j] = sub.mean()
    return B


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
    n = len(labels)
    clusters = load_clusters(labels)
    names = [c["label"] for c in clusters]

    grids = {}  # (rowname, channel) -> 35x35 block matrix
    H = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(H, 0)
    grids[("HUMAN", "JUDGE")] = block(zscore_offdiag(H), clusters)

    for m in MODELS:
        meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
        mid, massive = meta["mid_layer"], meta["massive_dims"]
        adj_order = sorted(meta["adjectives"].keys())
        reo = [adj_order.index(l) for l in labels]

        repo = hf.resolve(m).replace("/", "_")
        dr = torch.load(f"results/adjectives/acts/{repo}__pers.pt",
                        map_location="cpu", weights_only=False)
        adjR = [str(a).lower() for a in dr["adjectives"]]
        R = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
        R = R[[adjR.index(l) for l in labels]]
        S = cos_sim(winsorize(R, massive))
        np.fill_diagonal(S, 0)
        grids[(m, "REPRESENT")] = block(zscore_offdiag(S), clusters)

        jn = JUDGE_NAME.get(m, m)
        z = np.load(f"results/adjectives/introspect_full/{jn}_tom_likely_dir.npz",
                    allow_pickle=True)
        ja = [str(a).lower() for a in z["adjectives"]]
        B = z["B"].astype(float)
        B = 0.5 * (B + B.T)
        B = B[np.ix_([ja.index(l) for l in labels],
                     [ja.index(l) for l in labels])]
        np.fill_diagonal(B, 0)
        grids[(m, "JUDGE")] = block(zscore_offdiag(B), clusters)

        ze = np.load(f"results/persona_vectors/enact_mid/{m}.npz",
                     allow_pickle=True)
        ea = [str(a).lower() for a in ze["adjectives"]]
        E = ze["dir"].astype(np.float64)[[ea.index(l) for l in labels]]
        S = cos_sim(winsorize(E, massive))
        np.fill_diagonal(S, 0)
        grids[(m, "ENACT")] = block(zscore_offdiag(S), clusters)
        print(f"  {m} done", flush=True)

    # ---- figure ----
    rows = ["HUMAN"] + MODELS
    fig = make_subplots(
        rows=len(rows), cols=3, column_titles=CHANNELS, row_titles=rows,
        horizontal_spacing=0.03, vertical_spacing=0.012)
    href = grids[("HUMAN", "JUDGE")]
    hmask = ~np.eye(len(clusters), dtype=bool)
    for ri, rname in enumerate(rows):
        for ci, ch in enumerate(CHANNELS):
            key = (rname, ch)
            if rname == "HUMAN" and ch != "JUDGE":
                continue
            if key not in grids:
                continue
            Bm = grids[key]
            fig.add_trace(go.Heatmap(
                z=Bm[::-1], x=names, y=names[::-1],
                colorscale=DIV, zmin=-2, zmax=2, showscale=False),
                row=ri + 1, col=ci + 1)
            if rname != "HUMAN":
                r = np.corrcoef(Bm[hmask], href[hmask])[0, 1]
                fig.add_annotation(
                    text=f"r(HUMAN)={r:.2f}", xref="x domain", yref="y domain",
                    x=0.99, y=0.99, xanchor="right", yanchor="top",
                    showarrow=False, font=dict(size=9, color=INK),
                    row=ri + 1, col=ci + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # tick labels only on bottom row / left column
    for ci in range(3):
        fig.update_xaxes(showticklabels=True, tickfont=dict(size=5),
                         tickangle=45, row=len(rows), col=ci + 1)
    for ri in range(len(rows)):
        fig.update_yaxes(showticklabels=True, tickfont=dict(size=5),
                         row=ri + 1, col=1)
    fig.update_layout(
        title=dict(text="Adjective-facet dashboard — 35 human-derived trait "
                        "clusters × {REPRESENT, JUDGE, ENACT} × cohort; "
                        "z-scored block similarity, blue − / red +; HUMAN "
                        "reference top (its own correlation geometry)",
                   font=dict(size=14, color=INK)),
        width=1400, height=250 * len(rows) + 120,
        margin=dict(l=90, r=60, t=80, b=80),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    out = FIG / "adjective_facet_dashboard"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
