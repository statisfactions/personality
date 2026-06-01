#!/usr/bin/env python3
"""How many adjective factors are real? Scree + Horn parallel analysis (W13 §3.11).

Is 5 (the Big-Five convention) data-supported? Compare real eigenvalues to nulls
from structure-destroyed data (Horn 1965):
  human : raw 525-PDA ratings (700 x 523); null = permute each adjective column
          across respondents -> kills inter-adjective correlation.
  model : per-model adaptive-denoised adjective vectors (523 x hidden); null =
          permute each adjective's hidden entries (norm-preserving) -> kills
          inter-adjective cosine. Factor count = eigenvalues above the null 95th pct.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/scree_parallel.py
"""
import glob
import json

import numpy as np
import pyreadstat
import plotly.graph_objects as go

from adjective_corr_cluster import POR, DENY_LABELS
from adjective_geom import adaptive_denoise, LAYER_FRAC
import torch

REPS = 20
TOPN = 25


def varfracs(C):
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return ev / C.shape[0]


def human_real_null():
    df, meta = pyreadstat.read_por(POR)
    deny = {c for c in df.columns if (meta.column_names_to_labels.get(c) or c) in DENY_LABELS}
    cols = [c for c in df.columns if c.upper() != "ID" and c not in deny]
    X = df[cols].astype(float).values
    X = X[~np.isnan(X).any(1)]                     # complete cases
    real = varfracs(np.corrcoef(X, rowvar=False))
    nulls = []
    rng = np.random.default_rng(0)
    for _ in range(REPS):
        Xs = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        nulls.append(varfracs(np.corrcoef(Xs, rowvar=False)))
    return real, np.percentile(np.array(nulls), 95, axis=0)


def model_real_null(labels):
    rng = np.random.default_rng(0)
    reals, counts = [], []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        b = torch.load(p, weights_only=False)
        A = b["acts"].astype(np.float32); L = round((A.shape[1] - 1) * LAYER_FRAC)
        adj = b["adjectives"]; re = [adj.index(l) for l in labels]
        D, _, _ = adaptive_denoise(A[:, L, :][re])          # (523 x hidden) unit rows
        real = varfracs(D @ D.T)
        nrep = []
        for _ in range(5):
            Dn = np.array([rng.permutation(row) for row in D])   # norm-preserving
            nrep.append(varfracs(Dn @ Dn.T))
        null95 = np.percentile(np.array(nrep), 95, axis=0)
        counts.append(int(np.sum(real > null95)))
        reals.append(real)
    return np.mean(reals, 0), np.array(counts)


def main():
    labels = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]
    hr, hn = human_real_null()
    mr, mcounts = model_real_null(labels)
    n_h = int(np.sum(hr > hn))

    print(f"human factors above null (parallel analysis): {n_h}")
    print(f"  human cumvar @5: {hr[:5].sum():.2f}  @{n_h}: {hr[:n_h].sum():.2f}")
    print(f"model factors above null, per model: mean {mcounts.mean():.1f}, "
          f"range {mcounts.min()}–{mcounts.max()}")
    print(f"  model cumvar @5: {mr[:5].sum():.2f}")
    print(f"  human top8 varfrac: {', '.join(f'{v:.3f}' for v in hr[:8])}")
    print(f"  model top8 varfrac: {', '.join(f'{v:.3f}' for v in mr[:8])}")

    x = list(range(1, TOPN + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hr[:TOPN], name="human (real)",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=x, y=hn[:TOPN], name="human null (95%)",
                             line=dict(color="#d62728", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=x, y=mr[:TOPN], name="LLM cohort (real)",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_vline(x=5, line_dash="dash", line_color="gray",
                  annotation_text="Big Five = 5")
    fig.add_vline(x=n_h, line_dash="dot", line_color="#d62728",
                  annotation_text=f"human PA = {n_h}")
    fig.update_layout(
        title=dict(text="<b>Adjective scree: how many factors are real?</b>"
                   f"<br><sub>Horn parallel analysis. Human supports {n_h} factors "
                   f"(real &gt; null); 5 is a convention, not the elbow. Model falloff "
                   f"is gentler (model PA mean {mcounts.mean():.0f} factors).</sub>", x=0.02),
        xaxis_title="component", yaxis_title="variance fraction",
        width=820, height=540, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/scree_parallel.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=820, height=540, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
