#!/usr/bin/env python3
"""W16 figure: factor structure of judgment vs representation vs human (scree + varimax).

Left: scree overlay (uniform profile-correlation transform on all three, so the general
-factor comparison is apples-to-apples). The judgment (tom_likely) tracks the human's
dominant-general-factor-then-ladder shape; the representation is flat (no general factor
— the W14 co-equal pos/neg evaluative poles). Right: the k=6 Kaiser-varimax factors,
named, with whether each readout rotates to a clean trait structure or a goofy one.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/fig_judge_factor_scree.py
"""
import glob
import json
import os
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_human_match import repr_matrix

warnings.filterwarnings("ignore")
H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))


def dc(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, 1); g = np.nanmean(A)
    D = A - m[:, None] - m[None, :] + g; np.fill_diagonal(D, 0.0); return D


def cov2corr(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None)); return S / np.outer(d, d)


def judge(s):
    z = np.load(f"results/adjectives/introspect_full/{s}_tom_likely_dir.npz",
                allow_pickle=True)
    adj = list(z["adjectives"]); B = z["B"].astype(float)
    ri = [adj.index(w) for w in LABELS]; B = B[np.ix_(ri, ri)]
    return (B + B.T) / 2


def main():
    names = [os.path.basename(p).replace("_tom_likely_dir.npz", "") for p in
             sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz"))]
    TOMdc = np.nanmean([dc(judge(n)) for n in names], 0)
    REPR = np.nanmean([cov2corr(np.nan_to_num(repr_matrix(n))) for n in names], 0)

    series = [("HUMAN (525-PDA self-report)", HUMAN, "#000000"),
              ("TOM_LIKELY judgment (write)", TOMdc, "#1f77b4"),
              ("representation (read)", REPR, "#d62728")]

    fig = make_subplots(1, 1)
    for tag, M, c in series:
        P = np.corrcoef(np.nan_to_num(M))          # uniform profile-correlation
        ev = np.sort(np.linalg.eigvalsh(P))[::-1]
        pct = 100 * ev / P.shape[0]
        ratio = ev[0] / ev[1]
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=pct[:10], mode="lines+markers",
            name=f"{tag}   (PC1/PC2 = {ratio:.1f})",
            line=dict(color=c, width=2.5), marker=dict(size=7)))

    fig.update_xaxes(title_text="component", dtick=1, gridcolor="#eee")
    fig.update_yaxes(title_text="% variance", gridcolor="#eee")
    fig.update_layout(
        title=dict(text="<b>The judgment's factor structure resembles the human's; "
            "the representation's does not</b><br><sub>Scree of each readout under one "
            "uniform profile-correlation transform. Human and tom_likely judgment both "
            "have a single dominant evaluative general factor then a ladder of "
            "differentiated trait factors; the representation is flat — co-equal pos/neg "
            "evaluative poles, no general factor (the W14 '2-factor core'). Higher "
            "PC1/PC2 = stronger general factor.</sub>", x=0.01, font=dict(size=15)),
        legend=dict(x=0.55, y=0.95, font=dict(size=12)),
        width=1050, height=640, plot_bgcolor="white",
        font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/introspect_full/fig_judge_factor_scree.png"
    fig.write_html(out.replace(".png", ".html"), include_plotlyjs="cdn")
    fig.write_image(out, width=1050, height=640, scale=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
