#!/usr/bin/env python3
"""W16 figure: varimax simple-structure — judgment is clean, representation is goofy.

For each readout (HUMAN / tom_likely judgment / representation) take the k=6 Kaiser-
varimax loadings, align its 6 factors to the HUMAN factors (Hungarian match on |Tucker
congruence|, sign-fixed), and show a markers x factors loading heatmap under one shared
row order (marker adjectives = top loaders on the human factors, grouped by factor). A
clean readout lights up BLOCK-DIAGONAL; a goofy one smears mass off the blocks. Each
column header carries its Tucker congruence to the matched human factor — the
quantitative backbone of "clean vs goofy".

Conservative construction (no profile-correlation): HUMAN direct corr, REPR cohort-mean
cosine, TOM cohort-mean prevalence-corrected judgment with diag->1 (the same _phi
pipeline repr gets). This is the apples-to-apples-with-W14 version.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/fig_varimax_simple_structure.py
"""
import glob
import json
import os
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from adjective_overextraction import _phi, rotate, _signfix_sort, _tucker
from adjective_human_match import repr_matrix

warnings.filterwarnings("ignore")
H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))
K = 6
PER = 5            # marker adjectives per human factor


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


def varimax_load(C):
    L = rotate(_phi(C, K)); L, _ = _signfix_sort(L)
    return L                                       # (523, K), peak-positive, var-sorted


def align(L, Lh):
    """Permute+sign L's columns to best match human loadings Lh (Hungarian on |cong|)."""
    Cong = np.array([[_tucker(Lh[:, i], L[:, j]) for j in range(K)] for i in range(K)])
    ri, cj = linear_sum_assignment(-np.abs(Cong))      # human i -> readout cj[i]
    out = np.zeros_like(L); cong = np.zeros(K)
    for i, j in zip(ri, cj):
        s = np.sign(Cong[i, j]) or 1.0
        out[:, i] = s * L[:, j]; cong[i] = abs(Cong[i, j])
    return out, cong


def main():
    names = [os.path.basename(p).replace("_tom_likely_dir.npz", "") for p in
             sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz"))]
    REPR = np.nanmean([cov2corr(np.nan_to_num(repr_matrix(n))) for n in names], 0)
    TOMc = np.nanmean([dc(judge(n)) for n in names], 0); np.fill_diagonal(TOMc, 1.0)

    Lh = varimax_load(HUMAN)
    # markers + row order from the human factors; column (factor) labels = top human word
    rows, blocks, flabels = [], [], []
    for i in range(K):
        top = list(np.argsort(-Lh[:, i])[:PER])
        flabels.append(LABELS[top[0]])
        for t in top:
            rows.append(t); blocks.append(i)
    labs = [LABELS[r] for r in rows]
    colnames = [f"F{i+1}: {flabels[i]}" for i in range(K)]

    readouts = [("HUMAN", HUMAN), ("tom_likely judgment (WRITE)", None),
                ("representation (READ)", REPR)]
    # fill the judgment matrix
    readouts[1] = ("tom_likely judgment (WRITE)", TOMc)

    fig = make_subplots(1, 3, horizontal_spacing=0.06,
                        subplot_titles=[t for t, _ in readouts], shared_yaxes=True)
    means = {}
    for col, (tag, C) in enumerate(readouts, 1):
        L = varimax_load(C)
        if tag == "HUMAN":
            La, cong = Lh, np.ones(K)
        else:
            La, cong = align(L, Lh)
        S = La[rows, :]
        S = S / (np.abs(S).max() + 1e-9)            # per-readout scale for comparability
        means[tag] = float(np.mean(cong[1:] if tag == "HUMAN" else cong))
        hdr = [f"{colnames[i]}<br>cong {cong[i]:.2f}" if tag != "HUMAN"
               else colnames[i] for i in range(K)]
        fig.add_trace(go.Heatmap(
            z=S, x=hdr, y=labs, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            showscale=(col == 3), colorbar=dict(title="loading", len=0.9),
            xgap=1, ygap=0.5), row=1, col=col)
        fig.update_xaxes(tickfont=dict(size=8), tickangle=40, row=1, col=col)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=8))
    # block separators on the y axis
    for col in (1, 2, 3):
        for b in range(1, K):
            fig.add_hline(y=b * PER - 0.5, line=dict(color="#888", width=0.7),
                          row=1, col=col)

    mt = means["tom_likely judgment (WRITE)"]; mr = means["representation (READ)"]
    fig.update_layout(
        title=dict(text="<b>The judgment rotates to clean trait factors; the "
            "representation smears</b><br><sub>k=6 Kaiser-varimax loadings, marker "
            "adjectives = top loaders on the HUMAN factors (rows grouped by human "
            "factor; same order all panels). Each readout's factors aligned to the "
            "human's (Hungarian, sign-fixed); column congruence to the human factor "
            f"shown. Block-diagonal = clean. Mean factor congruence to human: judgment "
            f"{mt:.2f} vs representation {mr:.2f}. Construction: no profile-correlation "
            "(diag→1 direct).</sub>", x=0.01, font=dict(size=14)),
        width=1500, height=720, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/introspect_full/fig_varimax_simple_structure.png"
    fig.write_html(out.replace(".png", ".html"), include_plotlyjs="cdn")
    fig.write_image(out, width=1500, height=720, scale=2)
    print(f"mean factor-congruence to human: judgment {mt:.3f}  repr {mr:.3f}")
    print("per-factor (human-aligned):")
    for tag, C in [("judgment", TOMc), ("repr", REPR)]:
        _, cong = align(varimax_load(C), Lh)
        print(f"  {tag:9}", "  ".join(f"{colnames[i].split(':')[0]} {cong[i]:.2f}"
                                       for i in range(K)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
