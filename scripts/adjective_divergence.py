#!/usr/bin/env python3
"""Where do the (close) human and model adjective matrices actually disagree? (W14 §3)

The matrices correlate at r≈0.56 yet factor very differently, so the disagreement
must be localized. Per adjective, the relational agreement = correlation of its row
in the human matrix vs the cohort-mean model matrix. We characterize low- vs
high-agreement adjectives by (a) the model's evaluative-intensity axis (|PC1|) and
(b) human Big-Five membership.

Finding: agreement rises with evaluative intensity (r≈0.58). The model AGREES with
humans on evaluatively-charged dispositional words (cruel, loving, abusive) — it
gets N/A/E/C "for free" because those traits carry valence — and DISAGREES on the
evaluatively-flat ones: Openness (the one evaluatively-neutral trait; row-corr 0.39
vs N's 0.60) and the non-dispositional tail (Tall, Fat, Employed, Middle-aged).
That localizes "why O won't factor": the model organizes by evaluation, and O is
orthogonal to evaluation.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_divergence.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_geom import model_matrix
from adjective_factor_heatmap import varimax

BIGKEY = {"Troubled": "N", "Irritated": "N", "Exciting": "E", "Fascinating": "E",
          "Kind-hearted": "A", "Warm-hearted": "A", "Thorough": "C",
          "Responsible": "C", "Intelligent": "O", "Smart": "O"}
COLOR = {"N": "#d62728", "E": "#ff7f0e", "A": "#2ca02c", "C": "#1f77b4", "O": "#9467bd"}
# evaluatively-flat / non-dispositional adjectives worth labelling
LABELME = {"Tall", "Fat", "Chubby", "Middle-aged", "Employed", "Masculine", "Short",
           "Left-handed", "Predictable", "Dependent", "Liberal", "Big", "Slim",
           "Intelligent", "Creative", "Cruel", "Loving", "Abusive", "Bitter"}


def kfac(C, k):
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    L = varimax(V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None)), normalize=True)
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return L


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = np.array(human["labels"])
    np.fill_diagonal(H, 1.0)
    Cs = []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        _, C, _, _ = model_matrix(p, labels); Cs.append(C)
    M = np.mean(Cs, 0); np.fill_diagonal(M, 1.0)

    Hl = kfac(H, 5)
    hfac = np.argmax(np.abs(Hl), 1)
    big = np.array([BIGKEY.get(labels[np.argmax(np.abs(Hl[:, hfac[i]]))], "?")
                    for i in range(len(labels))])
    # human factor -> Big Five letter (by its own top adjective)
    fac2big = {j: BIGKEY.get(labels[np.argmax(np.abs(Hl[:, j]))], "?") for j in range(5)}
    big = np.array([fac2big[hfac[i]] for i in range(len(labels))])

    ev, V = np.linalg.eigh(M); o = np.argsort(ev)[::-1]
    evalax = np.abs(V[:, o[0]])                         # model evaluative-intensity axis
    ri = np.array([np.corrcoef(np.delete(H[i], i), np.delete(M[i], i))[0, 1]
                   for i in range(len(labels))])

    print("mean row-corr (M vs H) by human Big-Five factor:")
    for L in ["N", "E", "A", "C", "O"]:
        m = big == L
        print(f"  {L} (n={m.sum():3d}): {ri[m].mean():.3f}")
    print(f"corr(rowcorr, eval-intensity) = {np.corrcoef(ri, evalax)[0,1]:.3f}")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.10,
                        subplot_titles=(
        "Per adjective: evaluative intensity vs human-match (colour = human Big Five)",
        "Mean match by trait"))
    for L in ["N", "E", "A", "C", "O"]:
        m = big == L
        fig.add_trace(go.Scatter(
            x=evalax[m], y=ri[m], mode="markers", name=L,
            marker=dict(size=5, color=COLOR[L], opacity=0.6,
                        line=dict(width=0)), legendgroup=L), row=1, col=1)
    # label notable adjectives
    sel = [i for i, l in enumerate(labels) if l in LABELME]
    fig.add_trace(go.Scatter(
        x=evalax[sel], y=ri[sel], mode="markers+text",
        text=[labels[i] for i in sel], textposition="top center",
        textfont=dict(size=8), marker=dict(size=7, color="black", symbol="circle-open"),
        showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="model evaluative-intensity (|PC1| loading)", row=1, col=1)
    fig.update_yaxes(title_text="row-correlation, model vs human", row=1, col=1)
    means = [(L, (big == L).sum(), ri[big == L].mean()) for L in ["N", "A", "C", "E", "O"]]
    fig.add_trace(go.Bar(x=[m[0] for m in means], y=[m[2] for m in means],
                         marker_color=[COLOR[m[0]] for m in means],
                         text=[f"{m[2]:.2f}" for m in means], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="mean row-corr", range=[0, 0.7], row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Where the close matrices disagree: evaluatively-flat words, "
            "and Openness</b><br><sub>Agreement rises with evaluative intensity (r=0.58): "
            "the model matches humans on charged dispositional words (Cruel, Loving, "
            "Abusive) — getting N/A/E/C 'for free' — and diverges on evaluatively-neutral "
            "ones: Openness (the only valence-neutral trait, mean 0.39 vs N 0.60) and the "
            "non-dispositional tail (Tall, Fat, Employed). That is why O won't factor.</sub>",
            x=0.01),
        width=1180, height=600, font=dict(family="Helvetica, Arial"), plot_bgcolor="white")
    out = "results/adjectives/adjective_divergence.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1180, height=600, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
