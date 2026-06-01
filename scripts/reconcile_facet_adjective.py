#!/usr/bin/env python3
"""Reconcile the IPIP facet-geometry match with the adjective 2-factor core (W14 §2).

The puzzle: the model's *factor structure* over adjectives is a ~2-factor
evaluative core (no clean Big Five; W13 §3.11), yet its IPIP *facet geometry*
recovers the human facet correlation matrix at r≈0.56 (W9/W13 §3.9). How both?

Answer = the metric. Two lenses on the same representation:
  - RELATIONAL  : matrix-correlation of pairwise similarities to the human matrix.
                  High for facets AND adjectives (~0.56), at encoder-baseline level
                  — the Big Five RELATIONAL structure is present.
  - DIMENSIONAL : Tucker congruence of extracted varimax factors to the human Big
                  Five. Low/uneven (C weak, O absent) — because factor extraction
                  weights by variance and the model's variance concentrates on the
                  evaluative axis, so it surfaces evaluation first.
So "thin Big Five overlay" = low-variance-but-present, not absent: it contributes
fully to the relational r, but loses to evaluation in factor extraction.

Panel A: per-model relational r (facet vs adjective) + encoder baselines.
Panel B: cohort dimensional congruence to the human Big Five (the same models).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/reconcile_facet_adjective.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_geom import model_matrix
from adjective_factor_heatmap import varimax
from hf_logprobs import MODELS

BASE2SHORT = {}
for _short, _repo in MODELS.items():                 # keep the first (canonical) short
    BASE2SHORT.setdefault(_repo.split("/")[-1], _short)

FACET_BGE = 0.686    # §3.9 bge-large facet baseline (best mode); ADJ_BGE from §3.11
ADJ_BGE = 0.575


def corr_off(A, B):
    iu = np.triu_indices_from(A, 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def kfac(C, k):
    C = np.nan_to_num(C); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    L = varimax(V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None)), normalize=True)
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return L


def tuck(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def match(L0, L):
    k = L0.shape[1]
    M = np.array([[tuck(L0[:, i], L[:, j]) for j in range(k)] for i in range(k)])
    used, out = set(), np.zeros(k)
    for i in np.argsort(-M.max(1)):
        j = max((jj for jj in range(k) if jj not in used), key=lambda jj: M[i, jj])
        used.add(j); out[i] = M[i, j]
    return out


def main():
    # --- facet relational r (per model) ---
    Hf = json.load(open("instruments/ipip300_human_facet_correlations.json"))
    Hfac = np.array(Hf["correlation_matrix"], float); forder = Hf["facet_order"]
    fac = json.load(open("results/facets/ipip_facet_cluster.json"))
    facet_r = {}
    for name, d in fac.items():
        fn = d["facet_names"]; A = np.array(d["cosine_matrix"], float)
        idx = [fn.index(f) for f in forder] if set(fn) == set(forder) else list(range(30))
        facet_r[name] = corr_off(A[np.ix_(idx, idx)], Hfac)

    # --- adjective relational r (per model) + dimensional congruence ---
    Ha = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    Hadj = np.array(Ha["correlation_matrix"], float); labels = np.array(Ha["labels"])
    Href = kfac(Hadj, 5)
    htops = [", ".join(labels[np.argsort(-Href[:, j])[:2]]) for j in range(5)]
    adj_r, Cs = {}, []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        repo_base, C, _, _ = model_matrix(p, labels)
        short = BASE2SHORT.get(repo_base, repo_base)
        adj_r[short] = corr_off(C, Hadj); Cs.append(C)
    cong = np.mean([match(Href, kfac(C, 5)) for C in Cs], 0)   # per-model congruence, averaged

    # --- align models present in both ---
    common = [m for m in facet_r if m in adj_r]
    print(f"{'model':<10} facet_r  adj_r")
    for m in common:
        print(f"{m:<10} {facet_r[m]:+.3f}  {adj_r[m]:+.3f}")
    fr = np.mean([facet_r[m] for m in common]); ar = np.mean([adj_r[m] for m in common])
    print(f"\ncohort facet_r={fr:.3f}  adj_r={ar:.3f}")
    print(f"encoder baselines: bge facet={FACET_BGE}  bge adjective={ADJ_BGE}")
    print("dimensional congruence to human Big Five (cohort adjective):")
    for j, t in enumerate(htops):
        print(f"  [{t:<26}] {cong[j]:.2f}")

    # --- figure ---
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=(
        "A. RELATIONAL match (matrix-correlation r to human)",
        "B. DIMENSIONAL recovery (cohort)"))
    fig.add_trace(go.Bar(x=common, y=[facet_r[m] for m in common], name="facet geometry",
                         marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Bar(x=common, y=[adj_r[m] for m in common], name="adjective geometry",
                         marker_color="#ff7f0e"), row=1, col=1)
    fig.add_hline(y=FACET_BGE, line_dash="dot", line_color="#1f77b4", row=1, col=1,
                  annotation_text=f"bge facet {FACET_BGE:.2f}", annotation_font_size=9)
    fig.add_hline(y=ADJ_BGE, line_dash="dot", line_color="#ff7f0e", row=1, col=1,
                  annotation_text=f"bge adj {ADJ_BGE:.2f}", annotation_font_size=9)
    fig.update_yaxes(title_text="matrix-correlation r", range=[0, 0.75], row=1, col=1)
    tlabels = ["N", "E", "A", "C", "O"]            # human factor order from htops
    order = [0, 1, 2, 3, 4]
    fig.add_trace(go.Bar(x=[tlabels[i] for i in order], y=[cong[i] for i in order],
                         marker_color="#2ca02c", showlegend=False,
                         text=[f"{cong[i]:.2f}" for i in order], textposition="outside"),
                  row=1, col=2)
    fig.update_yaxes(title_text="factor congruence to Big Five", range=[0, 1.0], row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Same representation, two metrics: relational match is high "
            "and stimulus-invariant; dimensional Big-Five recovery is not</b><br><sub>"
            "A: the model reproduces the human pairwise structure for facets AND single "
            "adjectives (~0.56, at the encoder baseline) — the Big Five RELATIONAL "
            "structure is present. B: but factor extraction (variance-weighted) surfaces "
            "the evaluative axis first, so the extracted Big Five is weak (C) to absent "
            "(O). 'Thin overlay' = low-variance-but-present, not missing.</sub>", x=0.01),
        barmode="group", width=1180, height=560, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white", legend=dict(x=0.01, y=0.99))
    out = "results/adjectives/reconcile_facet_adjective.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1180, height=560, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
