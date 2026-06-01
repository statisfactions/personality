#!/usr/bin/env python3
"""Capstone: intensity over valence, for affect AND evaluation (W13 §3.11).

The model's first organizing axis for adjectives is INTENSITY/extremity ("how
loaded is this word"), not sign. Two domains show the same pattern:
  affect     : Cheerful ↔ Angry positively related (presence > valence)
  evaluation : Wonderful ↔ Awful positively related (extremity > good/bad)
Mechanism: distributional twins (same emphatic contexts) — associative similarity
dominates, symbolic sign is the residual. Humans keep both blocks ~independent.

RSA presence/valence (a, b) per block (see affect_rsa.py) for human, the LLM
cohort (adaptive denoise), and bge+mpnet — plotted as presence (x) vs valence (y):
humans live up-left (valence ≫ presence); text models collapse rightward.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/intensity_vs_valence.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer

from adjective_geom import model_matrix
from affect_rsa import rsa

HUMAN = "results/adjectives/escs_525pda_corr_raw.json"
ENCODERS = ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]

BLOCKS = {
    "affect": (
        ["Cheerful", "Happy", "Joyful", "Glad", "Pleased", "Delightful", "Lively",
         "Optimistic", "Good-humored", "Good-natured", "Positive", "Laughing"],
        ["Sad", "Unhappy", "Angry", "Irritated", "Moody", "Crabby", "Grumpy",
         "Upset", "Depressed", "Troubled", "Tense", "Nervous"]),
    "evaluation": (
        ["Wonderful", "Amazing", "Impressive", "Awesome", "Great", "Terrific",
         "Remarkable", "Admirable", "Excellent", "Fabulous", "Outstanding",
         "Magnificent", "Brilliant", "Exceptional", "Splendid", "Marvelous"],
        ["Awful", "Disgusting", "Terrible", "Bad", "Horrible", "Dreadful",
         "Atrocious", "Lousy", "Nasty", "Hideous", "Repulsive", "Vile", "Rotten",
         "Pathetic", "Appalling", "Revolting"]),
}


def main():
    human = json.load(open(HUMAN))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    idx = {l: i for i, l in enumerate(labels)}
    Cs = {}
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        name, C, _, _ = model_matrix(p, labels)
        Cs[name] = C
    texts = [f"someone who is {l.lower()}" for l in labels]
    encs = {}
    for repo in ENCODERS:
        emb = np.asarray(SentenceTransformer(repo).encode(
            texts, show_progress_bar=False, normalize_embeddings=True), float)
        De = emb - emb.mean(0); De /= (np.linalg.norm(De, axis=1, keepdims=True) + 1e-9)
        encs[repo.split("/")[-1]] = De @ De.T

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(BLOCKS),
                        horizontal_spacing=0.12)
    for col, (block, (POS, NEG)) in enumerate(BLOCKS.items(), 1):
        pi = [idx[w] for w in POS if w in idx]; ni = [idx[w] for w in NEG if w in idx]
        print(f"{block}: {len(pi)} pos / {len(ni)} neg")
        groups = {"human": [rsa(H, pi, ni)],
                  "LLM cohort": [rsa(C, pi, ni) for C in Cs.values()],
                  "encoder": [rsa(C, pi, ni) for C in encs.values()]}
        style = {"human": ("star", 22, "#d62728"),
                 "LLM cohort": ("circle", 10, "#1f77b4"),
                 "encoder": ("diamond", 13, "#2ca02c")}
        fig.add_trace(go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines",
            line=dict(dash="dot", color="lightgray"), showlegend=False,
            hoverinfo="skip"), row=1, col=col)
        for g, recs in groups.items():
            sym, sz, c = style[g]
            fig.add_trace(go.Scatter(
                x=[r["presence"] for r in recs], y=[r["valence"] for r in recs],
                mode="markers", marker=dict(symbol=sym, size=sz, color=c,
                                            line=dict(width=1, color="white")),
                name=g, legendgroup=g, showlegend=(col == 1)), row=1, col=col)
        fig.update_xaxes(title_text="presence / intensity  a", range=[0, 0.5], row=1, col=col)
        fig.update_yaxes(title_text="valence  b  (sign)", range=[-0.05, 0.42], row=1, col=col)
        fig.add_annotation(x=0.06, y=0.40, xref=f"x{col if col>1 else ''}",
                           yref=f"y{col if col>1 else ''}", text="sign wins ↑",
                           showarrow=False, font=dict(size=9, color="gray"))
        fig.add_annotation(x=0.42, y=0.05, xref=f"x{col if col>1 else ''}",
                           yref=f"y{col if col>1 else ''}", text="intensity wins →",
                           showarrow=False, font=dict(size=9, color="gray"))
    fig.update_layout(
        title=dict(text="<b>Intensity over valence — the same pattern for affect and "
                   "evaluation</b><br><sub>RSA presence (a) vs valence (b) per block; "
                   "dotted line is a=b. Humans sit ABOVE it (sign wins); every text model "
                   "sits ON/BELOW it (intensity wins). Cheerful≈Angry, Wonderful≈Awful — "
                   "distributional twins, sign is the residual.</sub>", x=0.02),
        width=1020, height=540, font=dict(family="Helvetica, Arial"), plot_bgcolor="white")
    out = "results/adjectives/intensity_vs_valence.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1020, height=540, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
