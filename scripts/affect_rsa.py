#!/usr/bin/env python3
"""RSA decomposition of the affect block: presence vs valence (W13 §3.11 follow-up).

Instead of projecting out an ad-hoc affect direction (cardinality-biased, and
projection can only lower cosines so it can't reach the human structure), we
DECOMPOSE the affect-block similarity matrix onto two template patterns and read
how much structure each carries:

  similarity(i,j) ~= a*presence(i,j) + b*valence(i,j) + residual
    presence(i,j) = +1 for every affect pair           (a = block elevation)
    valence(i,j)  = +1 same-valence, -1 opposite        (b = pos/neg opposition)

  a = (within + cross)/2   (presence strength)
  b = (within - cross)/2   (valence strength)
  R2 = fraction of the block's off-diagonal variance the two templates explain
       (residual = arousal sub-structure, word-specifics)

Balanced 12 pos / 12 neg, so the templates are composition-robust. Reported for
human (525-PDA), the LLM cohort (pers, center+top1), and bge+mpnet encoders.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/affect_rsa.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

from adjective_geom import model_matrix

HUMAN = "results/adjectives/escs_525pda_corr_raw.json"
POS = ["Cheerful", "Happy", "Joyful", "Glad", "Pleased", "Delightful", "Lively",
       "Optimistic", "Good-humored", "Good-natured", "Positive", "Laughing"]
NEG = ["Sad", "Unhappy", "Angry", "Irritated", "Moody", "Crabby", "Grumpy",
       "Upset", "Depressed", "Troubled", "Tense", "Nervous"]
ENCODERS = ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]


def rsa(C, pi, ni):
    """Decompose affect-block off-diagonal similarities into presence + valence."""
    idxs = pi + ni
    s = np.array([1] * len(pi) + [-1] * len(ni))  # valence sign per word
    pairs, val, sim = [], [], []
    for x in range(len(idxs)):
        for y in range(x + 1, len(idxs)):
            sim.append(C[idxs[x], idxs[y]])
            val.append(s[x] * s[y])
            pairs.append((x, y))
    sim = np.array(sim, float); val = np.array(val, float)
    a = sim.mean()                       # presence (intercept)
    b = ((sim * val).mean()) / (val ** 2).mean()  # valence coefficient
    resid = sim - (a + b * val)
    r2 = 1 - resid.var() / sim.var()
    within = sim[val > 0].mean(); cross = sim[val < 0].mean()
    return dict(presence=a, valence=b, R2=r2, within=within, cross=cross,
                val_share=b ** 2 / (a ** 2 + b ** 2))


def show(name, d):
    print(f"  {name:<26} presence(a)={d['presence']:+.3f}  valence(b)={d['valence']:+.3f}"
          f"  b/(a+b)={d['valence']/(abs(d['presence'])+abs(d['valence'])):.2f}"
          f"  R2={d['R2']:.2f}  [within {d['within']:+.2f}/cross {d['cross']:+.2f}]")


def main():
    human = json.load(open(HUMAN))
    H = np.array(human["correlation_matrix"], float)
    labels = human["labels"]; idx = {l: i for i, l in enumerate(labels)}
    pi = [idx[w] for w in POS if w in idx]; ni = [idx[w] for w in NEG if w in idx]
    print(f"affect block: {len(pi)} pos / {len(ni)} neg adjectives\n")

    print("=== presence vs valence (a = elevation, b = pos/neg opposition) ===")
    pts = []  # (name, rsa-dict, kind)
    hd = rsa(H, pi, ni); show("HUMAN 525-PDA", hd); pts.append(("human", hd, "human"))

    # LLM cohort (pers, adaptive denoise: center; remove top1 only for rogue-PC1)
    PROBES = ("aya", "falcon")
    cohort = []
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        name, C, removed, ipr = model_matrix(p, labels)
        cohort.append((name, rsa(C, pi, ni)))
    for name, d in cohort:
        show(name, d)
        kind = "probe" if any(k in name.lower() for k in PROBES) else "llm"
        pts.append((name, d, kind))
    agg = {k: np.mean([d[k] for _, d in cohort]) for k in cohort[0][1]}
    show(">>> LLM cohort mean", agg)

    # encoders (centered cosine)
    texts = [f"someone who is {l.lower()}" for l in labels]
    for repo in ENCODERS:
        emb = np.asarray(SentenceTransformer(repo).encode(
            texts, show_progress_bar=False, normalize_embeddings=True), float)
        De = emb - emb.mean(0); De /= (np.linalg.norm(De, axis=1, keepdims=True) + 1e-9)
        d = rsa(De @ De.T, pi, ni); nm = repo.split("/")[-1]
        show("ENC " + nm, d); pts.append((nm, d, "enc"))

    # ---- scatter: presence (x) vs valence (y); up-left = human-like ----
    style = {"human": ("star", 22, "#d62728"), "llm": ("circle", 11, "#1f77b4"),
             "probe": ("circle-open", 13, "#1f77b4"), "enc": ("diamond", 13, "#2ca02c")}
    fig = go.Figure()
    # y=x reference (presence == valence)
    fig.add_trace(go.Scatter(x=[0, 0.4], y=[0, 0.4], mode="lines",
        line=dict(dash="dot", color="lightgray"), hoverinfo="skip", showlegend=False))
    for kind in ["llm", "probe", "enc", "human"]:
        sub = [(n, d) for n, d, k in pts if k == kind]
        if not sub:
            continue
        sym, sz, col = style[kind]
        fig.add_trace(go.Scatter(
            x=[d["presence"] for _, d in sub], y=[d["valence"] for _, d in sub],
            mode="markers+text", text=[n for n, _ in sub], textposition="top center",
            textfont=dict(size=8),
            marker=dict(symbol=sym, size=sz, color=col,
                        line=dict(width=1, color="white")),
            name={"llm": "LLM cohort", "probe": "robustness probe",
                  "enc": "encoder", "human": "human (target)"}[kind]))
    fig.add_annotation(x=0.085, y=0.37, text="valence-dominated<br>(human-like)",
                       showarrow=False, font=dict(size=10, color="#d62728"))
    fig.update_layout(
        title=dict(text="<b>Affect-block structure: presence vs valence (RSA)</b>"
                   "<br><sub>a = block elevation (uniform 'all emotional' merge); "
                   "b = pos/neg opposition. Humans live up-left (valence ≫ presence); "
                   "every text model collapses toward the diagonal/right.</sub>", x=0.02),
        xaxis=dict(title="presence  a  (uniform merge →)", zeroline=True),
        yaxis=dict(title="valence  b  (pos/neg opposition ↑)", zeroline=True),
        width=820, height=720, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = Path("results/adjectives/affect_presence_vs_valence.html")
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.with_suffix(".png"), width=820, height=720, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
