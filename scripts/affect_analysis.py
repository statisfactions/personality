#!/usr/bin/env python3
"""Affect-center projection test + model-vs-human affect heatmap (pers framing).

The cohort fails the human positive/negative-affect opposition (Cheerful-Sad
-0.43 human vs ~+0.2 model). Hypothesis (rgb): the merge is an "affect-presence"
axis — the common component of all emotion words (pos AND neg) — that the model
over-weights. If so, projecting out the affect CENTROID should separate the
residual by valence and recover the human structure better, at least in the
affect region.

Per model (pers framing, center+top1 denoise at 2/3 depth):
  M      = cosine matrix
  M_proj = cosine after also projecting out unit(mean of AFFECT_DIR adjectives)
Compare each (cohort-mean) to human, globally (all 523) and in the affect block.

Also renders human | model affect-region heatmaps (shared hierarchical order).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/affect_analysis.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer

ENCODERS = ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]

ACTS = "results/adjectives/acts"
HUMAN = "results/adjectives/escs_525pda_corr_raw.json"
OUTDIR = Path("results/adjectives")
LAYER_FRAC = 2 / 3
FRAMING = "pers"

# affect-presence direction: pos + neg emotion words (common component = arousal)
AFFECT_DIR = ["Cheerful", "Happy", "Joyful", "Glad", "Pleased", "Excited",
              "Sad", "Unhappy", "Angry", "Irritated", "Moody", "Nervous",
              "Tense", "Crabby", "Grumpy", "Upset", "Depressed", "Troubled"]
# affect-region block for the heatmap + local recovery
WANT = ["Cheerful", "Joyful", "Happy", "Glad", "Good-humored", "Good-natured",
        "Positive", "Pleased", "Laughing", "Angry", "Sad", "Unhappy",
        "Irritated", "Moody", "Crabby", "Negative", "Nervous", "Tense",
        "Sympathetic", "Kind", "Warm", "Warm-hearted", "Kind-hearted"]


def denoise(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc - (Xc @ Vt[:1].T) @ Vt[:1]


def norm_rows(D):
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def upper_r(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def main():
    human = json.load(open(HUMAN))
    H = np.array(human["correlation_matrix"], float)
    labels = human["labels"]
    idx = {l: i for i, l in enumerate(labels)}
    aff_idx = [idx[w] for w in AFFECT_DIR if w in idx]
    want_idx = [idx[w] for w in WANT if w in idx]
    want_lab = [w for w in WANT if w in idx]

    Ms, Mps = [], []
    for p in sorted(glob.glob(f"{ACTS}/*__{FRAMING}.pt")):
        b = torch.load(p, weights_only=False)
        acts = b["acts"].astype(np.float32)
        L = round((acts.shape[1] - 1) * LAYER_FRAC)
        adj = b["adjectives"]
        reorder = [adj.index(l) for l in labels]
        D = denoise(acts[:, L, :])[reorder]            # aligned to human order
        a = D[aff_idx].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        Dp = D - np.outer(D @ a, a)
        Ms.append(norm_rows(D) @ norm_rows(D).T)
        Mps.append(norm_rows(Dp) @ norm_rows(Dp).T)
    M = np.mean(Ms, 0); Mp = np.mean(Mps, 0)

    # ---- encoders (bge + mpnet), centered cosine (raw, no PC removal per §3.9) ----
    Es, Eps = [], []
    texts = [f"someone who is {l.lower()}" for l in labels]
    for repo in ENCODERS:
        emb = np.asarray(SentenceTransformer(repo).encode(
            texts, show_progress_bar=False, normalize_embeddings=True), float)
        De = emb - emb.mean(0)
        a = De[aff_idx].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        Dep = De - np.outer(De @ a, a)
        Es.append(norm_rows(De) @ norm_rows(De).T)
        Eps.append(norm_rows(Dep) @ norm_rows(Dep).T)
    E = np.mean(Es, 0); Ep = np.mean(Eps, 0)

    def report(name, A):
        full = upper_r(A, H)
        wa = A[np.ix_(want_idx, want_idx)]; hw = H[np.ix_(want_idx, want_idx)]
        block = upper_r(wa, hw)
        ca = A[idx["Cheerful"], idx["Angry"]]; cs = A[idx["Cheerful"], idx["Sad"]]
        print(f"  {name:<28} full r={full:+.3f}  affect-block r={block:+.3f}  "
              f"Ch-Ang={ca:+.2f}  Ch-Sad={cs:+.2f}")
        return full, block

    print(f"=== affect-center projection (pers, cohort-mean of {len(Ms)} models) ===")
    print(f"  human affect-block internal coherence (ref): Ch-Ang={H[idx['Cheerful'],idx['Angry']]:+.2f}  "
          f"Ch-Sad={H[idx['Cheerful'],idx['Sad']]:+.2f}")
    report("LLM baseline", M)
    report("LLM + affect-center proj", Mp)
    report("encoder baseline", E)
    report("encoder + affect-center proj", Ep)

    # ---- heatmap: human | model (baseline), shared hierarchical order on human ----
    Hw = H[np.ix_(want_idx, want_idx)]
    d = 1 - Hw; np.fill_diagonal(d, 0); d = (d + d.T) / 2
    order = leaves_list(linkage(squareform(d, checks=False), method="average"))
    names = [want_lab[i] for i in order]
    def blk(A):
        return A[np.ix_(want_idx, want_idx)][np.ix_(order, order)]
    Hs = Hw[np.ix_(order, order)]

    # 2x2: [Human | LLM raw] / [Encoders | LLM affect-projected]
    panels = [(1, 1, "Human 525-PDA (correlation)", Hs),
              (1, 2, "LLM cohort (cosine, pers)", blk(M)),
              (2, 1, "Encoders bge+mpnet (cosine)", blk(E)),
              (2, 2, "LLM — affect-center projected out", blk(Mp))]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.11,
                        horizontal_spacing=0.09,
                        subplot_titles=[p[2] for p in panels])
    for r, c, _, Z in panels:
        last = (r == 2 and c == 2)
        fig.add_trace(go.Heatmap(z=Z, x=names, y=names, zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=False,
            showscale=last, colorbar=dict(title="r/cos", thickness=12)),
            row=r, col=c)
        fig.update_yaxes(autorange="reversed", row=r, col=c, tickfont=dict(size=7))
        fig.update_xaxes(tickangle=-60, tickfont=dict(size=7), row=r, col=c)
    fig.update_layout(
        title=dict(text="<b>Affect region: human opposition vs LLM & encoder merge</b>"
                   "<br><sub>Same adjectives + order. Human: pos/neg-affect blocks "
                   "RED (anti-correlated). LLM &amp; encoders: cross-block red gone "
                   "(the merge). LLM affect-projected: opposition returns — valence "
                   "masked, not absent. Encoder merging ⇒ the affect axis is "
                   "lexical, not LLM-specific.</sub>", x=0.02),
        width=1080, height=1040, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = OUTDIR / "affect_human_vs_model.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.with_suffix(".png"), width=1080, height=1040, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
