#!/usr/bin/env python3
"""How far back does the Wonderful≈Awful merge stretch? (W16 §1)

W15 found LLMs *represent* good/bad eval-antonyms as merged (associative twins in
the residual stream) while *judging* them opposite. The reading-group question:
the merge is in current transformer encoders too — how much further back? Is it a
transformer thing, a neural thing, or a property of the distributional hypothesis
itself?

This walks a regress of model classes, newest -> oldest, computing the SAME merge
statistic on the same 26 pole-spanning adjectives:

  LLM-judge      Qwen2.5-7B behavioral rating (the WRITE side; W15 §1)   <- splits
  LLM-repr       Qwen2.5-7B residual-stream cosine (the READ side)       <- merges
  bge-large      transformer encoder, retrieval-contrastive (2023)
  mpnet-base     transformer encoder, masked-LM + NLI (2021)
  glove.840B     STATIC count-predict vectors, 840B-token web (2014)     <- pre-transformer
  glove.6B       STATIC count-predict vectors, 6B-token wiki (2014)
  komninos       STATIC dependency-window word2vec (2016)
  human          525-PDA self-report correlations (the antonym truth)    <- splits

Statistic, z-scored WITHIN each matrix (the claim is purely relational — "are the
eval-antonyms positioned like synonyms, on this model's own scale"):
  synonym-z  : within-pole eval similarity (Wonderful~Amazing, Awful~Terrible)
  antonym-z  : pos_eval x neg_eval  (Wonderful~Awful)  -- THE merge test
  crossdim-z : eval x intellect/distress (different content dimension) -- a floor

Merge  => antonym-z > 0 and ~ synonym-z (antonyms sit where synonyms sit).
Split  => antonym-z < 0 (antonyms pushed apart), as in human self-report.

For single words the sentence-transformer average-word-embedding IS the static
word vector (glove/komninos) or the encoder's word encoding (bge/mpnet). Bare
word, lowercased — the canonical word-similarity setup, and the honest form for
GloVe (a carrier would average carrier tokens into the static vector).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_regress.py
"""
import json
import os

import numpy as np
import plotly.graph_objects as go

from adjective_introspection import ADJ, block_mean, offdiag
from adjective_geom import model_matrix
from hf_logprobs import MODELS

OUT = "results/adjectives/regress"

# encoder / static-embedding strata, newest -> oldest within the transformer and
# static blocks. sentence-transformers fetches+caches each on first use.
ENCODERS = [
    ("bge-large", "BAAI/bge-large-en-v1.5", "encoder"),
    ("mpnet-base", "sentence-transformers/all-mpnet-base-v2", "encoder"),
    ("glove.840B", "sentence-transformers/average_word_embeddings_glove.840B.300d", "static"),
    ("glove.6B", "sentence-transformers/average_word_embeddings_glove.6B.300d", "static"),
    ("komninos", "sentence-transformers/average_word_embeddings_komninos", "static"),
]

# which non-eval groups count as a different content DIMENSION for the floor
CROSSDIM = ["intellect", "distress"]


def zmat(A):
    """z-score the off-diagonal of a similarity matrix in place-ish (returns copy)."""
    o = offdiag(A)
    m, s = np.nanmean(o), np.nanstd(o) + 1e-9
    return (A - m) / s


def blocks(A):
    """(synonym-z, antonym-z, crossdim-z) on a single 26x26 similarity matrix."""
    Z = zmat(A)
    syn = np.mean([block_mean(Z, "pos_eval", "pos_eval"),
                   block_mean(Z, "neg_eval", "neg_eval")])
    ant = block_mean(Z, "pos_eval", "neg_eval")
    cross = np.mean([block_mean(Z, p, c)
                     for p in ("pos_eval", "neg_eval") for c in CROSSDIM])
    return float(syn), float(ant), float(cross)


def encoder_matrix(repo):
    """26x26 cosine matrix of bare-word embeddings from a sentence-transformer."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(repo)
    E = np.asarray(m.encode([w.lower() for w in ADJ], normalize_embeddings=True),
                   dtype=np.float64)
    return E @ E.T


def human_matrix():
    H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    lab = list(H["labels"]); C = np.array(H["correlation_matrix"], float)
    missing = [w for w in ADJ if w not in lab]
    if missing:
        raise SystemExit(f"human set missing {missing}")
    idx = [lab.index(w) for w in ADJ]
    return C[np.ix_(idx, idx)]


def llm_repr_matrix(short="Qwen7"):
    """Residual-stream cosine on the 26 adjectives (bare extraction if present)."""
    base = f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers"
    path = f"{base}_bare.pt" if os.path.exists(f"{base}_bare.pt") else f"{base}.pt"
    if not os.path.exists(path):
        return None
    full_labels = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]
    _, C, _, _ = model_matrix(path, np.array(full_labels))
    idx = [full_labels.index(w) for w in ADJ]
    return C[np.ix_(idx, idx)]


def llm_judge_matrix(short="Qwen7"):
    p = f"results/adjectives/introspect/{short}.json"
    if not os.path.exists(p):
        return None
    return np.array(json.load(open(p))["behavioral"], float)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []   # (label, kind, syn, ant, cross)

    # --- write side (judgment) and human: the two that should SPLIT ---
    judge = llm_judge_matrix()
    if judge is not None:
        s, a, c = blocks(judge); rows.append(("LLM-judge", "behavior", s, a, c))

    # --- read side: LLM representation ---
    rep = llm_repr_matrix()
    if rep is not None:
        s, a, c = blocks(rep); rows.append(("LLM-repr", "llm", s, a, c))

    # --- encoders + static embeddings ---
    for name, repo, kind in ENCODERS:
        try:
            M = encoder_matrix(repo)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        s, a, c = blocks(M); rows.append((name, kind, s, a, c))
        print(f"  {name:<12} syn-z {s:+.2f}  antonym-z {a:+.2f}  crossdim-z {c:+.2f}")

    # --- human reference ---
    s, a, c = blocks(human_matrix()); rows.append(("human", "human", s, a, c))

    # console
    print(f"\n{'stratum':<12}{'kind':<10}{'synonym-z':>10}{'antonym-z':>10}{'crossdim-z':>11}"
          "   merge?")
    for lab, kind, s, a, c in rows:
        print(f"{lab:<12}{kind:<10}{s:>10.2f}{a:>10.2f}{c:>11.2f}"
              f"   {'MERGED' if a > 0 else 'split'}")

    json.dump([{"stratum": l, "kind": k, "synonym_z": s, "antonym_z": a,
                "crossdim_z": c} for l, k, s, a, c in rows],
              open(f"{OUT}/regress.json", "w"), indent=2)

    # --- figure: one grouped-bar panel, ancient(static)->modern(LLM)->judge/human ---
    order = ["komninos", "glove.6B", "glove.840B", "mpnet-base", "bge-large",
             "LLM-repr", "LLM-judge", "human"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    xs = [r[0] for r in rows]
    fig = go.Figure()
    for k, (nm, col) in enumerate([("synonym_z", "#2ca02c"),
                                   ("antonym_z", "#d62728"),
                                   ("crossdim_z", "#7f7f7f")]):
        yy = [r[2 + k] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=yy, name={"synonym_z": "synonyms (within-pole)",
            "antonym_z": "eval ANTONYMS (Wonderful~Awful)",
            "crossdim_z": "different dimension (floor)"}[nm], marker_color=col))
    fig.add_hline(y=0, line_color="black", line_width=1)
    # bracket the two architectural eras
    fig.add_vrect(x0=-0.5, x1=2.5, fillcolor="#fff7e6", opacity=0.5, line_width=0,
                  annotation_text="static word vectors (pre-transformer, 2014–16)",
                  annotation_position="top left", annotation_font_size=10, layer="below")
    fig.add_vrect(x0=2.5, x1=4.5, fillcolor="#eef5ff", opacity=0.5, line_width=0,
                  annotation_text="transformer encoders", annotation_position="top left",
                  annotation_font_size=10, layer="below")
    fig.update_layout(
        title=dict(text="<b>The good/bad merge is a property of the distributional "
            "hypothesis, not the architecture</b><br><sub>z-scored block similarity on "
            "26 pole-spanning adjectives, each model on its own scale. Red ABOVE zero = "
            "the eval-antonyms (Wonderful~Awful) sit where synonyms sit (green) — merged. "
            "The merge is already complete in 2014 static GloVe vectors, persists through "
            "encoders and the LLM's resting representation, and only the LLM's JUDGMENT "
            "and human self-report push the antonyms apart (red below zero). The write "
            "side is the only thing that escapes the regress.</sub>", x=0.01),
        barmode="group", width=1180, height=560, plot_bgcolor="white",
        font=dict(family="Helvetica, Arial"), legend=dict(x=0.01, y=0.02),
        yaxis_title="block similarity (z within matrix)")
    fig.write_html(f"{OUT}/regress.html", include_plotlyjs="cdn")
    fig.write_image(f"{OUT}/regress.png", width=1180, height=560, scale=2)
    print(f"\nwrote {OUT}/regress.html and .png")


if __name__ == "__main__":
    main()
