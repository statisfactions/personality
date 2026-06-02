#!/usr/bin/env python3
"""Visualize the evaluative collapse sharpening — the "E–C join" (W15, exploratory).

The assistant shape (Big-Five E–C r=0.93) is: all desirable traits fuse into one
"good" bundle, all undesirable into one "bad" bundle, and the two pull apart. We
watch that sharpen across the three readouts on the same adjectives:
  representation (resting cosine) → semantic judgment → ToM (persona) judgment.

For each, the 26 adjectives are grouped into 3 positive groups (positive-evaluation,
warmth, intellect) and 3 negative (negative-evaluation, antagonism, distress) + a
neutral pair, and we show the group×group mean similarity (z-scored within each
readout) as a heatmap, valence-ordered so the 2-superblock structure is visible.
Left→right: the within-good block fuses (the join), the good×bad block goes cold
(the split), and PC1 (single-axis dominance) climbs.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_collapse.py \
           --models Qwen7,Gemma12
"""
import argparse
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hf_logprobs import MODELS
from adjective_geom import model_matrix
from adjective_introspection import ADJ, GROUPS, CACHE, offdiag, pc1frac

# valence-ordered groups: positive bundle | negative bundle | neutral
GORDER = ["pos_eval", "warm", "intellect", "neg_eval", "antagonism", "distress", "neutral"]
GLABEL = {"pos_eval": "pos-eval", "warm": "warmth", "intellect": "intellect",
          "neg_eval": "neg-eval", "antagonism": "antagon.", "distress": "distress",
          "neutral": "neutral"}
POS = ["pos_eval", "warm", "intellect"]
GIDX = {g: [ADJ.index(w) for w in GROUPS[g]] for g in GORDER}


def zmat(A):
    o = offdiag(A); m, s = np.nanmean(o), np.nanstd(o) + 1e-9
    return (A - m) / s


def group_mat(A):
    """7×7 mean z-scored similarity between groups (self-pairs excluded on diagonal)."""
    Z = zmat(A); G = np.zeros((7, 7))
    for a, ga in enumerate(GORDER):
        for b, gb in enumerate(GORDER):
            sub = Z[np.ix_(GIDX[ga], GIDX[gb])]
            if a == b:
                iu = np.triu_indices_from(sub, 1)
                G[a, b] = np.nanmean(sub[iu]) if len(iu[0]) else np.nan
            else:
                G[a, b] = np.nanmean(sub)
    return G


def join_split(A):
    Z = zmat(A)
    pos_pairs = [(GIDX[a], GIDX[b]) for i, a in enumerate(POS) for b in POS[i + 1:]]
    join = np.mean([np.nanmean(Z[np.ix_(ia, ib)]) for ia, ib in pos_pairs])
    split = np.nanmean(Z[np.ix_(GIDX["pos_eval"], GIDX["neg_eval"])])
    return float(join), float(split)


def corners(short, flab, idx):
    _, C, _, _ = model_matrix(
        f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers.pt", np.array(flab))
    R = C[np.ix_(idx, idx)]
    sem = np.array(json.load(open(f"{CACHE['semantic']}/{short}.json"))["behavioral"])
    tom = np.array(json.load(open(f"{CACHE['tom']}/{short}.json"))["behavioral"])
    return [("representation", R), ("semantic judgment", sem), ("ToM judgment", tom)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen7,Gemma12")
    args = ap.parse_args()
    models = args.models.split(",")
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    flab = list(human["labels"]); idx = [flab.index(w) for w in ADJ]

    ncol = 3
    labels = [GLABEL[g] for g in GORDER]
    # precompute every panel so metrics can go into the titles (avoids per-panel
    # annotations/shapes that trip kaleido)
    panels, titles, printout = {}, [], []
    for r, short in enumerate(models, 1):
        for c, (name, A) in enumerate(corners(short, flab, idx), 1):
            G = group_mat(A); jn, sp = join_split(A); p1 = pc1frac(A)
            panels[(r, c)] = G
            titles.append(f"{short} · {name}<br>join {jn:+.2f}  split {sp:+.2f}  PC1 {p1:.2f}")
            printout.append(f"  {short:<8} {name:<16} join={jn:+.2f} split={sp:+.2f} pc1={p1:.2f}")
    fig = make_subplots(rows=len(models), cols=ncol, horizontal_spacing=0.08,
                        vertical_spacing=0.2, subplot_titles=titles)
    fig.update_annotations(font_size=10)
    ylab = labels[::-1]                              # put the good bundle at the top
    cscale = [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]]   # blue(low)->red(high)
    for (r, c), G in panels.items():
        fig.add_trace(go.Heatmap(
            z=G[::-1], x=labels, y=ylab, zmin=-1.4, zmax=1.4, colorscale=cscale,
            showscale=(r == 1 and c == ncol)), row=r, col=c)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    fig.update_layout(
        title=dict(text="<b>The evaluative collapse sharpens: representation → "
            "judgment → person-reasoning</b><br><sub>Group×group similarity "
            "(z within readout), valence-ordered. The upper-left 3×3 is the GOOD "
            "bundle (pos-eval+warmth+intellect = the E–C join), lower-middle 3×3 the "
            "BAD bundle. Left→right the good bundle fuses (join↑), the good×bad "
            "corners go cold (split↓), and one axis takes over (PC1↑). Representation "
            "still half-merges good/bad; person-reasoning slams them apart.</sub>",
            x=0.01),
        width=1280, height=360 * len(models) + 120, plot_bgcolor="white",
        font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/evaluative_collapse.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1280, height=360 * len(models) + 120, scale=2)
    print(f"wrote {out} and .png")
    print("\n".join(printout))


if __name__ == "__main__":
    main()
