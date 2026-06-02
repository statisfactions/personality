#!/usr/bin/env python3
"""Training-stage trajectory of the read/write gap (W15 §3).

Reusable across a stage ladder: Qwen (Base -> Instruct) now; OLMo-2
(Base -> SFT -> DPO -> Instruct/RLVR) when downloaded. For each stage checkpoint
(all probed BARE so prompt format is constant), report three quantities on the
26-adjective subset:
  - representation merge: pos-eval x neg-eval cosine, z within the matrix (high =
    Wonderful~Awful merged in the resting geometry). Cosine, so entropy-immune.
  - behavioral split (decisiveness): raw within-good minus pos x neg gap on the 1-7
    rating scale (high = good judged decisively unlike bad). Raw, so NOT confounded
    by how peaked the rating distribution is.
  - confidence: mean digit-entropy of the judgments (low = decisive).
Separating these is the point: z-scored "split"/"collapse" magnitudes are confounded
by entropy across stages of differing decisiveness, so we read merge via cosine,
split via the raw gap, and confidence via entropy.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/training_stage_compare.py \
           --models Qwen7Base,Qwen7 --out qwen
"""
import argparse
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hf_logprobs import MODELS
from adjective_geom import model_matrix
from adjective_introspection import ADJ, GROUPS, block_mean, offdiag, pc1frac

POS = ["pos_eval", "warm", "intellect"]
GI = {g: [ADJ.index(w) for w in GROUPS[g]] for g in GROUPS}
STAGE_LABEL = {"Qwen7Base": "Base", "Qwen7": "Instruct",
               "Olmo2Base": "Base", "Olmo2SFT": "SFT", "Olmo2DPO": "DPO",
               "Olmo2Inst": "Instruct\n(RLVR)", "ZephyrSFT": "SFT", "ZephyrDPO": "DPO"}


def pn_z(A):
    o = offdiag(A); m, s = np.nanmean(o), np.nanstd(o) + 1e-9
    return (block_mean(A, "pos_eval", "neg_eval") - m) / s


def raw_gap(A):
    wp = np.mean([np.nanmean(A[np.ix_(GI[a], GI[b])])
                  for i, a in enumerate(POS) for b in POS[i:]])
    return float(wp - block_mean(A, "pos_eval", "neg_eval"))


def beh(short, mode):
    p = f"results/adjectives/introspect{'_tom' if mode == 'tom' else ''}/{short}_bare.json"
    try:
        d = json.load(open(p))
    except FileNotFoundError:
        return None
    return np.array(d["behavioral"]), d.get("mean_entropy", float("nan"))


def repr_merge(short, flab, idx):
    import os
    p = f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers.pt"
    if not os.path.exists(p):
        return None
    _, C, _, _ = model_matrix(p, np.array(flab))
    R = C[np.ix_(idx, idx)]
    return pn_z(R), pc1frac(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen7Base,Qwen7")
    ap.add_argument("--out", default="qwen")
    args = ap.parse_args()
    models = args.models.split(",")
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    flab = list(human["labels"]); idx = [flab.index(w) for w in ADJ]
    xs = [STAGE_LABEL.get(m, m) for m in models]

    rows = []
    print(f"{'stage':<14} {'repr-merge z':>12} {'gap sem/tom':>12} {'entropy sem/tom':>16}")
    for m in models:
        rm = repr_merge(m, flab, idx)
        sem, tom = beh(m, "semantic"), beh(m, "tom")
        r = {"x": STAGE_LABEL.get(m, m),
             "repr": rm[0] if rm else np.nan,
             "gap_sem": raw_gap(sem[0]) if sem else np.nan,
             "gap_tom": raw_gap(tom[0]) if tom else np.nan,
             "ent_sem": sem[1] if sem else np.nan, "ent_tom": tom[1] if tom else np.nan}
        rows.append(r)
        print(f"{r['x']:<14} {r['repr']:>12.2f} {r['gap_sem']:>5.2f}/{r['gap_tom']:<5.2f} "
              f"   {r['ent_sem']:>6.2f}/{r['ent_tom']:<6.2f}")

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09, subplot_titles=(
        "representation: Wonderful≈Awful merge", "judgment: good/bad split (raw gap)",
        "confidence: digit-entropy"))
    fig.add_trace(go.Scatter(x=xs, y=[r["repr"] for r in rows], mode="lines+markers",
        line=dict(color="#d62728", width=3), showlegend=False), row=1, col=1)
    for key, name, dash in (("sem", "semantic", "solid"), ("tom", "ToM", "dash")):
        fig.add_trace(go.Scatter(x=xs, y=[r[f"gap_{key}"] for r in rows],
            mode="lines+markers", name=name, line=dict(color="#1f77b4", width=2, dash=dash)),
            row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"ent_{key}"] for r in rows],
            mode="lines+markers", line=dict(color="#9467bd", width=2, dash=dash),
            showlegend=False), row=1, col=3)
    fig.update_yaxes(title_text="pos×neg cosine (z)", row=1, col=1)
    fig.update_yaxes(title_text="within-good − pos×neg (1–7)", range=[0, 3], row=1, col=2)
    fig.update_yaxes(title_text="mean entropy (nats)", range=[0, 1.6], row=1, col=3)
    fig.update_layout(
        title=dict(text="<b>Across training stages: the read/write gap is there from the "
            "start; tuning sharpens confidence</b><br><sub>All probed bare (format "
            "constant). Representation keeps the merge; the good/bad judgment gap is "
            "flat (split is pretrained); entropy collapses (tuning makes the same "
            "judgment decisive). Qwen has only two points — single-pair conclusions "
            "tend to get revised; OLMo's full ladder is the real test.</sub>", x=0.01),
        width=1240, height=480, plot_bgcolor="white", font=dict(family="Helvetica, Arial"),
        legend=dict(x=0.34, y=1.0, font=dict(size=9)))
    out = f"results/adjectives/training_stage_{args.out}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1240, height=480, scale=2)
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
