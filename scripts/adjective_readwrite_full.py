#!/usr/bin/env python3
"""Full 523x523 read-vs-write divergence, reported in PERCENTILES (W16 §1 cont.)

We have, at full adjective resolution, the representation (READ) cosine geometry
and the judgment (WRITE) similarity geometry. Earlier we z-scored each — but the
judgment EV distribution is floored at 1 and heavily right-skewed (most pairs
"completely different"), so z misleads (the antonym block sits at the absolute
floor yet z is only ~-0.3 because the SD is inflated by the right tail). rgb's
fix: report the CUMULATIVE — each pair's midrank percentile within its matrix's
off-diagonal distribution. Distribution-free, and says what we mean: "the antonym
pairs are at the Pth percentile of representation similarity but the Qth of
judgment similarity." Midrank ties so the floor cluster is handled honestly.

D = pctR - pctJ (percentile points): >0 rep ranks the pair higher than judgment
does (merged-in-read); <0 judgment ranks it higher (merged-in-write).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_readwrite_full.py --model Qwen7
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata
import plotly.graph_objects as go

from adjective_geom import model_matrix
from adjective_introspection import GROUPS
from hf_logprobs import MODELS

LABELS = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]


def pmat(A):
    """Symmetric percentile matrix in [0,1] over the off-diagonal (midrank ties).
    Diagonal = nan. Distribution-free; invariant to any monotone rescaling of A."""
    iu = np.triu_indices_from(A, 1)
    o = A[iu].astype(float)
    valid = ~np.isnan(o)
    pct = np.full(o.shape, np.nan)
    pct[valid] = rankdata(o[valid], method="average") / valid.sum()
    P = np.full_like(A, np.nan, dtype=float)
    P[iu] = pct
    P[(iu[1], iu[0])] = pct
    return P


def top_eig(M):
    w, V = np.linalg.eigh(M)
    o = np.argsort(np.abs(w))[::-1]
    return w[o], V[:, o]


def repr_matrix(short):
    base = f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers"
    p = f"{base}_bare.pt" if os.path.exists(f"{base}_bare.pt") else f"{base}.pt"
    _, C, _, _ = model_matrix(p, np.array(LABELS))
    return C


def judge_matrix(short):
    z = np.load(f"results/adjectives/introspect_full/{short}.npz", allow_pickle=True)
    adj = list(z["adjectives"]); B = z["B"]
    idx = [adj.index(w) for w in LABELS]
    return B[np.ix_(idx, idx)], adj


def block_pct(P, ga, gb):
    ia = [LABELS.index(w) for w in GROUPS[ga]]; ib = [LABELS.index(w) for w in GROUPS[gb]]
    return 100 * float(np.nanmean(P[np.ix_(ia, ib)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    args = ap.parse_args()
    short = args.model

    R = repr_matrix(short)
    Jraw, _ = judge_matrix(short)
    n = len(LABELS)
    iu = np.triu_indices(n, 1)

    # floor diagnostic: fraction of judged pairs at the bottom of the scale
    Jo = Jraw[iu]; floor_frac = float(np.mean(Jo <= np.nanmin(Jo) + 0.05))

    PR, PJ = pmat(R), pmat(Jraw)
    # rank (Spearman) agreement
    spear = float(np.corrcoef(PR[iu], PJ[iu])[0, 1])

    # antonym block percentiles — the headline, distribution-free
    aR = block_pct(PR, "pos_eval", "neg_eval")
    aJ = block_pct(PJ, "pos_eval", "neg_eval")
    synR = np.mean([block_pct(PR, "pos_eval", "pos_eval"), block_pct(PR, "neg_eval", "neg_eval")])
    synJ = np.mean([block_pct(PJ, "pos_eval", "pos_eval"), block_pct(PJ, "neg_eval", "neg_eval")])

    # warp structure on centered percentiles
    D = np.nan_to_num(PR - PJ, nan=0.0); np.fill_diagonal(D, 0.0)
    Dc = D.copy()
    wD, VD = top_eig(Dc)
    shear_frac = float(abs(wD[0]) / np.sum(np.abs(wD)))
    Rc = np.nan_to_num(PR - 0.5, nan=0.0); np.fill_diagonal(Rc, 0.0)
    _, VR = top_eig(Rc)
    eval_axis = VR[:, 0]
    pos = [LABELS.index(w) for w in GROUPS["pos_eval"]]
    if np.mean(eval_axis[pos]) < 0:
        eval_axis = -eval_axis
    align = abs(float(VD[:, 0] @ eval_axis) / (np.linalg.norm(VD[:, 0]) * np.linalg.norm(eval_axis)))

    print(f"\n===== {short}: full 523x523 read vs write (PERCENTILES) =====")
    print(f"judgment floor cluster (pairs at scale bottom): {100*floor_frac:.0f}% of all pairs")
    print(f"Spearman read/write agreement (off-diag):       {spear:+.3f}")
    print(f"\nANTONYM block (Wonderful/Awful... pos x neg eval):")
    print(f"  representation percentile: {aR:5.1f}   (synonyms {synR:.1f})")
    print(f"  judgment       percentile: {aJ:5.1f}   (synonyms {synJ:.1f})")
    print(f"  read/write split = {aR - aJ:+.1f} percentile points")
    print(f"\nwarp on percentile-diff D: top-mode {shear_frac:.3f}  eval-align {align:.3f}")

    Du = D[iu]; order = np.argsort(Du)
    def lab(k): return LABELS[iu[0][k]], LABELS[iu[1][k]]
    print(f"\nMERGED-IN-READ (rep pct >> judge pct) — top:")
    for k in order[::-1][:10]:
        a, b = lab(k); print(f"  {a:>15} ~ {b:<15} repP {100*PR[iu[0][k],iu[1][k]]:4.0f} judgeP {100*PJ[iu[0][k],iu[1][k]]:4.0f}")
    print(f"\nMERGED-IN-WRITE (judge pct >> rep pct) — top:")
    for k in order[:10]:
        a, b = lab(k); print(f"  {a:>15} ~ {b:<15} repP {100*PR[iu[0][k],iu[1][k]]:4.0f} judgeP {100*PJ[iu[0][k],iu[1][k]]:4.0f}")

    json.dump({"model": short, "floor_frac": floor_frac, "spearman_agreement": spear,
               "antonym_pct_read": aR, "antonym_pct_write": aJ,
               "antonym_split_pctpts": aR - aJ, "warp_top_mode": shear_frac,
               "warp_eval_alignment": align},
              open(f"results/adjectives/introspect_full/readwrite_{short}.json", "w"),
              indent=2, default=float)

    # figure: percentile scatter (distribution-free axes)
    val = eval_axis / (np.linalg.norm(eval_axis) + 1e-9)
    vp = np.outer(val, val)[iu]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=100 * PR[iu], y=100 * PJ[iu], mode="markers",
        marker=dict(size=3, color=vp, colorscale="RdBu", cmid=0, opacity=0.5,
                    colorbar=dict(title="valence<br>product")),
        text=[f"{lab(k)[0]}~{lab(k)[1]}" for k in range(len(Du))], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color="black", dash="dot"), showlegend=False))
    fig.update_layout(
        title=dict(text=f"<b>{short}: read vs write across all 523 adjectives "
            f"(percentiles)</b><br><sub>each point a pair: x = representation-similarity "
            f"percentile, y = judgment-similarity percentile. Diagonal = agree "
            f"(Spearman {spear:.2f}). Antonyms (Wonderful~Awful): read {aR:.0f}th pct vs "
            f"write {aJ:.0f}th. Horizontal-right plume = rep over-merges (lexical twins); "
            f"vertical-up = judgment over-merges (true synonyms). {100*floor_frac:.0f}% of "
            f"pairs sit at the judgment floor. Color = valence product.</sub>", x=0.01),
        xaxis_title="representation similarity (percentile)",
        yaxis_title="judgment similarity (percentile)",
        width=900, height=820, plot_bgcolor="white", font=dict(family="Helvetica, Arial"))
    fig.write_html(f"results/adjectives/introspect_full/readwrite_{short}.html", include_plotlyjs="cdn")
    fig.write_image(f"results/adjectives/introspect_full/readwrite_{short}.png", width=900, height=820, scale=2)
    print(f"\nwrote readwrite_{short}.png/.html/.json")


if __name__ == "__main__":
    main()
