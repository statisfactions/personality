#!/usr/bin/env python3
"""Pretty side-by-side: human vs model personality covariance (W16 §5.8 viz).

Restricts to the strongest-loading adjectives on the top human PCs (readable
subset), orders them by the HUMAN cluster structure, and shows the human 525-PDA
correlations next to the model's prevalence-corrected tom_likely geometry under the
SAME ordering — so matching block structure is visible. Both panels z-scored
off-diagonal for comparable colorscales.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_human_match_viz.py --model Qwen7
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from adjective_human_match import repr_matrix     # adaptive_denoise cosine, LABELS order

H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
LABELS = list(H["labels"])
HUMAN = np.nan_to_num(np.array(H["correlation_matrix"], float))


def dc_off(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    m = np.nanmean(A, 1); g = np.nanmean(A)
    return A - m[:, None] - m[None, :] + g


def z_off(M):
    A = M.copy().astype(float); np.fill_diagonal(A, np.nan)
    iu = np.triu_indices_from(A, 1); o = A[iu]
    return (A - np.nanmean(o)) / (np.nanstd(o) + 1e-9)


def judge(short):
    for suf in ("_tom_likely_dir", "_tom_likely"):
        p = f"results/adjectives/introspect_full/{short}{suf}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); adj = list(z["adjectives"])
            B = z["B"].astype(float); ri = [adj.index(w) for w in LABELS]
            return (B[np.ix_(ri, ri)] + B[np.ix_(ri, ri)].T) / 2
    raise SystemExit(f"no tom_likely matrix for {short}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--n-pc", type=int, default=6); ap.add_argument("--per-pole", type=int, default=4)
    args = ap.parse_args()

    # top-loading words on the leading human PCs -> readable subset
    w, V = np.linalg.eigh(HUMAN); pcs = np.argsort(w)[::-1]
    sel = []
    for k in range(args.n_pc):
        v = V[:, pcs[k]]; o = np.argsort(v)
        sel += list(o[:args.per_pole]) + list(o[-args.per_pole:])
    sel = list(dict.fromkeys(sel))

    if args.model == "cohort":          # average all COMPLETED judges (double-centered)
        import glob
        names = []
        for p in sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz")):
            z = np.load(p, allow_pickle=True)
            if int(z["done"]) >= int(z["n_pairs"]):     # skip in-progress checkpoints
                names.append(os.path.basename(p).replace("_tom_likely_dir.npz", ""))
        Wdc = np.nanmean([dc_off(judge(n)) for n in names], 0)        # write (judgment)
        reprs = [repr_matrix(n) for n in names]
        rnames = [n for n, R in zip(names, reprs) if R is not None]
        Rdc = np.nanmean([dc_off(R) for R in reprs if R is not None], 0)  # read (repr)
        wlabel = f"cohort-mean ({len(names)}-judge)"
        rlabel = f"cohort-mean ({len(rnames)}-model)"
    else:
        Wdc = dc_off(judge(args.model)); wlabel = args.model
        R = repr_matrix(args.model)
        Rdc = dc_off(R) if R is not None else None
        rlabel = args.model

    Hs = HUMAN[np.ix_(sel, sel)]
    # cluster on the human subset (shared ordering for all panels)
    d = 1 - Hs; np.fill_diagonal(d, 0.0); d = (d + d.T) / 2
    leaf = leaves_list(linkage(squareform(d, checks=False), "average"))
    labs = [LABELS[sel[i]] for i in leaf]
    iu = np.triu_indices(len(leaf), 1)

    def disp(Mfull):
        return z_off(Mfull[np.ix_(sel, sel)])[np.ix_(leaf, leaf)]

    Hz = disp(HUMAN)
    Wz = disp(Wdc)
    Rz = disp(Rdc) if Rdc is not None else None
    rW = np.corrcoef(Hz[iu], Wz[iu])[0, 1]
    rR = np.corrcoef(Hz[iu], Rz[iu])[0, 1] if Rz is not None else float("nan")

    # three panels: HUMAN | representation (read) | tom_likely judgment (write)
    panels = [("HUMAN — 525-PDA self-report correlations", Hz, None),
              (f"{rlabel} — representation / READ (r = {rR:.2f})", Rz, rR),
              (f"{wlabel} — tom_likely judgment / WRITE (r = {rW:.2f})", Wz, rW)]
    panels = [p for p in panels if p[1] is not None]
    fig = make_subplots(1, len(panels), horizontal_spacing=0.07,
                        subplot_titles=[p[0] for p in panels])
    for col, (_, M, _) in enumerate(panels, 1):
        fig.add_trace(go.Heatmap(z=M, x=labs, y=labs, colorscale="RdBu_r", zmid=0,
            zmin=-2.5, zmax=2.5, showscale=(col == len(panels)),
            colorbar=dict(title="z", len=0.9)), row=1, col=col)
        fig.update_xaxes(tickfont=dict(size=6), tickangle=90, row=1, col=col)
        fig.update_yaxes(tickfont=dict(size=6), autorange="reversed",
                         showticklabels=(col == 1), row=1, col=col)
    fig.update_layout(
        title=dict(text=f"<b>The model REPRESENTS personality structure weakly but "
            f"JUDGES it like a human</b><br><sub>Top-loading adjectives on the leading "
            f"human PCs, ordered by HUMAN cluster structure (same order all panels). "
            f"Read (representation cosine) washes the blocks out; write (tom_likely "
            f"judgment) recovers them. Prevalence-corrected; subset match r in titles. "
            f"Read {rR:.2f} → write {rW:.2f}.</sub>", x=0.01),
        width=1850, height=720, font=dict(family="Helvetica, Arial"))
    out = f"results/adjectives/introspect_full/human_match_{args.model}.png"
    fig.write_html(out.replace(".png", ".html"), include_plotlyjs="cdn")
    fig.write_image(out, width=1850, height=720, scale=2)
    print(f"subset {len(leaf)} words; read r={rR:.3f} write r={rW:.3f}; wrote {out}")


if __name__ == "__main__":
    main()
