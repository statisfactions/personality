#!/usr/bin/env python3
"""W12 §8.1b: per-facet row-correlation heatmap (30 facets × 10 cohort models).

For each facet F and model M, computes Pearson(H[F, ¬F], M[F, ¬F]) — the
correlation between F's row of human off-diagonal facet correlations and F's
row of model facet cosines, with the self-cell excluded. This is the same
diagnostic narrated as a table in W9 §7.6; the heatmap makes the cohort
homogeneity (and the Cheerf/Liberal divergences) citable at a glance.

Rows are sorted by mean row-r across the 10 cohort models (descending), so
the tightest-agreement facets (N cluster) appear at the top and the
worst-recovered (A:Sympath, O:Liberal) at the bottom. An eleventh "Mean"
column on the right shows the row mean.

Inputs:
  instruments/ipip300_human_facet_correlations.json
  results/facets/ipip_facet_cluster.json

Output:
  results/facets/facet_row_corr_heatmap.html

Usage: .venv/bin/python scripts/facet_row_corr_heatmap.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# Three column groups, drawn left→right with separators between them:
#   1. COHORT          — the original 10 LLMs, param-sorted (cohort baseline).
#   2. ROBUSTNESS PROBES — Aya + FalconMamba (W13 §8.2), held out from the
#                        cohort mean. A second axis of variation: Aya varies
#                        TRAINING DATA, FalconMamba varies ARCHITECTURE (a
#                        non-transformer SSM). They probe whether the facet
#                        structure is invariant to those, not just to scale.
#   3. ENCODERS        — sentence encoders (bge, mpnet) in their method-matched
#                        keyed-diff/raw mode (§3.9). They never attended
#                        causally or saw RLHF; rows where they FOLLOW the LLMs
#                        are shared lexical structure, rows where they DIVERGE
#                        localize the LLM-specific residual.
COHORT = ["Qwen", "Llama", "Phi4", "Gemma",          # 3–4B
          "Qwen7", "Llama8",                          # 7–8B
          "Gemma12", "Gemma27", "Gemma4", "Qwen32"]   # 12–32B  (mean baseline)
OUT_OF_COHORT = ["FalconMamba", "Aya"]                # held out from mean
ENCODERS = ["bge-large", "mpnet-base"]                # keyed-diff/raw

ENCODER_MODE = "keyed-diff/raw"   # negation-sensitive (matches model meandiff),
                                  # no projection (§3.9: correct for encoders)

EMBED_PATH = Path("results/facets/embedding_facet_baseline.json")
OUT_PATH = Path("results/facets/facet_row_corr_heatmap.html")


def row_correlation(H, M, i):
    """Pearson(H[i, ¬i], M[i, ¬i]) — self-cell excluded."""
    mask = np.ones(H.shape[0], dtype=bool)
    mask[i] = False
    return float(np.corrcoef(H[i, mask], M[i, mask])[0, 1])


def main():
    human = json.load(open("instruments/ipip300_human_facet_correlations.json"))
    H = np.array(human["correlation_matrix"])
    facet_lbls = human["facet_order"]
    n_human = human["n"]
    n_facets = len(facet_lbls)
    assert n_facets == 30 and H.shape == (30, 30)

    model_data = json.load(open("results/facets/ipip_facet_cluster.json"))
    embed_data = json.load(open(EMBED_PATH))
    assert embed_data["facet_order"] == facet_lbls, "encoder facet order drift"

    def model_row_corrs(model_short):
        """Per-facet row correlations for an LLM, reordered to facet_lbls."""
        model_lbls = model_data[model_short]["facet_names"]
        reorder = [model_lbls.index(lbl) for lbl in facet_lbls]
        M = np.array(model_data[model_short]["cosine_matrix"])[np.ix_(reorder, reorder)]
        return np.array([row_correlation(H, M, i) for i in range(n_facets)])

    def encoder_row_corrs(enc_tag):
        """Per-facet row correlations for an encoder (already in facet_lbls order)."""
        M = np.array(embed_data["encoders"][enc_tag]["modes"][ENCODER_MODE]["cosine_matrix"])
        return np.array([row_correlation(H, M, i) for i in range(n_facets)])

    cohort = [m for m in COHORT if m in model_data]
    ooc = [m for m in OUT_OF_COHORT if m in model_data]
    encoders = [e for e in ENCODERS if e in embed_data["encoders"]]

    cohort_R = np.column_stack([model_row_corrs(m) for m in cohort])      # (30, 10)
    ooc_R = np.column_stack([model_row_corrs(m) for m in ooc])            # (30, 2)
    enc_R = np.column_stack([encoder_row_corrs(e) for e in encoders])     # (30, 2)

    # Row order + cohort-LLM mean/SD computed from the cohort 10 ONLY.
    row_mean = np.nanmean(cohort_R, axis=1)
    row_sd = np.nanstd(cohort_R, axis=1)
    enc_mean = np.nanmean(enc_R, axis=1)
    order = np.argsort(-row_mean)
    labels_sorted = [facet_lbls[i] for i in order]

    # Assemble grouped columns:
    #   cohort(10) | ooc(2) | encoders(2) | LLM-mean | enc-mean
    blocks = [cohort_R, ooc_R, enc_R,
              row_mean[:, None], enc_mean[:, None]]
    Z = np.column_stack([b[order] for b in blocks])
    sd_sorted = row_sd[order]

    enc_short = {"bge-large": "bge", "mpnet-base": "mpnet"}
    col_meta = (  # (display label, hover-kind) per column, in Z order
        [(m, "cohort") for m in cohort]
        + [(m, "ooc") for m in ooc]
        + [(enc_short.get(e, e), "enc") for e in encoders]
        + [("LLM mean", "llm_mean"), ("enc mean", "enc_mean")]
    )
    col_labels = [c[0] for c in col_meta]

    hover = []
    for ri, lbl in enumerate(labels_sorted):
        row_h = []
        for ci, (name, kind) in enumerate(col_meta):
            v = Z[ri, ci]
            if kind == "llm_mean":
                row_h.append(f"{lbl}<br>cohort-LLM mean (n={len(cohort)}): {v:+.3f}<br>SD: {sd_sorted[ri]:.3f}")
            elif kind == "enc_mean":
                row_h.append(f"{lbl}<br>encoder mean (n={len(encoders)}): {v:+.3f}")
            elif kind == "ooc":
                row_h.append(f"{lbl} @ {name} (robustness probe, excl. from mean)<br>row r: {v:+.3f}")
            elif kind == "enc":
                row_h.append(f"{lbl} @ {name} ({ENCODER_MODE} encoder)<br>row r: {v:+.3f}")
            else:
                row_h.append(f"{lbl} @ {name}<br>row r: {v:+.3f}")
        hover.append(row_h)

    # Diverging colorscale centered at 0.
    vabs = max(0.6, float(np.nanmax(np.abs(Z))))

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=col_labels,
        y=labels_sorted,
        zmin=-vabs, zmax=vabs,
        colorscale="RdBu",
        reversescale=False,
        colorbar=dict(title="row r", thickness=14, len=0.85),
        hoverinfo="text",
        text=hover,
        xgap=1, ygap=1,
    ))

    # Group separators (white vlines) + a header label centered over each block.
    n_co, n_oc, n_en = len(cohort), len(ooc), len(encoders)
    bounds = [n_co, n_co + n_oc, n_co + n_oc + n_en]  # right edge of each LLM/enc block
    for b in bounds + [n_co + n_oc + n_en + 1]:       # +1 also splits the two mean cols
        fig.add_vline(x=b - 0.5, line_width=2, line_color="white")

    def group_label(text, x0, x1):
        fig.add_annotation(x=(x0 + x1 - 1) / 2, y=1.0, yref="paper", xref="x",
                           yshift=34, text=f"<b>{text}</b>", showarrow=False,
                           font=dict(size=12), xanchor="center")
    group_label("cohort LLMs", 0, n_co)
    group_label("robustness probes", n_co, n_co + n_oc)
    group_label("encoders", n_co + n_oc, n_co + n_oc + n_en)
    group_label("means", n_co + n_oc + n_en, n_co + n_oc + n_en + 2)

    fig.update_layout(
        title=dict(
            text=(f"<b>Per-facet row correlation vs human (N={n_human:,}): LLMs, "
                  f"robustness probes, and text encoders</b>"
                  f"<br><sub>For each facet F, Pearson(H[F,¬F], M[F,¬F]) — self-cell excluded. "
                  f"Encoders in {ENCODER_MODE} (§3.9). Cohort-LLM grand mean (n={n_co}) = "
                  f"{np.nanmean(row_mean):+.3f}. Rows where encoders FOLLOW the LLM red = shared "
                  f"lexical structure; rows where they DIVERGE = LLM-specific residual.</sub>"),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(tickangle=-30, side="top"),
        yaxis=dict(title="facet (high → low cohort-LLM r)",
                   tickfont=dict(size=11), autorange="reversed"),
        width=1000, height=900,
        margin=dict(l=110, r=40, t=200, b=40),
        font=dict(family="Helvetica, Arial, sans-serif"),
        plot_bgcolor="white",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_PATH, include_plotlyjs="cdn")
    png_path = OUT_PATH.with_suffix(".png")
    fig.write_image(png_path, width=1000, height=900, scale=2)
    print(f"wrote {OUT_PATH}")
    print(f"wrote {png_path}")
    print(f"  cohort-LLM grand mean row-r (n={n_co}): {np.nanmean(row_mean):+.3f}")
    print(f"  encoder grand mean row-r (n={n_en}):  {np.nanmean(enc_mean):+.3f}")
    # Divergence audit: rows where the cohort-LLM and encoder means disagree most.
    llm_m, enc_m = row_mean[order], enc_mean[order]
    gap = np.abs(llm_m - enc_m)
    print("  largest LLM-vs-encoder mean gap (candidate model-specific rows):")
    for i in np.argsort(-gap)[:5]:
        print(f"    {labels_sorted[i]:11} LLM {llm_m[i]:+.3f}  enc {enc_m[i]:+.3f}  gap {gap[i]:.3f}")


if __name__ == "__main__":
    main()
