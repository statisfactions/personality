#!/usr/bin/env python3
"""W11 visualization: compare Okada GFC-30 vs IPIP-NEO-GFC-60 TIRT recovery.

Two panels:
  1. Cohort-mean |r| by persona form (description / ipip_raw / ipip_reflowed),
     P=30 vs P=60 side-by-side bars. Shows the matched-vocabulary advantage
     and the reflow vocabulary-coupling effect.
  2. Per-model × per-form recovery, P=30 vs P=60 paired bars. Shows which
     models gained vs lost.

Output: results/persona/persona_w11_gfc_comparison.html
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
FORMS = ["description", "ipip_raw", "ipip_reflowed"]
FORM_LABELS = {
    "description": "W7 marker description",
    "ipip_raw": "IPIP-raw description",
    "ipip_reflowed": "IPIP-reflowed description",
}


def load_diag(model, form, instrument):
    if instrument == "P30":
        path = f"results/persona/persona_gfc_tirt_{model}_{form}.json"
    else:
        path = f"results/persona/persona_gfc_tirt_{model}_ipipneogfc60_hf_{form}.json"
    d = json.load(open(path))
    return [float(v) for v in d["diagonal_correlations"].values() if v != "NA"]


def mean_abs(vals):
    return float(np.mean(np.abs(vals)))


def main():
    # Build per-model × per-form × instrument table
    table = {(m, f, inst): mean_abs(load_diag(m, f, inst))
             for m in MODELS for f in FORMS for inst in ("P30", "P60")}

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}], [{}]],
        subplot_titles=[
            "Cohort mean |r| by persona form: Okada GFC-30 (P=30) vs IPIP-NEO-GFC-60 (P=60)",
            "Per-model × form recovery: paired bars (left=P=30 Okada, right=P=60 IPIP-NEO)",
        ],
        row_heights=[0.30, 0.70],
        vertical_spacing=0.12,
    )

    # ---- Panel 1: cohort means by form ----
    cohort_means_30 = [np.mean([table[(m, f, "P30")] for m in MODELS]) for f in FORMS]
    cohort_means_60 = [np.mean([table[(m, f, "P60")] for m in MODELS]) for f in FORMS]
    x_labels = [FORM_LABELS[f] for f in FORMS]

    fig.add_trace(go.Bar(
        x=x_labels, y=cohort_means_30, name="P=30 Okada GFC",
        marker_color="#999999",
        text=[f"{v:.3f}" for v in cohort_means_30],
        textposition="outside", textfont=dict(size=11),
        legendgroup="P30", showlegend=True,
        hovertemplate="<b>%{x}</b> (P=30): %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x_labels, y=cohort_means_60, name="P=60 IPIP-NEO-GFC",
        marker_color="#e41a1c",
        text=[f"{v:.3f}" for v in cohort_means_60],
        textposition="outside", textfont=dict(size=11),
        legendgroup="P60", showlegend=True,
        hovertemplate="<b>%{x}</b> (P=60): %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="cohort mean |r|", range=[0, 0.6], row=1, col=1)

    # Add annotations on the deltas
    for i, f in enumerate(FORMS):
        delta = cohort_means_60[i] - cohort_means_30[i]
        sign = "+" if delta >= 0 else ""
        fig.add_annotation(
            x=x_labels[i], y=max(cohort_means_30[i], cohort_means_60[i]) + 0.08,
            text=f"Δ = {sign}{delta:.3f}",
            showarrow=False, font=dict(size=11, color="#444444" if abs(delta) < 0.1 else "#cc0000"),
            row=1, col=1,
        )

    # ---- Panel 2: per-model × per-form paired bars ----
    # Group by (form, model). x-axis = model, faceted by form via offset.
    # Simpler: group by form, show each model twice (P30/P60 paired).
    grouped_x = []
    grouped_y_30 = []
    grouped_y_60 = []
    grouped_form_color = []
    for f in FORMS:
        for m in MODELS:
            grouped_x.append(f"{m}<br>({f.replace('description','desc').replace('ipip_','').replace('reflowed','refl')})")
            grouped_y_30.append(table[(m, f, "P30")])
            grouped_y_60.append(table[(m, f, "P60")])

    fig.add_trace(go.Bar(
        x=grouped_x, y=grouped_y_30, name="P=30 Okada GFC",
        marker_color="#999999",
        legendgroup="P30", showlegend=False,
        hovertemplate="<b>%{x}</b><br>P=30: %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=grouped_x, y=grouped_y_60, name="P=60 IPIP-NEO-GFC",
        marker_color="#e41a1c",
        legendgroup="P60", showlegend=False,
        hovertemplate="<b>%{x}</b><br>P=60: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Vertical separators between form groups
    for k in range(1, 3):
        fig.add_vline(x=k * len(MODELS) - 0.5, line_dash="dot",
                      line_color="black", opacity=0.3, row=2, col=1)
    fig.update_yaxes(title_text="cohort mean |r|", range=[0, 0.8], row=2, col=1)

    # Layout
    cohort_30 = np.mean(cohort_means_30)
    cohort_60 = np.mean(cohort_means_60)
    fig.update_layout(
        title=dict(
            text=(f"W11 §6: Okada GFC-30 vs IPIP-NEO-GFC-60 TIRT recovery — "
                  f"grand cohort |r| P=30 {cohort_30:.3f} vs P=60 {cohort_60:.3f} "
                  f"(matched-vocab Δ +0.02–0.04; reflow Δ −0.18 = vocab-coupling effect)"),
            x=0.5, font=dict(size=13),
        ),
        height=900, width=1500,
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=1.04,
                    xanchor="center", x=0.5),
    )

    out = Path("results/persona/persona_w11_gfc_comparison.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")
    print(f"  Cohort means by form: P=30 = {cohort_means_30}")
    print(f"                        P=60 = {cohort_means_60}")
    print(f"  Grand cohort: P=30 {cohort_30:.3f}  P=60 {cohort_60:.3f}")


if __name__ == "__main__":
    main()
