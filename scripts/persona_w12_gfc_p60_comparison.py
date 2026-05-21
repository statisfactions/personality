#!/usr/bin/env python3
"""W12 visualization: P=60 IPIP-NEO-GFC TIRT recovery across HONEST / FG-suffix /
FG-prefix conditions, full 10-model cohort.

Parallel to persona_w11_gfc_comparison.py (which compared P=30 vs P=60 on the
original 7-model cohort with HONEST condition only). This script holds the
instrument fixed at P=60 (the W11/W12 standard) and varies the condition
instead. Includes all 10 cohort models (W11 7 + W12 scaleup 3: Gemma27,
Qwen32, Gemma4) since all have P=60 data across all conditions.

Two panels:
  1. Cohort-mean |r| by persona form, three condition bars per form.
  2. Per-model × per-form recovery, three condition bars per (model, form).

Output: results/persona/persona_w12_gfc_p60_comparison.html
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7",
          "Gemma27", "Qwen32", "Gemma4"]
FORMS = ["description", "ipip_raw", "ipip_reflowed"]
FORM_LABELS = {
    "description": "W7 marker description",
    "ipip_raw": "IPIP-raw description",
    "ipip_reflowed": "IPIP-reflowed description",
}
CONDITIONS = ["honest", "fg_suffix", "fg_prefix"]
CONDITION_LABELS = {
    "honest": "honest",
    "fg_suffix": "fake-good suf.",
    "fg_prefix": "fake-good pre.",
}
CONDITION_COLORS = {
    "honest": "#377eb8",      # blue
    "fg_suffix": "#e41a1c",   # red
    "fg_prefix": "#ff7f00",   # orange
}


def tirt_path(model, form, condition):
    base = f"results/persona/persona_gfc_tirt_{model}_ipipneogfc60_hf_{form}"
    if condition == "honest":
        return f"{base}.json"
    if condition == "fg_suffix":
        return f"{base}_fake_good.json"
    if condition == "fg_prefix":
        return f"{base}_fake_good_fgpfx.json"
    raise ValueError(f"unknown condition: {condition}")


def load_diag(model, form, condition):
    path = tirt_path(model, form, condition)
    if not Path(path).exists():
        return None
    d = json.load(open(path))
    return [float(v) for v in d["diagonal_correlations"].values() if v != "NA"]


def mean_abs(vals):
    if vals is None or not vals:
        return float("nan")
    return float(np.mean(np.abs(vals)))


def main():
    table = {(m, f, c): mean_abs(load_diag(m, f, c))
             for m in MODELS for f in FORMS for c in CONDITIONS}

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}], [{}]],
        subplot_titles=[
            "Per-form distribution across 10 models (box + dot per model)",
            "Per-model × form recovery: three condition bars per (model, form)",
        ],
        row_heights=[0.32, 0.68],
        vertical_spacing=0.16,
    )

    # ---- Panel 1: per-form box plots of the 10-model distribution per condition ----
    cohort_by_cond = {c: [np.nanmean([table[(m, f, c)] for m in MODELS])
                          for f in FORMS]
                      for c in CONDITIONS}
    x_labels = [FORM_LABELS[f] for f in FORMS]

    for c in CONDITIONS:
        ys = []
        xs = []
        hover = []
        for f_i, f in enumerate(FORMS):
            for m in MODELS:
                v = table[(m, f, c)]
                if np.isnan(v):
                    continue
                ys.append(v)
                xs.append(x_labels[f_i])
                hover.append(f"{m} · {CONDITION_LABELS[c]} · |r|={v:.3f}")
        fig.add_trace(go.Box(
            x=xs, y=ys, name=CONDITION_LABELS[c],
            marker_color=CONDITION_COLORS[c],
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(size=5, opacity=0.7),
            line=dict(width=1.5),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=c, showlegend=True,
        ), row=1, col=1)
    fig.update_yaxes(title_text="per-model |r|", range=[0, 0.8], row=1, col=1)

    # ---- Panel 2: per-model × per-form × per-condition grouped bars ----
    grouped_x = []
    grouped_y = {c: [] for c in CONDITIONS}
    for f in FORMS:
        for m in MODELS:
            short_f = (f.replace("description", "desc")
                       .replace("ipip_", "")
                       .replace("reflowed", "refl"))
            grouped_x.append(f"{m}<br>({short_f})")
            for c in CONDITIONS:
                grouped_y[c].append(table[(m, f, c)])

    for c in CONDITIONS:
        fig.add_trace(go.Bar(
            x=grouped_x, y=grouped_y[c], name=CONDITION_LABELS[c],
            marker_color=CONDITION_COLORS[c],
            legendgroup=c, showlegend=False,
            hovertemplate="<b>%{x}</b><br>" + CONDITION_LABELS[c] + ": %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    # Vertical separators between form groups
    for k in range(1, len(FORMS)):
        fig.add_vline(x=k * len(MODELS) - 0.5, line_dash="dot",
                      line_color="black", opacity=0.3, row=2, col=1)
    fig.update_yaxes(title_text="mean |r|", range=[0, 0.9], row=2, col=1)

    grand = {c: float(np.nanmean(cohort_by_cond[c])) for c in CONDITIONS}

    # Per-model mean (across 3 forms) per condition → cohort-diversity range.
    per_model = {c: {m: float(np.nanmean([table[(m, f, c)] for f in FORMS]))
                     for m in MODELS} for c in CONDITIONS}
    model_range = {c: (min(per_model[c].values()), max(per_model[c].values()))
                   for c in CONDITIONS}
    best_model = {c: max(per_model[c], key=per_model[c].get) for c in CONDITIONS}
    worst_model = {c: min(per_model[c], key=per_model[c].get) for c in CONDITIONS}

    def summary(c):
        lo, hi = model_range[c]
        return (f"{CONDITION_LABELS[c]} mean {grand[c]:.3f} "
                f"(range {lo:.3f} [{worst_model[c]}] → {hi:.3f} [{best_model[c]}])")

    fig.update_layout(
        title=dict(
            text=("W12: IPIP-NEO-GFC-60 TIRT recovery across conditions "
                  f"(N={len(MODELS)} models, mean across 3 persona forms)<br>"
                  f"<sub>{summary('honest')} · "
                  f"{summary('fg_suffix')} · "
                  f"{summary('fg_prefix')}</sub>"),
            x=0.5, font=dict(size=13),
        ),
        height=900, width=1700,
        barmode="group",
        boxmode="group",
        legend=dict(orientation="h", yanchor="top", y=0.98,
                    xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.2)",
                    bordercolor="#cccccc", borderwidth=1),
        margin=dict(t=120),
    )

    out = Path("results/persona/persona_w12_gfc_p60_comparison.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")
    for c in CONDITIONS:
        print(f"  {CONDITION_LABELS[c]:>18s}: by form = "
              f"{[round(v, 3) for v in cohort_by_cond[c]]}, "
              f"grand = {grand[c]:.3f}, "
              f"model range = {model_range[c][0]:.3f} ({worst_model[c]}) "
              f"→ {model_range[c][1]:.3f} ({best_model[c]})")


if __name__ == "__main__":
    main()
