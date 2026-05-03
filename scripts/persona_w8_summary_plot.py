#!/usr/bin/env python3
"""Multi-panel W8 headline figure.

Builds two plotly HTML figures:
  - results/persona_w8_trajectory.html — cohort mean r per condition
    (W7 marker baseline, §3 raw, §5 raw, §5 reflowed) for rep + Likert,
    plus the matched gap.
  - results/persona_w8_per_model_gap.html — per-model matched-gap slope
    plot (raw → reflow).

Designed for the W8 reading-group framing: shows the trajectory of the
"Likert beats Rep" claim across vocabulary-coupling strip-downs and the
reflow ablation, highlighting the cohort-level finding (gap shrinks
+0.144 → ~+0.05–+0.08 under matched conditions, with model-level
variation).
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Llama", "Phi4", "Qwen", "Gemma12", "Llama8", "Qwen7"]
MODEL_LABELS = {
    "Gemma": "Gemma 4B", "Llama": "Llama 3B", "Phi4": "Phi4-mini",
    "Qwen": "Qwen 3B", "Gemma12": "Gemma 12B", "Llama8": "Llama 8B",
    "Qwen7": "Qwen 7B",
}
MODEL_COLORS = {
    "Gemma": "#4daf4a", "Llama": "#377eb8", "Phi4": "#e41a1c",
    "Qwen": "#984ea3", "Gemma12": "#a6d96a", "Llama8": "#74add1",
    "Qwen7": "#cab2d6",
}
TRAITS = ["A", "C", "E", "N", "O"]

# Result file patterns by (condition, mode)
PATTERNS = {
    ("W7", "rep"):           "results/persona_repr_mapping_{m}_response-position.json",
    ("W7", "likert"):        "results/persona_instrument_response_{m}.json",
    ("§3 raw", "rep"):       "results/persona_repr_mapping_{m}_response-position_ipip_raw.json",
    ("§3 raw", "likert"):    "results/persona_instrument_response_{m}_ipip_raw.json",
    ("§3 reflow", "rep"):    "results/persona_repr_mapping_{m}_response-position_ipip_reflowed.json",
    ("§3 reflow", "likert"): "results/persona_instrument_response_{m}_ipip_reflowed.json",
    ("§4 raw", "rep"):       "results/persona_repr_mapping_{m}_response-position_ipip_raw.json",  # same as §3 raw
    ("§4 raw", "likert"):    "results/persona_instrument_response_{m}_ipip_raw_target-ipip.json",
    ("§4 reflow", "rep"):    "results/persona_repr_mapping_{m}_response-position_ipip_reflowed.json",
    ("§4 reflow", "likert"): "results/persona_instrument_response_{m}_ipip_reflowed_target-ipip.json",
    ("§5 raw", "rep"):       "results/persona_repr_mapping_{m}_response-position_ipip_raw_dir-ipip.json",
    ("§5 raw", "likert"):    "results/persona_instrument_response_{m}_ipip_raw_target-ipip.json",  # same as §4 raw
    ("§5 reflow", "rep"):    "results/persona_repr_mapping_{m}_response-position_ipip_reflowed_dir-ipip.json",
    ("§5 reflow", "likert"): "results/persona_instrument_response_{m}_ipip_reflowed_target-ipip.json",
}

# The 5 conditions in the headline trajectory
CONDITIONS_TRAJECTORY = ["W7", "§3 raw", "§4 raw", "§5 raw", "§5 reflow"]

# Process labels for x-axis ticks: "persona form → measurement form".
# Two-line: top line is persona-side, bottom line is readout-side details.
CONDITION_LABELS = {
    "W7":         "Goldberg<br>→ Goldberg",
    "§3 raw":     "IPIP raw<br>→ Goldberg",
    "§4 raw":     "IPIP raw<br>→ IPIP target<br><sub>(Goldberg dir)</sub>",
    "§5 raw":     "IPIP raw<br>→ IPIP",
    "§5 reflow":  "IPIP reflow<br>→ IPIP",
}
CONDITION_TOOLTIPS = {
    "W7":         "Goldberg persona, Goldberg dir, Goldberg target",
    "§3 raw":     "IPIP-raw persona, Goldberg dir, Goldberg target (changes persona)",
    "§4 raw":     "IPIP-raw persona, Goldberg dir, IPIP target (also changes rating target)",
    "§5 raw":     "IPIP-raw persona, IPIP dir, IPIP target (also changes rep dir; fully matched)",
    "§5 reflow":  "IPIP-reflowed persona, IPIP dir, IPIP target (smooth prose, fully matched)",
}


def load_diag(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("diagonal_correlations")


def load_diag_mean(path):
    d = load_diag(path)
    if d is None:
        return None
    return float(np.mean(list(d.values())))


def collect_means():
    """Return {condition: {mode: {model: mean_r or None}}}."""
    out = {}
    # Collect for all conditions in the patterns dict, not just the trajectory.
    all_conds = sorted({c for (c, _) in PATTERNS.keys()})
    for cond in all_conds:
        out[cond] = {"rep": {}, "likert": {}}
        for mode in ("rep", "likert"):
            for m in MODELS:
                path = PATTERNS[(cond, mode)].format(m=m)
                out[cond][mode][m] = load_diag_mean(path)
    return out


def cohort_mean(d):
    vals = [v for v in d.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def make_trajectory_figure(data):
    """Cohort mean r per condition for rep + Likert, with gap shading."""
    rep_means = [cohort_mean(data[c]["rep"]) for c in CONDITIONS_TRAJECTORY]
    lik_means = [cohort_mean(data[c]["likert"]) for c in CONDITIONS_TRAJECTORY]
    gaps = [(l - r) if (l is not None and r is not None) else None
            for l, r in zip(lik_means, rep_means)]

    x_labels = [CONDITION_LABELS[c] for c in CONDITIONS_TRAJECTORY]
    tooltips = [CONDITION_TOOLTIPS[c] for c in CONDITIONS_TRAJECTORY]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cohort mean r per condition",
                        "Likert − Rep gap (cohort mean)"),
        vertical_spacing=0.1,
    )

    # Top: rep + likert lines
    fig.add_trace(go.Scatter(
        x=x_labels, y=rep_means,
        mode="lines+markers+text",
        text=[f"{v:.3f}" if v is not None else "" for v in rep_means],
        textposition="bottom center", textfont=dict(size=10),
        name="Rep r", line=dict(color="#377eb8", width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Rep r = %{y:+.3f}<br>%{customdata}<extra></extra>",
        customdata=tooltips,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_labels, y=lik_means,
        mode="lines+markers+text",
        text=[f"{v:.3f}" if v is not None else "" for v in lik_means],
        textposition="top center", textfont=dict(size=10),
        name="Likert r", line=dict(color="#e41a1c", width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Likert r = %{y:+.3f}<br>%{customdata}<extra></extra>",
        customdata=tooltips,
    ), row=1, col=1)

    # Bottom: gap bars
    fig.add_trace(go.Bar(
        x=x_labels, y=gaps,
        text=[f"{g:+.3f}" if g is not None else "" for g in gaps],
        textposition="outside",
        marker=dict(color=["#888" if g and g > 0 else "#cc6666" for g in gaps]),
        name="Gap (L−R)",
        hovertemplate="<b>%{x}</b><br>Gap = %{y:+.3f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="black",
                  opacity=0.5, row=2, col=1)

    fig.update_yaxes(title_text="diagonal r (cohort mean)", row=1, col=1,
                     range=[0.4, 1.0])
    fig.update_yaxes(title_text="Likert − Rep", row=2, col=1,
                     range=[-0.05, 0.20])
    fig.update_xaxes(title_text="condition", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=("W8: Likert vs Rep across vocabulary-coupling strip-down "
                  "(7-model cohort, 50 personas, seed=42)"),
            x=0.5, font=dict(size=14),
        ),
        height=700, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    return fig


def make_per_model_figure(data):
    """Per-model matched gap (§4 Likert − §5 Rep) under raw vs reflow."""
    raw_gap = {}
    ref_gap = {}
    for m in MODELS:
        l_raw = data["§4 raw"]["likert"].get(m)
        r_raw = data["§5 raw"]["rep"].get(m)
        l_ref = data["§4 reflow"]["likert"].get(m)
        r_ref = data["§5 reflow"]["rep"].get(m)
        raw_gap[m] = l_raw - r_raw if (l_raw is not None and r_raw is not None) else None
        ref_gap[m] = l_ref - r_ref if (l_ref is not None and r_ref is not None) else None

    fig = go.Figure()

    # Cohort-mean line
    cohort_raw = float(np.mean([v for v in raw_gap.values() if v is not None]))
    cohort_ref = float(np.mean([v for v in ref_gap.values() if v is not None]))
    fig.add_trace(go.Scatter(
        x=["raw", "reflow"], y=[cohort_raw, cohort_ref],
        mode="lines+markers",
        line=dict(color="black", width=4, dash="dash"),
        marker=dict(size=14, symbol="diamond", color="black"),
        name=f"Cohort ({cohort_raw:+.3f} → {cohort_ref:+.3f})",
        hovertemplate="cohort %{x}<br>gap = %{y:+.3f}<extra></extra>",
    ))

    # Per-model lines
    for m in MODELS:
        if raw_gap[m] is None or ref_gap[m] is None:
            continue
        fig.add_trace(go.Scatter(
            x=["raw", "reflow"], y=[raw_gap[m], ref_gap[m]],
            mode="lines+markers+text",
            text=[f"{raw_gap[m]:+.3f}", f"{ref_gap[m]:+.3f}"],
            textposition=["middle left", "middle right"],
            textfont=dict(size=9, color=MODEL_COLORS[m]),
            line=dict(color=MODEL_COLORS[m], width=2),
            marker=dict(size=10, color=MODEL_COLORS[m]),
            name=MODEL_LABELS[m],
            hovertemplate=f"<b>{MODEL_LABELS[m]}</b> %{{x}}<br>gap = %{{y:+.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5,
                  annotation_text="rep beats Likert ↓",
                  annotation_position="bottom right")

    fig.update_layout(
        title=dict(
            text=("Per-model matched gap (§4 Likert − §5 Rep): raw vs reflowed personas"
                  "<br><sub>Under matched IPIP conditions (IPIP dir + IPIP rating target)</sub>"),
            x=0.5, font=dict(size=13),
        ),
        xaxis=dict(title="persona prose form"),
        yaxis=dict(title="Likert r − Rep r", range=[-0.05, 0.15]),
        height=620, width=900,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


def main():
    data = collect_means()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    fig1 = make_trajectory_figure(data)
    out1 = out_dir / "persona_w8_trajectory.html"
    fig1.write_html(str(out1))
    print(f"Wrote {out1}")

    fig2 = make_per_model_figure(data)
    out2 = out_dir / "persona_w8_per_model_gap.html"
    fig2.write_html(str(out2))
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
