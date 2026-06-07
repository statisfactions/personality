#!/usr/bin/env python3
"""W16 §5.8 headline figure: read vs write human-match across the 12-model cohort.

Dumbbell plot — each model is a line from its REPRESENTATION (read) human-match to
its tom_likely JUDGMENT (write) human-match, both prevalence-corrected. The gap IS
the read/write dissociation; sorting by write-match puts FalconMamba (a non-
transformer SSM) on top with the cohort's widest gap. A second panel: a clean
grouped bar so the numbers are legible.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/fig_cohort_humanmatch.py
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_human_match import (HUMAN, double_center_off, match,
                                    repr_matrix, judge_matrix)

# short -> (display, family) ; family sets colour
COHORT = [
    ("FalconMamba", "Falcon-Mamba-7B (SSM)", "Falcon"),
    ("Qwen",        "Qwen2.5-3B",            "Qwen"),
    ("Phi4",        "Phi-4-mini (3.8B)",     "Phi"),
    ("Llama8",      "Llama-3.1-8B",          "Llama"),
    ("Llama",       "Llama-3.2-3B",          "Llama"),
    ("Qwen7",       "Qwen2.5-7B",            "Qwen"),
    ("Aya",         "Aya-expanse-8B",        "Aya"),
    ("Gemma12",     "Gemma-3-12B",           "Gemma"),
    ("Gemma",       "Gemma-3-4B",            "Gemma"),
    ("Gemma27",     "Gemma-3-27B",           "Gemma"),
    ("Qwen32",      "Qwen2.5-32B",           "Qwen"),
    ("Gemma4",      "Gemma-4-31B",           "Gemma"),
]
FAM_COLOR = {"Falcon": "#d62728", "Qwen": "#1f77b4", "Phi": "#9467bd",
             "Llama": "#2ca02c", "Aya": "#ff7f0e", "Gemma": "#8c564b"}


def corrected(M):
    return match(double_center_off(M), double_center_off(HUMAN))


def main():
    Hdc = double_center_off(HUMAN)
    rows = []
    for short, disp, fam in COHORT:
        rM, jM = repr_matrix(short), judge_matrix(short, "tom_likely")
        read = match(double_center_off(rM), Hdc) if rM is not None else np.nan
        write = match(double_center_off(jM), Hdc) if jM is not None else np.nan
        rows.append((disp, fam, read, write))
    rows.sort(key=lambda r: r[3])           # ascending -> top of plot = highest

    disp = [r[0] for r in rows]
    fams = [r[1] for r in rows]
    read = np.array([r[2] for r in rows])
    write = np.array([r[3] for r in rows])

    fig = make_subplots(
        1, 2, column_widths=[0.62, 0.38], horizontal_spacing=0.16,
        subplot_titles=("Read → Write human-match (dumbbell)",
                        "Write-side match (tom_likely)"))

    # panel 1: dumbbell
    for k, (d, f) in enumerate(zip(disp, fams)):
        c = FAM_COLOR[f]
        fig.add_trace(go.Scatter(
            x=[read[k], write[k]], y=[d, d], mode="lines",
            line=dict(color=c, width=2), showlegend=False,
            hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(
        x=read, y=disp, mode="markers", name="representation (read)",
        marker=dict(symbol="circle-open", size=11,
                    color=[FAM_COLOR[f] for f in fams], line=dict(width=2)),
        hovertemplate="read %{x:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(
        x=write, y=disp, mode="markers", name="tom_likely judgment (write)",
        marker=dict(symbol="circle", size=13,
                    color=[FAM_COLOR[f] for f in fams]),
        hovertemplate="write %{x:.3f}<extra></extra>"), 1, 1)

    # panel 2: write-match bar
    fig.add_trace(go.Bar(
        x=write, y=disp, orientation="h", showlegend=False,
        marker=dict(color=[FAM_COLOR[f] for f in fams]),
        text=[f"{w:.2f}" for w in write], textposition="outside",
        cliponaxis=False), 1, 2)

    fig.update_xaxes(title_text="corrected r to human 525-PDA",
                     range=[0.30, 0.86], row=1, col=1)
    fig.update_xaxes(title_text="corrected r", range=[0, 0.95], row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>What the model JUDGES matches human personality "
            "structure; what it REPRESENTS does not</b><br><sub>Prevalence-corrected "
            "match of each readout to the human inter-adjective correlation matrix "
            "(N=700 self-reports). The read→write gap is the dissociation; "
            "Falcon-Mamba — a non-transformer SSM — shows the widest gap "
            "(read 0.41 → write 0.80), so the split is not a transformer artifact.</sub>",
            x=0.01, font=dict(size=15)),
        legend=dict(orientation="h", y=-0.14, x=0.0),
        width=1350, height=620, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white", bargap=0.45)
    fig.update_xaxes(gridcolor="#eee", zeroline=False)

    out = "results/adjectives/introspect_full/fig_cohort_readwrite.png"
    fig.write_html(out.replace(".png", ".html"), include_plotlyjs="cdn")
    fig.write_image(out, width=1350, height=620, scale=2)
    gap = write - read
    print("model                read   write   gap")
    for d, r, w in sorted(zip(disp, read, write), key=lambda t: -t[2]):
        print(f"{d:22}{r:.3f}  {w:.3f}  {w-r:+.3f}")
    print(f"\nwidest read/write gap: {disp[int(np.nanargmax(gap))]} "
          f"({np.nanmax(gap):+.3f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
