"""Spectrum strips: what each model's read spectrum becomes when enacted
(W17 §12, story-first companion to the rpc_flow Sankey).

One horizontal strip per model. Segments are the top REPRESENT PCs in variance
order; segment width = that PC's share of total read variance, so the strip's
reach shows how concentrated the read geometry is. Segment color = the PC's
DOMINANT ENACT anchor (largest |cos| of its image, W v_k); opacity =
sp(in,out) faithfulness (§12.5). The unpainted tail is the rest of the read
spectrum (not fingerprinted).

Reads results/steer_map/rpc_fingerprints.json (massive-norm space).
Usage: PYTHONPATH=scripts python scripts/viz_rpc_strip.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go

FIG = Path("results/persona_vectors/figs")
SRC = "results/steer_map/rpc_fingerprints.json"
MODELS = ["llama3.2", "qwen2.5", "gemma3"]

AXES = ["valence", "E-PC1", "E-PC2", "E-PC3", "assist", "verbose", "format*"]
AXIS_LABEL = {"valence": "valence", "E-PC1": "E-PC1", "E-PC2": "E-PC2",
              "E-PC3": "E-PC3", "assist": "assistant", "verbose": "verbosity",
              "format*": "format (*)"}
AXIS_HEX = {"valence": "#2a78d6", "E-PC1": "#1baf7a", "E-PC2": "#eda100",
            "E-PC3": "#008300", "assist": "#4a3aa7", "verbose": "#e34948",
            "format*": "#e87ba4"}
INK, INK2 = "#0b0b0b", "#52514e"


def rgba(hexc, a):
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"rgba({r},{g},{b},{a:.2f})"


def main():
    data = json.load(open(SRC))
    fig = go.Figure()
    for model in MODELS:
        rows = data[model]["pcs"]
        x0 = 0.0
        for r in rows:
            ax, c = max(r["profile"].items(), key=lambda kv: abs(kv[1]))
            sp = r["sp_in_out"]
            alpha = 0.25 + 0.75 * max(0.0, min(sp, 0.85)) / 0.85
            w = r["var_share"]
            fig.add_trace(go.Bar(
                x=[w], y=[model], base=[x0], orientation="h",
                marker=dict(color=rgba(AXIS_HEX[ax], alpha),
                            line=dict(color="#fcfcfb", width=2)),
                width=0.55, showlegend=False,
                customdata=[[f"PC{r['pc']} · {r['in_neg'][0]}↔"
                             f"{r['in_pos'][-1]}<br>→ {AXIS_LABEL[ax]} "
                             f"(cos {c:+.2f}) · faithfulness {sp:+.2f}<br>"
                             f"image: [{', '.join(r['out_neg'][:3])}] ↔ "
                             f"[{', '.join(r['out_pos'][-3:])}]"]],
                hovertemplate="%{customdata[0]}<extra></extra>"))
            if w > 0.025:
                fig.add_annotation(x=x0 + w / 2, y=model, yref="y",
                                   text=f"PC{r['pc']}", showarrow=False,
                                   font=dict(size=9, color=INK))
            x0 += w
        fig.add_annotation(x=x0 + 0.004, y=model, yref="y", xanchor="left",
                           text=f"{x0:.0%} of read variance",
                           showarrow=False,
                           font=dict(size=10, color=INK2))
    # legend as swatch annotations (fixed anchor order)
    for ax in AXES:
        fig.add_trace(go.Bar(x=[None], y=[MODELS[0]], name=AXIS_LABEL[ax],
                             marker=dict(color=AXIS_HEX[ax]),
                             showlegend=True))
    fig.update_layout(
        title=dict(text="What the read spectrum becomes when enacted — "
                        "top-12 REPRESENT PCs, colored by dominant ENACT "
                        "anchor, opacity = faithfulness",
                   font=dict(size=14, color=INK)),
        barmode="overlay",
        xaxis=dict(title="cumulative share of REPRESENT variance",
                   range=[0, 0.62], tickformat=".0%",
                   showgrid=False, color=INK2),
        yaxis=dict(categoryorder="array", categoryarray=MODELS[::-1],
                   color=INK),
        legend=dict(orientation="h", y=-0.22, font=dict(size=10, color=INK)),
        width=1150, height=380, margin=dict(l=80, r=30, t=60, b=90),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
    )
    out = FIG / "rpc_strip"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
