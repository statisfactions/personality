"""Flow figure: REPRESENT PCs -> ENACT anchors, per model (W17 §12/§12.5).

Three side-by-side Sankey panels (llama3.2 / qwen2.5 / gemma3). Left nodes are
the top REPRESENT PCs labeled with their input-pole adjectives; right nodes are
interpretable ENACT anchors (valence, E-PC1..3, assistant axis, verbosity,
asterisk-format). Encodings:

  ribbon width    = transmitted mass  (gain x |cos(image, anchor)|)
  ribbon opacity  = sp(in,out) faithfulness (§12.5 — washed-out ribbons are
                    images that describe ENACT's manifold more than the R-PC)
  ribbon color    = destination anchor (validated categorical palette)

Links: anchors with |cos| >= 0.25, top-4 per PC. Hover (html) carries the
signed cosine, faithfulness, and the image's output-pole adjectives.

Reads results/steer_map/rpc_fingerprints.json (massive-norm space).
Usage: PYTHONPATH=scripts python scripts/viz_rpc_flow.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go

FIG = Path("results/persona_vectors/figs")
SRC = "results/steer_map/rpc_fingerprints.json"
MODELS = ["llama3.2", "qwen2.5", "gemma3"]
N_PCS = 10
COS_MIN = 0.25
MAX_LINKS = 4

AXES = ["valence", "E-PC1", "E-PC2", "E-PC3", "assist", "verbose", "format*"]
AXIS_LABEL = {"valence": "valence", "E-PC1": "E-PC1", "E-PC2": "E-PC2",
              "E-PC3": "E-PC3", "assist": "assistant", "verbose": "verbosity",
              "format*": "format (*)"}
# validated categorical palette (dataviz reference, slots 1-7, light mode)
AXIS_HEX = {"valence": "#2a78d6", "E-PC1": "#1baf7a", "E-PC2": "#eda100",
            "E-PC3": "#008300", "assist": "#4a3aa7", "verbose": "#e34948",
            "format*": "#e87ba4"}
LEFT_NODE = "#b5b4ad"
INK, INK2 = "#0b0b0b", "#52514e"


def rgba(hexc, a):
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"rgba({r},{g},{b},{a:.2f})"


def panel(rows):
    """Build sankey node/link arrays for one model."""
    labels, colors = [], []
    for r in rows[:N_PCS]:
        labels.append(f"PC{r['pc']} {r['in_neg'][0]}↔{r['in_pos'][-1]}")
        colors.append(LEFT_NODE)
    for ax in AXES:
        labels.append(AXIS_LABEL[ax])
        colors.append(AXIS_HEX[ax])
    src, dst, val, lcol, hover = [], [], [], [], []
    for i, r in enumerate(rows[:N_PCS]):
        prof = sorted(r["profile"].items(), key=lambda kv: -abs(kv[1]))
        picked = [(a, c) for a, c in prof if abs(c) >= COS_MIN][:MAX_LINKS]
        if not picked:
            picked = prof[:1]
        sp = r["sp_in_out"]
        alpha = 0.15 + 0.65 * max(0.0, min(sp, 0.85)) / 0.85
        for ax, c in picked:
            src.append(i)
            dst.append(N_PCS + AXES.index(ax))
            val.append(r["gain"] * abs(c))
            lcol.append(rgba(AXIS_HEX[ax], alpha))
            hover.append(
                f"PC{r['pc']} → {AXIS_LABEL[ax]}<br>"
                f"cos {c:+.2f} · gain {r['gain']:.2f} · "
                f"faithfulness sp {sp:+.2f}<br>"
                f"image poles: [{', '.join(r['out_neg'][:3])}] ↔ "
                f"[{', '.join(r['out_pos'][-3:])}]")
    return labels, colors, src, dst, val, lcol, hover


def main():
    data = json.load(open(SRC))
    fig = go.Figure()
    n = len(MODELS)
    for mi, model in enumerate(MODELS):
        labels, colors, src, dst, val, lcol, hover = panel(data[model]["pcs"])
        x0, x1 = mi / n + 0.015, (mi + 1) / n - 0.015
        fig.add_trace(go.Sankey(
            domain=dict(x=[x0, x1], y=[0, 0.92]),
            arrangement="snap",
            node=dict(label=labels, color=colors, pad=6, thickness=10,
                      line=dict(width=0)),
            link=dict(source=src, target=dst, value=val, color=lcol,
                      customdata=hover,
                      hovertemplate="%{customdata}<extra></extra>"),
            textfont=dict(color=INK, size=10),
        ))
        fig.add_annotation(x=(x0 + x1) / 2, y=0.985, xref="paper",
                           yref="paper", showarrow=False,
                           text=f"<b>{model}</b>",
                           font=dict(size=13, color=INK))
    fig.update_layout(
        title=dict(text="REPRESENT PCs → ENACT anchors — "
                        "width = transmitted mass, opacity = faithfulness "
                        "sp(in,out)", font=dict(size=14, color=INK)),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
            text="links: anchors with |cos| ≥ 0.25 (top 4 per PC) · "
                 "left labels: input-pole adjectives · washed-out ribbons "
                 "= image reflects ENACT's manifold more than the R-PC "
                 "(report_week17 §12.5) · hover for signed cos + "
                 "image poles",
            font=dict(size=10, color=INK2))],
        width=1500, height=640, margin=dict(l=10, r=10, t=70, b=60),
        paper_bgcolor="#fcfcfb",
    )
    out = FIG / "rpc_flow"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
