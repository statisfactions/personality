"""Reading-group chart: effective-dimension summary (W17).

The cohort-spectra curves are spaghetti at 7 models x 3 instruments. The
punchline is one number per (model, instrument): ENACT collapses (~5-12) while
REPRESENT/JUDGE/HUMAN stay high-dim (14-72). Horizontal dot plot, one row per
model, markers per instrument, HUMAN as a reference band. Reads the eff-dim
table written by viz_cohort_spectra.py.
"""
import csv
from pathlib import Path
import plotly.graph_objects as go
from hf_logprobs import display

FIG = Path("results/persona_vectors/figs")
COLOR = {"REPRESENT": "#4C78A8", "JUDGE": "#F58518", "ENACT": "#E45756"}
MARK = {"REPRESENT": "circle", "JUDGE": "square", "ENACT": "diamond"}


def main():
    rows = list(csv.DictReader(open(FIG / "cohort_spectra_effdim.csv")))
    human = next(float(r["eff_dim"]) for r in rows if r["model"] == "HUMAN")
    models = [r["model"] for r in rows if r["model"] != "HUMAN"]
    order = sorted(set(models),
                   key=lambda m: -next(float(r["eff_dim"]) for r in rows
                                       if r["model"] == m and r["instrument"] == "ENACT"))
    fig = go.Figure()
    # HUMAN reference line
    fig.add_vline(x=human, line=dict(color="black", width=2, dash="dot"))
    fig.add_annotation(x=human, y=1.02, yref="paper", text=f"HUMAN ({human:.0f})",
                       showarrow=False, font=dict(size=12), xanchor="center")
    for inst in ["REPRESENT", "JUDGE", "ENACT"]:
        xs, ys, txt = [], [], []
        for m in order:
            v = next((float(r["eff_dim"]) for r in rows
                      if r["model"] == m and r["instrument"] == inst), None)
            if v is None:
                continue
            xs.append(v); ys.append(m); txt.append(f"{v:.0f}")
        fig.add_trace(go.Scatter(
            x=xs, y=[display(m) for m in ys], mode="markers+text",
            name=inst, text=txt,
            textposition="top center", textfont=dict(size=9),
            marker=dict(symbol=MARK[inst], color=COLOR[inst], size=13,
                        line=dict(width=1, color="white"))))
    fig.update_xaxes(type="log", title="effective dimension (participation ratio, log)",
                     range=[0.5, 2.0])
    fig.update_yaxes(title="")
    fig.update_layout(
        title="Only ENACT collapses: enactment crushes the 523-adjective space "
              "onto ~1 axis;<br>representation, judgment, and humans stay high-"
              "dimensional",
        width=900, height=460, template="plotly_white",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
    out = FIG / "effdim_summary"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html (HUMAN ref={human:.0f})")


if __name__ == "__main__":
    main()
