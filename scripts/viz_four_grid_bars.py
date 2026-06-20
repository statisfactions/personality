"""Reading-group chart: four-grid HUMAN-match bars (W17).

Per model, how well each channel's 523x523 structure matches the human one
(double-centered Spearman). Grouped bars REPRESENT/JUDGE/ENACT. Headline:
JUDGE and ENACT match humans well (0.6-0.8), REPRESENT lags (0.43-0.58); the
merge/gaps live in representation. Reads four_grid_<model>.json.
"""
import json
from pathlib import Path
import plotly.graph_objects as go
from hf_logprobs import display

FIG = Path("results/persona_vectors/figs")
PV = Path("results/persona_vectors")
COLOR = {"REPRESENT": "#4C78A8", "JUDGE": "#F58518", "ENACT": "#E45756"}
STAT = "double_centered"   # interaction structure, prevalence stripped


def main():
    data = {}
    for p in sorted(PV.glob("four_grid_*.json")):
        m = p.stem.replace("four_grid_", "")
        pairs = json.load(open(p))["main"]["pairs"]
        data[m] = {ch: pairs[f"HUMAN|{ch}"][STAT]["spearman"]
                   for ch in COLOR}
    # order by ENACT match (shows the family pattern)
    order = sorted(data, key=lambda m: -data[m]["ENACT"])
    fig = go.Figure()
    for ch in COLOR:
        fig.add_trace(go.Bar(
            name=ch, marker_color=COLOR[ch],
            x=[display(m) for m in order],
            y=[round(data[m][ch], 3) for m in order],
            text=[f"{data[m][ch]:.2f}" for m in order],
            textposition="outside", textfont=dict(size=9)))
    fig.update_yaxes(title="match to human structure (double-centered Spearman)",
                     range=[0, 0.85])
    fig.update_layout(
        title="Models judge & enact human trait structure better than they "
              "represent it<br>(REPRESENT lags — the merge/gaps live in "
              "representation)",
        barmode="group", width=950, height=500, template="plotly_white",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        bargap=0.25, bargroupgap=0.05)
    out = FIG / "four_grid_humanmatch"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html ({len(data)} models)")
    for m in order:
        print(f"  {display(m):>14s}: R {data[m]['REPRESENT']:.2f}  "
              f"J {data[m]['JUDGE']:.2f}  E {data[m]['ENACT']:.2f}")


if __name__ == "__main__":
    main()
