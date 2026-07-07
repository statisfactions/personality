"""SAYS vs IS figures (W17 §15).

Figure 1 (says_vs_is): 2x5 small-multiples, one panel per ENACT-cohort model.
x = the adjective's projection on the model's massive-ablated ENACT assistant
axis (what the model IS, behaviorally); y = framing-mean SELF EV (what it SAYS
it is). Each panel is annotated with r against its OWN axis and the mean r
against the other nine models' axes — the near-equality is §15.7's
character-sheet result, visible per panel. The 2 largest positive and negative
residuals from the per-panel fit get direct labels (the model's biggest
over- and under-claims).

Figure 2 (says_framings): model x framing heatmap of says-is r (own ablated
axis), the framing-veridicality summary (`outputs` best, `assistant`-anchored
worst).

Reads results/adjectives/selfreport/<m>_self_full.json +
results/persona_vectors/<m>_pda_meta.json.
Usage: PYTHONPATH=scripts python scripts/viz_says_is.py
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIG = Path("results/persona_vectors/figs")
MODELS = ["llama3.2", "Llama8", "qwen2.5", "Qwen7", "Qwen32",
          "gemma3", "Gemma12", "Gemma27", "phi4", "Aya"]
FRAMINGS = ["direct", "assistant", "person", "pda", "observer", "outputs"]
BLUE, INK, INK2 = "#2a78d6", "#0b0b0b", "#52514e"
SURF = "#fcfcfb"


def load():
    says, says_fr, axis, enact = {}, {}, {}, {}
    adjs = None
    for m in MODELS:
        d = json.load(open(f"results/adjectives/selfreport/{m}_self_full.json"))
        if adjs is None:
            adjs = d["adjectives"]
        says_fr[m] = {f: np.array([d["results"][f][a]["ev"] for a in adjs])
                      for f in FRAMINGS}
        says[m] = np.mean([says_fr[m][f] for f in FRAMINGS], axis=0)
        meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
        aa = meta["assistant_axis"]["cos_to_adjectives_ablated"]
        axis[m] = np.array([aa[a] for a in adjs])
        # enactability percentile -> saturation channel (None if not judged yet)
        # raw judged trait presence in the persona rollouts (persona_ev),
        # NOT the baseline-subtracted shift: for this figure the question is
        # "does the trait reach the text", so already-at-ceiling assistant
        # words (helpful) stay saturated; only truly unexpressible ones fade.
        ep = f"results/adjectives/enactability/{m}_enactability.json"
        try:
            sc = json.load(open(ep))["scores"]
            vals = np.array([sc[a]["persona_ev"] for a in adjs])
            enact[m] = np.array([np.mean(vals < v) for v in vals])
        except FileNotFoundError:
            enact[m] = None
    return adjs, says, says_fr, axis, enact


def main():
    adjs, says, says_fr, axis, enact = load()

    # --- figure 1: small-multiples scatter -------------------------------
    fig = make_subplots(rows=2, cols=5, subplot_titles=MODELS,
                        horizontal_spacing=0.035, vertical_spacing=0.13)
    for k, m in enumerate(MODELS):
        row, col = k // 5 + 1, k % 5 + 1
        x, y = axis[m], says[m]
        if enact[m] is not None:
            op = 0.08 + 0.60 * enact[m]          # saturation = enactability pct
            cd = [f"{a} · persona-ev pct {p:.0%}" for a, p in zip(adjs, enact[m])]
        else:
            op = 0.35
            cd = adjs
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4, color=BLUE, opacity=op,
                        line=dict(width=0)),
            customdata=cd, hovertemplate="%{customdata}<br>is %{x:.2f} · "
                                         "says %{y:.2f}<extra></extra>",
            showlegend=False), row=row, col=col)
        r_own = np.corrcoef(x, y)[0, 1]
        r_oth = np.mean([np.corrcoef(axis[o], y)[0, 1]
                         for o in MODELS if o != m])
        ax = "x" if k == 0 else f"x{k+1}"
        ay = "y" if k == 0 else f"y{k+1}"
        fig.add_annotation(text=f"r own {r_own:.2f} · others {r_oth:.2f}",
                           xref=f"{ax} domain", yref=f"{ay} domain",
                           x=0.03, y=0.02, xanchor="left", yanchor="bottom",
                           showarrow=False, font=dict(size=9, color=INK2))
        # label the 2 biggest over- and under-claims vs the panel fit
        b = np.polyfit(x, y, 1)
        resid = y - (b[0] * x + b[1])
        for i in list(np.argsort(resid)[-2:]) + list(np.argsort(resid)[:2]):
            fig.add_annotation(x=x[i], y=y[i], text=adjs[i], showarrow=False,
                               yshift=7, xref=ax, yref=ay,
                               font=dict(size=8, color=INK))
    fig.update_xaxes(title_text="", showgrid=False, zeroline=True,
                     zerolinecolor="#d8d7d0", color=INK2, tickfont=dict(size=8))
    fig.update_yaxes(range=[1, 7.3], showgrid=False, color=INK2,
                     tickfont=dict(size=8))
    for k in (1, 6):
        fig.update_yaxes(title_text="SAYS (self EV)", title_font=dict(size=10),
                         row=(k - 1) // 5 + 1, col=1)
    for c in range(1, 6):
        fig.update_xaxes(title_text="IS (assistant-axis cos)",
                         title_font=dict(size=9), row=2, col=c)
    fig.update_layout(
        title=dict(text="SAYS vs IS per adjective · saturation = raw judged "
                        "trait presence in the persona rollouts (faded = the "
                        "word never reaches the text)",
                   font=dict(size=13, color=INK)),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.09, xref="paper", yref="paper", showarrow=False,
            text="labels: biggest over-/under-claims · 'r own/others': same "
                 "self-report vs own or other models' axes (§15.7) · "
                 "saturation available for llama3.2/qwen2.5/gemma3 so far",
            font=dict(size=10, color=INK2))],
        width=1500, height=650, margin=dict(l=60, r=20, t=70, b=50),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    fig.write_html(FIG / "says_vs_is.html")
    fig.write_image(FIG / "says_vs_is.png", scale=2)

    # --- figure 2: framing veridicality heatmap --------------------------
    Z = [[float(np.corrcoef(axis[m], says_fr[m][f])[0, 1]) for f in FRAMINGS]
         for m in MODELS]
    fig2 = go.Figure(go.Heatmap(
        z=Z, x=FRAMINGS, y=MODELS,
        colorscale=[[0, "#f2f6fc"], [1, "#1d5fb8"]],
        zmin=0.3, zmax=0.85,
        text=[[f"{v:.2f}" for v in row] for row in Z],
        texttemplate="%{text}", textfont=dict(size=10, color=INK),
        colorbar=dict(title="r", thickness=12)))
    fig2.update_layout(
        title=dict(text="says-is veridicality by framing — r(SELF EV, own "
                        "ablated assistant axis)", font=dict(size=13,
                                                             color=INK)),
        width=640, height=440, margin=dict(l=90, r=30, t=60, b=40),
        yaxis=dict(autorange="reversed", color=INK),
        xaxis=dict(color=INK),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    fig2.write_html(FIG / "says_framings.html")
    fig2.write_image(FIG / "says_framings.png", scale=2)
    print("saved says_vs_is + says_framings (.png/.html)")


if __name__ == "__main__":
    main()
