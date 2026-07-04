"""PC1 vs PC2 shape of all four grids across the cohort (W17).

For each (model, channel) and HUMAN: eigenvalues of the double-centered 523x523
similarity matrix -> PC1%, PC2% of variance. Scatter colored by channel, HUMAN
a black star, y=x diagonal (= no general factor, λ1=λ2). Shows: REPRESENT hugs
the diagonal (diffuse, no general factor, model-invariant); JUDGE sits with
HUMAN (human-like general factor); ENACT far right (dominant axis) but its shape
VARIES by model (Gemma3 near-diagonal = two-axis, the valence-flattening
signature).
"""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import four_grid_compare as fg
from hf_logprobs import display

FIG = Path("results/persona_vectors/figs")
COLOR = {"REPRESENT": "#4C78A8", "JUDGE": "#F58518", "ENACT": "#E45756"}
MODELS = ["llama3.2", "Llama8", "qwen2.5", "Qwen7", "Qwen32", "gemma3",
          "Gemma12", "Gemma27", "phi4", "Aya"]


def pc12(M, diag):
    M = M.astype(float).copy(); iu = ~np.eye(len(M), dtype=bool)
    M[np.isnan(M) & iu] = np.nanmean(M[iu]); np.fill_diagonal(M, diag)
    n = len(M); C = np.eye(n) - 1 / n; Md = C @ M @ C
    w = np.linalg.eigvalsh((Md + Md.T) / 2)
    pos = np.sort(w[w > 1e-9])[::-1]; tot = pos.sum()
    return 100 * pos[0] / tot, 100 * pos[1] / tot


def main():
    H, labels = fg.load_human()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 45], y=[0, 45], mode="lines",
                             line=dict(color="lightgray", dash="dash"),
                             name="λ1=λ2 (no general factor)", hoverinfo="skip"))
    for ch in COLOR:
        xs, ys, tx = [], [], []
        for m in MODELS:
            M = (fg.load_represent(m, labels)[0] if ch == "REPRESENT"
                 else fg.load_judge(m, labels) if ch == "JUDGE"
                 else fg.load_enact(m, labels, ablate=True))
            diag = 7.0 if ch == "JUDGE" else 1.0
            p1, p2 = pc12(M, diag)
            xs.append(p1); ys.append(p2); tx.append(display(m))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=ch,
            marker=dict(color=COLOR[ch], size=11, line=dict(width=1, color="white")),
            text=tx, hovertemplate="%{text}<br>PC1=%{x:.1f}% PC2=%{y:.1f}%<extra></extra>"))
        # label the ENACT outlier (Gemma) and spread
        if ch == "ENACT":
            gi = MODELS.index("gemma3")
            fig.add_annotation(x=xs[gi], y=ys[gi], text="Gemma3 (two-axis)",
                               font=dict(size=10), ax=30, ay=-20)
    h1, h2 = pc12(H, 1.0)
    fig.add_trace(go.Scatter(x=[h1], y=[h2], mode="markers+text", name="HUMAN",
                             marker=dict(color="black", symbol="star", size=18),
                             text=["HUMAN"], textposition="bottom center"))
    fig.update_xaxes(title="PC1 (% variance) — general-factor dominance", range=[0, 45])
    fig.update_yaxes(title="PC2 (% variance)", range=[0, 20])
    fig.update_layout(
        title="General-factor shape by grid: REPRESENT hugs the diagonal "
              "(diffuse), JUDGE sits with HUMAN,<br>ENACT is single-axis "
              "(except Gemma) — REPRESENT/JUDGE shape is model-invariant, "
              "ENACT's varies",
        width=950, height=560, template="plotly_white",
        legend=dict(y=0.98, x=0.98, xanchor="right"))
    out = FIG / "pc12_shape"
    fig.write_html(f"{out}.html"); fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html  (HUMAN PC1={h1:.1f}% PC2={h2:.1f}%)")


if __name__ == "__main__":
    main()
