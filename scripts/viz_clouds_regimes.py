"""Adjective clouds (PC1/PC2, colored by human valence) for REPRESENT vs ENACT,
one model per ENACT regime (W17).

Three ENACT regimes by general-factor shape: Gemma (two-axis, λ1/λ2≈1.2),
Llama (mid-compress, ≈2.2), everything-else (single-axis, ≈3-4). Does the
CLOUD shape differ, esp. for Gemma? 3 rows (regime exemplars) x 2 cols
(REPRESENT, ENACT). Classical-MDS 2D embedding of the double-centered cosine
matrix (same matrices as four_grid/pc12: ablated ENACT, adaptive-denoised
REPRESENT). Color = human valence (mean corr with pos_eval minus neg_eval).
"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import four_grid_compare as fg
from adjective_introspection import GROUPS
from hf_logprobs import display

FIG = Path("results/persona_vectors/figs")
REGIME = {"gemma3": "two-axis", "llama3.2": "mid-compress", "qwen2.5": "single-axis"}


def mds(C):
    """Classical-MDS 2D coords + λ1/λ2 from a cosine similarity matrix."""
    C = C.astype(float).copy(); np.fill_diagonal(C, 1.0)
    n = len(C); H = np.eye(n) - 1 / n; G = H @ C @ H
    w, V = np.linalg.eigh((G + G.T) / 2)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    coords = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))
    return coords, w[0] / w[1]


def main():
    H, labels = fg.load_human()
    idx = {l: i for i, l in enumerate(labels)}
    pos = [idx[w] for w in GROUPS["pos_eval"]]
    neg = [idx[w] for w in GROUPS["neg_eval"]]
    val = np.array([np.nanmean(H[i, pos]) - np.nanmean(H[i, neg])
                    for i in range(len(labels))])

    models = list(REGIME)

    def emb(C):
        coords, ratio = mds(C)
        if np.corrcoef(coords[:, 0], val)[0, 1] < 0:
            coords[:, 0] *= -1
        return coords, ratio

    # precompute embeddings + ratios, bake ratios into titles (avoids fragile
    # annotation indexing under colspan)
    cells = [("HUMAN (reference)", H, 1, 1)]
    for r, m in enumerate(models, start=2):
        cells.append((f"{display(m)} — REPRESENT", fg.load_represent(m, labels)[0], r, 1))
        cells.append((f"{display(m)} — ENACT", fg.load_enact(m, labels, ablate=True), r, 2))
    embs = [(t, emb(C), rr, cc) for t, C, rr, cc in cells]
    titles = [f"{t}  (λ1/λ2={e[1]:.1f})" for t, e, _, _ in embs]

    fig = make_subplots(
        rows=len(models) + 1, cols=2, subplot_titles=titles,
        specs=[[{"colspan": 2}, None]] + [[{}, {}]] * len(models),
        horizontal_spacing=0.08, vertical_spacing=0.06)
    for i, (_, (coords, _), r, c) in enumerate(embs):
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode="markers",
            marker=dict(size=4, color=val, colorscale="RdBu_r", cmid=0,
                        showscale=(i == 0),
                        colorbar=dict(title="human<br>valence", len=0.25, y=0.88)),
            text=labels, hoverinfo="text", showlegend=False), r, c)
    fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
    # HUMAN spans both columns (colspan) -> shrink its x-domain to one panel
    # width, centered, so it has the same aspect ratio as the model panels.
    fig.update_xaxes(domain=[0.27, 0.73], row=1, col=1)
    fig.update_layout(
        title="Adjective clouds (PC1×PC2, colored by human valence): HUMAN vs "
              "REPRESENT vs ENACT.<br>HUMAN = valence × (dominance↔warmth); "
              "Llama ENACT recovers it; Gemma adds warmth/competence; Qwen adds "
              "extraversion",
        width=900, height=1180, template="plotly_white")
    out = FIG / "clouds_regimes"
    fig.write_html(f"{out}.html"); fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
