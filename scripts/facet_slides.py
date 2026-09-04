"""Five-space facet slides: HUMAN / SELF / REPRESENT / JUDGE / ENACT (interview aid).

v2 partition: 44 trait blocks (instruments/trait_blocks_44.json — no size
cap, majors included).

One 16:9 slide per space, each a 44-block heatmap (same clusters,
branch order, z scale, and construction as adjective_facet_cohort.py), titled
with the narrative beat it carries. A sixth summary slide shows the
PC1-removed row for all five spaces plus the congruence bar — the
"JUDGE >> ENACT > REPRESENT >> SELF" ranking in one visual.

Reuses adjective_facet_cohort's cached cohort channel sims (no GPU needed).

Usage: PYTHONPATH=scripts python scripts/facet_slides.py
Out:   results/persona_vectors/figs/slides/slide{1..6}_*.png (+ .html)
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import adjective_facet_cohort as afc

OUT = Path("results/persona_vectors/figs/slides")
INK, INK2, SURF = afc.INK, afc.INK2, afc.SURF
DIV = afc.DIV

BEATS = {
    "HUMAN": dict(
        num=1,
        title="Personality measurement is a theory of human variance",
        sub=("Saucier 525-PDA: 700 human self-ratings on 523 adjectives, "
             "aggregated to 44 trait blocks (pole-respecting Ward on the "
             "human correlations; coherence-filtered, no size cap — the "
             "large blocks are the trait majors: warmth, anxiety, anger, "
             "honesty). This block structure is what the Big Five compress."),
    ),
    "SELF": dict(
        num=2,
        title="Ask models the same questions: the population is tiny — "
              "and not human-shaped",
        sub=("Same construction with models as respondents: n = 10 — each "
             "model a 523-vector of self-rating EVs (6 framings averaged), "
             "so the correlation estimate is rank-9. Raw congruence is a "
             "desirability freebie; remove each matrix's top component and "
             "SELF ties REPRESENT at the bottom of the four channels."),
    ),
    "REPRESENT": dict(
        num=3,
        title="Models represent something richer — but human-shaped "
              "mostly in the top component",
        sub=("Residual-stream cosine between adjective activations "
             "(persona framing, mid layer), 10-model cohort mean. More "
             "structure than SELF, but beyond the shared evaluative axis "
             "it organizes traits its own way (eval-antonyms merge, "
             "affect blocks)."),
    ),
    "JUDGE": dict(
        num=4,
        title="Asked about trait relations, models reconstruct human "
              "structure",
        sub=("“How likely is a {X} person to be {Y}?” — EV over "
             "the answer distribution, symmetrized, 10-model cohort mean. "
             "Explicit judgment recovers the human covariance the other "
             "channels lose — including with the top component removed."),
    ),
    "ENACT": dict(
        num=5,
        title="But models play the traits out at lower fidelity",
        sub=("Cosine between persona-vector directions extracted from each "
             "model's own trait-enacting rollouts, 10-model cohort mean. "
             "Enactment keeps some human structure — less than judgment. "
             "Part is difficulty of enacting; part may be intrinsic."),
    ),
}


def build_grids():
    """Replicates adjective_facet_cohort.main()'s data path (no render)."""
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    # 44-block standing partition (v2: no size cap, majors included);
    # the frozen 35-cluster file remains for W17-W18 continuity
    tc = json.load(open("instruments/trait_blocks_44.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    H0 = np.array(h["correlation_matrix"], float)

    def block(S):
        k = len(clusters)
        B = np.zeros((k, k))
        for i, ci in enumerate(clusters):
            ii = [labels.index(m) for m in ci["members"]]
            for j, cj in enumerate(clusters):
                jj = [labels.index(m) for m in cj["members"]]
                sub = S[np.ix_(ii, jj)]
                B[i, j] = (sub[~np.eye(len(ii), dtype=bool)].mean()
                           if i == j else sub.mean())
        return B

    names = []
    for c in clusters:
        hidx = [labels.index(m) for m in c["members"]]
        sub = H0[np.ix_(hidx, hidx)]
        names.append(c["members"][int(np.argmax(sub.mean(1)))])

    Hm = H0.copy()
    np.fill_diagonal(Hm, 0)

    # SELF respondents = models (framings averaged out as a measurement
    # facet — the a priori design; the pooled 10x6=60 construction mixes
    # framing-manipulation variance into "individual differences").
    # n=10 -> the correlation estimate is rank-9; that IS the small-n point.
    resp = []
    for m in afc.MODELS:
        d = json.load(open(f"results/adjectives/selfreport/{m}_self_full.json"))
        resp.append(np.mean([[d["results"][f][a]["ev"] for a in labels]
                             for f in afc.FRAMINGS], axis=0))
    S = np.corrcoef(np.array(resp).T)
    np.fill_diagonal(S, 0)

    cz = np.load(afc.CACHE)
    mats = {"HUMAN": afc.zscore_offdiag(Hm), "SELF": afc.zscore_offdiag(S),
            **{ch: cz[ch] for ch in ("REPRESENT", "JUDGE", "ENACT")}}
    grids = {ch: block(M) for ch, M in mats.items()}
    grids_p = {ch: block(afc.zscore_offdiag(afc.remove_pc1(M)))
               for ch, M in mats.items()}
    return grids, grids_p, names, clusters


def congruence(grids, ch):
    k = grids[ch].shape[0]
    m = ~np.eye(k, dtype=bool)
    return np.corrcoef(grids[ch][m], grids["HUMAN"][m])[0, 1]


def wrap(s, width=125):
    lines, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


def slide(ch, grids, grids_p, names, branches):
    beat = BEATS[ch]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=grids[ch][::-1], x=names, y=names[::-1],
                             colorscale=DIV, zmin=-2, zmax=2,
                             colorbar=dict(thickness=12, len=0.7, x=0.72,
                                           tickfont=dict(size=10, color=INK2),
                                           title=dict(text="z", font=dict(
                                               size=10, color=INK2)))))
    # branch separators
    edges = np.cumsum([0] + [sum(1 for b in branches if b == br)
                             for br in sorted(set(branches))])
    n = len(names)
    for e in edges[1:-1]:
        fig.add_shape(type="line", x0=e - 0.5, x1=e - 0.5, y0=-0.5,
                      y1=n - 0.5, line=dict(color=INK2, width=0.6))
        fig.add_shape(type="line", y0=n - e - 0.5, y1=n - e - 0.5, x0=-0.5,
                      x1=n - 0.5, line=dict(color=INK2, width=0.6))
    stats = ""
    if ch != "HUMAN":
        stats = (f"r(HUMAN) = {congruence(grids, ch):.2f} raw   |   "
                 f"{congruence(grids_p, ch):.2f} with top component removed")
    fig.add_annotation(text=stats, xref="paper", yref="paper", x=0.475,
                       y=1.022, xanchor="center", yanchor="bottom",
                       showarrow=False, font=dict(size=15, color=INK))
    fig.add_annotation(text=wrap(beat["sub"]), xref="paper", yref="paper",
                       x=0, y=1.155, xanchor="left", yanchor="top",
                       align="left", showarrow=False,
                       font=dict(size=14, color=INK2))
    fig.update_layout(
        title=dict(text=f"<b>{beat['title']}</b>",
                   font=dict(size=22, color=INK), x=0.0375, y=0.965),
        width=1600, height=900,
        margin=dict(l=60, r=60, t=150, b=140),
        paper_bgcolor=SURF, plot_bgcolor=SURF,
        xaxis=dict(tickfont=dict(size=9, color=INK2), tickangle=45,
                   domain=[0.28, 0.67], constrain="domain"),
        yaxis=dict(tickfont=dict(size=9, color=INK2), scaleanchor="x",
                   constrain="domain"))
    fig.add_annotation(text=f"{ch}", xref="paper", yref="paper", x=0.90,
                       y=0.5, xanchor="center", showarrow=False,
                       font=dict(size=30, color=INK2))
    return fig


def summary_slide(grids_p, grids=None, cap=None):
    order = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
    fig = make_subplots(rows=1, cols=6, column_titles=order + ["congruence"],
                        horizontal_spacing=0.015,
                        specs=[[{}] * 5 + [{"type": "bar"}]])
    for ci, ch in enumerate(order):
        fig.add_trace(go.Heatmap(z=grids_p[ch][::-1], colorscale=DIV,
                                 zmin=-2, zmax=2, showscale=False),
                      row=1, col=ci + 1)
        if ch != "HUMAN":
            r = congruence(grids_p, ch)
            fig.add_annotation(text=f"r={r:.2f}", xref="x domain",
                               yref="y domain", x=0.97, y=-0.13,
                               xanchor="right", showarrow=False,
                               font=dict(size=13, color=INK),
                               row=1, col=ci + 1)
    chans = ["SELF", "REPRESENT", "JUDGE", "ENACT"]
    cols = ["#a08c5b", "#1d5fb8", "#2e7d43", "#c93a3a"]
    faded = ["rgba({},{},{},0.30)".format(int(c[1:3], 16), int(c[3:5], 16),
                                          int(c[5:7], 16)) for c in cols]
    if grids is not None:  # before (raw, faded) behind after (pc1-removed)
        fig.add_trace(go.Bar(x=chans,
                             y=[congruence(grids, c) for c in chans],
                             marker_color=faded, name="raw"), row=1, col=6)
    fig.add_trace(go.Bar(x=chans, y=[congruence(grids_p, c) for c in chans],
                         marker_color=cols, name="top removed"), row=1, col=6)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=11), row=1, col=6)
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=10),
                     range=[0, 1], row=1, col=6)
    _top = (cap["paras"][0] if cap else
            "All five 44-block grids with the top eigencomponent of the "
            "underlying 523² matrix removed; bars = congruence with HUMAN "
            "(off-diagonal r).")
    fig.add_annotation(text=_top,
                       xref="paper", yref="paper", x=0, y=1.21,
                       xanchor="left", yanchor="top", align="left",
                       showarrow=False, font=dict(size=13, color=INK2))
    _bot = (cap["paras"][1] if cap and len(cap["paras"]) > 1 else
            "Aggregation-free check — full-resolution 523² item level, top "
            "component removed: SELF 0.20, REPRESENT 0.31, JUDGE 0.54, "
            "ENACT 0.49. Same ranking; blocks are visualization, not "
            "load-bearing.")
    fig.add_annotation(text=_bot,
                       xref="paper", yref="paper", x=0, y=-0.22,
                       xanchor="left", yanchor="top", align="left",
                       showarrow=False, font=dict(size=12, color=INK2))
    fig.update_layout(
        title=dict(text="<b>" + (cap["title"] if cap else
                        "Remove each space's top component: judgment "
                        "reconstructs human structure; enactment trails; "
                        "self-report and representation weakest") + "</b>",
                   font=dict(size=19, color=INK), x=0.019, y=0.945),
        width=1600, height=500, margin=dict(l=30, r=30, t=120, b=80),
        paper_bgcolor=SURF, plot_bgcolor=SURF, showlegend=False)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grids, grids_p, names, clusters = build_grids()
    branches = [c["branch"] for c in clusters]
    for ch in ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]:
        fig = slide(ch, grids, grids_p, names, branches)
        stem = OUT / f"slide{BEATS[ch]['num']}_{ch.lower()}"
        fig.write_html(f"{stem}.html")
        fig.write_image(f"{stem}.png", scale=2)
        print(f"saved {stem}.png")
    fig = summary_slide(grids_p)
    fig.write_html(OUT / "slide6_summary.html")
    fig.write_image(OUT / "slide6_summary.png", scale=2)
    print(f"saved {OUT}/slide6_summary.png")
    for ch in ["SELF", "REPRESENT", "JUDGE", "ENACT"]:
        print(f"  {ch:9s} raw r={congruence(grids, ch):.3f}  "
              f"pc1-removed r={congruence(grids_p, ch):.3f}")


if __name__ == "__main__":
    main()
