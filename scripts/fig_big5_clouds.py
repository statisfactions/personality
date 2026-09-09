"""Two-object Big5 comparison figure (v9: Ten Berge frame + treatment toggle).

Three glyphs: (1) elevation strip — raw mean self-rating per respondent;
(2) cloud — both populations scored through a FIXED human varimax-5 ruler
whose loadings are orthogonalized to the constant vector (Ten Berge 1999:
"partialling the mean as an alternative to ipsatization"), so scores are
elevation-invariant and the ipsative constraint plane never forms;
(3) stacked deviation-norm bars (C&G scatter, split in-/off-Big5).

Cloud treatments (dropdown, with axis triples): "shape" = ipsatized
respondents (C&G shape; amplitude-normalized) and "raw" = human-mean-
centered profiles (elevation-blind by the ruler, but amplitude left in —
the view that shows the model amplitude deficit directly).

Usage: .venv/bin/python scripts/fig_big5_clouds.py
Out:   results/persona_vectors/figs/fig_big5_clouds.{html,png}
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pkit

OUT = "results/persona_vectors/figs/fig_big5_clouds"
HCOL, MCOL = "#8a8a8a", "#c93a3a"

M, lab = pkit.load.load_human()
lab_l = [l.lower() for l in lab]
R = pkit.load.self_matrix(which="cohort")
ix = [lab_l.index(a) for a in R.columns]
M = M[:, ix]
adj = list(R.columns)
Mi, Ri = pkit.measures.ipsatize(M), pkit.measures.ipsatize(R.values)

# ---- ruler: raw-human varimax-5, mean-partialled (Ten Berge), Loewdin ----
w, v = pkit.axes.eig_axes(M, kmax=5)
L = pkit.axes.varimax(v * np.sqrt(w))
u = np.ones(len(adj)) / np.sqrt(len(adj))
Lp = L - np.outer(u, u @ L)
Uo, sv, Vt = np.linalg.svd(Lp, full_matrices=False)
Q = Uo @ Vt                                   # orthonormal, all columns ⊥ u
MARKERS = {"A": "kind-hearted", "E": "exciting", "N": "troubled",
           "C": "thorough", "O": "intelligent"}
axmap = {}
for name, m_ in MARKERS.items():
    i = adj.index(m_)
    k = int(np.abs(Q[i]).argmax())
    if Q[i, k] < 0:
        Q[:, k] = -Q[:, k]
    axmap[name] = k

# ---- treatments: shape (ipsatized) and raw (human-mean-centered) --------
def scored(Zh, Zm):
    Sh, Sm = Zh @ Q, Zm @ Q
    hmu, hsd = Sh.mean(0), Sh.std(0)
    return (Sh - hmu) / hsd, (Sm - hmu) / hsd, hmu, hsd

TREAT = {}
TREAT["shape"] = scored(Mi, Ri)
TREAT["raw"] = scored(M - M.mean(0), R.values - M.mean(0))

# core/degraded roster: fixed, from the shape treatment
Sh0, Sm0, _, _ = TREAT["shape"]
med = np.median(Sm0, 0)
mad = np.median(np.abs(Sm0 - med), 0) * 1.4826
core = np.sqrt((((Sm0 - med) / mad) ** 2).sum(1)) < 3.5

def view_sets(tr):
    """The six scene point-sets for one treatment, in trace order."""
    Sh, Sm, hmu, hsd = TREAT[tr]
    flat = (np.zeros_like(u) - (M - M.mean(0)).mean(0)) if tr == "raw" else None
    # flat (shapeless) profile: scores of a constant profile. Q ⊥ u makes it
    # c-independent: shape treatment -> -hmu/hsd; raw -> (-human mean)@Q std.
    if tr == "shape":
        fp = -hmu / hsd
    else:
        fp = ((-(M.mean(0) - M.mean())) @ Q - hmu) / hsd
    return [Sh, Sm[core], Sm[~core], fp[None, :],
            Sh.mean(0, keepdims=True), Sm[core].mean(0, keepdims=True)]

fig = make_subplots(
    rows=1, cols=3, column_widths=[0.26, 0.50, 0.18],
    specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "xy"}]],
    subplot_titles=["elevation × scatter (C&G)",
                    "cloud — fixed human Big5 ruler, elevation-invariant "
                    "(Ten Berge)",
                    "deviation norm, relative<br>to human median"])

# (1) C&G elevation x scatter plane (per respondent: mean rating vs
# within-respondent SD); model whiskers = elevation +/- SD, the rating range
h_el, h_sd = M.mean(1), M.std(1)
m_el, m_sd = R.values.mean(1), R.values.std(1)
fig.add_trace(go.Scatter(x=h_el, y=h_sd, mode="markers",
                         marker=dict(size=3, color=HCOL, opacity=.25),
                         name="human (n=700)"), row=1, col=1)
# whiskers first (under the dots), core models only to limit clutter
fig.update_xaxes(title_text="elevation (mean rating, 1-7)", row=1, col=1)
fig.update_yaxes(title_text="scatter (within-respondent SD)", row=1, col=1)

# panel 1 model layers (need the core/degraded split)
for i in np.where(core)[0]:
    fig.add_trace(go.Scatter(x=[m_el[i] - m_sd[i], m_el[i] + m_sd[i]],
                             y=[m_sd[i], m_sd[i]], mode="lines",
                             line=dict(color="rgba(201,58,58,.25)", width=1),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=m_el[core], y=m_sd[core], mode="markers",
                         marker=dict(size=6, color=MCOL, opacity=.9),
                         text=[n for n, c in zip(R.index, core) if c],
                         hoverinfo="text",
                         name=f"model core (n={core.sum()})"), row=1, col=1)
fig.add_trace(go.Scatter(x=m_el[~core], y=m_sd[~core], mode="markers",
                         marker=dict(size=7, color="rgba(201,58,58,0)",
                                     line=dict(color="#8a4444", width=2),
                                     symbol="circle-open"),
                         text=[n for n, c in zip(R.index, core) if not c],
                         hoverinfo="text",
                         name="degraded"), row=1, col=1)

# (2) cloud — default view: shape · A/C/O
ps = view_sets("shape")
ka, kc, ko = axmap["A"], axmap["C"], axmap["O"]
styles = [
    dict(marker=dict(size=2, color=HCOL, opacity=.25), name="human"),
    dict(marker=dict(size=4, color=MCOL, opacity=.9), name="model (core)",
         text=[n for n, c in zip(R.index, core) if c], hoverinfo="text"),
    dict(marker=dict(size=4, color="rgba(201,58,58,0)",
                     line=dict(color="#b98080", width=2),
                     symbol="circle-open"), name="degraded instruments",
         text=[n for n, c in zip(R.index, core) if not c], hoverinfo="text"),
    dict(marker=dict(size=3, color="#6b6b6b", symbol="cross"),
         mode="markers+text", text=["flat profile (no shape)"],
         textposition="bottom center",
         textfont=dict(size=10, color="#6b6b6b")),
    dict(marker=dict(size=9, color="#444444", symbol="diamond"),
         mode="markers+text", text=["human centroid"],
         textposition="bottom center",
         textfont=dict(size=11, color="#444444")),
    dict(marker=dict(size=9, color="#7a2020", symbol="diamond"),
         mode="markers+text", text=["core-model centroid"],
         textposition="top center", textfont=dict(size=11, color="#7a2020")),
]
for S, st in zip(ps, styles):
    st.setdefault("mode", "markers")
    fig.add_trace(go.Scatter3d(x=S[:, ka], y=S[:, kc], z=S[:, ko],
                               showlegend=False, **st), row=1, col=2)
fig.update_scenes(xaxis_title="A (human SDs)", yaxis_title="C (human SDs)",
                  zaxis_title="O (human SDs)",
                  camera=dict(eye=dict(x=1.7, y=-1.5, z=0.7)),
                  aspectmode="cube")
for tr in ("shape", "raw"):
    Sh, Sm, _, _ = TREAT[tr]
    print(f"[{tr}] core centroid (human SDs): "
          + " ".join(f"{a}:{Sm[core, axmap[a]].mean():+.2f}" for a in "ACO")
          + " | core SD ratios: "
          + " ".join(f"{a}:{Sm[core, axmap[a]].std():.2f}" for a in "AENCO"))

# dropdown: 5 triples x 2 treatments
scene_trace_idx = [i for i, t in enumerate(fig.data) if t.type == "scatter3d"]
TRIPLES = [("A", "C", "O"), ("A", "E", "N"), ("E", "N", "O"),
           ("A", "C", "N"), ("C", "E", "O")]
buttons = []
for tr in ("shape", "raw"):
    sets_ = view_sets(tr)
    for trip in TRIPLES:
        ks = [axmap[a] for a in trip]
        data_update = {c: [S[:, k].tolist() for S in sets_]
                       for c, k in zip(("x", "y", "z"), ks)}
        layout_update = {f"scene.{ax}axis.title.text": f"{a} (human SDs)"
                         for ax, a in zip(("x", "y", "z"), trip)}
        buttons.append(dict(label=f"{'/'.join(trip)} · {tr}", method="update",
                            args=[data_update, layout_update,
                                  scene_trace_idx]))
fig.update_layout(updatemenus=[dict(buttons=buttons, x=0.25, y=0.90,
                                    xanchor="left", showactive=True)])

# (3) stacked deviation-norm bars (C&G scatter split; shape frame)
def parts(Z):
    D = Z - Z.mean(0)
    on = D @ Q
    off = D - on @ Q.T
    return (np.median(np.linalg.norm(on, axis=1)),
            np.median(np.linalg.norm(off, axis=1)))
hon, hoff = parts(Mi)
mon, moff = parts(Ri[core])
scale = hon + hoff
hon, hoff, mon, moff = hon / scale, hoff / scale, mon / scale, moff / scale
xlab = ["human", f"model core (n={core.sum()})"]
fig.add_trace(go.Bar(x=xlab, y=[hon, mon], name="in-Big5",
                     marker_color="#1d5fb8"), row=1, col=3)
fig.add_trace(go.Bar(x=xlab, y=[hoff, moff], name="off-Big5",
                     marker_color="#9db8dd"), row=1, col=3)
for xi, (a, b) in enumerate([(hon, hoff), (mon, moff)]):
    fig.add_annotation(text=f"{b/(a+b):.0%} off", x=xi, y=a + b, yshift=10,
                       showarrow=False, font=dict(size=11), row=1, col=3)
fig.add_annotation(
    text=("E's loading vector is the only one that differs materially between "
          "raw- and ipsatized-derived solutions (congruence .67 vs .77–.90 "
          "for A/C/O/N);<br>E positions are correspondingly convention-"
          "dependent. Ruler loadings are mean-partialled (Ten Berge 1999), "
          "so all views are elevation-invariant; 'raw' keeps amplitude in. "
          "Cloud positions equal the gain model's implied trait scores "
          "exactly (shape = direction convention, raw = amplitude-in; "
          "r=1.0000 per axis) — the table and the cloud are one object."),
    xref="paper", yref="paper", x=0.30, y=-0.24, xanchor="left",
    showarrow=False, font=dict(size=11, color="#666666"))
fig.update_layout(barmode="stack", width=1500, height=560,
                  paper_bgcolor="#f5f4ef", plot_bgcolor="#f5f4ef",
                  title=dict(text="<b>Two objects: level and shape — models "
                                  "vs humans on the human Big5 ruler</b>",
                             x=0.02, font=dict(size=18)),
                  legend=dict(orientation="h", y=-0.12))
fig.write_html(OUT + ".html")
fig.write_image(OUT + ".png", scale=2)
print("wrote", OUT + ".{html,png}")
