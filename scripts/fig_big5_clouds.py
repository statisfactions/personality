"""Two-object Big5 comparison figure (2026-09-06 design, ledgered).

Three glyphs: (1) elevation strip — raw mean self-rating per respondent;
(2) shape cloud — ipsatized profiles of both populations scored through
the FIXED raw-human varimax-5 ruler (unweighted; A/C/O axes, the
treatment-stable rulers); (3) stacked deviation-norm bars — median
norm of deviation from own population mean, split in-/off-Big5.

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
Mi, Ri = pkit.measures.ipsatize(M), pkit.measures.ipsatize(R.values)

w, v = pkit.axes.eig_axes(M, kmax=5)
L = pkit.axes.varimax(v * np.sqrt(w))
# symmetric (Loewdin) orthogonalization: varimax loading columns are NOT
# orthogonal (unequal sqrt-eigenvalue scaling + rotation; cos up to .47
# here) — this is the closest orthonormal frame to them, order-independent
U, sv, Vt = np.linalg.svd(L, full_matrices=False)
Q = U @ Vt
adj = list(R.columns)
MARKERS = {"A": "kind-hearted", "E": "exciting", "N": "troubled",
           "C": "thorough", "O": "intelligent"}
axmap = {}
for name, m_ in MARKERS.items():
    i = adj.index(m_)
    k = int(np.abs(Q[i]).argmax())
    if Q[i, k] < 0:
        Q[:, k] = -Q[:, k]
    axmap[name] = k
Sh, Sm = Mi @ Q, Ri @ Q                      # unweighted, common frame

fig = make_subplots(
    rows=1, cols=3, column_widths=[0.16, 0.56, 0.22],
    specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "xy"}]],
    subplot_titles=["elevation<br>(raw mean rating)",
                    "shape cloud — fixed human Big5 ruler (A / C / O)",
                    "deviation norm<br>(in vs off Big5)"])

# (1) elevation strip
rng = np.random.default_rng(0)
fig.add_trace(go.Scatter(x=rng.normal(0, .06, len(M)), y=M.mean(1), mode="markers",
                         marker=dict(size=3, color=HCOL, opacity=.25),
                         name="human (n=700)"), row=1, col=1)
fig.add_trace(go.Scatter(x=1 + rng.normal(0, .06, len(R)), y=R.values.mean(1),
                         mode="markers", marker=dict(size=5, color=MCOL, opacity=.8),
                         name=f"model (n={len(R)})"), row=1, col=1)
fig.update_xaxes(tickvals=[0, 1], ticktext=["human", "model"], row=1, col=1)
fig.update_yaxes(title_text="mean rating (1-7)", row=1, col=1)

# (2) A/C/O cloud
ka, kc, ko = axmap["A"], axmap["C"], axmap["O"]
fig.add_trace(go.Scatter3d(x=Sh[:, ka], y=Sh[:, kc], z=Sh[:, ko], mode="markers",
                           marker=dict(size=2, color=HCOL, opacity=.25),
                           name="human", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter3d(x=Sm[:, ka], y=Sm[:, kc], z=Sm[:, ko], mode="markers",
                           marker=dict(size=4, color=MCOL, opacity=.9),
                           text=list(R.index), hoverinfo="text",
                           name="model", showlegend=False), row=1, col=2)
for S, col, nm in [(Sh, "#444444", "human centroid"),
                   (Sm, "#7a2020", "model centroid")]:
    fig.add_trace(go.Scatter3d(x=[S[:, ka].mean()], y=[S[:, kc].mean()],
                               z=[S[:, ko].mean()], mode="markers+text",
                               marker=dict(size=9, color=col, symbol="diamond"),
                               text=[nm], textposition="top center",
                               textfont=dict(size=11, color=col),
                               showlegend=False), row=1, col=2)
print("centroid offset (model - human): "
      f"A {Sm[:, ka].mean()-Sh[:, ka].mean():+.2f}  "
      f"C {Sm[:, kc].mean()-Sh[:, kc].mean():+.2f}  "
      f"O {Sm[:, ko].mean()-Sh[:, ko].mean():+.2f}")
fig.update_scenes(xaxis_title="A (kind-hearted+)", yaxis_title="C (thorough+)",
                  zaxis_title="O (intelligent+)",
                  camera=dict(eye=dict(x=1.7, y=-1.5, z=0.7)),
                  aspectmode="cube")

# (3) stacked deviation-norm bars
def parts(Z):
    D = Z - Z.mean(0)
    on = D @ Q
    off = D - on @ Q.T
    return (np.median(np.linalg.norm(on, axis=1)),
            np.median(np.linalg.norm(off, axis=1)))
hon, hoff = parts(Mi)
mon, moff = parts(Ri)
fig.add_trace(go.Bar(x=["human", "model"], y=[hon, mon], name="in-Big5",
                     marker_color="#1d5fb8"), row=1, col=3)
fig.add_trace(go.Bar(x=["human", "model"], y=[hoff, moff], name="off-Big5",
                     marker_color="#9db8dd"), row=1, col=3)
for xi, (a, b) in enumerate([(hon, hoff), (mon, moff)]):
    fig.add_annotation(text=f"{b/(a+b):.0%} off", x=xi, y=a + b, yshift=10,
                       showarrow=False, font=dict(size=11), row=1, col=3)
fig.update_layout(barmode="stack", width=1500, height=560,
                  paper_bgcolor="#f5f4ef", plot_bgcolor="#f5f4ef",
                  title=dict(text="<b>Two objects: level and shape — models vs humans on the human Big5 ruler</b>",
                             x=0.02, font=dict(size=18)),
                  legend=dict(orientation="h", y=-0.12))
fig.write_html(OUT + ".html")
fig.write_image(OUT + ".png", scale=2)
print("wrote", OUT + ".{html,png}")
print(f"in/off norms: human {hon:.1f}/{hoff:.1f}  model {mon:.1f}/{moff:.1f}")
