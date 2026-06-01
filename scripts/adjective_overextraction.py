#!/usr/bin/env python3
"""C&C-style over-extraction diagnostics on the adjective geometry (W13 §3.11).

Two analyses from Cutler & Condon (2022) supplement, run on human 525-PDA vs the
LLM cohort, extended from their k≤5 to our k=1→10:

1. Rotation-stability curve (their Figs 6–8, p.16–17). For each k, varimax-rotate
   the top-k components and measure Tucker congruence back to the UNROTATED
   (nested, variance-ordered) components, matched 1:1. High congruence = rotation
   only tilts axes (real structure). Falling congruence as k grows = rotation must
   scramble the hierarchy to manufacture factors = "obfuscation of the unrotated
   structure after rotation" = over-extraction. Their human result: identical
   (≥.95) through k=2, degrades after 3.

2. Bass-ackwards tree (their Fig 2, p.12; Goldberg 2006). Extract k=1..10 varimax
   components; edges = correlation between a level-k factor's loadings and a
   level-(k+1) factor's loadings across the 523 adjectives (|r|≥.3 shown, primary
   edge solid). Box = top adjective + variance fraction. Shows whether the model,
   like the human, shows "list-type" stability (early factors preserved, new
   factors split off the smallest) or reshuffles.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_overextraction.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adjective_factor_heatmap import varimax
from adjective_geom import model_matrix

KMAX = 10
EPS = 1e-9


def _phi(C, k):
    """Top-k eigen-loadings of C (variance-scaled), unrotated."""
    C = np.nan_to_num(C); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    return V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None))


def _signfix_sort(L):
    """Make each column's peak loading positive; sort columns by variance desc."""
    for j in range(L.shape[1]):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    var = (L ** 2).sum(0)
    o = np.argsort(-var)
    return L[:, o], var[o]


def rotate(Phi, kaiser=True):
    """Varimax with Kaiser normalization by default (SPSS / C&C convention)."""
    if Phi.shape[1] < 2:
        return Phi
    return varimax(Phi.copy(), normalize=kaiser)


def _tucker(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def rotation_profile(C, kaiser=True):
    """prof[k] = array of length k: |Tucker congruence| of each UNROTATED component
    (indexed by variance rank) to its best varimax match. Low at rank 0 = the
    general factor got split by rotation; low in the tail = over-extraction."""
    prof = {}
    for k in range(2, KMAX + 1):
        U = _phi(C, k)
        R = rotate(U, kaiser=kaiser)
        M = np.array([[abs(_tucker(U[:, i], R[:, j])) for j in range(k)]
                      for i in range(k)])                      # i=unrotated, j=rotated
        cong, used = np.zeros(k), set()
        for i in np.argsort(-M.max(1)):           # resolve strongest rows first
            j = max((jj for jj in range(k) if jj not in used), key=lambda jj: M[i, jj])
            used.add(j); cong[i] = M[i, j]
        prof[k] = cong
    return prof


def levels(C):
    """levels[k-1] = (L 523xk varimax, var k). Sign-fixed, variance-sorted."""
    out = []
    for k in range(1, KMAX + 1):
        L = rotate(_phi(C, k)) if k > 1 else _phi(C, 1)
        out.append(_signfix_sort(L))
    return out


def _zc(A):
    return (A - A.mean(0)) / (A.std(0) + EPS)


def bass_ackwards(C, labels):
    """nodes[(k,j)]=dict(var, top, x); edges=[(k, parent_j, child_j, r, primary)]."""
    lv = levels(C)
    n = C.shape[0]
    nodes = {}
    for k, (L, var) in enumerate(lv, 1):
        for j in range(k):
            nodes[(k, j)] = dict(var=float(var[j]), top=labels[int(np.argmax(L[:, j]))])
    edges, parent = [], {}
    for k in range(2, KMAX + 1):
        Rp = _zc(lv[k - 2][0]); Rc = _zc(lv[k - 1][0])
        Rmat = (Rp.T @ Rc) / n                      # (k-1) x k correlation
        for j in range(k):
            col = Rmat[:, j]; p = int(np.argmax(np.abs(col)))
            parent[(k, j)] = p
            edges.append((k, p, j, float(col[p]), True))
            for i in range(k - 1):
                if i != p and abs(col[i]) >= 0.30:
                    edges.append((k, i, j, float(col[i]), False))
    # lineage layout: x of each node ~ near its primary parent
    xpos = {(1, 0): 0.5}
    for k in range(2, KMAX + 1):
        kids = sorted(range(k), key=lambda j: (xpos[(k - 1, parent[(k, j)])],
                                               -nodes[(k, j)]["var"]))
        for idx, j in enumerate(kids):
            xpos[(k, j)] = (idx + 0.5) / k
    for (k, j), x in xpos.items():
        nodes[(k, j)]["x"] = x
    # lineage = which k=2 factor each node descends from (via primary parents)
    lineage = {(1, 0): -1}
    for k in range(2, KMAX + 1):
        for j in range(k):
            ak, aj = k, j
            while ak > 2:
                aj = parent[(ak, aj)]; ak -= 1
            lineage[(k, j)] = aj
    return nodes, edges, lineage


# ---- figure 1: per-component preservation profile (over-extraction) ----
def fig_profile(H, Cs):
    """Heatmap of congruence-to-unrotated per (k, component variance-rank), human
    vs cohort-mean. A wide band of high cells = supported dimensions; a low rank-1
    cell = general-factor splitting; a low tail = over-extraction. Returns the
    per-k mean profiles (human, cohort)."""
    ph = rotation_profile(H)
    cohort = [rotation_profile(C) for C in Cs.values()]
    pm = {k: np.mean([c[k] for c in cohort], 0) for k in range(2, KMAX + 1)}

    def grid(prof):
        G = np.full((KMAX - 1, KMAX), np.nan)
        for k in range(2, KMAX + 1):
            G[k - 2, :k] = prof[k]
        return G

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("HUMAN 525-PDA", "LLM cohort (mean)"))
    for col, prof in ((1, ph), (2, pm)):
        G = grid(prof)
        fig.add_trace(go.Heatmap(
            z=G, x=[f"PC{i+1}" for i in range(KMAX)],
            y=[str(k) for k in range(2, KMAX + 1)], zmin=0.3, zmax=1.0,
            colorscale="RdYlBu", showscale=(col == 2),
            colorbar=dict(title="congruence<br>to unrotated", x=1.02),
            text=[[("" if np.isnan(v) else f"{v:.2f}") for v in row] for row in G],
            texttemplate="%{text}", textfont=dict(size=8),
            hoverongaps=False), row=1, col=col)
        fig.update_xaxes(title_text="unrotated component (variance rank)",
                         side="bottom", row=1, col=col)
        fig.update_yaxes(title_text="k extracted", autorange="reversed", row=1, col=col)
    fig.update_layout(
        title=dict(text="<b>Over-extraction profile: which components survive "
            "varimax?</b><br><sub>Tucker congruence of each unrotated component to "
            "its best varimax match (Kaiser-normalized / SPSS convention). Blue = "
            "preserved (real); red = rotation had to fabricate it. Low PC1 = a "
            "strong general factor that rotation splits; a growing red tail = "
            "over-extraction. C&C land on 2–3 robust dimensions.</sub>", x=0.02),
        width=1120, height=560, font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/overextraction_profile.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1120, height=560, scale=2)
    print(f"wrote {out} and .png")
    return (np.array([ph[k].mean() for k in range(2, KMAX + 1)]),
            np.array([pm[k].mean() for k in range(2, KMAX + 1)]))


def compare_conventions(H, M, labels):
    """Does the SPSS/Kaiser knob change our verdict? Print, for human and model:
    (a) mean rotation-profile congruence at k=5 under raw vs Kaiser varimax, and
    (b) the k=5 factor top-2 adjectives under each."""
    for tag, C in (("HUMAN", H), ("MODEL", M)):
        print(f"\n[{tag}] raw vs Kaiser-normalized varimax @ k=5:")
        U = _phi(C, 5)
        for kaiser in (False, True):
            R = rotate(U, kaiser=kaiser)
            L, _ = _signfix_sort(R)
            tops = ["+".join(labels[i] for i in np.argsort(-L[:, j])[:2]) for j in range(5)]
            mc = rotation_profile(C, kaiser=kaiser)[5].mean()
            print(f"   {'Kaiser' if kaiser else 'raw   '}: mean-cong={mc:.2f}  "
                  f"factors={tops}")


# ---- figure 2: bass-ackwards tree (human vs cohort) ----
_FILL = {0: "#aed4ea", 1: "#f7caa3", -1: "#dddddd"}   # node fill by k=2 lineage
_EDGE = {0: "#2471a3", 1: "#c0681c", -1: "#999999"}   # edge color by k=2 lineage


def _draw_tree(fig, nodes, edges, lineage, col):
    # secondary (cross-loading) edges first, as a faint wash — unlabeled
    for k, p, j, r, primary in edges:
        if primary:
            continue
        fig.add_trace(go.Scatter(
            x=[nodes[(k - 1, p)]["x"], nodes[(k, j)]["x"]], y=[-(k - 1), -k],
            mode="lines", line=dict(color="rgba(150,150,150,0.16)", width=0.6, dash="dot"),
            showlegend=False, hoverinfo="skip"), row=1, col=col)
    # primary lineage edges, colored by child lineage; dashed = polarity inversion
    for k, p, j, r, primary in edges:
        if not primary:
            continue
        lin = lineage[(k, j)]; flip = r < 0
        x0, x1 = nodes[(k - 1, p)]["x"], nodes[(k, j)]["x"]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[-(k - 1), -k], mode="lines",
            line=dict(color=_EDGE[lin], width=2.0, dash=("dash" if flip else "solid")),
            showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0.62 * x1 + 0.38 * x0], y=[-(k - 0.38)],
            mode="text", text=[f"{r:+.2f}"],
            textfont=dict(size=7.5, color=("#c0392b" if flip else "#555")),
            showlegend=False, hoverinfo="skip"), row=1, col=col)
    keys = list(nodes)
    xs = [nodes[key]["x"] for key in keys]; ys = [-key[0] for key in keys]
    fills = [_FILL[lineage[key]] for key in keys]
    txt = [f"{nodes[key]['top']}<br>{nodes[key]['var']:.2f}" for key in keys]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=txt, textposition="middle center",
        textfont=dict(size=9), marker=dict(symbol="square", size=26,
        color=fills, line=dict(color="#2c3e50", width=1)),
        showlegend=False, hoverinfo="skip"), row=1, col=col)


def fig_tree(H, M, labels):
    hn, he, hl = bass_ackwards(H, labels)
    mn, me, ml = bass_ackwards(M, labels)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05,
                        subplot_titles=("HUMAN 525-PDA", "LLM cohort (mean)"))
    _draw_tree(fig, hn, he, hl, 1)
    _draw_tree(fig, mn, me, ml, 2)
    for c in (1, 2):
        fig.update_xaxes(visible=False, range=[-0.05, 1.05], row=1, col=c)
        fig.update_yaxes(visible=False, range=[-KMAX - 0.5, -0.5], row=1, col=c)
    fig.update_layout(
        title=dict(text="<b>Bass-ackwards trees (k=1→10): human vs model</b><br>"
            "<sub>Goldberg (2006), C&C Fig 2 extended. Box = top adjective + variance "
            "fraction. <b>Node colour = which k=2 factor it descends from</b> "
            "(blue / orange) — two clean colour blocks = trackable trunks. "
            "Primary edge labelled with its loading-correlation; <b>dashed edge = "
            "polarity inversion</b> (the trunk flips sign, red label). Faint grey = "
            "secondary cross-loadings (|r|≥.3).</sub>", x=0.01),
        width=2000, height=1300, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/bassackwards_tree.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=2000, height=1300, scale=2)
    print(f"wrote {out} and .png")
    return hn, he, mn, me


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = human["labels"]
    Cs = {}
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        name, C, _, _ = model_matrix(p, labels)
        Cs[name] = C
    M = np.mean(list(Cs.values()), 0)

    hc, mc = fig_profile(H, Cs)
    print("\nrotation profile (mean congruence to unrotated), k=2..10:")
    print("  human :", ", ".join(f"{v:.2f}" for v in hc))
    print("  cohort:", ", ".join(f"{v:.2f}" for v in mc))
    compare_conventions(H, M, labels)

    hnodes, hedges, mnodes, medges = fig_tree(H, M, labels)
    for tag, nodes, edges in (("HUMAN", hnodes, hedges), ("MODEL", mnodes, medges)):
        print(f"\n{tag} tree — primary splits (parent spawns >1 child):")
        for k in range(2, KMAX + 1):
            kids_by_parent = {}
            for (kk, pp, cc, r, pr) in edges:
                if kk == k and pr:
                    kids_by_parent.setdefault(pp, []).append((cc, r))
            for pp, kids in kids_by_parent.items():
                if len(kids) > 1:
                    par = nodes[(k - 1, pp)]["top"]
                    ch = ", ".join(f"{nodes[(k, c)]['top']}({r:+.2f})" for c, r in kids)
                    print(f"  L{k-1}->{k}: [{par}] -> {ch}")


if __name__ == "__main__":
    main()
