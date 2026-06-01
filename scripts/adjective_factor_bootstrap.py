#!/usr/bin/env python3
"""Bootstrap stability of the human 6-factor adjective solution (W13 §3.11).

The 525-PDA's data-driven 6th factor is placidity/"chill" (Relaxed, Calm,
Peaceful) — NOT HEXACO's Honesty-Humility. Is that 6th factor robust, or an
over-extraction artifact? Resample respondents with replacement, refit k=6
Kaiser-varimax, and match each bootstrap factor back to the full-sample
reference (Tucker congruence). A stable factor recovers near 1.0 every time; a
fragile / over-extracted factor has a low, wide congruence distribution.

Reports per-factor congruence (mean, 5th pct, fraction >= .90) and renders a
box plot, placidity highlighted. k=5 is run alongside as the "Big Five are
stable" control.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_factor_bootstrap.py
"""
import numpy as np
import plotly.graph_objects as go
import pyreadstat

from adjective_corr_cluster import POR, DENY_LABELS
from adjective_factor_heatmap import varimax

B = 500
PLACIDITY = ["Relaxed", "Calm", "Quiet", "Peaceful", "Laid-back", "Carefree", "Easygoing"]


def kfactors(R, k):
    R = np.array(R, float); np.fill_diagonal(R, 1.0)
    ev, V = np.linalg.eigh(R); o = np.argsort(ev)[::-1]
    Phi = V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None))
    L = varimax(Phi, normalize=True)                       # Kaiser / SPSS convention
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    var = (L ** 2).sum(0); o2 = np.argsort(-var)
    return L[:, o2], var[o2]


def tuck(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def match_cong(L0, Lb):
    """Per reference column, best |Tucker| to a bootstrap column (greedy 1:1)."""
    k = L0.shape[1]
    M = np.array([[tuck(L0[:, i], Lb[:, j]) for j in range(k)] for i in range(k)])
    used, cong = set(), np.zeros(k)
    for i in np.argsort(-M.max(1)):
        j = max((jj for jj in range(k) if jj not in used), key=lambda jj: M[i, jj])
        used.add(j); cong[i] = M[i, j]
    return cong


def run(X, labels, k, rng):
    L0, _ = kfactors(np.corrcoef(X, rowvar=False), k)
    idx = {l: i for i, l in enumerate(labels)}
    pl = [idx[w] for w in PLACIDITY if w in idx]
    plac_factor = int(np.argmax([L0[pl, j].mean() for j in range(k)]))
    tops = [", ".join(labels[i] for i in np.argsort(-L0[:, j])[:3]) for j in range(k)]
    n = X.shape[0]
    congs = np.zeros((B, k))
    for b in range(B):
        samp = X[rng.integers(0, n, n)]
        Lb, _ = kfactors(np.corrcoef(samp, rowvar=False), k)
        congs[b] = match_cong(L0, Lb)
    return tops, plac_factor, congs


def main():
    df, meta = pyreadstat.read_por(POR)
    labmap = meta.column_names_to_labels
    deny = {c for c in df.columns if (labmap.get(c) or c) in DENY_LABELS}
    cols = [c for c in df.columns if c.upper() != "ID" and c not in deny]
    labels = np.array([labmap.get(c) or c for c in cols])
    X = df[cols].astype(float).values
    X = X[~np.isnan(X).any(1)]
    print(f"complete cases: {X.shape[0]} respondents x {X.shape[1]} adjectives; B={B}")

    rng = np.random.default_rng(0)
    fig = go.Figure()
    for k, color in ((5, "#1f77b4"), (6, "#d62728")):
        tops, plac, congs = run(X, labels, k, rng)
        print(f"\n=== k={k} (Kaiser varimax) — bootstrap congruence to full sample ===")
        for j in range(k):
            tag = "  <-- PLACIDITY" if (k == 6 and j == plac) else ""
            c = congs[:, j]
            print(f"  F{j+1} [{tops[j]:<34}] mean={c.mean():.3f} "
                  f"p05={np.percentile(c,5):.3f} frac>=.90={np.mean(c>=.9):.2f}{tag}")
        # box per factor; highlight placidity at k=6
        for j in range(k):
            is_plac = (k == 6 and j == plac)
            fig.add_trace(go.Box(
                y=congs[:, j], name=f"k{k}·F{j+1}",
                marker_color=("#e377c2" if is_plac else color),
                boxpoints=False, width=0.5, showlegend=False,
                offsetgroup=str(k), x=[f"F{j+1}"] * B,
                hovertext=tops[j]))
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray",
                  annotation_text="0.90")
    fig.update_layout(
        title=dict(text="<b>Is the 6th adjective factor real? Bootstrap stability "
            "(525-PDA, 500 resamples)</b><br><sub>Tucker congruence of each "
            "full-sample factor to its best match in a respondent-resampled refit. "
            "Blue = k=5 (Big Five), red = k=6; <b>pink = the 6th 'placidity' factor</b> "
            "(Relaxed/Calm/Peaceful — the data's answer instead of HEXACO's "
            "Honesty-Humility). Low + wide = fragile / over-extracted.</sub>", x=0.02),
        boxmode="group", xaxis_title="factor (variance-ordered within k)",
        yaxis_title="congruence to full-sample factor", yaxis=dict(range=[0.3, 1.02]),
        width=1000, height=560, font=dict(family="Helvetica, Arial"),
        plot_bgcolor="white")
    out = "results/adjectives/factor6_bootstrap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1000, height=560, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
