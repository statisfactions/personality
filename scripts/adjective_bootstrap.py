#!/usr/bin/env python3
"""Adjective-resampling stability — the model-side twin of the W14 §1 human bootstrap.

W14 §1 rejected the human 6th factor (placidity) by resampling RESPONDENTS: it
recovered at congruence ≥.90 in only 3% of refits. The model has one "respondent",
so the symmetric test resamples the dimension we have many of — ADJECTIVES. We
subsample adjectives (m-out-of-n, no replacement — with-replacement duplicates
words and fabricates perfect-correlation pairs), refit the model's varimax factors,
and ask which recover. Plus subsample CIs on the headline scalars.

This is the right tool where bootstrapping the 12 (correlated) models is not:
- "Is structure X a cohort-mean averaging artifact?"  -> per-model point estimates
  (no bootstrap; the spread is the answer).
- "Is structure X robust to the word sample?"         -> adjective subsample (here).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_bootstrap.py
"""
import glob
import json

import numpy as np
import plotly.graph_objects as go

from adjective_geom import model_matrix
from adjective_factor_heatmap import varimax

K = 6
FRAC = 0.8
B = 300


def kfac(C, k):
    C = np.nan_to_num(C); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]
    L = varimax(V[:, o][:, :k] * np.sqrt(np.clip(ev[o][:k], 0, None)), normalize=True)
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    var = (L ** 2).sum(0); order = np.argsort(-var)
    return L[:, order]


def factor_stability(C, labels, rng, K, FRAC, B):
    """Per-factor congruence of the full-set varimax factors to their refit on an
    adjective subsample. Returns (factor labels = top-2 positive markers, congs[B,K])."""
    n = len(labels); k = int(FRAC * n)
    L0 = kfac(C, K)
    fnames = [", ".join(labels[np.argsort(-L0[:, j])[:2]]) for j in range(K)]  # top + markers
    congs = np.zeros((B, K))
    for b in range(B):
        ix = rng.choice(n, k, replace=False)
        congs[b] = match(L0[ix], kfac(C[np.ix_(ix, ix)], K))
    return fnames, congs


def tuck(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def match(L0sub, Lb):
    k = L0sub.shape[1]
    M = np.array([[tuck(L0sub[:, i], Lb[:, j]) for j in range(k)] for i in range(k)])
    used, out = set(), np.zeros(k)
    for i in np.argsort(-M.max(1)):
        j = max((jj for jj in range(k) if jj not in used), key=lambda jj: M[i, jj])
        used.add(j); out[i] = M[i, j]
    return out


def pc1(A):
    ev, V = np.linalg.eigh(np.nan_to_num(A)); return V[:, np.argmax(ev)]


def main():
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); labels = np.array(human["labels"])
    np.fill_diagonal(H, 1.0)
    Cs = {}
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        n, C, _, _ = model_matrix(p, labels); np.fill_diagonal(C, 1.0); Cs[n] = C
    M = np.mean(list(Cs.values()), 0); np.fill_diagonal(M, 1.0)
    n = len(labels); k = int(FRAC * n); rng = np.random.default_rng(0)

    # --- human & model factor stability under the SAME adjective subsample ---
    out_data = {}
    for tag, C in (("HUMAN", H), ("MODEL", M)):
        fnames, congs = factor_stability(C, labels, np.random.default_rng(0), K, FRAC, B)
        out_data[tag] = (fnames, congs)
        print(f"{tag} factor stability under adjective subsample ({int(FRAC*100)}%, B={B}):")
        for j in range(K):
            c = congs[:, j]
            print(f"  F{j+1} [{fnames[j]:<28}] mean={c.mean():.3f}  frac>=.90={np.mean(c>=.9):.2f}")
        print()

    # --- scalar subsample CIs + per-model point estimates ---
    v = np.array([abs(pc1(H[np.ix_(ix, ix)]) @ pc1(M[np.ix_(ix, ix)]))
                  for ix in (rng.choice(n, k, replace=False) for _ in range(B))])
    print(f"human-PC1 x model-PC1 congruence: {v.mean():.3f}  95% CI "
          f"[{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")
    print("  per-model point estimates (NOT bootstrap): " +
          " ".join(f"{abs(pc1(H)@pc1(C)):.2f}" for C in Cs.values()))

    # --- figure: human vs model factor recovery, both adjective-resampled ---
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("HUMAN 525-PDA", "LLM cohort"))
    for col, tag in ((1, "HUMAN"), (2, "MODEL")):
        fnames, congs = out_data[tag]
        for j in range(K):
            stable = congs[:, j].mean() >= 0.9
            fig.add_trace(go.Box(y=congs[:, j], name=f"F{j+1} {fnames[j].split(',')[0]}",
                marker_color=("#2ca02c" if stable else "#d62728"), boxpoints=False,
                width=0.5, showlegend=False), row=1, col=col)
        fig.add_hline(y=0.90, line_dash="dot", line_color="gray", row=1, col=col)
        fig.update_yaxes(range=[0.2, 1.02], title_text="congruence to full set", row=1, col=col)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=8))
    fig.update_layout(
        title=dict(text="<b>Which adjective factors are real? Adjective-resampling — "
            "the model-side twin of the W14 §1 human test</b><br><sub>Per-factor "
            "congruence to the full-set factor under an 80% adjective subsample (B=300). "
            "Green = stable (mean ≥.90). Human: the Big Five hold, the data-driven 6th "
            "(placidity) is fragile — same verdict the respondent bootstrap gave, now via "
            "word-resampling. Model: only the evaluative factors are robust.</sub>", x=0.01),
        width=1200, height=560, plot_bgcolor="white", font=dict(family="Helvetica, Arial"))
    out = "results/adjectives/adjective_bootstrap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1000, height=540, scale=2)
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
