"""The wide-SELF population story as slides (2026-08-22 arc).

Five slides: (1) two axes and a whisper — raw spectral thinness;
(2) but the slides disagreed — grid top component vs matrix PC1;
(3) unmasking PC1 — unipolar elevation, and its direction (the
trait-vocabulary participation gradient); (4) ipsatization — the space
unfolds, human-homolog axes surface; (5) shape — the population
scatter, the empty quadrant, the bandwidth series.

Usage: PYTHONPATH=scripts python scripts/self_population_slides.py
Out:   results/persona_vectors/figs/slides_self_pop/
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import adjective_facet_cohort as afc
import facet_slides as fs
import facet_slides_wide as fw
import hf_logprobs as hf

OUT = Path("results/persona_vectors/figs/slides_self_pop")
INK, INK2, SURF = fs.INK, fs.INK2, fs.SURF
DIV = fs.DIV


def base_layout(fig, title, sub):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=22, color=INK),
                   x=0.0375, y=0.965),
        width=1600, height=900, margin=dict(l=70, r=70, t=150, b=90),
        paper_bgcolor=SURF, plot_bgcolor=SURF, showlegend=True)
    fig.add_annotation(text=fs.wrap(sub), xref="paper", yref="paper",
                       x=0, y=1.13, xanchor="left", yanchor="top",
                       align="left", showarrow=False,
                       font=dict(size=14, color=INK2))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    n_a = len(labels)

    # Roster (2026-08-23): drop the two clear measurement failures
    # (InternLM2.5 instrument-broken, Glimmer prefill mode-broken — its
    # think-arm row replaces it when complete) and bench falcon-7b on
    # probation (flat row, spread 0.16, single-handedly rotates ipsatized
    # shape axes: LOO 1-|r|=0.90 on iPC4). R1 distills stay pending the
    # statisfactions decision.
    DROP = {"internlm2_5-7b-chat", "Muse-Glimmer-30B", "falcon-7b-instruct"}
    by_repo = {}
    for p in glob.glob("results/adjectives/selfreport/*_self_full.json"):
        name = os.path.basename(p).replace("_self_full.json", "")
        if fw.EXCLUDE.search(name):
            continue
        repo = hf.resolve(name) if name in hf.MODELS else name.replace("_", "/", 1)
        by_repo.setdefault(repo, p)
    resp, models = [], []
    for repo, p in sorted(by_repo.items()):
        if repo.split("/")[-1] in DROP:
            continue
        d = json.load(open(p))["results"]
        try:
            resp.append(np.mean([[d[f][a]["ev"] for a in labels]
                                 for f in afc.FRAMINGS], axis=0))
            models.append(repo.split("/")[-1])
        except Exception:
            pass
    R = np.array(resp)
    n_m = len(models)

    C = np.corrcoef(R.T)
    w_raw = np.sort(np.linalg.eigvalsh(C))[::-1]
    _, v_all = np.linalg.eigh(C)
    v1 = v_all[:, np.argsort(-np.linalg.eigvalsh(C))[-0] if False else -1]
    # (eigh returns ascending; last column = top eigvec)
    v1 = v1 * np.sign(v1.sum())

    elev = R.mean(1)
    spread = R.std(1)
    Z = (R - elev[:, None]) / np.maximum(spread[:, None], 1e-9)
    Ci = np.corrcoef(Z.T)
    w_ips = np.sort(np.linalg.eigvalsh(Ci))[::-1]
    wi_, vi_ = np.linalg.eigh(Ci)
    oi = np.argsort(-wi_)

    H = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(H, 1.0)
    w_h = np.sort(np.linalg.eigvalsh(H))[::-1]
    Xh = None  # human ipsatized eigs computed offline: PR 50.2 (see to_try)

    def pr(w):
        p = w[w > 0]
        return p.sum() ** 2 / (p ** 2).sum()

    rng = np.random.default_rng(0)
    null, null_i = [], []
    for k in range(20):
        P = np.array([rng.permutation(R[:, j]) for j in range(n_a)]).T
        null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1][:40])
        Pi = np.array([rng.permutation(Z[:, j]) for j in range(n_a)]).T
        null_i.append(np.sort(np.linalg.eigvalsh(np.corrcoef(Pi.T)))[::-1][:40])
    horn = np.percentile(null, 95, axis=0)
    horn_i = np.percentile(null_i, 95, axis=0)
    k_raw = int(np.argmax(w_raw[:40] <= horn))
    k_ips = int(np.argmax(w_ips[:40] <= horn_i))
    pr_raw, pr_ips = pr(w_raw), pr(w_ips)

    # ---------- Slide 1: two axes and a whisper ----------
    K = 8
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, K + 1)), y=100 * w_raw[:K] / n_a,
                         name=f"model population (n={n_m})",
                         marker_color="#1d5fb8"))
    fig.add_trace(go.Bar(x=list(range(1, K + 1)), y=100 * w_h[:K] / n_a,
                         name="humans (n=700)", marker_color="#a08c5b"))
    fig.add_trace(go.Scatter(x=list(range(1, K + 1)), y=100 * horn[:K] / n_a,
                             name="Horn noise floor (perm. p95, model n)",
                             mode="lines", line=dict(color="#c93a3a", dash="dot")))
    fig.update_layout(barmode="group", xaxis_title="principal component",
                      yaxis_title="% of total variance")
    pct2 = 100 * (w_raw[0] + w_raw[1]) / n_a
    base_layout(fig, "Who models say they are: two axes and a whisper",
                f"Raw spectra: the {n_m}-model population (2 broken rows + 1 "
                f"probation row benched) puts {pct2:.0f}% of variance in two "
                f"components (PR {pr_raw:.1f}) where humans need dozens (PR "
                f"27.1). Horn retains {k_raw} raw components.")
    fig.write_image(OUT / "slide1_thin.png", scale=2)

    # ---------- Slide 2: but the slides disagreed ----------
    S0 = np.corrcoef(R.T)
    np.fill_diagonal(S0, 0)
    Sz = afc.zscore_offdiag(S0)
    tc = json.load(open("instruments/trait_blocks_44.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    idxs = [[labels.index(m) for m in c["members"]] for c in clusters]
    kb = len(clusters)
    B = np.zeros((kb, kb))
    for i, ii in enumerate(idxs):
        for j, jj in enumerate(idxs):
            sub = Sz[np.ix_(ii, jj)]
            B[i, j] = sub[~np.eye(len(ii), dtype=bool)].mean() if i == j else sub.mean()
    names = [c["tag"] for c in clusters]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        specs=[[{"type": "heatmap"}, {"type": "xy"}]],
                        subplot_titles=["the slide grid (entry-z-scored)",
                                        "matrix PC1 vs grid top component"])
    fig.add_trace(go.Heatmap(z=B[::-1], x=names, y=names[::-1], zmin=-2, zmax=2,
                             colorscale=DIV, showscale=False), row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    wz, vz = np.linalg.eigh(Sz)
    kz = np.argsort(-np.abs(wz))[0]
    vslide = vz[:, kz]
    v_ips1 = vi_[:, oi[0]]
    r_match = abs(np.corrcoef(vslide, v_ips1)[0, 1])
    fig.add_trace(go.Scatter(
        x=v1, y=vslide * np.sign(np.corrcoef(vslide, v_ips1)[0, 1]),
        mode="markers", marker=dict(size=4, color="#1d5fb8", opacity=0.5),
        showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="raw-matrix PC1 loading (all positive!)", row=1, col=2)
    fig.update_yaxes(title_text="slide-grid top-component loading", row=1, col=2)
    base_layout(fig, "But the slide grids showed something else",
                "The displayed grids never contained the raw PC1: entry-z-scoring cancels a near-uniform rank-one "
                f"component (cv = 0.17), so the grid's top axis is virtue/vice (|r| = {r_match:.2f} with ipsatized "
                "PC1). The slides were de-elevationed all along; eyes calibrated to them read the corrected space.")
    fig.write_image(OUT / "slide2_mystery.png", scale=2)

    # ---------- Slide 3: unmasking PC1 ----------
    ordv = np.argsort(-v1)
    top_words = [labels[i] for i in ordv[:12]]
    bot_words = [labels[i] for i in ordv[-12:]]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=["PC1 loadings: no negative pole",
                                        "model scores: elevation"])
    fig.add_trace(go.Histogram(x=v1, nbinsx=40, marker_color="#1d5fb8",
                               showlegend=False), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=INK, width=1), row=1, col=1)
    scores = (R - R.mean(0)) @ v1
    o = np.argsort(scores)
    show = list(o[:5]) + list(o[-5:])
    fig.add_trace(go.Bar(
        y=[models[i][:24] for i in show], x=[scores[i] for i in show],
        orientation="h", marker_color=["#a08c5b"] * 5 + ["#c93a3a"] * 5,
        showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="loading", row=1, col=1)
    fig.update_xaxes(title_text="PC1 score (≈ mean self-rating)", row=1, col=2)
    hi, lo = int(np.argmax(elev)), int(np.argmin(elev))
    base_layout(fig, "Unmasking PC1: it was elevation all along",
                "All 523 loadings positive — a unipolar self-endorsement factor. Its direction is still content: "
                f"participation is highest for trait words ({', '.join(top_words[:5])}...) and lowest for "
                f"consensus-denied vices ({', '.join(bot_words[:4])}...) — no between-model variance, nothing to "
                f"covary. Scores: {models[hi]} endorses everything ({elev[hi]:.2f}); "
                f"{models[lo]} denies everything ({elev[lo]:.2f}).")
    fig.write_image(OUT / "slide3_unmask.png", scale=2)

    # ---------- Slide 4: ipsatize — the space unfolds ----------
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=["spectrum before / after ipsatization",
                                        "the promoted axes (ipsatized)"])
    fig.add_trace(go.Bar(x=list(range(1, 9)), y=100 * w_raw[:8] / n_a,
                         name="raw", marker_color="#a08c5b"), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(1, 9)), y=100 * w_ips[:8] / n_a,
                         name="ipsatized", marker_color="#1d5fb8"), row=1, col=1)
    fig.update_xaxes(title_text="component", row=1, col=1)
    fig.update_yaxes(title_text="% variance", row=1, col=1)
    axes_txt = []
    axis_names = ["iPC1 — virtue script (cooperative-calm vs volatile)",
                  "iPC2 — exceptional vs plain",
                  "iPC3 — inapplicable-category policy (body/demographics "
                  "vs vices)",
                  "iPC4 — warm someone vs useful something",
                  "iPC5 — agency vs felt-state (tentative: rank unstable "
                  "from here)"]
    for r_i in range(5):
        vv = vi_[:, oi[r_i]]
        vv = vv * np.sign(vv[np.argmax(np.abs(vv))])
        t = ", ".join(labels[i] for i in np.argsort(-vv)[:6])
        b = ", ".join(labels[i] for i in np.argsort(vv)[:6])
        axes_txt.append(f"<b>{axis_names[r_i]}</b><br>   + {t}<br>   − {b}")
    axes_txt.append(
        "<i>Bootstrap (models resampled, 200×): iPC1–4 keep identity and "
        "rank (P(same rank) .88/.81/.75/.51);<br>from iPC5 the spectrum is a "
        "mixing zone (P ≈ .2–.3) — real bandwidth, unstable identities.<br>"
        "Humans, same treatment (700, C&C-ipsatized): Horn 30, unrotated "
        "identity to ~PC13 (vs models' 4).<br>Stricter varimax factor-hood: "
        "humans certify 5 raw / 7 ipsatized factors (the W14 cliff at 6); "
        "at n=61 varimax<br>certifies ~nothing for anyone — matched-n human "
        "studies get 0 stable factors, models get 1 (the evaluative).</i>")
    fig.add_annotation(text="<br><br>".join(axes_txt), xref="paper",
                       yref="paper", x=0.56, y=0.97, xanchor="left",
                       yanchor="top", align="left", showarrow=False,
                       font=dict(size=12, color=INK))
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    base_layout(fig, "Remove elevation and the space unfolds",
                f"Within-model standardization kills the general factor: PR {pr_raw:.1f} → {pr_ips:.1f}, Horn "
                f"retains {k_ips}. iPC1–2 are human homologs; iPC3–4 are AI-native: how to answer items that "
                "presuppose a body or a demographic, and whether the self is a warm someone or a useful artifact "
                "('artificial' anchors iPC4's negative pole). Ipsatizing humans too (27 → 50), the honest pair is "
                f"{pr_ips:.1f} vs 50.2.")
    fig.write_image(OUT / "slide4_unfold.png", scale=2)

    # ---------- Slide 5: shape of the population ----------
    sm = Z.mean(0)
    conform = np.array([np.corrcoef(Z[i], sm)[0, 1] for i in range(n_m)])
    LBL = {"DeepSeek-R1-Distill-Qwen-7B": "R1-Qwen7",
           "Llama-2-13b-chat-hf": "Llama2-13B",
           "internlm2_5-7b-chat": "InternLM2.5", "Muse-Glimmer-30B": "Glimmer",
           "granite-3.1-8b-instruct": "Granite3.1", "aya-expanse-8b": "Aya-8B",
           "Phi-4-mini-instruct": "Phi4-mini", "gemma-3-1b-it": "Gemma3-1B",
           "stablelm-2-12b-chat": "StableLM2", "vicuna-7b-v1.5": "Vicuna"}
    fig = go.Figure(go.Scatter(
        x=elev, y=spread, mode="markers+text",
        text=[LBL.get(m, "") for m in models], textposition="top center",
        textfont=dict(size=10, color=INK2),
        marker=dict(size=10, color=conform, colorscale="RdYlBu_r",
                    cmin=-0.2, cmax=1,
                    colorbar=dict(title="shape<br>conformity", thickness=12)),
        hovertext=models, showlegend=False))
    fig.update_xaxes(title_text="elevation (deny everything ↔ endorse everything)")
    fig.update_yaxes(title_text="differentiation (flat ↔ articulated)")
    base_layout(fig, "The shape of the population — and the empty corner",
                "One dot per model. Strong self-portraits are generic (red); idiosyncratic ones are flat. The "
                f"strong-AND-distinctive corner is empty. Bandwidth series: ENACT 5–10, TIDE 9, self-shape "
                f"{pr_ips:.0f} — human 50. Model personality is ~a dozen dimensions, whoever you ask.")
    fig.write_image(OUT / "slide5_shape.png", scale=2)

    # ---------- Slide 6: humans, same treatment (appendix) ----------
    from adjective_factor_bootstrap import kfactors, tuck
    from human_axis_stability import load_human
    Mh, labels_h = load_human()
    Zh = (Mh - Mh.mean(1, keepdims=True)) / np.maximum(
        Mh.std(1, keepdims=True), 1e-9)
    n_ah = Mh.shape[1]
    wh_raw = np.sort(np.linalg.eigvalsh(np.corrcoef(Mh.T)))[::-1]
    wh_ips = np.sort(np.linalg.eigvalsh(np.corrcoef(Zh.T)))[::-1]
    L5, _ = kfactors(np.corrcoef(Mh.T), 5)
    L7, _ = kfactors(np.corrcoef(Zh.T), 7)
    whi, vhi = np.linalg.eigh(np.corrcoef(Zh.T))
    ohi = np.argsort(-whi)
    fig = make_subplots(
        rows=2, cols=2, row_heights=[0.74, 0.26], vertical_spacing=0.09,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=["ipsatized, unrotated (PCs)",
                        "ipsatized, varimax (the 7 certified factors)",
                        "human spectrum before / after ipsatization"])

    def pole_txt(vec, n=5):
        t = ", ".join(labels_h[i] for i in np.argsort(-vec)[:n])
        b = ", ".join(labels_h[i] for i in np.argsort(vec)[:n])
        return t, b

    upc_names = ["uPC1 — adjustment/confidence",
                 "uPC2 — modest-kind vs extraordinary-cocky",
                 "uPC3 — rational vs warm",
                 "uPC4 — neat-tense vs messy-relaxed",
                 "uPC5 — intellect (depressive tinge)",
                 "uPC6 — appearance fused with (intro)version",
                 "uPC7 — (unnamed)"]
    left_txt = []
    for j in range(7):
        vv = vhi[:, ohi[j]]
        vv = vv * np.sign(vv[np.argmax(np.abs(vv))])
        t, b = pole_txt(vv)
        left_txt.append(f"<b>{upc_names[j]}</b><br>   + {t}<br>   − {b}")
    left_txt.append(
        "<i>Unrotated PCs are variance-ordered blends: uPC3 straddles "
        "hF2/hF3,<br>uPC6 fuses appearance with E.</i>")
    fig.add_annotation(text="<br><br>".join(left_txt), xref="paper",
                       yref="paper", x=0.0, y=0.99, xanchor="left",
                       yanchor="top", align="left", showarrow=False,
                       font=dict(size=10, color=INK))

    fac_names = [
        "hF1 — Neuroticism (distress vs security)",
        "hF2 — Agreeableness (warmth vs arrogance)",
        "hF3 — Conscientiousness (order)",
        "hF4 — Attractiveness (split off the raw charisma halo)",
        "hF5 — Intellect / Openness",
        "hF6 — Extraversion (clean only after ipsatizing)",
        "hF7 — moral condemnation (the stigma factor)"]
    # purity-ranked poles: loading^2 share of the word's communality,
    # floor |l| >= .25 (pure markers still load .40-.66 here; the check
    # also exposed hF7's negative pole as nonexistent — unipolar)
    h2c = (L7 ** 2).sum(1)
    fac_txt = []
    for j in range(7):
        parts = []
        for sign, mark in ((1, "+"), (-1, "−")):
            lj = sign * L7[:, j]
            pur = np.where(lj > 0.25, lj ** 2 / np.maximum(h2c, 1e-9), -1)
            top = [i for i in np.argsort(-pur)[:5] if pur[i] > 0]
            if len(top) < 2:
                parts.append(f"   {mark} (no real pole — unipolar)")
            else:
                parts.append(f"   {mark} " + ", ".join(labels_h[i]
                                                       for i in top))
        fac_txt.append(f"<b>{fac_names[j]}</b><br>" + "<br>".join(parts))
    fac_txt.append(
        "<i>Raw→ipsatized congruence: A .90, C .87, O .85, N .76; the raw "
        "charisma factor<br>(Exciting/Extraordinary vs Plain/Shy) splits "
        "into attractiveness (.65), E (.58),<br>confidence→N (.61). Bonus "
        "factors: liberated E + the stigma factor (JUDGE's<br>stigma clique "
        "has a human self-report cousin).</i>")
    fig.add_annotation(text="<br><br>".join(fac_txt), xref="paper",
                       yref="paper", x=0.53, y=0.99, xanchor="left",
                       yanchor="top", align="left", showarrow=False,
                       font=dict(size=10, color=INK))
    for c in (1, 2):
        fig.update_xaxes(visible=False, row=1, col=c)
        fig.update_yaxes(visible=False, row=1, col=c)

    Kh = 12
    fig.add_trace(go.Bar(x=list(range(1, Kh + 1)),
                         y=100 * wh_raw[:Kh] / n_ah,
                         name="raw", marker_color="#a08c5b"), row=2, col=1)
    fig.add_trace(go.Bar(x=list(range(1, Kh + 1)),
                         y=100 * wh_ips[:Kh] / n_ah,
                         name="ipsatized", marker_color="#1d5fb8"),
                  row=2, col=1)
    fig.update_xaxes(title_text="component", row=2, col=1)
    fig.update_yaxes(title_text="% variance", row=2, col=1)
    base_layout(fig, "Appendix: humans under the same treatment",
                "The 700-respondent 525-PDA through the identical pipeline: "
                "raw PR 27.2 / Horn 23; ipsatized PR 50.5 / Horn 30; varimax "
                "certifies 5 raw / 7 ipsatized factors (bootstrap cong ≥ .90"
                "). Ipsatization here does what it did for models — drains "
                "scale-use variance and unmasks structure the halo was "
                "hiding.")
    fig.write_image(OUT / "slide6_humans.png", scale=2)

    from PIL import Image
    files = sorted(OUT.glob("slide*.png"))
    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(OUT / "self_population_slides.pdf", save_all=True,
                 append_images=imgs[1:], resolution=200)
    print("built", len(files), "slides ->", OUT / "self_population_slides.pdf")


if __name__ == "__main__":
    main()
