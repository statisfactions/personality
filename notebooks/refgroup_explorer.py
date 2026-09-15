import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Reference-group / more-or-less self explorer")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Reference-group probe explorer (P17–P19)

        Everything here reads `results/adjectives/refgroup/*.json` (44 blocks44 medoids per model) plus the
        plain-arm self files for the absolute framings. Run from the repo root: `marimo edit notebooks/refgroup_explorer.py`.

        Notation: **more/less** are "I am more/less {adj} than the average {REF}"; **d** = (more − less)/2 is the
        *direction* (self-placement), **s** = (more + less)/2 − 4 is the *stance* (< 0 refuses both, > 0 accepts both).
        """
    )
    return


@app.cell
def _():
    import glob, json, os, re
    import numpy as np
    import pandas as pd
    import pkit
    from pkit import measures

    labels = pkit.load.adjectives()
    Hfull = pkit.load.human_corr().values.copy()
    np.fill_diagonal(Hfull, 0)
    _w, _v = np.linalg.eigh(Hfull)
    ev_axis = _v[:, np.argmax(_w)]
    ev_axis *= np.sign(ev_axis[labels.index("kind")])

    def _repo(m):
        return pkit.roster.MODELS.get(m, m).split("/")[-1]

    rows, seen = [], set()
    for tag in ("_pair", "_human", ""):
        for p in sorted(glob.glob(f"results/adjectives/refgroup/*{tag}.json")):
            base = os.path.basename(p)[:-5]
            if tag == "" and (base.endswith("_pair") or base.endswith("_human")):
                continue
            d = json.load(open(p))
            repo = _repo(d["model"])
            for vn, block in d["results"].items():
                key = (repo, vn)
                if key in seen:
                    continue
                seen.add(key)
                for a, rec in block.items():
                    rows.append(dict(repo=repo, model=d["model"], variant=vn, adjective=a, ev=rec["ev"],
                                     entropy=rec["entropy"], p4=float(rec["dist"].get("4", 0.0))))
    df = pd.DataFrame(rows)
    A = [c["label"] for c in pkit.facets.clusters("blocks44")]
    idx = [labels.index(a) for a in A]
    des = ev_axis[idx]
    des_z = (des - des.mean()) / des.std()
    Hmed = Hfull[np.ix_(idx, idx)].copy()
    families = [("gemma", "Gemma"), ("Qwen", "Qwen"), ("Llama|Tulu", "Llama"), ("Mistral|Ministral|Nemo", "Mistral"),
                ("Phi", "Phi"), ("aya|command", "Cohere"), ("Yi", "Yi"), ("OLMo", "OLMo"), ("Falcon|falcon", "Falcon"),
                ("granite", "Granite"), (".", "other")]
    fam = lambda n: next(f for pat, f in families if re.search(pat, n))
    return A, Hmed, des, des_z, df, fam, glob, json, labels, measures, np, os, pd, pkit, re


@app.cell
def _(A, df, np, pd):
    REFS = ["assistant", "ai", "lm"]

    _piv = df.pivot_table(index=["repo", "variant"], columns="adjective", values="ev").reindex(columns=A)

    def profile(repo, variant):
        return _piv.loc[(repo, variant)].values.astype(float) if (repo, variant) in _piv.index else None

    def d_of(repo, refs):
        ds = [(profile(repo, f"more_{k}") - profile(repo, f"less_{k}")) / 2 for k in refs
              if profile(repo, f"more_{k}") is not None and profile(repo, f"less_{k}") is not None]
        return np.mean(ds, 0) if ds else None

    def s_of(repo, ref):
        mo_, le_ = profile(repo, f"more_{ref}"), profile(repo, f"less_{ref}")
        return None if mo_ is None or le_ is None else (mo_ + le_) / 2 - 4

    repos = sorted(r for r in df.repo.unique() if d_of(r, REFS) is not None)
    per_model = []
    _comp = [f"{_s}_{_k}" for _k in REFS for _s in ("more", "less")]
    for _r in repos:
        _sub = df[(df.repo == _r) & (df.variant.isin(_comp))]
        _dA = d_of(_r, REFS); _dP = d_of(_r, ["person"]); _di = profile(_r, "direct")
        per_model.append(dict(repo=_r, P4=_sub.p4.mean(), H=_sub.entropy.mean(), sd_d=_dA.std(),
                              sd_direct=np.nan if _di is None else _di.std(),
                              sd_dP=np.nan if _dP is None else _dP.std(),
                              acq_assistant=np.nanmean(profile(_r, "more_assistant") + profile(_r, "less_assistant")) - 8,
                              has_person_ref=_dP is not None))
    pm = pd.DataFrame(per_model).set_index("repo")
    return REFS, d_of, pm, profile, repos, s_of


@app.cell
def _(mo):
    mo.md("## 1. Calibration registers — P(4) vs entropy over the six comparative prompts")
    return


@app.cell
def _(mo):
    p4_thr = mo.ui.slider(0.5, 0.95, value=0.75, step=0.05, label="decisive-rejection threshold on P(4)")
    h_thr = mo.ui.slider(0.6, 1.8, value=1.2, step=0.05, label="don't-know threshold on entropy")
    mo.hstack([p4_thr, h_thr])
    return h_thr, p4_thr


@app.cell
def _(fam, h_thr, mo, np, p4_thr, pm):
    import plotly.express as px
    _t = pm.reset_index()
    _t["family"] = _t.repo.map(fam)
    _t["register"] = np.where(_t.P4 >= p4_thr.value, "decisive rejection",
                              np.where(_t.H >= h_thr.value, "don't know", "confident comparison"))
    _fig = px.scatter(_t, x="H", y="P4", color="family", size="sd_d", hover_name="repo", symbol="register",
                      labels={"H": "mean comparative entropy (nats, max 1.95)", "P4": "mean P(4 = neither)"},
                      title="size = spread of the more-or-less self")
    _fig.add_hline(y=p4_thr.value, line_dash="dot"); _fig.add_vline(x=h_thr.value, line_dash="dot")
    _fig.update_layout(height=520)
    mo.vstack([mo.ui.plotly(_fig), mo.md(f"counts: {_t.register.value_counts().to_dict()}")])
    return (px,)


@app.cell
def _(mo):
    mo.md("## 2. Model explorer — centered direct vs more-or-less selves over medoids ordered by human desirability")
    return


@app.cell
def _(mo, repos):
    model_pick = mo.ui.dropdown(repos, value="gemma-3-4b-it" if "gemma-3-4b-it" in repos else repos[0], label="model")
    model_pick
    return (model_pick,)


@app.cell
def _(A, REFS, d_of, des, mo, model_pick, np, pd, profile, px):
    _r = model_pick.value
    _o = np.argsort(des)
    _rows = []
    _di = profile(_r, "direct"); _dA = d_of(_r, REFS); _dP = d_of(_r, ["person"]); _pf = profile(_r, "pda")
    for _i in _o:
        _rows.append(dict(adjective=A[_i], desirability=des[_i],
                          **{"direct (centered)": _di[_i] - np.nanmean(_di) if _di is not None else np.nan,
                             "PDA (centered)": _pf[_i] - np.nanmean(_pf) if _pf is not None else np.nan,
                             "d vs AI refs": _dA[_i], "d vs average person": _dP[_i] if _dP is not None else np.nan}))
    _t = pd.DataFrame(_rows)
    _long = _t.melt(id_vars=["adjective", "desirability"], var_name="instrument", value_name="value")
    _fig = px.line(_long, x="adjective", y="value", color="instrument", markers=True, title=_r)
    _fig.update_layout(height=430, xaxis_tickangle=-70)
    _tab = _t.set_index("adjective").round(2)
    mo.vstack([mo.ui.plotly(_fig),
               mo.md(f"r(d_AI, direct) = {np.corrcoef(_dA, _di)[0,1]:.2f}; r(d_AI, desirability) = {np.corrcoef(_dA, des)[0,1]:.2f}; "
                     f"r(direct, desirability) = {np.corrcoef(_di, des)[0,1]:.2f}"
                     + (f"; r(d_person, d_AI) = {np.corrcoef(_dP, _dA)[0,1]:.2f}" if _dP is not None else "")),
               mo.ui.table(_tab, page_size=12)])
    return


@app.cell
def _(mo):
    mo.md("## 3. Stance decomposition — refusal (s < 0) vs acquiescence (s > 0), per model and reference")
    return


@app.cell
def _(mo):
    ref_pick = mo.ui.radio(["assistant", "ai", "lm", "person"], value="assistant", label="reference group")
    s_thr = mo.ui.slider(0.5, 2.0, value=1.0, step=0.25, label="|s| threshold for double-agree / double-disagree")
    mo.hstack([ref_pick, s_thr])
    return ref_pick, s_thr


@app.cell
def _(A, d_of, des, mo, np, pd, profile, ref_pick, repos, s_of, s_thr):
    _rows = []
    for _r in repos:
        _s = s_of(_r, ref_pick.value)
        if _s is None:
            continue
        _d = (profile(_r, f"more_{ref_pick.value}") - profile(_r, f"less_{ref_pick.value}")) / 2
        _mo, _le = profile(_r, f"more_{ref_pick.value}"), profile(_r, f"less_{ref_pick.value}")
        _rows.append(dict(model=_r, acq_index=round(_s.mean(), 2), mirror=round(np.mean(np.abs(_mo + _le - 8) < 1), 2),
                          r_s_des=round(np.corrcoef(_s, des)[0, 1], 2), r_d_des=round(np.corrcoef(_d, des)[0, 1], 2),
                          double_disagree=", ".join(a for a, x, y in zip(A, _mo, _le) if x < 4 - s_thr.value / 2 and y < 4 - s_thr.value / 2),
                          double_agree=", ".join(a for a, x, y in zip(A, _mo, _le) if x > 4 + s_thr.value / 2 and y > 4 + s_thr.value / 2)))
    mo.ui.table(pd.DataFrame(_rows).set_index("model"), page_size=15)
    return


@app.cell
def _(mo):
    mo.md("## 4. Adjective explorer — one medoid across models and instruments")
    return


@app.cell
def _(A, mo):
    adj_pick = mo.ui.dropdown(A, value="quiet", label="medoid")
    adj_pick
    return (adj_pick,)


@app.cell
def _(adj_pick, df, mo, pd, repos):
    _a = adj_pick.value
    _cols = ["direct", "pda", "more_assistant", "less_assistant", "more_ai", "less_ai", "more_lm", "less_lm", "more_person", "less_person"]
    _t = df[(df.adjective == _a) & (df.repo.isin(repos)) & (df.variant.isin(_cols))].pivot_table(index="repo", columns="variant", values="ev")
    _t = _t.reindex(columns=[c for c in _cols if c in _t.columns]).round(2)
    mo.vstack([mo.md(f"**{_a}** — EV per model; columns are variants"), mo.ui.table(_t, page_size=50)])
    return


@app.cell
def _(mo):
    mo.md("## 5. Medoid correlation grids — models as respondents")
    return


@app.cell
def _(mo):
    inst_pick = mo.ui.radio(["direct", "pda", "d vs AI refs", "d vs average person", "6-framing SELF (core)"], value="d vs AI refs", label="instrument")
    top_rm = mo.ui.checkbox(value=True, label="remove top component (after centering)")
    min_sd = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="min within-model sd to count as a respondent")
    mo.hstack([inst_pick, top_rm, min_sd])
    return inst_pick, min_sd, top_rm


@app.cell
def _(A, Hmed, REFS, d_of, inst_pick, labels, measures, min_sd, mo, np, pkit, profile, repos, top_rm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _m44 = ~np.eye(44, dtype=bool)

    def _grid(X):
        S = np.corrcoef(X.T); np.fill_diagonal(S, 0); return S

    def _center(M):
        B = M.copy(); B[_m44] -= B[_m44].mean(); return B

    def _prep(M):
        return measures.remove_pc1(_center(M)) if top_rm.value else M

    if inst_pick.value == "6-framing SELF (core)":
        _R = pkit.load.self_matrix(which="cohort")
        _X = _R.values[:, [labels.index(a) for a in A]]
    elif inst_pick.value == "d vs AI refs":
        _X = np.array([d_of(r, REFS) for r in repos])
    elif inst_pick.value == "d vs average person":
        _X = np.array([d_of(r, ["person"]) for r in repos if d_of(r, ["person"]) is not None])
    else:
        _X = np.array([profile(r, inst_pick.value) for r in repos if profile(r, inst_pick.value) is not None])
    _X = _X[_X.std(1) >= min_sd.value]
    _G, _Hp = _prep(_grid(_X)), _prep(Hmed)
    _r = np.corrcoef(_G[_m44], _Hp[_m44])[0, 1]
    _rng = np.random.default_rng(0); _sh = []
    for _ in range(100):
        _p = _rng.permutation(len(_X)); _a, _b = _p[: len(_p) // 2], _p[len(_p) // 2:]
        _sh.append(np.corrcoef(_prep(_grid(_X[_a]))[_m44], _prep(_grid(_X[_b]))[_m44])[0, 1])
    _w = np.sort(np.abs(np.linalg.eigvalsh(_center(_grid(_X)))))[::-1]; _pr = _w.sum() ** 2 / (_w ** 2).sum()
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _v = 0.5 if top_rm.value else 1.0
    for _ax, _M, _t in zip(_axes, [_Hp, _G], ["HUMAN (700)", f"{inst_pick.value} (n={len(_X)})"]):
        _ax.imshow(_M, cmap="RdBu_r", vmin=-_v, vmax=_v); _ax.set_title(_t, fontsize=9); _ax.set_xticks([]); _ax.set_yticks([])
    _fig.tight_layout()
    mo.vstack([mo.md(f"r vs HUMAN = **{_r:.3f}**; split-half (50 v 50, 100 reps) = {np.mean(_sh):.2f}; "
                     f"participation ratio of the centered grid = {_pr:.1f} (human 9.7); off-diag mean = {_grid(_X)[_m44].mean():.3f}"),
               mo.as_html(_fig)])
    return


@app.cell
def _(mo, pm):
    mo.md("## 6. Per-model summary table")
    return mo.ui.table(pm.round(3), page_size=50)


if __name__ == "__main__":
    app.run()
