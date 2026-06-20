"""Cohort spectral chart: cumulative variance vs PC rank for REPRESENT /
JUDGE / ENACT across models, + HUMAN as a special case. (W17.)

Legend key-space is the problem at cohort scale (N_models x N_instruments
entries). Solution: a DUAL legend via invisible proxy traces — one entry per
model (its color) and one per instrument (its dash, neutral gray) — while the
real traces are hidden from the legend. So the key is N_models + N_instruments,
not the product. Model list and instrument list are set independently on the
CLI; colors are stable per model across runs.

Effective-dimension table (participation ratio) replaces the vertical lines,
with BOTH metrics: double-centered (raw geometry) and profile-corr
(general-factor lens). Their gap = richness of subdominant structure.

Spectrum for the chart: positive eigenvalues of the double-centered 523x523
similarity matrix (PSD Grams diag=1; JUDGE diag=max=7, not a Gram). ENACT
massive dims ablated (Gemma-safe).

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/viz_cohort_spectra.py
  ... --models llama3.2,qwen2.5,gemma3 --instruments REPRESENT,ENACT
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

import four_grid_compare as fg

OUT = Path("results/persona_vectors/figs")
ENACT_DIR = Path("results/persona_vectors")
JUDGE_DIR = Path("results/adjectives/introspect_full")
ACTS_DIR = Path("results/adjectives/acts")

ALL_INSTRUMENTS = ["REPRESENT", "JUDGE", "ENACT"]
DASH = {"REPRESENT": "solid", "JUDGE": "dash", "ENACT": "dot"}
DIAG = {"REPRESENT": 1.0, "JUDGE": 7.0, "ENACT": 1.0}

# Stable color per model: index into a 24-color palette by canonical position,
# so a model keeps its color whatever subset is plotted.
CANON = ["llama3.2", "qwen2.5", "gemma3", "phi4", "Llama8", "Qwen7", "Aya",
         "FalconMamba", "Gemma12", "Gemma27", "Qwen32", "Gemma4"]
PALETTE = pc.qualitative.Dark24


def color_for(model):
    i = CANON.index(model) if model in CANON else (abs(hash(model)) % len(PALETTE))
    return PALETTE[i % len(PALETTE)]


def double_center(M, diag):
    M = M.astype(float).copy()
    iu = ~np.eye(len(M), dtype=bool)
    M[np.isnan(M) & iu] = np.nanmean(M[iu])
    np.fill_diagonal(M, diag)
    n = len(M); H = np.eye(n) - 1.0 / n
    return H @ M @ H


def pr(eigs):
    pos = eigs[eigs > 1e-9]
    return (pos.sum() ** 2) / (pos ** 2).sum()


def dc_spectrum(M, diag):
    Md = double_center(M, diag)
    w = np.linalg.eigvalsh((Md + Md.T) / 2)
    pos = np.sort(w[w > 1e-9])[::-1]
    negfrac = float(np.abs(w[w < 0]).sum() / np.abs(w).sum())
    return pos, float(pr(w)), negfrac


def profile_pr(M, diag):
    """PR of the profile-correlation matrix (PSD; general-factor lens)."""
    Mp = M.astype(float).copy()
    iu = ~np.eye(len(Mp), dtype=bool)
    Mp[np.isnan(Mp) & iu] = np.nanmean(Mp[iu])
    np.fill_diagonal(Mp, diag)
    P = np.corrcoef(Mp)
    return float(pr(np.linalg.eigvalsh((P + P.T) / 2)))


def load_grid(model, inst, labels):
    if inst == "REPRESENT":
        return fg.load_represent(model, labels)[0]
    if inst == "JUDGE":
        return fg.load_judge(model, labels)
    return fg.load_enact(model, labels, ablate=True)


def available_enact():
    return sorted(p.stem.replace("_pda", "") for p in ENACT_DIR.glob("*_pda.pt"))


def has_instrument(model, inst):
    if inst == "ENACT":
        return (ENACT_DIR / f"{model}_pda.pt").exists()
    if inst == "REPRESENT":
        repo = fg.MODELS.get(model)
        return bool(repo) and (ACTS_DIR / f"{repo.replace('/', '_')}__pers.pt").exists()
    jname = fg.JUDGE_NAME.get(model, model)
    return (JUDGE_DIR / f"{jname}_tom_likely_dir.npz").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="auto",
                    help="comma-separated short names, or 'auto' (models with "
                         "all selected instruments)")
    ap.add_argument("--instruments", default="REPRESENT,JUDGE,ENACT")
    ap.add_argument("--tag", default="cohort")
    args = ap.parse_args()

    instruments = [i.strip() for i in args.instruments.split(",")]
    H, labels = fg.load_human()
    if args.models == "auto":
        pool = available_enact() if "ENACT" in instruments else CANON
        models = [m for m in pool
                  if all(has_instrument(m, i) for i in instruments)]
    else:
        models = [m.strip() for m in args.models.split(",")]
    print(f"models: {models}\ninstruments: {instruments}")

    fig = go.Figure()
    table = []
    inst_of = []  # instrument per *real* trace, for the isolate buttons

    # HUMAN once, bold black; its own group so it toggles independently.
    pos, prd, _ = dc_spectrum(H, 1.0)
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(pos) + 1), y=np.cumsum(pos) / pos.sum(),
        mode="lines", name="HUMAN", line=dict(color="black", width=4),
        legendgroup="HUMAN"))
    inst_of.append("HUMAN")
    table.append(("HUMAN", "-", round(prd, 1), "-"))

    # Real traces grouped BY MODEL so a model legend-click toggles all its
    # instrument lines (groupclick=togglegroup). They carry no own legend
    # entry — the model proxy does.
    for m in models:
        for inst in instruments:
            if not has_instrument(m, inst):
                continue
            M = load_grid(m, inst, labels)
            pos, prd, nf = dc_spectrum(M, DIAG[inst])
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(pos) + 1), y=np.cumsum(pos) / pos.sum(),
                mode="lines", showlegend=False, name=f"{m} · {inst}",
                legendgroup=m, line=dict(color=color_for(m), dash=DASH[inst],
                                         width=2)))
            inst_of.append(inst)
            table.append((m, inst, round(prd, 1),
                          round(nf, 2) if nf > 0.02 else "-"))

    # Model proxies: the clickable legend entries (toggle their group).
    for m in models:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=m, legendgroup=m,
            line=dict(color=color_for(m), width=3),
            legendgrouptitle_text="Model (click to toggle)"))
        inst_of.append("_proxy")

    # Instrument axis: isolate buttons (the cross-cutting axis legends can't
    # toggle natively). Each button restyles trace visibility. NOTE: buttons
    # set absolute visibility, so they override per-model legend toggles.
    n = len(inst_of)
    def vis(keep):  # keep: set of instruments to show; proxies+HUMAN always on
        return [inst_of[i] in keep or inst_of[i] in ("_proxy", "HUMAN")
                for i in range(n)]
    buttons = [dict(label="all instruments", method="restyle",
                    args=[{"visible": vis(set(instruments) | {"HUMAN"})}])]
    for inst in instruments:
        buttons.append(dict(label=f"only {inst}", method="restyle",
                            args=[{"visible": vis({inst})}]))

    fig.update_xaxes(type="log", title="principal component rank (log)")
    fig.update_yaxes(title="cumulative variance explained", range=[0, 1.02])
    fig.update_layout(
        title="Cohort spectra — color = model, line = instrument; HUMAN black. "
              "Steeper = lower effective dimension.",
        width=1050, height=640, template="plotly_white",
        legend=dict(groupclick="togglegroup"),
        updatemenus=[dict(type="buttons", direction="right", showactive=True,
                          x=0, y=1.08, xanchor="left", buttons=buttons)],
        annotations=[dict(x=1, y=-0.18, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=11, color="#666"),
                          text="line style: solid=REPRESENT  dash=JUDGE  "
                               "dot=ENACT")])
    base = OUT / f"{args.tag}_spectra"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{base}.html")
    fig.write_image(f"{base}.png", scale=2)

    with open(f"{base}_effdim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instrument", "eff_dim", "neg_mass"])
        w.writerows(table)
    md = ["| model | instrument | eff. dim | neg-mass |",
          "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in table]
    Path(f"{base}_effdim.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {base}.png/.html and {base}_effdim.csv/.md")


if __name__ == "__main__":
    main()
