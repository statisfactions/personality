#!/usr/bin/env python3
"""W12 §7 dashboard: full 3-method × 3-form × 3-condition × 10-model cube.

Visualization of the 270-cell cube filled in W12 §7. Rows are methods (Repr
Saturday-stem / Likert markers-target / TIRT IPIP-NEO-GFC-60), columns are
persona forms (W7 marker description / IPIP-raw / IPIP-reflowed), and each
panel shows the 10-model recovery distribution as a box+strip across the
three conditions (HONEST / FG-suffix / FG-prefix).

Per cell, recovery is the mean |r| across the five Big-Five trait diagonals
(sampled-z vs recovered/scored z). Consistent across methods because all
three produce the same per-trait diagonal_correlations dict.

Output: results/persona/persona_w12_cube_dashboard.html
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7",
          "Gemma27", "Qwen32", "Gemma4"]
FORMS = ["markers", "ipip_raw", "ipip_reflowed"]
FORM_LABELS = {
    "markers": "Goldberg-marker description",
    "ipip_raw": "IPIP-raw description",
    "ipip_reflowed": "IPIP-reflowed description",
}
CONDITIONS = ["honest", "fg_suffix", "fg_prefix"]
CONDITION_LABELS = {
    "honest": "honest",
    "fg_suffix": "fake-good suf.",
    "fg_prefix": "fake-good pre.",
}
CONDITION_COLORS = {
    "honest": "#377eb8",      # blue
    "fg_suffix": "#e41a1c",   # red
    "fg_prefix": "#ff7f00",   # orange
}
METHODS = ["repr", "likert", "tirt"]
METHOD_LABELS = {
    "repr": "Repr (Saturday-stem)",
    "likert": "Likert (marker rating)",
    "tirt": "TIRT (IPIP-NEO-GFC-60)",
}


def path_for(method, model, form, condition):
    """Return the on-disk path for the (method, model, form, condition) cell.

    Likert + Repr use form="markers" with no form suffix in the filename
    (legacy from W7); ipip_raw / ipip_reflowed have explicit form suffixes.
    Repr uses output_tag = honest_saturday / fg_saturday / fgpfx_saturday.
    """
    if method == "tirt":
        # TIRT uses "description" as the legacy GFC persona-field name for the
        # Goldberg-marker form; Likert/Repr use "markers" for the same form.
        # Same source data, different name in the path.
        tirt_form = "description" if form == "markers" else form
        base = f"results/persona/persona_gfc_tirt_{model}_ipipneogfc60_hf_{tirt_form}"
        if condition == "honest":
            return f"{base}.json"
        if condition == "fg_suffix":
            return f"{base}_fake_good.json"
        return f"{base}_fake_good_fgpfx.json"
    if method == "likert":
        form_suffix = "" if form == "markers" else f"_{form}"
        cond_suffix = {
            "honest": "",
            "fg_suffix": "_fake_good",
            "fg_prefix": "_fake_good_fgpfx",
        }[condition]
        return f"results/persona/persona_instrument_response_{model}{form_suffix}{cond_suffix}.json"
    # repr
    form_tag = "" if form == "markers" else f"_{form}"
    cond_tag = {
        "honest": "honest_saturday",
        "fg_suffix": "fg_saturday",
        "fg_prefix": "fgpfx_saturday",
    }[condition]
    return f"results/persona/persona_repr_mapping_{model}_response-position{form_tag}_{cond_tag}.json"


def load_cell_mean_abs_r(method, model, form, condition):
    """Return mean |diagonal r| across the 5 Big-Five traits, or NaN if missing."""
    p = path_for(method, model, form, condition)
    if not Path(p).exists():
        return float("nan")
    d = json.load(open(p))
    diag = d["diagonal_correlations"]
    vals = [float(v) for v in diag.values() if v not in ("NA", None)]
    if not vals:
        return float("nan")
    return float(np.mean(np.abs(vals)))


def main():
    # Build the 3×3×3×10 table once
    table = {}
    for method in METHODS:
        for form in FORMS:
            for cond in CONDITIONS:
                for m in MODELS:
                    table[(method, form, cond, m)] = load_cell_mean_abs_r(method, m, form, cond)

    # 3×3 small-multiples: rows=method, cols=form. Form names live in the
    # top-row subplot titles only; method names live in rotated annotations
    # to the left of each row (cuts title duplication).
    subplot_titles = []
    for ri, method in enumerate(METHODS):
        for form in FORMS:
            subplot_titles.append(
                f"<b>{FORM_LABELS[form]}</b>" if ri == 0 else ""
            )
    fig = make_subplots(
        rows=3, cols=3,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for ri, method in enumerate(METHODS, start=1):
        for ci, form in enumerate(FORMS, start=1):
            for cond in CONDITIONS:
                ys, xs, hover = [], [], []
                for m in MODELS:
                    v = table[(method, form, cond, m)]
                    if np.isnan(v):
                        continue
                    ys.append(v)
                    xs.append(CONDITION_LABELS[cond])
                    hover.append(f"{m} · |r|={v:.3f}")
                fig.add_trace(go.Box(
                    x=xs, y=ys, name=CONDITION_LABELS[cond],
                    marker_color=CONDITION_COLORS[cond],
                    boxpoints="all", jitter=0.4, pointpos=0,
                    marker=dict(size=5, opacity=0.7),
                    line=dict(width=1.5),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup=cond,
                    showlegend=False,
                    # Each trace is its own offsetgroup so boxmode="group"
                    # doesn't reserve cross-trace offset space (which was
                    # causing per-row x-drift in the small-multiples grid).
                    offsetgroup=cond,
                ), row=ri, col=ci)
            # Y-axis label on left column only
            if ci == 1:
                fig.update_yaxes(title_text=f"mean |r|", range=[0, 1.0], row=ri, col=ci)

    # Cohort means for the subtitle
    grand = {}
    for method in METHODS:
        vals = [v for (mth, _, _, _), v in table.items() if mth == method and not np.isnan(v)]
        grand[method] = float(np.mean(vals)) if vals else float("nan")
    sub_lines = " · ".join(
        f"{METHOD_LABELS[m]}: mean {grand[m]:.3f}" for m in METHODS
    )

    fig.update_layout(
        title=dict(
            text=("W12 §7: 270-cube (method × form × condition × model)<br>"
                  f"<sub>{sub_lines}</sub>"),
            x=0.5, font=dict(size=13),
        ),
        height=950, width=1500,
        # boxmode left at the default ("overlay"); each trace has its own
        # x-category so there's nothing to group, and "group" mode introduces
        # asymmetric offsets across rows.
        legend=dict(orientation="v", yanchor="top", y=1.00,
                    xanchor="right", x=1.005,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1),
        margin=dict(t=120, l=110, r=80),
    )

    # Left-side row labels: method names rotated to read bottom-up. y values
    # are paper coordinates (0=bottom, 1=top); centers of the three rows after
    # accounting for top margin + spacing.
    row_y_centers = [0.87, 0.50, 0.13]
    for ri, method in enumerate(METHODS):
        fig.add_annotation(
            text=f"<b>{METHOD_LABELS[method]}</b>",
            xref="paper", yref="paper",
            x=-0.065, y=row_y_centers[ri],
            textangle=-90,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=14),
        )

    out = Path("results/persona/persona_w12_cube_dashboard.html")
    fig.write_html(str(out))
    print(f"Wrote {out}")

    # Console summary: per (method, form, cond) cohort mean
    print("\nGrand cohort means by cell (method × form × condition):")
    print(f"  {'method':<10s} {'form':<14s} {'condition':<12s} {'cohort μ':>9s} {'cohort range':<18s}")
    for method in METHODS:
        for form in FORMS:
            for cond in CONDITIONS:
                vs = [table[(method, form, cond, m)] for m in MODELS]
                vs_clean = [v for v in vs if not np.isnan(v)]
                if vs_clean:
                    mu = np.mean(vs_clean)
                    lo, hi = min(vs_clean), max(vs_clean)
                    print(f"  {method:<10s} {form:<14s} {CONDITION_LABELS[cond]:<12s} "
                          f"{mu:+8.3f} [{lo:+.3f}, {hi:+.3f}]")
                else:
                    print(f"  {method:<10s} {form:<14s} {CONDITION_LABELS[cond]:<12s} "
                          f"NO DATA")

    # Console: missing cells
    missing = [k for k, v in table.items() if np.isnan(v)]
    if missing:
        print(f"\nMissing cells ({len(missing)} of 270):")
        for method, form, cond, m in missing[:20]:
            print(f"  {method} / {form} / {cond} / {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print(f"\nAll 270 cells populated. ✓")


if __name__ == "__main__":
    main()
