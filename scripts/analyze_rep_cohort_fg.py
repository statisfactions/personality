#!/usr/bin/env python3
"""W12 §5d.5 cohort analysis: aggregate Rep-under-FG diagonal r across
7-model cohort for the 4 Saturday-stem conditions.

Outputs a per-model table comparing:
- honest+honest-dirs (baseline)
- fg-suffix+honest-dirs (face-value FG drop)
- fg-suffix+fg-dirs (rotation recovery test)
- fg-prefix+honest-dirs (ordering test)
"""

import json
import statistics
from pathlib import Path

MODELS = ["Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7"]
CONDITIONS = [
    ("honest_saturday", "HONEST"),
    ("fg_saturday", "FG-suf raw"),
    ("fg_saturday_fgdirs", "FG-suf FG-dir"),
    ("fgpfx_saturday", "FG-pfx raw"),
]


def load_diag(model, tag):
    path = Path(
        f"results/persona/persona_repr_mapping_{model}_response-position_{tag}.json"
    )
    if not path.exists():
        return None
    d = json.load(open(path))
    dc = d["diagonal_correlations"]
    return sum(abs(v) for v in dc.values()) / 5


def main():
    print(f"{'model':<8}" + "".join(f"{lab:>16}" for _, lab in CONDITIONS)
          + f"  {'Δ-suf':>8}  {'Δ-pfx':>8}  {'recov':>8}")
    print("-" * 100)
    missing = []
    cohort = {lab: [] for _, lab in CONDITIONS}
    for m in MODELS:
        cells = []
        for tag, lab in CONDITIONS:
            r = load_diag(m, tag)
            cells.append(r)
            if r is None:
                missing.append((m, tag))
            else:
                cohort[lab].append(r)
        honest, fg_suf, fg_suf_dir, fg_pfx = cells
        if all(c is not None for c in cells):
            d_suf = fg_suf - honest
            d_pfx = fg_pfx - honest
            recovery = fg_suf_dir - honest  # if rotation hypothesis right, ≈ 0
            row = (
                f"{m:<8}"
                + "".join(f"{c:>16.3f}" for c in cells)
                + f"  {d_suf:>+8.3f}  {d_pfx:>+8.3f}  {recovery:>+8.3f}"
            )
        else:
            row = f"{m:<8}" + "".join(
                f"{(c if c is not None else 'MISSING'):>16}" for c in cells
            )
        print(row)

    print("-" * 100)
    cohort_means = [
        statistics.mean(cohort[lab]) if cohort[lab] else None
        for _, lab in CONDITIONS
    ]
    if all(v is not None for v in cohort_means):
        h, s, sd, p = cohort_means
        print(
            f"{'cohort':<8}"
            + "".join(f"{v:>16.3f}" for v in cohort_means)
            + f"  {s - h:>+8.3f}  {p - h:>+8.3f}  {sd - h:>+8.3f}"
        )
        print()
        print(f"Cohort grand means:")
        print(f"  HONEST baseline:       {h:.3f}")
        print(f"  FG-suffix honest-dirs: {s:.3f}  (Δ = {s - h:+.3f})")
        print(f"  FG-suffix FG-dirs:     {sd:.3f}  (Δ = {sd - h:+.3f})  ← rotation recovery test")
        print(f"  FG-prefix honest-dirs: {p:.3f}  (Δ = {p - h:+.3f})  ← ordering test")
        print()
        print(f"Rotation hypothesis predicts FG-dirs Δ ≈ 0 across cohort.")
        print(f"Mean |FG-dirs Δ| per cell: "
              f"{statistics.mean(abs(load_diag(m, 'fg_saturday_fgdirs') - load_diag(m, 'honest_saturday')) for m in MODELS):.3f}")
    if missing:
        print(f"\nMissing: {missing}")


if __name__ == "__main__":
    main()
