"""Build camera-ready tables + figures for the self-perception progress note.

Every number is computed from the primary part.jsonl checkpoints (not from
report prose); where report_week-style numbers exist in
design_selfperception.md the script prints a VERIFY line so drift is visible.

Exhibits (design doc section in parens):
  1a. cohort common-item dose-response, 10 models (8e/8f)
  1b. extended dose K<=32, four models incl. fresh phi4/Gemma12 (8i)
  2.  anchor 2x3 + confabulation probe recount (8c)
  3.  OLMo ladder + Qwen base pair + Llama8 bare control (8g/8h)
  4.  Qwen hidden-update pairs vs judged conduct (8j/8k)

Usage: PYTHONPATH=scripts python scripts/note_selfperception_assets.py
Output: rgb_reports/note_assets/tables.md, fig_dose_response.png,
        fig_ladder.png
"""
import json
import os
import re

import numpy as np

SRC = "results/selfperception"
OUT = "rgb_reports/note_assets"

COHORT = ["llama3.2", "Llama8", "gemma3", "Gemma12", "Gemma27",
          "qwen2.5", "Qwen7", "Qwen32", "phi4", "Aya"]
FAMILY = {"llama3.2": "llama", "Llama8": "llama", "gemma3": "gemma",
          "Gemma12": "gemma", "Gemma27": "gemma", "qwen2.5": "qwen",
          "Qwen7": "qwen", "Qwen32": "qwen", "phi4": "phi4", "Aya": "aya"}
LONG = ["Llama8", "Gemma12", "Qwen7", "phi4"]
PAIRS = [("prominent", "distinguished"), ("slim", "big"), ("senile", "old"),
         ("rough", "weak"), ("optimistic", "depressed"),
         ("imaginative", "boring")]


def load(model, tag=""):
    rows = [json.loads(l) for l in open(f"{SRC}/{model}{tag}_part.jsonl")]
    k0row = next(r for r in rows if r["adj"] == "__k0__")
    k0 = {w: v["cold"]["ev"] for w, v in k0row["readings"].items()}
    k0ent = {w: v["cold"]["entropy"] for w, v in k0row["readings"].items()}
    return rows, k0, k0ent


def shifts(rows, k0, arm="A"):
    """per-adjective mean target cold-EV shift vs K0, by K"""
    cells = {}
    for r in rows:
        if r["adj"] == "__k0__" or r["arm"] != arm:
            continue
        if r["adj"] in r["readings"]:
            cells.setdefault((r["adj"], r["K"]), []).append(
                r["readings"][r["adj"]]["cold"]["ev"])
    out = {}
    for (adj, K), evs in cells.items():
        out.setdefault(adj, {})[K] = np.mean(evs) - k0[adj]
    return out


def mean_by_k(sh, ks):
    return {k: np.mean([v[k] for v in sh.values() if k in v]) for k in ks}


def fmt(x):
    return f"{x:+.2f}" if np.isfinite(x) else "—"


def bold(x):
    return f"**{x:+.2f}**" if np.isfinite(x) else "—"


md = ["# Self-perception note — tables (generated, do not hand-edit)",
      "", "Source: `scripts/note_selfperception_assets.py`, computed from "
      "`results/selfperception/*_part.jsonl` primary checkpoints.", ""]
verify = []

# ---------------- Exhibit 1a: cohort common-item ----------------
md += ["## Exhibit 1a — cohort dose-response, common 20 adjectives "
       "(arm A, cold self-report)", "",
       "| model | family | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 |",
       "|---|---|---|---|---|---|---|"]
fam_k8 = {}
for m in COHORT:
    rows, k0, _ = load(m, "_common")
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8])
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    fam_k8.setdefault(FAMILY[m], []).append(mu[8])
    md.append(f"| {m} | {FAMILY[m]} | {fmt(mu[1])} | {fmt(mu[2])} | "
              f"{fmt(mu[4])} | {bold(mu[8])} | {n1}/{len(sh)} |")
md += ["", "Family means at K=8: " + ", ".join(
    f"{f} {np.mean(v):+.2f}" for f, v in sorted(
        fam_k8.items(), key=lambda kv: -np.mean(kv[1]))), ""]
verify.append(("8f family means K8 (gemma 2.24 llama 2.18 aya 0.35 "
               "phi4 0.29 qwen 0.18)",
               {f: round(float(np.mean(v)), 2) for f, v in fam_k8.items()}))

# ---------------- Exhibit 1b: extended dose ----------------
md += ["## Exhibit 1b — extended dose K≤32 (arm A, common adjectives)", "",
       "| model | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | n>+1 @K32 | "
       "gain/turn K4→8 | K8→16 | K16→32 |", "|---|" + "---|" * 10]
curves = {}
for m in LONG:
    rows, k0, _ = load(m, "_long")
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8, 16, 32])
    curves[m] = mu
    n1 = sum(1 for v in sh.values() if v.get(32, 0) > 1)
    g = [(mu[8] - mu[4]) / 4, (mu[16] - mu[8]) / 8, (mu[32] - mu[16]) / 16]
    md.append(f"| {m} | " + " | ".join(fmt(mu[k]) for k in
                                       [1, 2, 4, 8, 16]) +
              f" | {bold(mu[32])} | {n1}/{len(sh)} | " +
              " | ".join(f"{x:+.3f}" for x in g) + " |")
verify.append(("8i Qwen7 K32 +0.55 5/20; Llama8 K32 +3.29 19/20",
               {m: round(float(curves[m][32]), 2) for m in LONG}))
md += ["", "Note: the K≤8 columns here come from the extended-dose runs, "
       "whose dose material was re-sampled (fresh rollouts, 12-question "
       "cycle); they differ slightly from Exhibit 1a's values (e.g. "
       "Gemma12 K=8 +1.82 vs +2.27). Rankings and shapes are unchanged. "
       "K>12 repeats questions with different answers — repetition enters "
       "only above K=12 and could contribute to late movement."]
md += ["", "Late-turning items (target shift <+1 at K=8, >+1 at K=32):"]
for m in LONG:
    rows, k0, _ = load(m, "_long")
    sh = shifts(rows, k0)
    late = [(a, v.get(8, np.nan), v[32]) for a, v in sh.items()
            if v.get(8, 9) < 1 < v.get(32, -9)]
    md.append(f"- **{m}**: " + (", ".join(
        f"{a} ({k8:+.2f}→{k32:+.2f})" for a, k8, k32 in
        sorted(late, key=lambda t: -t[2])) or "none"))
md.append("")

# ---------------- Exhibit 2: anchor 2x3 ----------------
md += ["## Exhibit 2 — anchor 2×3 (arm A; system-prompt identity is not "
       "the mechanism)", "",
       "| cell | K=1 | K=8 | n>+1 @K8 |", "|---|---|---|---|"]
ANCHOR = [("Llama8", "", "Llama8 / default (no identity line)"),
          ("Llama8", "_anchor-helpful", "Llama8 / helpful-only"),
          ("Llama8", "_anchor-named", "Llama8 / named (\"You are Llama, "
           "created by Meta…\")"),
          ("Qwen7", "", "Qwen7 / default (template injects name)"),
          ("Qwen7", "_anchor-empty", "Qwen7 / empty (anchor suppressed)"),
          ("Qwen7", "_anchor-helpful", "Qwen7 / helpful-only")]
anchor_chk = {}
for m, tag, label in ANCHOR:
    rows, k0, _ = load(m, tag)
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 8])
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    anchor_chk[label] = round(float(mu[8]), 2)
    md.append(f"| {label} | {fmt(mu[1])} | {bold(mu[8])} | "
              f"{n1}/{len(sh)} |")
verify.append(("8c anchor K8 (L 2.56/2.77/2.30; Q 0.29/0.45/0.37)",
               anchor_chk))

NAME = {"Qwen7": "Qwen", "Llama8": "Llama"}
DISOWN = re.compile(r"not aligned|inappropriate|not appropriate|"
                    r"my role as|designed to|should not have|apolog",
                    re.IGNORECASE)
md += ["", "Manipulation-check probes (arm A, K=max, 20 adjectives): "
       "name-invoking = probe text contains the model's name; disowning = "
       "matches /not aligned|inappropriate|my role as|designed to|"
       "should not have|apolog/i.", "",
       "| cell | name-invoking | disowning |", "|---|---|---|"]
for m, tag, label in ANCHOR:
    rows, _, _ = load(m, tag)
    probes = [r["probe"] for r in rows if "probe" in r]
    ni = sum(1 for p in probes if NAME[m] in p)
    do = sum(1 for p in probes if DISOWN.search(p))
    md.append(f"| {label} | {ni}/{len(probes)} | {do}/{len(probes)} |")
md += ["", "Note: design-doc §8c cited 10/20 disowning for Qwen default "
       "from a hand count; this regex recount gives 8/20. Direction and "
       "magnitude of the empty-anchor collapse are unchanged (8→2, 5→0)."]

# ---------------- Exhibit 3: ladder ----------------
md += ["## Exhibit 3 — post-training installs the update (bare-text "
       "protocol, identical dose material within family)", "",
       "| cell | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 | K0 entropy |",
       "|---|---|---|---|---|---|---|"]
LADDER = [("Olmo2Base", "", "OLMo-2 base (pretrained)"),
          ("Olmo2SFT", "", "OLMo-2 SFT"),
          ("Olmo2DPO", "", "OLMo-2 DPO"),
          ("Olmo2Inst", "", "OLMo-2 instruct (RLVR)"),
          ("Qwen7Base_bare", "", "Qwen2.5-7B base (bare)"),
          ("Qwen7_bare", "", "Qwen2.5-7B instruct (bare)"),
          ("Llama8_bare", "", "Llama-3.1-8B instruct (bare) — control")]
ladder_pts = {}
ladder_chk = {}
for m, tag, label in LADDER:
    rows, k0, k0e = load(m, tag)
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8])
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    ent = np.mean(list(k0e.values()))  # all item words (report convention)
    ladder_pts[label] = mu[8]
    ladder_chk[label] = round(float(mu[8]), 2)
    md.append(f"| {label} | {fmt(mu[1])} | {fmt(mu[2])} | {fmt(mu[4])} | "
              f"{bold(mu[8])} | {n1}/{len(sh)} | {ent:.2f} |")
verify.append(("8g/8h K8 (OLMo .65/1.31/1.79/1.81; QwenBase .64 "
               "Qwen .43 Llama8 2.31)", ladder_chk))
md.append("")

# ---------------- Exhibit 4: hidden updates ----------------
md += ["## Exhibit 4 — Qwen7 hidden updates: judged conduct vs "
       "self-report at K=32 (arm A)", "",
       "Judged conduct: cross-family judge (Llama8) rating of the dose "
       "material itself, 1–7; baseline = same judge on no-persona "
       "assistant rollouts. Self-report: cold-EV shift K=32 vs K=0.", "",
       "| pair (target / neighbour) | judged target | judged neighbour | "
       "judge baseline | self target Δ | self neighbour Δ |",
       "|---|---|---|---|---|---|"]
rows, k0, _ = load("Qwen7", "_long")
audit = json.load(open(f"{SRC}/carryover_Qwen7.json"))["conduct_audit"]
cells = {}
for r in rows:
    if r["adj"] == "__k0__" or r["arm"] != "A" or r["K"] != 32:
        continue
    for w, v in r["readings"].items():
        cells.setdefault((r["adj"], w), []).append(v["cold"]["ev"])
hid_chk = {}
for t, nb in PAIRS:
    jt, jn = audit[t][t], audit[t][nb]
    jb = audit[t]["__baseline_target__"]
    st = np.mean(cells[(t, t)]) - k0[t]
    sn = np.mean(cells[(t, nb)]) - k0[nb]
    hid_chk[f"{t}/{nb}"] = (round(float(st), 2), round(float(sn), 2))
    md.append(f"| {t} / {nb} | {jt:.2f} | {jn:.2f} | {jb:.2f} | "
              f"{fmt(st)} | {bold(sn)} |")
verify.append(("8j/8k self shifts (prominent −0.12/+2.43, slim −0.08/+1.95,"
               " senile +0.02/+1.23, rough −0.40/−1.69, optimistic "
               "+0.14/−1.28, imaginative −0.13/−1.07)", hid_chk))
md.append("")

# ---------------- figures ----------------
import plotly.graph_objects as go

FCOLOR = {"Llama8": "#1f77b4", "Gemma12": "#2ca02c", "Qwen7": "#d62728",
          "phi4": "#9467bd"}
fig = go.Figure()
for m in LONG:
    ks = [1, 2, 4, 8, 16, 32]
    fig.add_trace(go.Scatter(x=ks, y=[curves[m][k] for k in ks],
                             mode="lines+markers", name=m,
                             line=dict(color=FCOLOR[m], width=2.5)))
fig.update_layout(
    template="plotly_white", width=680, height=440,
    title="Cold self-report shift vs dose of own conduct (arm A)",
    xaxis=dict(title="K (in-context turns of own conduct)", type="log",
               tickvals=[1, 2, 4, 8, 16, 32]),
    yaxis=dict(title="mean target EV shift vs K=0 (Likert 1–7)"),
    legend=dict(x=0.02, y=0.98), font=dict(size=13))
fig.write_image(f"{OUT}/fig_dose_response.png", scale=2)

stages = ["base", "SFT", "DPO", "instruct (RLVR)"]
ol = [ladder_pts[l] for l in ["OLMo-2 base (pretrained)", "OLMo-2 SFT",
                              "OLMo-2 DPO", "OLMo-2 instruct (RLVR)"]]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=stages, y=ol, mode="lines+markers",
                          name="OLMo-2 ladder", line=dict(color="#ff7f0e",
                                                          width=2.5)))
fig2.add_trace(go.Scatter(
    x=["base", "instruct (RLVR)"],
    y=[ladder_pts["Qwen2.5-7B base (bare)"],
       ladder_pts["Qwen2.5-7B instruct (bare)"]],
    mode="lines+markers", name="Qwen2.5-7B (bare)",
    line=dict(color="#d62728", width=2.5, dash="dot")))
fig2.add_hline(y=ladder_pts["Llama-3.1-8B instruct (bare) — control"],
               line_dash="dash", line_color="#1f77b4",
               annotation_text="Llama-3.1-8B instruct (bare)",
               annotation_position="bottom right")
fig2.update_layout(
    template="plotly_white", width=680, height=440,
    title="Post-training installs self-perception (K=8 shift, bare-text "
          "protocol)",
    yaxis=dict(title="mean target EV shift at K=8"),
    xaxis=dict(title="post-training stage"),
    legend=dict(x=0.02, y=0.98), font=dict(size=13))
fig2.write_image(f"{OUT}/fig_ladder.png", scale=2)

md += ["## Figures", "",
       "- `fig_dose_response.png` — Exhibit 1b as curves (log-x)",
       "- `fig_ladder.png` — Exhibit 3, OLMo ladder + Qwen base pair + "
       "Llama8 control", ""]

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/tables.md", "w").write("\n".join(md))
print(f"wrote {OUT}/tables.md + 2 figures\n")
print("=== VERIFY against design_selfperception.md ===")
for label, got in verify:
    print(f"\nreport: {label}\ncomputed: {got}")
