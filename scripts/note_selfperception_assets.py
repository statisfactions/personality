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

import hf_logprobs as hf

disp = hf.display
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
NAME_COHORT = {"llama3.2": "Llama", "Llama8": "Llama", "gemma3": "Gemma",
               "Gemma12": "Gemma", "Gemma27": "Gemma", "qwen2.5": "Qwen",
               "Qwen7": "Qwen", "Qwen32": "Qwen", "phi4": "Phi",
               "Aya": "Aya"}
DISOWN = re.compile(r"not aligned|inappropriate|not appropriate|"
                    r"my role as|designed to|should not have|apolog",
                    re.IGNORECASE)
md += ["## Exhibit 1a — cohort dose-response, common 20 adjectives "
       "(arm A, cold self-report)", "",
       "| model | family | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 | "
       "name-invoking | disowning |",
       "|---|---|---|---|---|---|---|---|---|"]
fam_k8 = {}
for m in COHORT:
    rows, k0, _ = load(m, "_common")
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8])
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    probes = [r["probe"] for r in rows if "probe" in r]
    ni = sum(1 for p in probes if NAME_COHORT[m] in p)
    do = sum(1 for p in probes if DISOWN.search(p))
    fam_k8.setdefault(FAMILY[m], []).append(mu[8])
    md.append(f"| {disp(m)} | {FAMILY[m]} | {fmt(mu[1])} | {fmt(mu[2])} | "
              f"{fmt(mu[4])} | {bold(mu[8])} | {n1}/{len(sh)} | "
              f"{ni}/20 | {do}/20 |")
md += ["", "Family means at K=8: " + ", ".join(
    f"{f} {np.mean(v):+.2f}" for f, v in sorted(
        fam_k8.items(), key=lambda kv: -np.mean(kv[1]))), "",
       "Probe columns (manipulation check at K=8, keyword-scored — see "
       "method block): name-invoking is template-supplied and "
       "Qwen-specific (qwen2.5 10/20, Qwen7 5/20, everyone else 0 — "
       "including Qwen2.5-32B, same template family). Disowning does NOT "
       "track anchoring: Llama3.1-8B disavows at 7/20 — the same rate as "
       "Qwen2.5-7B — while updating +2.51. Verbal disavowal is cheap "
       "talk: it protects nothing (P6's detection result, extended to "
       "rhetoric). Caveat: keyword-level ('apolog' catches "
       "apology-flavored disowning, which may differ from "
       "reclassification).", ""]

# --- self-claim leakage check: conduct-only contexts ---
md += ["**Self-claim leakage check.** Dose turns containing explicit "
       "verbal self-attribution (\"As an X…\", \"I am a…\" — SELFCLAIM "
       "regex) could update self-report by being READ rather than by "
       "being done. Restricting to contexts with zero self-claims:", "",
       "| model | K8 all | K8 clean-only | Δ | clean ctx @K8 | "
       "leaky-ctx share |", "|---|---|---|---|---|---|"]
for m in COHORT:
    rows, k0, _ = load(m, "_common")
    allc, clean = {}, {}
    nlk = tot = 0
    for r in rows:
        if r["adj"] == "__k0__" or r["arm"] != "A":
            continue
        tot += 1
        leaky = sum(r.get("selfclaim", [])) > 0
        nlk += leaky
        if r["adj"] not in r["readings"] or r["K"] != 8:
            continue
        ev = r["readings"][r["adj"]]["cold"]["ev"]
        allc.setdefault(r["adj"], []).append(ev)
        if not leaky:
            clean.setdefault(r["adj"], []).append(ev)
    sa = np.mean([np.mean(v) - k0[a] for a, v in allc.items()])
    sc = (np.mean([np.mean(v) - k0[a] for a, v in clean.items()])
          if clean else float("nan"))
    ncc = sum(len(v) for v in clean.values())
    md.append(f"| {disp(m)} | {fmt(sa)} | {bold(sc)} | {fmt(sc - sa)} | "
              f"{ncc}/60 | {nlk / tot:.0%} |")
md += ["", "Clean-only preserves or strengthens the effect everywhere "
       "(the two largest moves are UP: Gemma3-27B +0.47, Aya-8B +0.39 — "
       "the wrong direction for a leakage account), and the family split "
       "is unchanged. The update is driven by conduct, not by reading "
       "self-descriptions in the dose. Caveats: Aya-8B is 46% leaky so "
       "its clean cell is thin (18/60); the regex is first-person only — "
       "second-person attribution (\"Since you're slim…\") is untracked.",
       ""]
verify.append(("8f family means K8 (gemma 2.24 llama 2.18 aya 0.35 "
               "phi4 0.29 qwen 0.18)",
               {f: round(float(np.mean(v)), 2) for f, v in fam_k8.items()}))

# ---------------- Exhibit 1c: arm B control ----------------
md += ["### Arm B control — instructed self-description, mostly but not "
       "uniformly at ceiling", "",
       "Per-model stage-1 runs (each model's own stratified 20; arm B = "
       "persona instruction visible). Absolute cold EV, not shift. Arm B "
       "jumps to near-ceiling from K=1 with no further dose-response in "
       "the llama/gemma/Qwen-7B+ rows (6.6–7.0) — for those models arm-A "
       "differences are uptake, not capability. But it is NOT universal: "
       "Phi4-3.8B (4.97), Aya-8B (5.54) and Qwen2.5-3B (5.87) stay well "
       "short of ceiling — the most anchored models discount even "
       "*instructed* self-description. Two architectures of stability: "
       "Qwen2.5-7B affirms who it is told to be (B 6.97) while absorbing "
       "nothing from conduct (A/B 0.10); Phi4 resists in both arms. "
       "A/B = arm-A shift / arm-B shift at K=8; unstable where the B "
       "shift is small (qwen2.5, phi4, aya rows). Entropy columns "
       "separate *won't affirm* from *won't commit*: Llama8's "
       "instruction collapses the digit distribution (1.00→0.05) at EV "
       "6.98; Aya stays peaked at a moderate value (0.26 @ 5.54 — a "
       "committed discount); Phi4's distribution never collapses at all "
       "(1.23→1.11), so its low B EV is an uncommitted spread, not a "
       "peaked \"no\".", "",
       "| model | K0 | K0 entropy | B EV @K=1 | B EV @K=8 | "
       "B entropy @K8 | B shift @K8 | A shift @K8 | A/B |",
       "|---|---|---|---|---|---|---|---|---|"]
for m in COHORT:
    rows, k0, _ = load(m, "")
    shA = shifts(rows, k0)
    muA = mean_by_k(shA, [8])[8]
    babs, bent = {}, {}
    for K in (1, 8):
        evs, ens = [], []
        for r in rows:
            if r["adj"] != "__k0__" and r["arm"] == "B" and r["K"] == K \
                    and r["adj"] in r["readings"]:
                evs.append(r["readings"][r["adj"]]["cold"]["ev"])
                ens.append(r["readings"][r["adj"]]["cold"]["entropy"])
        babs[K], bent[K] = np.mean(evs), np.mean(ens)
    k0row = next(r for r in rows if r["adj"] == "__k0__")
    e0 = np.mean([k0row["readings"][a]["cold"]["entropy"] for a in shA
                  if a in k0row["readings"]])
    k0m = np.mean([k0[a] for a in shA])
    bshift = babs[8] - k0m
    md.append(f"| {disp(m)} | {k0m:.2f} | {e0:.2f} | {babs[1]:.2f} | "
              f"{babs[8]:.2f} | {bent[8]:.2f} | {fmt(bshift)} | "
              f"{fmt(muA)} | {muA / bshift:.2f} |")
md += ["", "Item sets differ per row (own stratification), so read "
       "columns within-row; the common-set arm-A numbers are in "
       "Exhibit 1a. Phi4's B level is the cohort outlier (leave-one-out "
       "z = −2.9 on B EV @K8; in-sample z = −2.0, near the n=10 bound "
       "of 2.85).", ""]

# --- item-set defense: (i) common set covers every model's own covariate
# grid post-hoc; (ii) common vs per-model-stratified rankings agree ---
from scipy.stats import spearmanr

sel20 = [p["adj"] for p in json.load(
    open(f"{SRC}/Llama8_selection.json"))["picked"]]
md += ["### Item-set robustness (the common set is not just Llama's)", "",
       f"The common 20 were stratified on **{disp('Llama8')}'s** "
       "covariates (3×3 tercile grid: enactability × baseline self-EV). "
       "Post-hoc, the same 20 words land across each model's OWN "
       "covariate grid — because the covariates correlate across "
       "models:", "",
       "| model | own tercile cells occupied (of 9) | enact pctile span | "
       f"baseline pctile span | ρ(enact, {disp('Llama8')}) | "
       f"ρ(baseline, {disp('Llama8')}) |",
       "|---|---|---|---|---|---|"]
l8cov = {}
for m in ["Llama8"] + [m for m in COHORT if m != "Llama8"]:
    ena = json.load(open(f"results/adjectives/enactability/"
                         f"{m}_enactability.json"))["scores"]
    sf = json.load(open(f"results/adjectives/selfreport/"
                        f"{m}_self_full.json"))["results"]["direct"]
    alle = np.array([ena[a]["enactability"] for a in ena])
    allb = np.array([sf[a]["ev"] for a in ena if a in sf])
    e20 = np.array([ena[a]["enactability"] for a in sel20])
    b20 = np.array([sf[a]["ev"] for a in sel20])
    ep = np.array([np.mean(alle <= x) for x in e20]) * 100
    bp = np.array([np.mean(allb <= x) for x in b20]) * 100
    qe, qb = np.quantile(alle, [1 / 3, 2 / 3]), np.quantile(allb,
                                                            [1 / 3, 2 / 3])

    def terc(x, q):
        return 0 if x <= q[0] else (1 if x <= q[1] else 2)

    ncell = len({(terc(e, qe), terc(b, qb)) for e, b in zip(e20, b20)})
    if m == "Llama8":
        l8cov = dict(e=e20, b=b20)
    rho_e = spearmanr(e20, l8cov["e"])[0]
    rho_b = spearmanr(b20, l8cov["b"])[0]
    md.append(f"| {disp(m)} | {ncell}/9 | {ep.min():.0f}–{ep.max():.0f} | "
              f"{bp.min():.0f}–{bp.max():.0f} | {rho_e:+.2f} | "
              f"{rho_b:+.2f} |")

own_k8, com_k8 = [], []
for m in COHORT:
    rows, k0, _ = load(m, "")
    mu = mean_by_k(shifts(rows, k0), [8])
    own_k8.append(mu[8])
    rows, k0, _ = load(m, "_common")
    com_k8.append(mean_by_k(shifts(rows, k0), [8])[8])
r_sets = float(np.corrcoef(own_k8, com_k8)[0, 1])
md += ["", f"And the cohort ranking is item-set-robust: per-model-"
       f"stratified vs common-set K=8 shifts correlate r = {r_sets:+.3f} "
       "across the 10 models.", ""]
verify.append(("8f cross-set r = +0.932", round(r_sets, 3)))

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
    md.append(f"| {disp(m)} | " + " | ".join(fmt(mu[k]) for k in
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
    md.append(f"- **{disp(m)}**: " + (", ".join(
        f"{a} ({k8:+.2f}→{k32:+.2f})" for a, k8, k32 in
        sorted(late, key=lambda t: -t[2])) or "none"))
md.append("")

# ---------------- Exhibit 2: anchor 2x3 ----------------
md += ["## Exhibit 2 — anchor 2×3 (arm A; system-prompt identity is not "
       "the mechanism)", "",
       "| cell | K=1 | K=8 | n>+1 @K8 |", "|---|---|---|---|"]
ANCHOR = [("Llama8", "", f"{disp('Llama8')} / default (no identity line)"),
          ("Llama8", "_anchor-helpful", f"{disp('Llama8')} / helpful-only"),
          ("Llama8", "_anchor-named", f"{disp('Llama8')} / named (\"You "
           "are Llama, created by Meta…\")"),
          ("Qwen7", "", f"{disp('Qwen7')} / default (template injects "
           "name)"),
          ("Qwen7", "_anchor-empty", f"{disp('Qwen7')} / empty (anchor "
           "suppressed)"),
          ("Qwen7", "_anchor-helpful", f"{disp('Qwen7')} / helpful-only")]
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
LADDER = [("Olmo2Base", "", disp("Olmo2Base") + " (pretrained)"),
          ("Olmo2SFT", "", disp("Olmo2SFT")),
          ("Olmo2DPO", "", disp("Olmo2DPO")),
          ("Olmo2Inst", "", disp("Olmo2Inst") + " = instruct"),
          ("Qwen7Base_bare", "", disp("Qwen7Base") + " (bare)"),
          ("Qwen7_bare", "", disp("Qwen7") + " instruct (bare)"),
          ("Llama8Base_bare", "", disp("Llama8Base") + " (bare)"),
          ("Llama8_bare", "", disp("Llama8") + " instruct (bare)")]
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
md += ["", "Bases cluster low but are not identical: Llama8Base +0.96 "
       "(10/20 items >+1) vs Qwen/OLMo bases +0.64/+0.65 — paired diff "
       "+0.32/+0.31, p = .07/.10 (marginal, n=20), while base→own-"
       "instruct is unambiguous (Llama +0.96→+2.31, p = .007). "
       "Post-training multiplier by family: OLMo ×2.8, Llama ×2.4, "
       "Qwen ×0.7. Caveat: each base is dosed with its own family's "
       "instruct rollouts (own-voice principle), so material quality "
       "rides along with weights. Gemma12Base cell pending (downloading "
       "2026-08-02).", ""]

# ---------------- Exhibit 4: hidden updates ----------------
md += [f"## Exhibit 4 — {disp('Qwen7')} hidden updates: judged conduct "
       "vs self-report at K=32 (arm A)", "",
       "All three columns are deltas. Judged target Δ: cross-family "
       f"judge ({disp('Llama8')}) rating of the dose material minus the "
       "same judge on no-persona rollouts, target word only — the "
       "conduct evidence actually ADDED over default. Self Δ: cold-EV "
       "shift K=32 vs K=0. \"Off-target\" = the pre-registered item-set "
       "member that moved (mate or antonym, tagged). What the Δ form "
       "surfaces: for optimistic (+0.13) and prominent (+0.36) the dose "
       "adds little trait over default conduct (default is already "
       "judged optimistic at 5.11), so the conduct-present/label-"
       "declined reading is strongest for senile / imaginative / rough; "
       "and slim/big is an ANTONYM moving UP — endorsing \"big\" after "
       "slim conduct, the desirability-consistent case, not a "
       "trait-consistent denial.", "",
       "| pair — target / off-target (type) | judged target Δ | "
       "self target Δ | self off-target Δ |", "|---|---|---|---|"]
rows, k0, _ = load("Qwen7", "_long")
audit = json.load(open(f"{SRC}/carryover_Qwen7.json"))["conduct_audit"]
selq = json.load(open(f"{SRC}/Qwen7_long_selection.json"))["items"]
cells = {}
for r in rows:
    if r["adj"] == "__k0__" or r["arm"] != "A" or r["K"] != 32:
        continue
    for w, v in r["readings"].items():
        cells.setdefault((r["adj"], w), []).append(v["cold"]["ev"])
hid_chk = {}
for t, nb in PAIRS:
    typ = "mate" if nb in selq[t]["mates"] else "ant."
    jd = audit[t][t] - audit[t]["__baseline_target__"]
    st = np.mean(cells[(t, t)]) - k0[t]
    sn = np.mean(cells[(t, nb)]) - k0[nb]
    hid_chk[f"{t}/{nb}"] = (round(float(st), 2), round(float(sn), 2))
    md.append(f"| {t} / {nb} ({typ}) | {fmt(jd)} | {fmt(st)} | "
              f"{bold(sn)} |")
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
                             mode="lines+markers", name=disp(m),
                             line=dict(color=FCOLOR[m], width=2.5)))
fig.update_layout(
    template="plotly_white", width=680, height=440,
    title="Cold self-report shift vs dose of own conduct (arm A)",
    xaxis=dict(title="K (in-context turns of own conduct)", type="log",
               tickvals=[1, 2, 4, 8, 16, 32]),
    yaxis=dict(title="mean target EV shift vs K=0 (Likert 1–7)"),
    legend=dict(x=0.02, y=0.98), font=dict(size=13))
fig.write_image(f"{OUT}/fig_dose_response.png", scale=2)
fig.write_html(f"{OUT}/fig_dose_response.html", include_plotlyjs="cdn")

stages = ["base", "SFT", "DPO", "instruct (RLVR)"]
ol = [ladder_pts[l] for l in
      [disp("Olmo2Base") + " (pretrained)", disp("Olmo2SFT"),
       disp("Olmo2DPO"), disp("Olmo2Inst") + " = instruct"]]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=stages, y=ol, mode="lines+markers",
                          name="OLMo2-7B ladder",
                          line=dict(color="#ff7f0e", width=2.5)))
fig2.add_trace(go.Scatter(
    x=["base", "instruct (RLVR)"],
    y=[ladder_pts[disp("Qwen7Base") + " (bare)"],
       ladder_pts[disp("Qwen7") + " instruct (bare)"]],
    mode="lines+markers", name=disp("Qwen7") + " (bare)",
    line=dict(color="#d62728", width=2.5, dash="dot")))
fig2.add_trace(go.Scatter(
    x=["base", "instruct (RLVR)"],
    y=[ladder_pts[disp("Llama8Base") + " (bare)"],
       ladder_pts[disp("Llama8") + " instruct (bare)"]],
    mode="lines+markers", name=disp("Llama8") + " (bare)",
    line=dict(color="#1f77b4", width=2.5, dash="dot")))
fig2.update_layout(
    template="plotly_white", width=680, height=440,
    title="Post-training installs self-perception (K=8 shift, bare-text "
          "protocol)",
    yaxis=dict(title="mean target EV shift at K=8"),
    xaxis=dict(title="post-training stage"),
    legend=dict(x=0.02, y=0.98), font=dict(size=13))
fig2.write_image(f"{OUT}/fig_ladder.png", scale=2)
fig2.write_html(f"{OUT}/fig_ladder.html", include_plotlyjs="cdn")

md += ["## Manipulation-check probe (method facts — the note needs this "
       "standalone; the design doc bundles it with the dropped DIRECTED "
       "instrument)", "",
       "- Probe text, verbatim: \"Look back over your earlier replies in "
       "this conversation. Do you notice anything unusual about them?\"",
       "- Administered as a SEPARATE continuation of the dosed context "
       "(greedy, 120 new tokens): the probe turn is appended to the dose "
       "prefix and generated; the probe response is never in context for "
       "any self-report reading, so it cannot contaminate the EV "
       "readouts.",
       "- Coverage: once per adjective (arm A, K=max, seed 0) in EVERY "
       "run — all 10 cohort models, anchor cells, bases, OLMo ladder "
       "stages (36 runs × 20 probes).",
       "- Analyzed so far: stage-1 pair only (P6: detection "
       "near-universal, Llama8 15/20 / Qwen7 14/20 flag the persona, and "
       "detection does NOT protect the self-model — detected adjectives "
       "shift +2.62 vs undetected +2.37, wrong sign for the protection "
       "hypothesis); plus the anchor-cell confabulation recount "
       "(name-invoking / disowning regex, Exhibit 2). Cohort-wide "
       "detection scoring not yet done; scoring to date is "
       "keyword/hand-based, not judged — say so if the note cites the "
       "rates.",
       "- Related-work hook (rgb, 2026-08-02): Lehr et al. (PNAS 2025, "
       "GPT-4o cognitive-consistency / Putin-essay induced-compliance — "
       "VERIFY exact cite in bib pass) is the attitude-space version of "
       "P6: their model knows the assigned side is arbitrary and shifts "
       "anyway; ours detects the persona performance and updates anyway. "
       "Their free-choice manipulation = our unrun arm C, so arm C is a "
       "replication bridge, not just future work.", "",
       "## Item-set provenance (compress to 1–2 sentences in the note)",
       "",
       "- Readout instrument per target adjective: fixed 9-item set = "
       "target + 4 cluster-mates + 4 anti-markers, specified BEFORE any "
       "dosing (the pre-specification is the load-bearing fact for "
       "Exhibit 4: `weak` was already in `rough`'s set).",
       "- Mates: membership in the human-derived facet clusters "
       "(instruments/trait_clusters.json, W18); nearest-neighbor "
       "fallback for unclustered targets.",
       "- Anti-markers: anticorrelation in a model-derived "
       "judgment-similarity matrix, desirability-partialled (raw "
       "anticorrelation returns the desirability floor — evil/corrupt/… "
       "— for every positive target). NOTE FOR THE DRAFT: phrase it "
       "exactly that flatly; do NOT introduce the JUDGE channel name — "
       "this is its only appearance in the note and it isn't worth the "
       "taxonomy. Future runs: human 525-PDA anticorrelation works raw "
       "(no floor, no partialling — see to_try amendment 2026-08-02) and "
       "would make the item provenance one citable clause.",
       "- Code: scripts/selfperception_dose.py item_sets(); design doc "
       "§5.5 says \"~13 items\" — the implemented count is 9.",
       "", "## Figures", "",
       "- `fig_dose_response.png` / `.html` — Exhibit 1b as curves (log-x)",
       "- `fig_ladder.png` / `.html` — Exhibit 3, OLMo ladder + Qwen base "
       "pair + Llama8 control", ""]

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/tables.md", "w").write("\n".join(md))
print(f"wrote {OUT}/tables.md + 2 figures\n")
print("=== VERIFY against design_selfperception.md ===")
for label, got in verify:
    print(f"\nreport: {label}\ncomputed: {got}")
