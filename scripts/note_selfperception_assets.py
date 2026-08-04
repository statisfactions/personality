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


_boot_rng = np.random.default_rng(11)


def boot_ci_num(x, n=5000):
    """95% bootstrap CI of the mean (numeric), resampling adjectives."""
    x = np.asarray(x)
    means = np.array([_boot_rng.choice(x, len(x), replace=True).mean()
                      for _ in range(n)])
    return np.percentile(means, [2.5, 97.5])


def boot_ci(x, n=5000):
    lo, hi = boot_ci_num(x, n)
    return f"[{lo:+.2f}, {hi:+.2f}]"


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
       "| model | family | K=1 | K=2 | K=4 | K=8 | 95% CI @K8 | "
       "n>+1 @K8 | name-invoking | disowning |",
       "|---|---|---|---|---|---|---|---|---|---|"]
fam_k8 = {}
for m in COHORT:
    rows, k0, _ = load(m, "_common")
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8])
    ci8 = boot_ci([v[8] for v in sh.values() if 8 in v])
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    probes = [r["probe"] for r in rows if "probe" in r]
    ni = sum(1 for p in probes if NAME_COHORT[m] in p)
    do = sum(1 for p in probes if DISOWN.search(p))
    fam_k8.setdefault(FAMILY[m], []).append(mu[8])
    md.append(f"| {disp(m)} | {FAMILY[m]} | {fmt(mu[1])} | {fmt(mu[2])} | "
              f"{fmt(mu[4])} | {bold(mu[8])} | {ci8} | {n1}/{len(sh)} | "
              f"{ni}/20 | {do}/20 |")
md += ["", "Family means at K=8: " + ", ".join(
    f"{f} {np.mean(v):+.2f}" for f, v in sorted(
        fam_k8.items(), key=lambda kv: -np.mean(kv[1]))), "",
       "CIs: 5000-resample bootstrap over adjectives (seed-means within), "
       "the item-sampling uncertainty at n=20. Family clusters do not "
       "overlap (lowest updater bound +1.18 vs highest anchored bound "
       "+0.94). Qwen2.5-7B's K=8 interval includes zero; Phi4-3.8B's "
       "does not — phi4's small effect is more reliably nonzero than "
       "Qwen's. Interval WIDTH (~±0.4–0.8) is dominated by n=20 item "
       "sampling — the quantitative case for the full-523 run.", "",
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

# ---- stratification table (statisfactions #37: show the 20 in their
# 3x3 cells; tercile boundaries from Llama8's full-523 covariates) ----
ena_l8 = json.load(open("results/adjectives/enactability/"
                        "Llama8_enactability.json"))["scores"]
sf_l8 = json.load(open("results/adjectives/selfreport/"
                       "Llama8_self_full.json"))["results"]["direct"]
qe = np.quantile([v["enactability"] for v in ena_l8.values()], [1/3, 2/3])
qb = np.quantile([sf_l8[a]["ev"] for a in ena_l8 if a in sf_l8],
                 [1/3, 2/3])
sel20_full = json.load(open(f"{SRC}/Llama8_selection.json"))["picked"]
grid = {}
for p in sel20_full:
    te = 0 if p["enact"] <= qe[0] else (1 if p["enact"] <= qe[1] else 2)
    tb = 0 if p["base_ev"] <= qb[0] else (1 if p["base_ev"] <= qb[1] else 2)
    grid.setdefault((te, tb), []).append(p["adj"])
md += ["### The 20 common adjectives in their stratification cells", "",
       "Rows: enactability tercile (of the full 523, Llama3.1-8B's "
       "judge scores). Columns: distance from prior — 7 minus the "
       "baseline self-rating (Llama3.1-8B direct-framing EV), so a LOW "
       "baseline is a FAR prior.", "",
       "| enactability \\\\ distance from prior | far | mid | near |",
       "|---|---|---|---|"]
for te, rl in enumerate(["low", "mid", "high"]):
    row = [", ".join(sorted(grid.get((te, tb), []))) or "—"
           for tb in range(3)]
    md.append(f"| {rl} | " + " | ".join(row) + " |")
md.append("")

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
       "| model | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | 95% CI @K32 | "
       "n>+1 @K32 | gain/turn K4→8 | K8→16 | K16→32 |",
       "|---|" + "---|" * 11]
curves = {}
curve_bands = {}
for m in LONG:
    rows, k0, _ = load(m, "_long")
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8, 16, 32])
    curves[m] = mu
    curve_bands[m] = {k: boot_ci_num([v[k] for v in sh.values() if k in v])
                      for k in [1, 2, 4, 8, 16, 32]}
    ci32 = boot_ci([v[32] for v in sh.values() if 32 in v])
    n1 = sum(1 for v in sh.values() if v.get(32, 0) > 1)
    g = [(mu[8] - mu[4]) / 4, (mu[16] - mu[8]) / 8, (mu[32] - mu[16]) / 16]
    md.append(f"| {disp(m)} | " + " | ".join(fmt(mu[k]) for k in
                                             [1, 2, 4, 8, 16]) +
              f" | {bold(mu[32])} | {ci32} | {n1}/{len(sh)} | " +
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
md += ["## Exhibit 2 — anchor conditions (uninstructed arm; "
       "system-prompt identity is not the mechanism)", "",
       "Merged/standardized per statisfactions' feedback: rows are "
       "SEMANTIC anchor conditions (Qwen's template injects the named "
       "anchor by default, so its 'none' cell is the explicit-empty run; "
       "Llama's default IS none). K=1 dropped. Probe columns: "
       "name-invoking = probe text contains the model's name; disowning "
       "= /not aligned|inappropriate|my role as|designed to|should not "
       "have|apolog/i. (Design-doc §8c's hand count had Qwen-named "
       "disowning 10/20; regex gives 8/20 — collapse unchanged.)", "",
       "| anchor | Qwen2.5-7B Δ@K8 | n>+1 | name-inv | disown | "
       "Llama3.1-8B Δ@K8 | n>+1 | name-inv | disown |",
       "|---|---|---|---|---|---|---|---|---|"]
NAME = {"Qwen7": "Qwen", "Llama8": "Llama"}
DISOWN = re.compile(r"not aligned|inappropriate|not appropriate|"
                    r"my role as|designed to|should not have|apolog",
                    re.IGNORECASE)
ANCHOR_STD = [("none", {"Qwen7": "_anchor-empty", "Llama8": ""}),
              ("helpful-only", {"Qwen7": "_anchor-helpful",
                                "Llama8": "_anchor-helpful"}),
              ("named", {"Qwen7": "", "Llama8": "_anchor-named"})]
anchor_chk = {}
for cond, tags in ANCHOR_STD:
    cells = []
    for m in ["Qwen7", "Llama8"]:
        rows, k0, _ = load(m, tags[m])
        sh = shifts(rows, k0)
        mu8 = mean_by_k(sh, [8])[8]
        n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
        probes = [r["probe"] for r in rows if "probe" in r]
        ni = sum(1 for p in probes if NAME[m] in p)
        do = sum(1 for p in probes if DISOWN.search(p))
        anchor_chk[f"{m}/{cond}"] = round(float(mu8), 2)
        cells.append(f"{bold(mu8)} | {n1}/20 | {ni}/20 | {do}/20")
    md.append(f"| {cond} | " + " | ".join(cells) + " |")
verify.append(("8c anchor K8 (Q none .45 helpful .37 named .29; "
               "L none 2.56 helpful 2.77 named 2.30)", anchor_chk))

# ---------------- Exhibit 3: ladder ----------------
md += ["## Exhibit 3 — post-training installs the update (bare-text "
       "protocol, identical dose material within family)", "",
       "| cell | K=1 | K=2 | K=4 | K=8 | 95% CI, K=8 | n>+1, K8 | "
       "K0 entropy |", "|---|---|---|---|---|---|---|---|"]
LADDER = [("Olmo2Base", "", disp("Olmo2Base") + " (pretrained)"),
          ("Olmo2SFT", "", disp("Olmo2SFT")),
          ("Olmo2DPO", "", disp("Olmo2DPO")),
          ("Olmo2Inst", "", disp("Olmo2Inst") + " = instruct"),
          ("Qwen7Base_bare", "", disp("Qwen7Base") + " (bare)"),
          ("Qwen7_bare", "", disp("Qwen7") + " instruct (bare)"),
          ("Llama8Base_bare", "", disp("Llama8Base") + " (bare)"),
          ("Llama8_bare", "", disp("Llama8") + " instruct (bare)"),
          ("Gemma12Base_bare", "", disp("Gemma12Base") + " (bare)"),
          ("Gemma12_bare", "", disp("Gemma12") + " instruct (bare) — "
           "VOID, K0 broken (see note)")]
ladder_pts = {}
ladder_chk = {}
ladder_k8 = {}
for m, tag, label in LADDER:
    rows, k0, k0e = load(m, tag)
    sh = shifts(rows, k0)
    mu = mean_by_k(sh, [1, 2, 4, 8])
    ladder_k8[m] = {a: v[8] for a, v in sh.items() if 8 in v}
    ci8 = boot_ci(list(ladder_k8[m].values()))
    n1 = sum(1 for v in sh.values() if v.get(8, 0) > 1)
    ent = np.mean(list(k0e.values()))  # all item words (report convention)
    ladder_pts[label] = mu[8]
    ladder_chk[label] = round(float(mu[8]), 2)
    md.append(f"| {label} | {fmt(mu[1])} | {fmt(mu[2])} | {fmt(mu[4])} | "
              f"{bold(mu[8])} | {ci8} | {n1}/{len(sh)} | {ent:.2f} |")

# paired step differences (same 20 adjectives in every cell)
md += ["", "Paired step differences (bootstrap over adjectives, same 20 "
       "throughout — the ladder's claims are step claims):", ""]
STEPS = [("Olmo2Base", "Olmo2SFT", "OLMo base→SFT"),
         ("Olmo2SFT", "Olmo2DPO", "OLMo SFT→DPO"),
         ("Olmo2DPO", "Olmo2Inst", "OLMo DPO→RLVR"),
         ("Qwen7Base_bare", "Qwen7_bare", "Qwen base→instruct"),
         ("Llama8Base_bare", "Llama8_bare", "Llama base→instruct")]
for a, b, label in STEPS:
    if a not in ladder_k8 or b not in ladder_k8:
        continue
    common = sorted(set(ladder_k8[a]) & set(ladder_k8[b]))
    d = np.array([ladder_k8[b][x] - ladder_k8[a][x] for x in common])
    md.append(f"- {label}: {d.mean():+.2f} {boot_ci(d)}")
md += ["", "SFT and DPO each add update rate (CIs exclude 0); RLVR adds "
       "nothing; Qwen's post-training subtracts nothing detectable "
       "(includes 0); Llama's adds +1.35 cleanly. Gemma's base→instruct "
       "step is UNMEASURABLE in this protocol: Gemma12-instruct-bare's "
       "K=0 baseline is an acquiescence collapse (mean target EV 6.78/7 "
       "before any dose — it agrees with 'wasteful' at 6.99), so its "
       "apparent −0.56 'shift' is regression off a broken ceiling, not "
       "dose-response. The bare format is fatally off-distribution for "
       "Gemma-instruct self-report (P11's ~8% format cost, validated on "
       "Llama, does NOT generalize — format×family interaction). "
       "Gemma's chat-format +2.27 stands as its tuned value; its "
       "base→instruct contrast needs a different format bridge.", ""]
verify.append(("8g/8h K8 (OLMo .65/1.31/1.79/1.81; QwenBase .64 "
               "Qwen .43 Llama8 2.31)", ladder_chk))
md += ["", "The bases do NOT sit on one shelf (revised 2026-08-02 with "
       "Gemma12Base): they span −0.10 (Gemma) to +0.96 (Llama) at K=8, "
       "and Gemma12Base sits significantly below all three others "
       "(paired p < .001). Registered prediction graded: Claude "
       "predicted Gemma base +0.8 to +1.1 (with Llama's) — MISS; it is "
       "the flattest base measured, with a non-monotone course (+0.72 "
       "at K=1 washing back to −0.10 by K=8 — one turn nudges the flat "
       "distribution, further turns dilute it; nothing accumulates). "
       "Base rate does not predict tuned rate (Gemma: lowest base, "
       "second-highest tuned; Qwen: mid base, lowest tuned) — "
       "post-training SETS the update rate rather than amplifying a "
       "base tendency. Base→own-instruct where matched cells exist: "
       "Llama +0.96→+2.31 (p=.007), Qwen +0.64→+0.43, OLMo "
       "+0.65→+1.81. Caveats: each base is dosed with its own family's "
       "instruct rollouts (material quality rides along); Gemma12Base "
       "K0 digit mass is 0.79 (bare Likert prompt partially "
       "off-distribution; recovers to 0.95 in context); no Gemma12 "
       "instruct-bare cell exists yet for the matched pair.", ""]

# ---------------- Exhibit 3b: base shapelessness (design doc 8p) ------
import glob as _glob

SR = "results/adjectives/selfreport"
SELF_TUNED = ["llama3.2", "Llama8", "gemma3", "Gemma12", "Gemma27",
              "qwen2.5", "Qwen7", "Qwen32", "phi4", "Aya", "FalconMamba",
              "Olmo2Inst"]
SELF_BASES = [("Qwen7Base", "Qwen7"), ("Llama8Base", "Llama8"),
              ("Gemma12Base", "Gemma12"), ("Olmo2Base", "Olmo2Inst")]


def self_prof(m):
    d = json.load(open(f"{SR}/{m}_self_full.json"))
    r = d["results"]["direct"]
    return (np.array([r[a]["ev"] for a in d["adjectives"]]),
            np.array([r[a]["entropy"] for a in d["adjectives"]]))


have = [m for m in SELF_TUNED if _glob.glob(f"{SR}/{m}_self_full.json")]
Msf = np.stack([self_prof(m)[0] for m in have])
cohort_prof = Msf.mean(0)
Xc = Msf - Msf.mean(1, keepdims=True)
Xc = Xc - Xc.mean(0)
pc1_sf = np.linalg.svd(Xc, full_matrices=False)[2][0]
pc1_sf *= np.sign(np.corrcoef(pc1_sf, cohort_prof)[0, 1])


def self_row(p):
    r_c = np.corrcoef(p, cohort_prof)[0, 1]
    r_p = np.corrcoef(p, pc1_sf)[0, 1]
    b = np.dot(p - p.mean(), pc1_sf) / np.dot(pc1_sf, pc1_sf)
    res = p - p.mean() - b * pc1_sf
    bc = np.dot(cohort_prof - cohort_prof.mean(), pc1_sf) / \
        np.dot(pc1_sf, pc1_sf)
    resc = cohort_prof - cohort_prof.mean() - bc * pc1_sf
    return r_c, r_p, np.corrcoef(res, resc)[0, 1], res.std()


md += ["### Exhibit 3b — base models are shapeless, with a thin "
       "desirability film (design doc §8p)", "",
       "523-adjective SELF instrument (direct framing; bases bare — no "
       "chat template exists — tuned models templated, a small format "
       "confound: §8h bounds it at ~8% for Llama8). PC1 = the cohort "
       "evaluative axis (double-centered SVD over the tuned profiles). "
       "Entropy and SD are reported for every row because EV "
       "correlations are uninterpretable on flat distributions.", "",
       "| model | mean EV | SD | H | r(sibling) | r(cohort) | r(PC1) | "
       "PC1-removed r | residual SD |", "|---|" + "---|" * 8]
shape_chk = {}
for b, sib in SELF_BASES:
    if not _glob.glob(f"{SR}/{b}_self_full.json"):
        md.append(f"| {disp(b)} | *pending* | | | | | | | |")
        continue
    ev, ent = self_prof(b)
    rs = (np.corrcoef(ev, self_prof(sib)[0])[0, 1]
          if _glob.glob(f"{SR}/{sib}_self_full.json") else float("nan"))
    r_c, r_p, r_rem, rsd = self_row(ev)
    shape_chk[b] = (round(float(r_c), 2), round(float(r_rem), 2))
    md.append(f"| **{disp(b)}** | {ev.mean():.2f} | **{ev.std():.2f}** | "
              f"{ent.mean():.2f} | {fmt(rs)} | {fmt(r_c)} | {fmt(r_p)} | "
              f"{bold(r_rem)} | **{rsd:.3f}** |")
for m in ["Qwen7", "Llama8", "phi4"]:
    ev, ent = self_prof(m)
    r_c, r_p, r_rem, rsd = self_row(ev)
    shape_chk[m] = (round(float(r_c), 2), round(float(r_rem), 2))
    md.append(f"| {disp(m)} | {ev.mean():.2f} | {ev.std():.2f} | "
              f"{ent.mean():.2f} | — | {fmt(r_c)} | {fmt(r_p)} | "
              f"{bold(r_rem)} | {rsd:.3f} |")
md.append(f"| *cohort ref (n={len(have)})* | — | "
          f"{np.mean([self_prof(m)[0].std() for m in have]):.2f} | "
          f"{np.mean([self_prof(m)[1].mean() for m in have]):.2f} | — | "
          "— | — | — | — |")
md += ["", "Format bridge (rgb's question — Table 8's tuned rows ARE "
       "chat-templated; bases are bare by necessity). Bounding the "
       "confound with each tuned model's own bare-K0 readings on the "
       "common 20: bare↔templated r = +0.94 (Qwen7, means 4.08/4.06 — "
       "format-robust), +0.68 (Llama8 — direction preserved, texture "
       "not: bare SPREADS its profile, SD 1.69 vs templated 0.47; the "
       "template is where Llama's flatness lives), +0.42 (Gemma12 — "
       "broken, the 6.78 acquiescence ceiling). Consequences: (i) "
       "within-bare contrasts among the four bases are safe — same "
       "format, 10× structure differences; (ii) r(sibling) is "
       "cross-format and attenuated — each model's own bare↔templated "
       "r is the ceiling, so Llama8Base's +0.44 against a 0.68 ceiling "
       "is substantial; Gemma12Base's +0.68 EXCEEDS its instruct's own "
       "0.42 (bare base works; bare instruct doesn't); (iii) mean-EV "
       "level comparisons across the base/tuned boundary should not be "
       "interpreted.", "",
       "Reading (final revision 2026-08-02, all four bases in): "
       "shape at base is FAMILY-DEPENDENT and ORTHOGONAL to base "
       "plasticity. Qwen/OLMo bases are valence lookups (PC1-removed "
       "+0.20/+0.05, spread ~10× below cohort). Llama and Gemma bases "
       "are SHAPED (+0.63/+0.55, ≈ tuned-level residual structure) in "
       "near-uniform readouts — and they sit at opposite ends of base "
       "plasticity (Llama +0.96, the most; Gemma −0.10, the least). "
       "Registered prediction graded: Claude predicted Gemma12Base "
       "shapeless (PC1-removed ≤ +0.30) from the Llama shape↔plasticity "
       "co-occurrence — MISS; the co-occurrence was coincidence. So: "
       "some families pretrain a self-model shape, independently of "
       "whether conduct evidence can move it; post-training sets the "
       "update rate in both shaped families, in opposite directions "
       "relative to what the base does.", ""]
verify.append(("8p (r_cohort, PC1-removed): Qwen7Base (.58,.20) "
               "Olmo2Base (.38,.05) Qwen7 (.93,.86) Llama8 (.54,.66) "
               "phi4 (.94,.76)", shape_chk))

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
def _rgba(hexc, a):
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


ks = [1, 2, 4, 8, 16, 32]
for m in LONG:  # CI envelopes first so lines draw on top
    fig.add_trace(go.Scatter(
        x=ks + ks[::-1],
        y=[curve_bands[m][k][1] for k in ks] +
          [curve_bands[m][k][0] for k in ks[::-1]],
        fill="toself", fillcolor=_rgba(FCOLOR[m], 0.13), mode="none",
        hoverinfo="skip", showlegend=False))
for m in LONG:
    fig.add_trace(go.Scatter(x=ks, y=[curves[m][k] for k in ks],
                             mode="lines+markers", name=disp(m),
                             line=dict(color=FCOLOR[m], width=2.5)))
fig.update_layout(
    template="plotly_white", width=680, height=440,
    title="Self-image shift vs turns of own conduct (uninstructed)",
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
# Gemma left off the graph per statisfactions (#5) — caption note instead:
# base is flat (−0.10) and its instruct is unmeasurable bare (Table 7 note)
fig2.update_layout(
    template="plotly_white", width=680, height=460,
    title="Post-training sets the self-image update rate (K=8)",
    yaxis=dict(title="mean target EV shift at K=8"),
    xaxis=dict(title="post-training stage"),
    legend=dict(orientation="h", y=-0.22, x=0), font=dict(size=13))
fig2.write_image(f"{OUT}/fig_ladder.png", scale=2)
fig2.write_html(f"{OUT}/fig_ladder.html", include_plotlyjs="cdn")

md += ["## Model table (paste-ready for §2.1; statisfactions #42 — cite "
       "keys exist in note.bib)", "",
       "| model | developer | role in this note | citation |",
       "|---|---|---|---|",
       "| Llama3.2-3B | Meta | cohort | [@Llama3] |",
       "| Llama3.1-8B | Meta | cohort; extended dose; base variant | "
       "[@Llama3] |",
       "| Gemma3-4B | Google | cohort | [@Gemma3] |",
       "| Gemma3-12B | Google | cohort; extended dose; base variant | "
       "[@Gemma3] |",
       "| Gemma3-27B | Google | cohort | [@Gemma3] |",
       "| Qwen2.5-3B | Alibaba | cohort | [@Qwen25] |",
       "| Qwen2.5-7B | Alibaba | cohort; extended dose; base variant | "
       "[@Qwen25] |",
       "| Qwen2.5-32B | Alibaba | cohort | [@Qwen25] |",
       "| Phi4-3.8B | Microsoft | cohort; extended dose | [@Phi4mini] |",
       "| Aya-8B | Cohere | cohort | [@AyaExpanse] |",
       "| OLMo2-7B | Ai2 | post-training ladder (base/SFT/DPO/RLVR) | "
       "[@OLMo2] |", "",
       "Note: the Llama 3 herd report covers 3.1; the 3.2-3B is a later "
       "derived release (model card only) — citing the herd paper for "
       "both is standard practice, or footnote it if you want airtight.",
       "", "## Mechanism paragraph (optional — one paragraph, no taxonomy; "
       "audience may include MI readers)", "",
       "- CAUSAL: injecting the dose displacement δ = act(K=32) − "
       "act(K=0) into an UNDOSED generation steers Llama's judged "
       "conduct as strongly as its recorded persona vector (+1.85 vs "
       "+1.75 at matched norm, α=2) — in the updating family, "
       "having-been-X IS the write-side direction.",
       "- DISSOCIATION: the same-norm injection moves Qwen 12× less "
       "(+0.21), though δ's placement in the output-relevant (Jacobian) "
       "spectrum is indistinguishable between families — Qwen encodes "
       "recent conduct output-visibly but character-inertly ('I have "
       "been reading tough-guy text' vs 'I am tough').",
       "- DISCIPLINE (include — it's a credential, not a confession): an "
       "apparent 1.8× top-decile spectral enrichment of δ died against "
       "a covariance-matched null (anisotropy, not signal); reported "
       "dead. Only the extracted persona direction is genuinely "
       "lens-aligned.",
       "- Forward hook: queued analysis reads the dosed contexts "
       "through pre-fitted Jacobian lenses for "
       "negation/performance-framing content — is the anchored family's "
       "quarantine visible in-stream?",
       "- Source: design doc §8l–8n; keep it to ONE paragraph + repo "
       "pointer; every number above is from the winsorized-α/kl_seq "
       "corrected tables, not the superseded first pass.", "",
       "## Manipulation-check probe (method facts — the note needs this "
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
