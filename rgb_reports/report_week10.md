# Week 10 — GFC/TIRT as the third readout, and where to go next

## 0. One-line summary

Ran statisfactions's scaffolded GFC/TIRT pipeline end-to-end across the
7-model cohort × 3 persona forms, confirming TIRT recovers persona z's
at substantially lower magnitude than Rep or Likert on open 3-12B
models (cohort mean |r|: ~0.27 description, ~0.19 ipip_raw, ~0.46
ipip_reflowed — vs Rep ~0.65 and Likert ~0.71 in matched conditions).
A small-fit diagnostic (N=25 vs N=50) showed item-parameter noise is
NOT the bottleneck: doubling personas changes cohort |r| by −0.008.
The bottleneck is per-persona inference precision — 30 paired
comparisons triangulating 5 traits per persona. Conclusion: scaling
TIRT recovery requires **more pairs, not more personas**. W11 plan
is a new desirability-matched GFC instrument built on IPIP-NEO-300
items with open-model raters (replacing Okada's Goldberg-100 + GPT-5/
Gemini), targeting 60-90 pairs vs Okada's 30, treated as a separate
instrument from Okada-GFC-30.

## 1. The GFC/TIRT pipeline end-to-end

statisfactions wired Thurstonian IRT on Okada GFC-30 as a third readout
into the W8 trajectory plot at commit `1b4bbd7` (W9 §10 of his track),
leaving the inference + fits as a runnable pipeline. Stages:

  1. 21 HF GFC-30 inferences (7 cohort models × 3 persona forms,
     50 personas × 30 paired comparisons each = 1500 prompts per file).
     ~95 min total on M5 Max bf16.
  2. 21 Stan TIRT fits via `psychometrics/gfc_tirt/fit_tirt_okada_indep.R`
     (Okada Appendix-D priors: θ ~ N(0, I_5) independent, a_pos ~
     HN(0, 0.5), κ ~ N(0, 1.5)). ~3 min total once Stan compiled.
  3. Regenerated trajectory plot with the TIRT third readout populated.

Commit `169fcfb` lands all 21 inference JSONs, 21 sidecar recovery JSONs,
and the regenerated plot.

## 2. Headline numbers

Cohort mean |r| (recovery of sampled persona z) by persona form × readout:

| Form               | Rep (W8 §6 / §8) | Likert (W8 §6 / §8) | TIRT (W10) |
|---|---|---|---|
| W7 description     | ~0.73            | ~0.89               | **0.266**  |
| ipip_raw (§4/§5)   | 0.65             | 0.71                | **0.190**  |
| ipip_reflowed (§5) | 0.67             | 0.74                | **0.461**  |

Three things stand out:

- **TIRT recovery is structurally much lower** than Rep or Likert on
  3-12B open models. Per-persona information: Likert gives ~690 bits
  per persona (300 items × 5 categories), GFC gives ~84 bits (30
  pairs × 7 categories). Each GFC response is `θ_trait_A −
  θ_trait_B + noise` — a difference between traits, not a level —
  so absolute θ recovery requires triangulating from many such
  differences.
- **The clean "step function with two flat segments" prediction
  doesn't hold.** Predicted: TIRT same for {§3-raw, §4-raw, §5-raw}
  (one shared ipip_raw persona); same for {§3-reflow, §5-reflow}
  (shared ipip_reflowed). Instead all three persona forms produce
  distinct recovery levels, AND ipip_raw is *lower* than description
  for most models. Surprising direction.
- **Reflow boosts TIRT substantially** (cohort Δ +0.27 in mean |r|).
  Much bigger than reflow's effect on Rep (+0.015) or Likert (+0.039).
  Smooth prose evidently carries more usable signal for the TIRT
  fitter than choppy first-person IPIP statements do.

Per-model magnitudes mostly fall in statisfactions's predicted range
(|r| ≈ 0.20–0.35; commit `1b4bbd7` flagged this floor explicitly):
Gemma 12B is cohort top at ~0.50-0.60, Qwen 7B ~0.30-0.54, Llama 3B
the weak point at ~0.10-0.15.

Sign flips appear as expected from week07: Phi4 ipip_reflowed has C
at −0.53 (the {A, C} graded pattern); Llama 8B has A consistently
negative across all forms; Gemma family has scattered flips on C
and E under reflow. Per-trait, not whole-vector (in contrast to
statisfactions's week11 binary-mode finding which had Gemma-27B /
Llama-70B whole-vector reflecting).

## 3. Visual fix: reorder + rescale

The trajectory plot at commit `169fcfb` had two cosmetic issues:

- **y-range [0.4, 1.0]** cut off the entire TIRT trace
  (cohort means 0.13-0.46). Widened to [0.0, 1.0]. statisfactions
  flagged this as a TODO in his scaffold commit.
- **Persona-form interleaving** — original `CONDITIONS_TRAJECTORY =
  ["W7", "§3 reflow", "§4 raw", "§5 raw", "§5 reflow"]` put reflow
  between description and raw, creating four step transitions on the
  TIRT line where there should structurally be two. Reordered to
  group adjacent same-form positions: `["W7", "§4 raw", "§5 raw",
  "§3 reflow", "§5 reflow"]`. TIRT now reads cleanly as three
  distinct heights with hv-step segments connecting same-form repeats.

Committed at `f4b9baf`.

## 4. The "more personas would help?" diagnostic

Considered scaling to N=400 (after reflowing the remaining 350
personas via Sonnet; ~26 min API, ~$2 — done at this commit).
Before committing the ~13 hour overnight inference run, did the
cheap diagnostic: re-fit the existing N=50 GFC inferences at N=25
(R script subsets to s1-s25), compute recovery, compare.

Cohort mean |r| (description + ipip_raw conditions, since
ipip_reflowed N=25 hit only s25 of the originally-50 reflowed
personas and returned NA):

  - N=25: 0.236
  - N=50: 0.228
  - Δ = −0.008

Per-cell correlation between N=25 and N=50 signed cohort recoveries:
r = 0.919.

Doubling the persona count produced a negligible (and in fact
slightly negative) change in recovery, AND the two cohort sizes
produce highly correlated per-cell results. **Item-parameter noise
is not the bottleneck** for TIRT recovery on our cohort.

What this means structurally: in the Stan model, `a_pos[j]` and
`kappa[p]` are population-level — each estimated from all
N × relevant_pairs observations pooled. With N=50, they're already
well-identified. The persona-level `theta[i, d]` is bandwidth-limited
by *that persona's* 30 responses, regardless of how well the item
parameters are known. The per-persona ceiling is what dominates.

Practical: don't scale N for TIRT recovery. Scale P (number of pairs)
instead.

## 5. Okada methodology, in 60 seconds

Pulled the full Appendix C from Okada 2026 (arXiv 2602.17262v2) to
think about extending P. Key facts:

  - **Item bank: Goldberg-1999 IPIP-100 Big-Five markers**, of which
    2 voting items were dropped → J = 98. NOT IPIP-NEO-300. Items
    are short trait-adjective phrases ("Cut others to pieces"),
    not behavioral statements.
  - **Desirability ratings**: 30 reps × 2 rater LLMs (GPT-5 +
    Gemini 2.5 Pro) → 60 ratings per item, averaged. Within-rater
    ICC = .999 for the 30-rep mean. Between-rater r = .993. Validated
    against Britz et al. 2023 human norms at r ≈ 0.95.
  - **Pair selection**: two-stage mixed-integer optimization that
    (i) minimizes the max within-pair desirability gap and (ii) among
    minimax-optimal solutions, minimizes total squared mismatch,
    subject to: cross-domain only, each item used ≤ 1 time, 12 per
    domain (= 2P/5), 3 per trait-pair combination (= P/10), 40-60%
    mixed-keying, ≥30% each sign per domain. Result: max |Δsd| =
    0.18, mean 0.03 on 1-9 scale.
  - **Expansion ceiling within Okada's item pool**: ⌊98/2⌋ = 49 pairs
    maximum. Published 30 uses 60 items; remaining 38 → up to 19 more
    pairs, less under all the balance constraints.
  - **TIRT setup**: matches statisfactions's `tirt_okada_indep.stan`
    exactly. Independent θ prior (no estimated trait correlation
    matrix in the IRT step), half-normal item discriminations,
    N(0, 1.5) thresholds, ordered_logistic likelihood.
  - **Their personas**: z ~ N(0, Σ) with van der Linden 2010 empirical
    Big Five correlation matrix. Worth checking whether our
    `synthetic_personas.json` sampling matches Σ vs uses I_5 — if
    different, the "recovery r" metric is slightly differently scaled.
  - **HONEST vs FAKE-GOOD**: Okada's main SDR analysis. We haven't
    used the fake-good condition in our cohort yet; would be a clean
    extension testing whether Rep / Likert / TIRT differ in SDR
    immunity.

Full bibliography entry already lives in `rgb_reports/bibliography.md`
under "Forced-choice personality measurement".

## 6. W11 plan — IPIP-NEO-GFC-N, open-rater construction

Plan emerging from rgb (2026-05-13): build a new desirability-matched
GFC instrument on the IPIP-NEO-300 item pool, using open cohort models
as desirability raters (not frontier raters). Treat as a separate
instrument from Okada-GFC-30, not a competitor. Two methodological
arguments at once:

  - Okada's methodology extends to a larger item pool with longer
    behavioral statements (vs short trait adjectives).
  - Okada's methodology doesn't require frontier raters — open 3-12B
    models can serve as desirability raters.

### Phase A — validate cohort as raters

Rate the 60 Goldberg items in Okada's Table 3 (whose published
desirability scores anchor the comparison) using each of the 7
cohort models with the Okada Appendix A prompt + 25-item-block
protocol, 10 replications per model. Pass criterion: r > 0.85
between cohort-model-mean and Okada-published. Compute: ~15 min
on M5 Max bf16. If a subset of cohort models pass and others
don't, document the scale dependence; if all pass, the "open
models suffice as raters" claim is established.

### Phase B — rate IPIP-NEO-300

Same protocol on 300 IPIP-NEO items, ~70 min compute. Prompt
adaptation needed: Okada's wording targets "trait or characteristic"
on short adjectives; IPIP-NEO items are first-person behavioral
statements ("I love large parties"). Two options:

  - **De-personalize** ("I love" → "Loves"; "I am" → "Being"). Most
    Okada-preserving.
  - **Keep first-person, adjust prompt** to rate the tendency
    described. Looser semantics, more interpretive load on the model.

Option A is cleaner for the methodology argument; we can validate
the choice by also rating a few IPIP items in their original
first-person form and comparing.

### Phase C — build the new instrument

Run Okada's two-stage MIP from Appendix C on the cohort-rater
desirability scores + IPIP-NEO domain/keying labels. Open question:
target P. Constraints suggest P ≤ 150 (each item used ≤ once →
⌊300/2⌋). With balance constraints (P/10 per cross-trait combo
= 6+ pairs per combo), realistically 60-90 pairs feasible.
Likely sweet spot is P = 60 (doubles Okada's 30, achievable, still
fits 25 minutes-ish of per-persona inference).

### Phase D — inference + TIRT on the new instrument

Same `run_gfc_hf.py` pipeline against the new pair set. N=50 personas
to start; ~3-5 hours inference depending on P. Then TIRT fits, recovery
analysis. Compare to Okada-GFC-30 cohort means: does P=60 close the
gap to Rep/Likert? Per the diagnostic in §4, doubling pairs should
roughly halve the per-persona θ variance.

## 7. Paper crystallization

statisfactions and rgb are scheduled to chat about paper shape.
Outline scaffold at `rgb_reports/paper_outline.md`. With W10's TIRT
numbers in hand and a Phase D result on a new instrument, the
three-readout argument firms up considerably:

  - Likert > Rep on open models in matched conditions, but mostly
    vocabulary-coupling — W8 §6/§8
  - TIRT bandwidth-limited on open models at GFC-30; bigger inventory
    scales recovery — W10 §4, W11 Phase D
  - All three readouts recover human Big Five facet structure at
    cohort r ≈ +0.56 vs human norms — W9 §7 (so far Rep only; Likert
    + TIRT analogs would need scoring against the human matrix
    too)

If FAKE-GOOD goes in, that's a fourth analytical dimension on top of
the readout × persona × cohort cube — but probably belongs in a
follow-up paper rather than this one.

## 8. Status

Commits since W9:
  - `0b7e7d5` (statisfactions) — GFC pipeline path comment fix
  - `1b4bbd7` (statisfactions) — Scaffold GFC/TIRT line for W8
    vocabulary-coupling trajectory plot (the W10 piece we ran)
  - `ce96230` — Paper crystallization outline scaffold
  - `169fcfb` — W10 GFC/TIRT pipeline output across cohort
  - `f4b9baf` — Trajectory plot reorder + y-range widen
  - Reflow of remaining 350 personas → `synthetic_personas_ipip.json`
    now has all 400 reflowed (uncommitted as of writing this report)
  - W10 N=25 diagnostic fits at `psychometrics/gfc_tirt/*_n25_indep_fit.rds`
    and `results/persona/persona_gfc_tirt_*_n25.json` (uncommitted)

Open before Phase A starts:
  - Commit the reflow expansion + N=25 diagnostic
  - Push the W10 stack so statisfactions sees the trajectory plot
    + writeup before the paper chat
  - Phase A validation script (Okada Appendix A protocol against
    cohort models on 60 Goldberg Table 3 items)
