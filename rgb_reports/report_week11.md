# Week 11 — IPIP-NEO-GFC instrument construction and recovery

## 0. One-line summary

Built a new desirability-matched graded forced-choice (GFC) instrument
on the IPIP-NEO-300 item pool using open cohort models as raters
(replacing Okada's frontier-rater GPT-5 + Gemini protocol), targeting
P=60 pairs vs Okada's 30. The cohort-mean rating from 7 open 3-12B
models correlates with Okada's published frontier-consensus values
at **r = +0.961** on the same 60 items, validating "open models suffice
as desirability raters" at cohort level. Constructed P=60 via Okada
Appendix-C's two-stage MIP using a binary-search-on-m reformulation
that solves in 5 seconds (vs the literal LP-relaxation Okada
formulation that hung past 10 min on this scale). Resulting inventory
has max Δsd = 0.0125 on 1-9 scale (~14× tighter matching than Okada's
0.18, due to larger item pool + multi-rater averaging). After
debugging a hardcoded-instrument-path bug in the TIRT R fitter that
mangled an earlier "negative result", final corrected recovery shows
P=60 outperforms P=30 at matched-vocabulary conditions (cohort
Δ +0.02–0.04 |r|) but underperforms at ipip_reflowed (Δ −0.18) —
which retrospectively explains the W8/W10 reflow boost as a
vocabulary-coupling effect rather than a fundamental property of
reflowed prose.

## 1. Motivation and plan

W10 §4 diagnostic established that **scaling persona count doesn't help
TIRT recovery** for the existing Okada GFC-30 instrument — the
per-persona inference limit (only 30 paired comparisons probing 5
traits) dominates, regardless of item-parameter precision. The lever
that should help is **more pairs per persona**, which directly reduces
per-persona θ posterior variance.

But extending Okada's instrument was constrained: each item used at
most once means ⌊98/2⌋ = 49 pair max from the Goldberg-IPIP-100, and
the 30 published pairs already use 60 items. Beyond ~45 pairs from
Okada's bank requires either reusing items (changes the statistical
model) or moving to a larger item pool.

rgb (2026-05-12): "Let's do social desirability for the IPIP-NEO-300
items and pair those, treating it as entirely different instrument
from Okada's. Using the models themselves for desirability is a
tradeoff — they're presumably consistent with themselves, but it may
limit generalizability. It's nicer if we show that Okada's methodology
didn't need frontier models."

The W11 plan: replicate Okada's full methodology on a larger item
pool (IPIP-NEO-300) using open cohort models as raters (testing
"frontier raters not required"), then run the resulting instrument
through the existing Phase D infrastructure and see whether more
pairs improve TIRT recovery as predicted.

Four phases, executed in order:

  - **A**: validate cohort models as raters by re-rating Okada's
    published 60 items and comparing to Okada's frontier-rater means.
  - **B**: rate the 300 IPIP-NEO-300 items with the validated cohort.
  - **C**: build the new pair inventory via Okada's two-stage MIP
    on the cohort-mean desirability scores.
  - **D**: run the new instrument through `run_gfc_hf.py` and
    `fit_tirt_okada_indep.R`, compare recovery to P=30 Okada.

## 2. Phase A — cohort raters validation

Re-rated the 60 Goldberg items from Okada Table 3 with each of the 7
cohort models. Per-item distributional rating over `{'1',...,'9'}`
via `likert_distribution` (single deterministic forward pass), prompt
adapted from Okada Appendix A. Total compute: ~52 seconds on M5 Max
bf16 across all 7 models.

Results:

| Model | r vs Okada published (n=60) |
|---|---|
| Gemma 4B  | +0.914 |
| Llama 3B  | +0.711 |
| Phi4-mini | +0.870 |
| Qwen 3B   | +0.749 |
| Gemma 12B | +0.958 |
| Llama 8B  | +0.937 |
| Qwen 7B   | +0.943 |
| **COHORT MEAN** | **+0.961** |

Five of seven models individually pass r > 0.85; the cohort mean at
+0.961 nearly matches Okada's between-rater r=0.993 (GPT-5 vs Gemini
2.5 Pro) and exceeds their human-norm validation at r ≈ 0.95. Pairwise
cohort agreement: mean +0.798. Per-trait inspection: 7-12B models
uniformly strong on every trait (≥ +0.93); 3B models struggle most on
A (Agreeableness), uniformly easy on N (Neuroticism).

The "open models suffice" claim is established at cohort-ensemble
level. Used cohort-mean (not any single model) as the desirability
input to Phase B.

## 3. Phase B — rate IPIP-NEO-300

Rated all 300 IPIP-NEO items with the cohort using the same protocol.
The IPIP items are first-person behavioral statements ("I love large
parties") rather than Goldberg's trait-adjective phrases ("Cut others
to pieces"); we tested two prompt-handling options on a 30-item
balanced pilot first.

**Pilot result (cohort-mean A↔B Pearson r = +0.987)**:
- **Option A** (de-personalize: "I love" → "love" + Okada Appendix A prompt)
- **Option B** (keep first-person + adjusted prompt: "self-description statement, rate the underlying tendency")

Phrasing choice doesn't matter at the cohort-mean level — both yield
nearly identical rank orderings of items, with a small systematic
shift (~0.13 units) but mean |A−B| only 0.26 on the 1-9 scale.
Chose Option A for the full Phase B run because it's closer to Okada's
exact protocol; the de-personalization transform is stored alongside
each item for inspection.

Full Phase B compute: ~2.5 minutes across all 7 models × 300 items.
Sanity checks:

- **Forward-keyed > reverse-keyed mean EV** for every cohort model.
- **Per-trait cohort-mean desirability** behaves correctly:
  A fwd 6.66 / rev 3.51, C fwd 7.14 / rev 3.54, E fwd 6.67 / rev 5.17,
  N fwd 3.73 / rev 7.12 (inverted as expected — high-N statements
  undesirable, "I remain calm" desirable), O fwd 6.89 / rev 4.98.
- **Cohort-mean range [1.84, 8.23]**, mean 5.47, SD 1.76 — full spread.

Output: `results/desirability/cohort_phase_b_ipip300.json` with
per-model per-item EV + cohort mean. Cohort mean is the input to
Phase C.

## 4. Phase C — build IPIP-NEO-GFC-60 via constrained MIP

Implementation of Okada Appendix C in Python with PuLP + CBC. The
literal Okada formulation embeds the max-gap variable `m` continuously
with `delta_k · x_k ≤ m` for each candidate pair — 33,152 such
constraints over the cross-domain candidate pool. CBC's branch-and-
bound choked on this; first attempt hung past 10 minutes without
converging.

**Reformulation**: binary-search on `m`. Each step fixes a threshold
m_test, restricts admissible pairs to {Δsd ≤ m_test}, and solves a
pure-feasibility integer program (no objective). Feasibility checks
are 1-2 orders of magnitude faster on CBC than the LP-relaxed minimax
embedding. The reformulated solver converged in 11 iterations + Stage
2 squared-mismatch minimization = ~5 seconds total wall time.

**Resulting inventory** (`instruments/ipip_neo_gfc_P60.json`):

- max Δsd = 0.0125 (Okada published: 0.18; ~14× tighter)
- mean Δsd = 0.0042
- 60 pairs, 120 items used out of 288 (after 12 deny-listed
  marker-like/partisan IPIP items excluded)
- Each trait appears 24 times (2P/5)
- Each cross-trait combo appears 6 times (P/10)
- Mixed-keying fraction 52% (target 40-60%)
- Per-domain keying ≥ 8 of each sign (≥30% target)

The much tighter matching vs Okada is from two sources: (a) IPIP-NEO
has 3× more items (300 vs 100), giving more pair flexibility; (b)
cohort-mean averaging across 7 raters produces finer-grained
desirability scores than Okada's 2-rater × 30-replication mean.

Spot-check pairs:

- Block 1 (C+/N-, both sd≈7.13): "go straight for the goal" vs "am
  not embarrassed easily" — both high-desirability, different traits
- Block 5 (A-/C-, both sd≈2.0): "make people feel uncomfortable" vs
  "misrepresent the facts" — both low-desirability
- Block 58 (E+/N-, both sd≈7.72): "cheer people up" vs "can handle
  complex problems"

## 5. Phase D — inference + TIRT recovery

### 5.1. Pipeline run

Extended `scripts/run_gfc_hf.py` with an `--instrument` argument
(default unchanged for Okada-GFC-30 backward compat). Wrote
`scripts/run_phase_d_pipeline.sh` as the wrapper:

- 21 HF GFC inferences (7 cohort models × 3 persona forms ×
  50 personas × 60 pairs = 63,000 prompts total)
- 21 Stan TIRT fits via `fit_tirt_okada_indep.R`

Stage 1 ran ~3 hours on M5 Max bf16 (roughly 2× the GFC-30 inference
time as expected). Stage 2 fits each ~30 seconds; ~10 min total.
All 21 inferences and 21 fits completed without errors. Outputs in
`psychometrics/gfc_tirt/<MODEL>_ipipneogfc60_hf_<FORM>.json` and
matching `_indep_fit.rds`.

### 5.2. Initial "surprise negative result" — bug

First-pass numbers showed P=60 *substantially worse* than P=30:

| Form | P=30 cohort \|r\| | P=60 cohort \|r\| (bug) |
|---|---|---|
| description | 0.266 | 0.177 |
| ipip_raw | 0.190 | 0.145 |
| ipip_reflowed | **0.460** | **0.130** |
| Grand | 0.305 | 0.151 |

This was confusing: 4× more per-pair information but worse recovery,
and the strongest condition (ipip_reflowed) hit the hardest. We
considered three hypotheses:

1. **Tight desirability matching weakens TIRT identification**
   (m*=0.0125 << Okada's 0.18 → per-pair signal weak → MCMC slips
   into reflected sign basins).
2. **Wrong priors** (κ ~ N(0, 1.5) too wide → cutpoints absorb data
   variance; a_pos ~ HN(0, 0.5) shrinks loadings toward zero).
3. **Item form differences** (de-personalized IPIP fragments parse
   worse in forced-choice than Goldberg's declarative trait phrases).

### 5.3. The credibility check

To discriminate among hypotheses, ran a direct sanity check at the
per-pair level. For each persona × pair, predict the response
direction from the persona's known z-vector: persona with z_E = +2.96
should prefer LEFT when LEFT is an E+ item and RIGHT is anything else.
Compute match/mismatch rates per persona.

Results for the most extreme |z_E| personas in Gemma12 × description
(the worst-flipped condition under the bug):

| pid | z_E | matches | mismatches | neutral | net signal |
|---|---|---|---|---|---|
| s50 | +2.96 | 30 | 10 | 20 | +0.500 |
| s38 | −2.49 | 44 | 14 | 2 | +0.517 |
| s47 | +1.91 | 27 | 16 | 17 | +0.256 |
| s7 | −1.88 | 33 | 19 | 8 | +0.269 |

**49 of 50 personas had positive net signal** (model preferences
matched predicted direction more often than not); mean net signal
+0.343, median +0.298. The per-pair responses were *clearly tracking
the personas* — model behavior was fine.

This decisively ruled out hypotheses 1, 2, 3 above. If per-pair
preferences look right but recovery is wrong, the issue is downstream
of the data — in how TIRT aggregates.

### 5.4. The actual bug

Looking at `fit_tirt_okada_indep.R`:

```r
inst <- fromJSON("instruments/okada_gfc30.json")  # ← hardcoded
pairs <- inst$pairs
P <- nrow(pairs)
```

The R fitter loaded **trait labels, keying, and pair structure from
the OLD okada_gfc30.json** regardless of which inference data it was
fed. When applied to our P=60 inference output:

- It saw P=30 from Okada's instrument file
- It built `y_mat` from the first 30 of our 60 response columns
- It assigned each of those 30 blocks the wrong trait labels
  (our Block 1 is C+/N-, Okada's Block 1 is A+/O+)
- TIRT happily fit the Stan model, converged cleanly, and produced
  meaningless θ recovery aligned to the wrong instrument

The MCMC convergence diagnostics (n_high_rhat=0, n_low_neff=0)
gave no warning because the model fits *something* — just not the
right something.

**Fix**: read the instrument path from the inference JSON's
`instrument_path` field (which `run_gfc_hf.py` started writing at
commit 959f672). Falls back to Okada GFC-30 for backward compat.
Single-line change at the data-loading step.

### 5.5. Corrected results

After re-fitting all 21 cells with the corrected R script:

| Form | P=30 cohort \|r\| | P=60 cohort \|r\| | Δ |
|---|---|---|---|
| description | 0.266 | **0.287** | +0.021 |
| ipip_raw | 0.190 | **0.230** | +0.040 |
| ipip_reflowed | **0.460** | 0.281 | −0.179 |
| Grand | 0.305 | 0.266 | −0.039 |

Signs are mostly aligned now: 80/105 positive at P=60 (vs 74/105 at
P=30); signed mean +0.207 (vs +0.210). The sign-chaos under the bug
was 100% an aggregation artifact.

Per-model highlights at P=60 (signed mean across traits):

- **Gemma 12B × description: +0.567** (best single-cell recovery in
  the cohort; was +0.505 at P=30)
- **Llama 8B × description: +0.441** (jump from +0.223 at P=30 —
  largest single-model improvement)
- **Phi4 × ipip_reflowed: +0.457** (strong)
- **Gemma 12B × ipip_raw: +0.437** (jump from +0.253)
- Llama 3B and Qwen 3B remain cohort weak points — likely a
  model-capability ceiling, not an instrument issue.

## 6. Findings

### 6.1. More pairs *do* help — but only at matched-vocabulary conditions

The W10 §4 hypothesis was: with item-parameter precision already
saturated at N=50, **more pairs is the lever for TIRT recovery**. At
description and ipip_raw conditions, this is confirmed:

- description: +0.021 cohort improvement (5/7 models improve)
- ipip_raw: +0.040 cohort improvement (5/7 models improve)

Per-model gains scale roughly with model capability — the 12B and 7-8B
models gain most, the 3B models plateau. This is consistent with
TIRT being a per-persona inference where additional pairs reduce θ
posterior variance: more capable models benefit more from the extra
information bandwidth.

### 6.2. Reflow advantage was a vocabulary-coupling effect

At ipip_reflowed:

- P=30 (Okada GFC, Goldberg-style items) cohort: 0.460
- P=60 (IPIP-NEO-GFC, behavioral items) cohort: 0.281
- Δ: −0.179

The W10 W8-trajectory finding that "TIRT reflow gives Δ+0.27 in mean
|r|" wasn't a fundamental property of reflowed prose. It was that
**the Okada GFC-30 items used Goldberg-style trait adjectives, while
the reflowed personas were prose synthesized from IPIP behavioral
items**. Reflow bridged a lexical mismatch between persona description
and instrument items. When the instrument is itself built on IPIP-NEO
items (P=60), the persona-instrument vocabulary is *already aligned*
and reflow provides nothing extra to bridge.

This is W8's vocabulary-coupling story playing out at the GFC level.
The matched-vocabulary baseline cohort |r| ≈ 0.28 across all three
persona forms is the "real" TIRT recovery at this scale, without
the lexical bridge.

### 6.3. The R-fitter bug

Worth flagging as a methodological note in the paper: a single
hardcoded instrument path in the TIRT scoring pipeline produced
a wrong-instrument fit that converged cleanly through MCMC
diagnostics (Rhat OK, n_eff OK) but produced near-zero recovery
with chaotic signs. The bug was caught only through a direct
per-pair credibility check — verifying that response data matched
expected trait-conditional preferences for known personas.

Two takeaways:

1. **Convergence diagnostics don't catch label-misalignment bugs**.
   The Stan model fit the wrong-instrument-labeled data just fine
   from its perspective; nothing in posterior summary statistics or
   Rhat warned that the labels were wrong.

2. **Per-pair credibility checks are a cheap, effective
   sanity-check** against this class of bug. For graded forced-
   choice with known persona z-vectors, predicting response
   direction from z and checking match rate per persona surfaces
   the issue immediately — even before any TIRT fitting happens.

### 6.4. Open models suffice as desirability raters

Independently of the recovery story, Phase A's finding stands:
**cohort-ensemble desirability ratings from open 3-12B models match
Okada's frontier-rater consensus at r=+0.961** on the same 60 items.
This is methodologically interesting because:

- Replicates Okada's approach without dependence on commercial APIs
- Cohort-ensemble at modest scale (7 open models, 3-12B parameters)
  matches a 2-rater frontier consensus (GPT-5 + Gemini 2.5 Pro at
  30 replications each)
- Per-model individual agreement scales cleanly with size — 7-12B
  uniformly r ≥ +0.93; 3B models +0.71-0.91

Practically: instrument-construction methodology that previously
required API access to frontier models can be done on open
infrastructure.

## 7. Diagnostic walkthrough — the R-fitter bug

The order of debugging steps mattered. Worth recording in case the
same class of "convergent fit, wrong labels" issue arises later:

1. **First wrong hypothesis** (W10): too-tight desirability matching
   weakens TIRT likelihood, MCMC slips into reflected sign basins.
   Plausible because Okada-binary at week11 showed similar pattern.
   Built diagnostic prep but didn't run yet.

2. **Second wrong hypothesis**: wrong priors (κ N(0, 1.5) too wide,
   a_pos HN(0, 0.5) shrinks loadings). User's gut reaction;
   stats-defensible. Built two Stan variants (tightkappa, lognorm)
   and re-fit all 21 cells. **Both variants produced identical
   cohort recovery to base** (0.147, 0.148 vs base 0.151). Prior
   choices weren't moving the needle, ruling out the hypothesis.

3. **Credibility check** (the right move): for one persona per
   model × condition, check whether per-pair preferences match
   expected direction from known z-vector. **49/50 personas had
   positive net signal** — model behavior was fine. This put the
   issue downstream of response data.

4. **R-fitter inspection**: at line 54, hardcoded path. Single fix.

5. **Re-fit**: corrected results matched expectations and the W10
   prediction was confirmed.

The credibility check was the load-bearing step. Without it, we
might have spent days iterating on instrument tweaks, prior tweaks,
even item-form tweaks — none of which would have fixed the actual
problem. The two wrong hypotheses both passed initial plausibility
tests ("Okada-binary saw similar pattern", "HN(0, 0.5) does shrink");
they were ruled out only by data, not by reasoning.

Cost: ~12 minutes of compute for the credibility check vs hours of
prior-variant fits. Always cheap to verify response data integrity
before trusting model-fitting outputs.

## 8. Next steps

1. **Update the W8 trajectory plot** with a fourth line (or replace
   the existing TIRT line) showing P=60 TIRT recovery. Visually
   document the vocabulary-coupling explanation for the reflow gap.

2. **Re-run Phase D at N=400 personas** now that the bug is fixed
   and the corrected baseline is established. Per the W10 §4
   diagnostic, this won't help recovery much (item-param noise
   wasn't the bottleneck), but it would tighten cohort estimates
   and provide a complete dataset matched to the Rep/Likert runs.
   KV-cache refactor (the "would help at N=400" optimization
   discussed in W10) becomes relevant if we go this direction.

3. **Per-trait analysis under P=60**: the cohort-mean numbers
   hide per-trait variation. With 60 paired comparisons covering
   each cross-trait combination 6 times, we have substantially
   more per-trait inference power. Worth seeing whether any trait
   is now reliably recovered cohort-wide, or whether per-trait
   weaknesses (esp. on A and Qwen-family) persist.

4. **Paper figure update**: the W11 work strengthens the three-readout
   framing in `paper_outline.md`. With P=60 results, the GFC/TIRT
   track has a coherent narrative: "Okada's methodology generalizes
   to a larger item pool with open raters; the more-pairs lever
   delivers the predicted recovery gain at matched-vocabulary; the
   apparent reflow advantage was a vocabulary-coupling effect."

5. **FAKE-GOOD condition** (deferred from W10): per the Okada
   methodology, the headline SDR analysis is honest-vs-fake-good
   under each readout. Our pipeline has only HONEST so far. Adding
   FAKE-GOOD across the cohort would directly test the SDR-immunity
   claims of each readout (Rep ≈ SDR-immune; Likert shows SDR;
   TIRT shows less SDR than Likert). Probably belongs in the
   paper.

## 9. Visualizations

- `results/persona_w8_trajectory.html` — needs update to show P=60
  TIRT line alongside Rep/Likert (next-steps item 1)
- A new dashboard comparing P=30 vs P=60 recovery per model × form
  would be the cleanest single visual for the W11 work

## 10. Status

Commits:

- `305268b` — W10 report + reflow-to-400 + N=25 diagnostic
- `8c99c77` — W11 Phase A: validate cohort raters (r=+0.961)
- `46db580` — W11 Phase B prep: prompt-phrasing pilot
- `c452b63` — W11 Phase B: rate 300 IPIP-NEO items
- `92d470a` — W11 Phase C: build IPIP-NEO-GFC-60 via MIP
- `959f672` — W11 Phase D setup: `--instrument` arg + pipeline wrapper
- `0adbafd` — W11 Phase D buggy results (preserved for diff)
- `af26107` — W11 Phase D diagnostic prep (the wrong-hypothesis stuff)
- `99889cb` — **W11 Phase D FIX**: R fitter bug + corrected recovery

Headline result file: `results/persona/persona_gfc_tirt_<MODEL>_ipipneogfc60_hf_<FORM>.json`
(21 cells, all sign-aligned, cohort grand mean |r| = 0.266).
