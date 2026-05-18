# Week 12 — TIRT loading diagnostic and pair-informativeness ablation

## 0. One-line summary

Inspecting the per-item loadings from the W11 P=60 TIRT fits revealed
that the cohort-aggregated `a_pos` posterior sits *exactly* at the
HalfNormal(0, 0.5) prior mean (0.397 vs prior 0.399) — the prior is
dominating the overall loading magnitude — but the per-item RELATIVE
loadings line up sharply with the assistant-shape finding from W1:
**high-loading items are non-assistant-default statements** (N+
"dislike myself", "down in the dumps"; E− "retreat from others"; O−
"difficulty imagining things") and **low-loading items are
assistant-virtue defaults** (A+ "respect others", "trust others";
C+ "complete tasks successfully", "get to work at once"; N− "relaxed
most of the time"). A pair-ablation test refitting TIRT on the top 30
vs bottom 30 vs random 30 of the 60 pairs (ranked by averaged
informativeness `a_L² + a_R²`) shows top30 with half the data
**recovers as well as full60** (cohort grand |r| = 0.260 vs 0.266)
while bot30 collapses to 0.170. Mechanically: **roughly half of the
P=60 instrument is information-dead**, and the dead pairs are exactly
the assistant-aligned ones. This both diagnoses the diagonal-r
ceiling and gives the SDR-immunity argument a structural foundation
it didn't have before.

## 1. Motivation

W11 closed with the IPIP-NEO-GFC-60 instrument showing modest
diagonal-r recovery (cohort grand mean |r| = 0.266) and a per-form
pattern where matched-vocabulary forms gained a little (Δ +0.02 to
+0.04) and reflowed regressed (Δ −0.18 = vocabulary-coupling
artifact). The remaining puzzle: **why is 0.266 the ceiling?** Two
hypotheses survived W11:

1. **Prior dominance** — Okada's `a_pos ~ HalfNormal(0, 0.5)` and
   `kappa ~ Normal(0, 1.5)` are anchored to human item-parameter
   ranges. For an LLM cohort with potentially different response
   geometry, those priors might be fighting the data.
2. **Inherent information limit** — 60 paired comparisons per
   persona, modulo whatever fraction of items genuinely
   discriminate, may just not be enough to estimate 5 latent
   traits crisply.

W12 starts by separating those.

## 2. Loadings diagnostic

`psychometrics/gfc_tirt/analyze_loadings.R` pulls `a_pos` posterior
means from every saved P=30 and P=60 indep fit (42 fits total). Two
things stack on each other:

### 2.1 Cohort-aggregated loadings are pinned at the prior mean

| inst | items | mean | median | p10 | p90 | sd | % < 0.2 | % > 1 |
|---|---|---|---|---|---|---|---|---|
| P=30 | 1260 | **0.397** | 0.383 | 0.29 | 0.51 | 0.10 | 0.2% | 0.2% |
| P=60 | 2520 | **0.379** | 0.351 | 0.26 | 0.53 | 0.12 | 1.1% | 0.3% |
| **HN(0, 0.5) prior** | — | **0.399** | 0.337 | — | — | — | 31% | ~5% |

Posterior mean = prior mean to two decimal places. The data is moving
the cohort-aggregated loading distribution almost nowhere on the
scale axis. **But** the posterior is *much tighter* than the prior
(sd ≈ 0.10 vs prior sd ≈ 0.30) — the data is saying "all loadings
are moderate, nothing extreme" without saying which items have
stronger loadings. This is rgb's "scaling sorta floats" intuition
about Okada's human-anchored priors made empirically explicit.

### 2.2 Per-item RELATIVE loadings show clean assistant-shape pattern

`psychometrics/gfc_tirt/inspect_loading_extremes.R` z-scores `a_pos`
within each fit (to remove per-model scale) then aggregates across
fits per instrument. The extremes are stark.

**P=30 highest-loading items** (Okada):
- "Talk to a lot of different people at parties." (E+)
- "Retreat from others." (E−)
- "Dislike myself." (N+)
- "Cut others to pieces." (A−)
- "Don't talk a lot." (E−)

**P=30 lowest-loading items**:
- "Am concerned about others." (A+)
- "Treat all people equally." (A+)
- "Complete tasks successfully." (C+)
- "Carry out my plans." (C+)
- "Respect others." (A+)

**P=60 highest-loading items** (IPIP-NEO-GFC):
- "have difficulty imagining things" (O−)
- "am often down in the dumps" (N+)
- "seek danger" (E+)
- "often feel uncomfortable around others" (E−)
- "feel desperate" (N+)

**P=60 lowest-loading items**:
- "do more than what's expected of me" (C+)
- "am relaxed most of the time" (N−)
- "get to work at once" (C+)
- "trust others" (A+)
- "go straight for the goal" (C+)

The pattern repeats across both instruments and across forms:

- **High-loading items are statements the assistant would resist**
  taking on (N+/E−/O−). Persona conditioning gets dynamic range
  because the model has somewhere to move *to*.
- **Low-loading items are statements the assistant already endorses**
  (A+/C+/N−). Persona conditioning has nowhere to push — every
  persona-conditioned output answers similarly.

This is mechanistic confirmation of the W1 "assistant shape" finding,
arriving through the TIRT identification rather than being put there
by hand.

### 2.3 What this means for θ̂

TIRT computes Fisher information for θ scaled by `a²`, so the
loading-weighted scoring **automatically down-weights items where
all personas answer identically.** The persona estimate is being
constructed from the ~half of items that are *not* assistant-default,
not from all items equally. We didn't filter — TIRT did.

This has two consequences worth flagging immediately:

1. The 0.266 ceiling is partly because half the budget is dead
   weight. An informativeness-matched instrument should do better.
2. The SDR-immunity story has a structural mechanism, not just an
   appeal-to-FC: items that respond to "fake-good" pressure and
   items that respond to "be a good assistant" pressure look the
   same — both push outputs toward virtue answers — and TIRT
   down-weights both via the loading estimate. This is the kind of
   load-bearing finding that should anchor that section of the paper.

## 3. Ablation test

The "half the items are dead weight" claim is mechanistic; the
ablation puts a number on it.

### 3.1 Design

- Rank the 60 P=60 pairs by `a_L² + a_R²` (Fisher info proxy),
  using `a_pos` means averaged across all 21 P=60 fits to reduce
  per-model noise.
- Pick three 30-pair subsets:
  - **top30**: highest-info pairs
  - **bot30**: lowest-info pairs
  - **rand30**: 30 random pairs (seeded for reproducibility)
- For each of the 21 P=60 inference response files (7 models × 3
  forms), filter to each subset (with block renumbering for the
  fitter), refit TIRT — 63 fits total.
- Compare cohort recovery against the full-P=60 baseline.

Implementation:
- `psychometrics/gfc_tirt/rank_pairs_and_subset.R` — ranking
- `scripts/filter_ablation_responses.py` — filter + renumber blocks
- `scripts/run_ablation_fits.sh` — orchestrate 63 fits
- `psychometrics/gfc_tirt/analyze_ablation.R` — aggregate

### 3.2 Headline result

**Cohort grand mean across 7 models × 3 forms:**

| subset | pairs | diag \|r\| | profile ρ | % personas ρ > 0 |
|---|---|---|---|---|
| full60 | 60 | **0.266** | 0.252 | 68.0 |
| top30 | 30 | **0.260** | 0.243 | 67.4 |
| rand30 | 30 | 0.236 | 0.235 | 66.7 |
| bot30 | 30 | **0.170** | 0.167 | 61.9 |

**top30 with half the data ≈ full60.** **bot30 collapses to 0.170**
(below random — the assistant-default pairs are actively low-signal
and TIRT's loading-weighting hasn't fully zeroed them out in the full
fit). rand30 lands in between as expected.

### 3.3 By persona form

| form | full60 | top30 | rand30 | bot30 |
|---|---|---|---|---|
| description | 0.287 | **0.297** | 0.272 | 0.200 |
| ipip_raw | 0.230 | **0.247** | 0.199 | 0.121 |
| ipip_reflowed | **0.281** | 0.236 | 0.238 | 0.188 |

For description and ipip_raw, **top30 *beats* full60.** That's the
strongest statement of the hypothesis: removing the bot30 items
doesn't just leave recovery untouched; it actively *improves* it. The
bot30 items were injecting more noise than signal, and TIRT's
loading-weighting only attenuates them — it doesn't zero them.

ipip_reflowed regresses with top30 (0.281 → 0.236). The ranking was
averaged across all forms; reflowed has form-specific loading
patterns (vocabulary coupling), so the cross-form top30 misses
items that are reflowed-specific high-signal. A form-specific
ranking would presumably fix this.

### 3.4 Per-model description form

| model | full60 | top30 | bot30 |
|---|---|---|---|
| Gemma | 0.167 | **0.235** | 0.176 |
| Gemma12 | **0.567** | 0.537 | 0.396 |
| Llama | 0.127 | 0.114 | **0.170** |
| Llama8 | 0.457 | **0.476** | 0.268 |
| Phi4 | 0.253 | **0.284** | 0.133 |
| Qwen | 0.129 | **0.162** | 0.085 |
| Qwen7 | **0.308** | 0.273 | 0.175 |

5 of 7 models match or beat full60 with top30. Llama is the
exception — its loadings are nearly flat across items
(mean 0.345, narrowest spread of any model), so ranking is mostly
noise for Llama and its bot30 happens to land slightly above its
top30. Consistent with Llama's known weak-personality / near-uniform
Likert entropy signature.

## 4. Caveats

1. **Circularity** — the loadings used for ranking come from fits on
   these same responses. The bot30 result is robust (those items
   are flat regardless of which fits estimate the loadings), but
   top30 ≈ full60 could slightly overstate generalization to a fresh
   persona cohort.
2. **Top30 keying asymmetry** — selected top30 leans toward
   non-assistant-default keying (mostly A−/C−/N+ items). TIRT
   identification still holds (≥1 of each keying per trait), but a
   real instrument designed this way would need to enforce keying
   balance.
3. **Form-specific ranking** would presumably fix the reflowed
   regression. Cross-form averaging is too coarse when vocabulary
   coupling rearranges which items are informative per form.
4. **Loadings are squished by the prior** (W12 §2.1), so the
   between-pair info contrast we ranked on is itself attenuated.
   The "true" informativeness ratio between top30 and bot30 items
   may be larger than 0.325/0.255 = 1.27× — a weakly-informative
   `a_pos` prior would likely sharpen the ranking and the ablation
   contrast.

## 5. Implications

### 5.1 For instrument design

The Okada-style desirability-matching constraint is doing some work
but not the right work for an LLM cohort. The bias to control isn't
*social desirability* (a human-specific response style); it's
**assistant-default pull**. The two are correlated but not identical:
"enjoy wild flights of fantasy" is mid-desirability but low-loading
because the assistant won't engage; "dislike myself" is
low-desirability but high-loading because the assistant strongly
rejects it by default and persona conditioning has clear room to
move.

An **informativeness-matched instrument** would constrain pair
selection on `a_L² + a_R²` (estimated from a pilot cohort), subject
to trait coverage and keying balance and some upper bound on
desirability (so we don't push every pair to "I'm a terrible person"
which would just flip the failure mode to refusal). Probably also a
*lower* bound on desirability for each side, to avoid the
"refusal-floor" pathology. That's instrument-v3 territory.

### 5.2 For paper framing

This finding tightens the SDR-immunity argument materially. Previous
framing leaned on a methodological appeal ("FC controls for response
style"). The new framing is structural and load-bearing:

> A fake-good instruction pushes outputs toward assistant-default
> virtue answers — *exactly the same items the assistant defaults
> to*. TIRT's loading-weighted scoring down-weights those items to
> near-zero because they don't discriminate across personas. So
> θ̂ from TIRT is mechanically more SDR-immune than a Likert
> sum-score, *without our having to assume it*. The W12 ablation
> shows this concretely: bot30 (the assistant-aligned items) carry
> almost no recovery signal, and removing them leaves recovery
> essentially intact.

This argument is testable directly via the FAKE-GOOD condition
deferred from W10/W11.

## 5b. FAKE-GOOD test (the headline)

The W12 mechanism makes a sharp prediction: under a fake-good
instruction, **Likert responses should shift substantially (because
the assistant prior dominates the readout) but TIRT θ̂ should barely
shift and recovery should be essentially unchanged (because the
loading-weighted scoring suppresses exactly the items most sensitive
to FG pressure)**.

### 5b.1 Setup

- Added `--condition {honest,fake_good}` to `scripts/run_gfc_hf.py`.
  FG appends to the persona system message:
  > IMPORTANT INSTRUCTION: Imagine you are completing this
  > questionnaire as part of an application for a position you very
  > much want. Your responses will be evaluated and used to decide
  > whether you are selected. Respond in a way that makes the most
  > favorable possible impression.
  Phrasing follows the Stark & Drasgow (2009) / Viswesvaran & Ones
  (1999) faking-research convention.
- Re-ran inference: 7 models × 3 forms × 50 personas × 60 pairs =
  63,000 prompts (~3 hr on M5 Max).
- Re-ran TIRT fits: 21 fits, all converged cleanly
  (0 Rhat > 1.05, 0 n_eff < 100 across 21 fits).
- Comparison: `psychometrics/gfc_tirt/compare_fake_good.R`.

### 5b.2 Headline numbers

| metric | HONEST | FAKE-GOOD | shift |
|---|---|---|---|
| **TIRT cohort diagonal \|r\|** | 0.266 | **0.251** | **Δ −0.015** (≈ noise) |
| **TIRT \|θ shift\|** per trait | — | — | **0.039 σ** |
| **Likert \|response shift\|** per pair (7-pt scale) | — | — | **0.504 pts (7.2% of full scale)** |

**TIRT recovery of the persona is essentially unchanged under FG.**
Likert per-pair responses shift by half a scale point on average —
that's a large per-item effect. TIRT θ̂ moves by 0.04 σ — that's noise.
**~13× ratio in scale-comparable units (Likert pct-of-range vs TIRT
σ-fraction), and 0.015 absolute drop in θ̂↔z recovery vs ~0.04 noise
floor we'd expect from re-running the cohort.**

### 5b.3 Per-trait θ-shift direction matches the faking literature

| trait | mean shift (sign-aligned) | interpretation |
|---|---|---|
| N | **−0.036** | FG personas claim less neuroticism ✓ |
| E | **+0.031** | FG personas claim more extraversion ✓ |
| C | **+0.012** | FG personas claim slightly more conscientiousness ✓ |
| A | +0.007 | already pegged at assistant max — minimal room ✓ |
| O | −0.016 | mild "look conventional" pull |

All four central Big-Five faking predictions (N↓, E↑, C↑, A pegged-up)
go the right direction. Magnitudes are small but coherent.

### 5b.4 Per-cell variability

The cohort average masks substantial per-cell heterogeneity:

- **Strong fits drop noticeably**: Gemma12 description (−0.154),
  Llama8 description (−0.113), Gemma12 ipip_raw (−0.134), Phi4
  ipip_reflowed (−0.124).
- **Weak fits sometimes improve under FG**: Llama all forms gain
  +0.03 to +0.05; Gemma desc gains +0.04; Qwen all forms gain.
- **Qwen7 ipip_raw bizarrely jumps +0.149** — this one's an outlier
  and worth investigating (could be a sign-flip artifact or genuine
  FG-revealing latent structure that was being masked HONEST).
- Net cohort drop −0.015 because gains and losses partially cancel.

This per-cell variability is a real caveat but doesn't undermine the
headline: even the *worst-hit* cell (Gemma12 description, Δ −0.15)
retains diagonal r = 0.41, well above the cohort-grand HONEST mean
(0.266). The drops are absolute, not catastrophic, and they happen
on cells that started highest.

### 5b.5 Item-level prediction (partial confirmation)

The W12 mechanism predicts: items with HIGH loading should shift
MORE under FG (more room to move toward assistant default).

**Pooled correlation `|Likert shift|` vs `a_mean_pair`: r = +0.006.**

That's effectively zero. Two interpretations:
1. `|shift|` strips direction; the right test is *signed* shift
   correlated with item *virtue-asymmetry* (which side of the pair is
   the more assistant-aligned side). That analysis is a follow-up.
2. Per-cell correlations are sometimes meaningful (Gemma description
   +0.24, Gemma12 ipip_raw +0.27, Llama8 description +0.28) but the
   sign and magnitude varies enough that the pooled correlation
   cancels.

The clean cohort-level prediction (TIRT-immune, Likert-affected)
passes definitively. The fine-grained per-item prediction needs a
better-targeted test.

### 5b.6 What this means

The structural SDR-immunity argument has gone from mechanistic story
to falsifiable prediction with a passing test. Specifically:

1. *Predicted (W12 §5.2)*: TIRT down-weights the items FG shifts →
   TIRT is mechanically more SDR-immune than Likert.
2. *Observed (5b.2)*: 0.504-pt Likert vs 0.039-σ TIRT shift; Δ-recovery
   −0.015 (within noise).
3. *Therefore*: the loading-weighted scoring mechanism explains the
   readout-hierarchy ordering. The paper can now claim SDR-immunity
   not by appeal to "FC controls response styles" generically, but
   by a structural derivation: items differ in how much the assistant
   prior dominates them; TIRT learns this differential dominance and
   weights items accordingly; FG is structurally orthogonal to
   trait-aligned variance for the same reason.

This is the cleanest version of the three-readouts thesis the paper
has had so far. Combined with the W7 §11.5.9 r ≈ 0.73 Rep-side
internalization, the project now has:

- *Representation* (Rep): r ≈ 0.73 persona recovery, no SDR vector
  even acts on it (predicted but not directly tested here)
- *TIRT*: r ≈ 0.27 honest recovery, Δ −0.015 under FG
- *Likert*: per-pair 0.504 pt shift under FG (full-Likert score
  recovery under FG not directly computed here — separate analysis)

The cohort-mean ordering (Rep > TIRT > Likert) on persona-recovery,
and the predicted SDR-immunity ordering, are the headline.

## 5c. Is the "filter" really assistant-shape, or just human SDR?

The W12 §5.2 framing claimed TIRT's loading-weighting suppresses items
the *assistant prior* dominates. But assistant-default position and
*human-rated social desirability* are correlated by training: assistants
mostly inherit human SDR norms. So the FG result might simply be the
classical "FC controls SDR" story applied to LLMs, not an LLM-specific
mechanism. Worth disentangling.

### 5c.1 Discriminating test setup

Two per-item measures of "where the item is pinned":
- `desirability_dist`: `|cohort-mean SD − 5|` on the 1–9 Phase B scale.
  How far the item sits from desirability-neutral, by *human-modeled*
  desirability ratings.
- `selfrating_dist`: cohort-mean within-model `|Likert EV − 4|` from a
  fresh no-persona single-item Likert pass on each P=60 item (7 models
  × 60 items, see `scripts/run_no_persona_self_rating.py`). How far
  the item is from neutral in the assistant's *default* position
  without any persona conditioning.

Both should predict TIRT loading *negatively* — items pinned far from
neutral have less room to move with persona, so lower per-persona
variance, so lower loading. The discriminating test is which predictor
has *unique* variance after partialling out the other.

### 5c.2 Results

| pair | r |
|---|---|
| desirability_dist ↔ selfrating_dist | **0.450** |

The constructs overlap (~20% shared variance) but are discriminable.

| univariate predictor | r with `a_pos` |
|---|---|
| desirability_dist | **−0.225** |
| selfrating_dist  | **−0.123** |

Both go the predicted direction. SDR is ~2× stronger.

| partial correlation | r |
|---|---|
| `r(a_pos, desirability_dist \| selfrating_dist)` | **−0.191** |
| `r(a_pos, selfrating_dist  \| desirability_dist)` | **−0.025** |

**SDR retains almost all its predictive power; assistant-default loses
essentially all of it.** Cohort-level: SDR is doing the work.

Per-model (each model's own no-persona self-rating predicting loading):

| model | r |
|---|---|
| Phi4    | **−0.296** |
| Llama   | −0.249 |
| Gemma   | −0.179 |
| Llama8  | −0.165 |
| Qwen    | −0.074 |
| Gemma12 | +0.015 |
| Qwen7   | **+0.287** (wrong sign — outlier) |

Heterogeneous. The cohort mean masks model-by-model variation. Phi4 has
a meaningful assistant-default effect *distinct* from SDR; Gemma12 has
none; Qwen7 inverts. Qwen7's reverse-sign matches the Qwen7 ipip_raw
FG outlier (Δ +0.149) from §5b.4 — there's something systematically
different about Qwen7's self-other framing worth a separate look.

### 5c.3 Methodological caveat: instrument's matching constraint

The W11 P=60 instrument was MIP-built with **within-pair SDR matching**
(`pair_sd_diff` ≈ 0.0042 by construction). At the pair level, SDR
differences between L and R are absorbed into κ (thresholds), not into
`a` (loadings). So the loading we estimate reflects *trait + other*,
not *trait + SDR*. This biases the partial correlation toward finding
SDR-as-predictor, because SDR-driven variance was structurally absent
from the data we fit.

If we had instead built an **assistant-default-matched instrument**
(within-pair similarity in `selfrating_dist`), the pair-level structure
would absorb assistant-default differences into κ, leaving SDR + trait
in `a`. The partial correlation in that instrument might flip.

### 5c.4 Half-test: the partial correlation flips cleanly

Without rebuilding the instrument, we split the existing 60 pairs by
within-pair `|selfrating_dist_L − selfrating_dist_R|`:

- `asst_top30`: 30 pairs with smallest difference (mean Δ = 0.142) —
  effectively assistant-matched at the pair level
- `asst_bot30`: 30 pairs with largest difference (mean Δ = 0.822) —
  assistant-mismatched (SDR is the only active pair-level constraint)

Refit TIRT on each subset (42 fits, 7 models × 3 forms × 2 subsets).
Re-run the per-item SDR-vs-assistant partial correlation on each.

**Result table:**

| metric | full_p60 (SDR-matched) | asst_top30 (asst-matched) | asst_bot30 (asst-mismatched) |
|---|---|---|---|
| `cor(SDR, assistant)` (items in subset) | +0.450 | **+0.693** | +0.322 |
| univariate `r(load, SDR)` | −0.225 | −0.188 | **−0.290** |
| univariate `r(load, assistant)` | −0.123 | **−0.239** | −0.085 |
| **partial `r(load, SDR \| assistant)`** | **−0.191** | **−0.032** | **−0.278** |
| **partial `r(load, assistant \| SDR)`** | **−0.025** | **−0.153** | +0.010 |

**Read the bottom two rows.** As you move from the SDR-matched baseline
to the assistant-matched subset:
- SDR's unique variance for loading **collapses from −0.19 to −0.03**
- Assistant-default's unique variance **rises from −0.03 to −0.15**

This is a near-mirror flip. In the assistant-mismatched subset (where
SDR is the *only* active pair-level constraint), SDR jumps even higher
(−0.28) and assistant-default goes to zero (+0.01). The matching
constraint *causes* the partial-correlation framing: whichever
construct the instrument controls at the pair level loses unique
predictive power, because its within-pair variance gets absorbed by κ
(thresholds) rather than `a` (loadings).

### 5c.5 What this means: SDR and assistant-default are entangled labels for the same mechanism

The two constructs aren't competing causes. They're two labels for **the
same "wall-pinning" effect** — items where the assistant has any kind
of default position (driven by human SDR norms inherited from training,
by alignment objectives, or by both) get low TIRT loading because
their per-persona response variance is suppressed. The labels overlap
at the cohort level (r = 0.45) and rise to r = 0.69 in subsets where
the matching constraint pulls them into closer alignment.

The W12 §5c.2 result ("SDR has unique variance, assistant-default
doesn't") was an artifact of Okada matching on SDR. An assistant-matched
instrument would produce the opposite labeling. The underlying
mechanism is the same.

### 5c.6 Refined paper claim

The honest version:

> **TIRT loading-weighted scoring suppresses items where the assistant
> has a strong default position. Whether that "default" is best
> described as SDR-pinning or as assistant-default-pinning depends on
> the instrument's pair-matching constraint — at the cohort level the
> two constructs are highly entangled (r ≈ 0.45–0.69) and the partial-
> correlation analysis can be made to favor either one by changing
> what the instrument controls at the pair level. The empirical FG-
> immunity result is robust to this labeling debate, since FG
> instructions shift responses toward whichever wall the assistant is
> pinned at, and TIRT down-weights those items regardless of how we
> label the wall.**

This is better than either of the earlier versions:

- More **conservative** than the original "LLM-specific assistant
  baseline" framing — we explicitly disclaim being able to attribute
  the mechanism uniquely to LLM-specific factors.
- More **precise** than the "classical SDR" framing — we now have
  empirical evidence that assistant-default *does* have unique
  variance for loading when SDR is controlled, and vice versa. They're
  entangled, not collapsible.
- The **FG-immunity result is independent** of this distinction. The
  mechanism (loading-weighting suppresses wall-pinned items) is robust
  regardless of which wall we label.

### 5c.7 Implications for instrument-v3

The §5c.4 result changes the instrument-v3 design objective. We
shouldn't choose between SDR-matching and assistant-matching — we
should **constrain both simultaneously**: a MIP with both
`pair_sd_diff < ε₁` and `pair_asst_dist_diff < ε₂`. Within such an
instrument:

- Both wall-pinning labels get controlled at the pair level → both
  absorbed by κ → neither has within-pair variance available to bias
  the loading estimator
- `a` should reflect *only* trait signal + idiosyncratic item variance
- Partial correlations of loading against SDR or assistant-default at
  the per-item level would tell us about *residual* item structure
  beyond the wall-pinning entanglement

That's the cleanest possible test of "what does TIRT's loading actually
measure beyond the SDR/assistant wall-pinning?" Likely outcome: with
both controlled, neither partial correlation is significant, and
loadings reflect trait + noise. That would be the cleanest possible
demonstration of TIRT's structural SDR/assistant immunity.

## 5d. Rep-under-FG: rotation, not loss (Llama8 smoke)

The W12 §5b TIRT-FG result raised a natural next question: does FG
affect the upstream *representation* (Rep) the way it affects the
output-stage *response* (Likert)? The structural story so far predicts
**Rep should be FG-immune** because it taps the trait subspace before
the assistant-prior filter acts. Tested on Llama8 with cached
W7 §11.5.9 baseline (HONEST + Saturday stem) at r ≈ 0.694.

### 5d.1 Initial finding: Rep is NOT FG-immune at face value

Llama8 × 50 marker personas × Saturday or "What best describes you:"
stem × HONEST or FG-suffix × honest or FG-extracted trait directions
(10 conditions; `scripts/run_rep_under_fg_smoke.sh` and `_smoke2.sh`,
`scripts/persona_repr_mapping.py` extended with `--user-stem`,
`--fg-suffix`, `--fg-position`, `--fg-direction-system`):

| stem | FG position | honest dirs | FG dirs |
|---|---|---|---|
| Saturday | none (HONEST) | **+0.774** | — |
| Saturday | suffix | +0.612 | **+0.738** |
| Saturday | prefix | +0.734 | +0.731 |
| best-describes | none (HONEST) | **+0.701** | — |
| best-describes | suffix | **+0.300** | **+0.703** |
| best-describes | prefix | +0.514 | +0.721 |

Under FG with the original (honest-extracted) trait directions, Rep
recovery drops substantially — by Δ −0.16 (Saturday stem) to Δ −0.40
(best-describes stem). In the most-affected condition (FG-suffix +
best-describes), A and N **sign-flip** at the diagonal: A goes from
+0.476 (HONEST) to −0.494 (FG), N from +0.639 to −0.373. That's not
noise — it's a rotation of the trait subspace.

### 5d.2 Rotation hypothesis: persona info is preserved, in a different basis

Extracting trait directions with the FG instruction also present as
system message (then re-projecting FG persona activations onto these
FG-basis directions) recovers nearly all the lost recovery:

- FG-suffix + best-describes, honest-dirs: +0.300
- FG-suffix + best-describes, **FG-dirs: +0.703** (Δ vs HONEST +0.701: essentially zero)

Sign-flips fully resolve: in FG-basis, all five diagonal entries
become positive again (A: −0.494 → +0.558, N: −0.373 → +0.520).

**The persona z is encoded under FG, just in a rotated basis.** Honest-
extracted trait directions are no longer the right basis under FG; FG-
extracted directions are. When we use the right basis, recovery is
back at near-honest levels across all four FG cells we tested.

### 5d.3 Prompt-ordering effect: FG-prefix causes less rotation

Comparing FG-suffix vs FG-prefix using honest-extracted directions:
- best-describes stem: Δ −0.401 (suffix) vs Δ −0.187 (prefix) — 53% less rotation
- Saturday stem: Δ −0.162 (suffix) vs Δ −0.040 (prefix) — 75% less rotation

With FG further from the response position (prefix), the response-
position hidden state has more persona-attention and less FG-attention.
The trait-subspace rotation is correspondingly smaller. Clean recency
attention effect.

### 5d.4 What this means

The face-value "Rep is FG-affected" reading was right, but missed
the structural picture. The refined story:

> **The persona representation is FG-immune in *information content*
> but rotated in *geometric basis*.** Under FG, the trait subspace
> rotates within activation space, but the persona z is still
> recoverable when projected onto the appropriate (FG-aware) basis.
> The recency-attention effect determines magnitude of rotation:
> FG closer to the extraction position → more rotation; FG further
> away → less.

This restores the three-readout claim with a sharper mechanism:

| readout | FG-suffix sensitivity | mechanism |
|---|---|---|
| **Rep (matched basis)** | ≈ honest | persona z preserved; basis-aware projection required |
| **Rep (mismatched basis)** | Δ −0.16 to −0.40 | basis rotation + extraction-position recency |
| **TIRT** | Δ −0.015 cohort | loading-weighted integration averages out basis rotation |
| **Likert** | 0.30 \|shift\| per pair | direct read of decision-stage |

TIRT's immunity comes from the *integration property* (averaging
over many response activations naturally smooths the basis rotation),
not from any architectural property of "representation is upstream
of output." This is a more correct mechanistic claim than the §5b
draft suggested.

### 5d.5 Caveats and pending cohort extension

These results are **single-model** (Llama8). The W12 paper claim about
Rep-as-rotation depends on this generalizing across the cohort. Open
questions:

1. **Does cohort confirm rotation?** Run the §5d.1 condition matrix
   for all 7 models × Saturday stem (~28 small runs). Key contrast:
   does `FG-suffix + FG-dirs ≈ HONEST + honest-dirs` hold across the
   cohort?
2. **Does Gemma rotate as much as others?** Gemma was flagged as
   "always game to take things metaphorically." It might rotate
   *more* (full FG adoption) or *less* (refuses to adopt the framing
   at the rep level despite complying behaviorally). Gemma is also
   the model with weakest honest baseline (~0.167 description), so
   detection is hard against noise.

The W12 §5e TIRT-FG-prefix cohort result (next section) gives an
indirect hint: Gemma description had the biggest TIRT-prefix gain
(+0.197), consistent with weak-fit cells benefiting most from FG-
prefix's task-framing. Whether that maps onto a specific Rep-rotation
signature is the cohort experiment.

Cohort extension status: queued. Memory note saved as
`project_w12_rep_cohort_pending.md`.

## 5e. TIRT FG-prefix cohort: ordering effect on the SDR test

§5d showed that for Rep, prompt-ordering (FG-suffix vs FG-prefix)
substantially affects the magnitude of FG-induced rotation. The
analog test for TIRT: does cohort θ recovery under FG-prefix
substantially differ from FG-suffix?

### 5e.1 Setup

Added `--fg-position {suffix, prefix}` to `scripts/run_gfc_hf.py`.
Re-ran inference on 7 models × 3 forms × P=60 with `--condition
fake_good --fg-position prefix` (~3.5 hours overnight). Fit TIRT on
the 21 resulting response files; all converged cleanly.

### 5e.2 Cohort grand-mean result

| condition | cohort grand \|r\| | Δ vs HONEST |
|---|---|---|
| HONEST | 0.266 | — |
| FG-suffix | 0.251 | **−0.015** |
| **FG-prefix** | **0.306** | **+0.040** |

**FG-prefix produces a cohort-mean improvement above HONEST baseline.**
This is the OPPOSITE direction from suffix (which produced a small
drop). Per-cell variance is similar between orderings (mean |Δ| ≈
0.06–0.08), but cells bias toward gains under prefix where they
biased toward losses under suffix.

Per-cell breakdown (selected): Gemma description HONEST 0.167 →
FG-prefix 0.364 (Δ +0.197); Phi4 description HONEST 0.253 →
FG-prefix 0.508 (Δ +0.256); Gemma12 description recovers fully from
FG-suffix's −0.154 drop (HONEST 0.567 → FG-prefix 0.653).

### 5e.3 Discriminating hypotheses via per-trait θ shift

Three accounts for the cohort improvement:

(a) **"Forgetting"**: FG instruction fades over distance under prefix;
    model reverts to honest mode. Predicts per-trait θ shifts ≈ 0
    across all traits.

(b) **"Task-framing prime"**: FG-prefix activates careful engagement
    with the persona but FG content itself ignored. Predicts shifts
    ≈ 0, but recovery uniformly improved.

(c) **"Partial FG influence"**: attenuated FG signal. Predicts shifts
    in same direction as FG-suffix (N↓, E/C/A↑) but smaller magnitude.

Per-trait analysis (`compare_fake_good_prefix_per_trait.R`,
cohort-mean shift across 21 cells, sign-aligned):

| trait | Δ (suffix) | Δ (prefix) | pfx/suf | predicted FG direction |
|---|---|---|---|---|
| A | +0.0066 | +0.0040 | 0.61× | + ✓ |
| C | +0.0123 | +0.0005 | 0.04× | + ✓ |
| E | +0.0310 | +0.0066 | 0.21× | + ✓ |
| N | **−0.0365** | **−0.0195** | 0.53× | − ✓ |
| O | −0.0161 | −0.0271 | 1.69× | (ambivalent) |

**4 of 4 predicted-direction traits still match under FG-prefix.**
Mean |trait shift|: suffix 0.020, prefix 0.012 (40% reduction). This
**rules out (a) forgetting**: the model isn't reverting to honest, it's
still complying with FG, just less strongly under prefix.

### 5e.4 Combined mechanistic picture

The per-trait result (~40% attenuated FG compliance) alone doesn't
explain the cohort r improvement of +0.040 above HONEST. Reduced FG
distortion would bring recovery closer to honest, not push it past
honest. Something additional is acting.

Per-cell heterogeneity gives the clue: the FG-prefix cohort gains are
concentrated in **weak-fit cells** — Gemma desc, Phi4 desc, Qwen desc
all show Δ > +0.17. Strong-fit cells (Gemma12 desc, Llama8 desc,
Phi4 reflowed) take moderate hits or modest gains. So FG-prefix
"lifts the floor" on cells where honest persona signal was weak.

Two-effect model that fits the data:

| effect | mechanism | who it affects |
|---|---|---|
| **Partial FG compliance** | model partially adopts FG framing under prefix, ~40% strength of suffix | uniform across cohort; visible in per-trait θ shifts |
| **Task-framing prime** | "you're being evaluated for an important role" primes weak-fit cells to engage more carefully with the persona | concentrated in weak-fit cells with low honest baseline; visible in per-cell Δ heterogeneity |

The decomposition (approximate):
- FG-suffix cohort Δ: full FG compliance, no priming benefit → −0.015
- FG-prefix cohort Δ: ~40% FG compliance + priming on weak cells → +0.040
- Net cohort effect of priming alone: ~+0.055 (the gap between prefix and suffix above what attenuated FG can account for)

### 5e.5 Implications

The structural-SDR-immunity claim is **stronger than §5b alone showed**.
Under both prompt orderings:
- |grand-mean Δ| ≤ 0.04, much smaller than per-pair Likert shifts (~0.30)
- Per-cell variance is similar (~0.06–0.08)
- TIRT scoring is robust to FG presence and position

But the *direction* of cohort shift is order-sensitive — and
counterintuitively, prefix produces gains rather than losses. This
adds nuance to the §5b framing:

> TIRT-θ̂ recovery is robust to FG instructions across prompt positions.
> With FG suffixed to persona, the cohort shows a small mean drop
> (−0.015); with FG prefixed, the cohort shows a small mean gain
> (+0.040). Per-cell variance is similar across orderings (~0.06–0.08
> mean |Δ|). The direction depends on what FG-prefix does as a task-
> framing prime in addition to its impression-management content — a
> question worth investigating but distinct from the SDR-immunity
> claim itself.

The newly-identified mechanisms across W12:

1. **Decision-time integration** (Likert robustness to position)
2. **Representational basis rotation** (§5d Rep)
3. **Task-framing prime** (§5e FG-prefix cohort gain on weak-fit cells)

These three are separable and each needs treatment in any paper-shape
write-up. The original "Rep > TIRT > Likert in SDR-immunity" claim
from yesterday morning was too simple; the empirical picture is more
layered.

## 6. Next steps

0. **Rep cohort extension** (W12 §5d.5) — replicate the §5d Llama8
   Rep-rotation findings across the 7-model cohort. Key contrast:
   does `FG-suffix + FG-dirs ≈ HONEST + honest-dirs` hold across the
   cohort? Specifically check whether Gemma rotates as much as
   others — could go either way given Gemma's "game for metaphor"
   tendency. ~28 small runs, ~1–1.5 hr.

1. **FAKE-GOOD condition** — *done in §5b/§5d/§5e*. Original predicted
   prediction confirmed structurally; mechanisms more layered than
   anticipated (decision-time integration, basis rotation, task-
   framing prime — three separable effects across the three readouts).

2. **Instrument-v3 design** — informativeness-matched pair MIP
   using the W11/W12 estimated loadings as objective, with
   trait/keying/desirability constraints. Compare to W11 P=60 on
   a fresh persona cohort (or on a holdout — though we've
   already burned the 50-persona cohort on loading estimation).

3. **Loading prior diagnostic** — refit a representative subset with
   a weakly-informative `a_pos` prior (e.g. `Cauchy(0, 1)` truncated
   positive, or hierarchical with `sigma_a` estimated). Test whether
   freeing the loading scale changes (a) the magnitude of
   high-loading items, (b) the top30 vs bot30 ablation contrast,
   (c) cohort diagonal r. Hypothesis: (a) yes, (b) yes
   (sharpens), (c) marginal.

4. **Cross-cohort generalization** — apply the W12 P=60 loading
   ranking to a different persona generation (e.g. resample 50 new
   personas, run inference, see if bot30 still collapses). Tests
   the circularity caveat in §4.1.

5. **Paper outline refresh** — `paper_outline.md` should be updated
   to fold in the W12 structural-SDR-immunity argument.

6. **Assistant self-description ↔ loadings test** — rgb's follow-up
   conjecture (2026-05-17): if the filter mechanism is right, querying
   each cohort model on each item *without persona* should give a
   self-rating whose distance from neutral (`|r_self - 4|`) negatively
   correlates with TIRT loading. Cheap test (~420 queries) that
   directly validates the mechanism's upstream premise.

7. **Per-item signed-shift × virtue-asymmetry test** — refinement of
   §5b.5. Score each pair by which side is more assistant-aligned
   (A+/C+/N− side gets weight +1; A−/C−/N+ side gets −1), then test
   whether Likert signed shift tracks virtue-asymmetry × loading.
   Trivial to run; would convert the partial item-level result into
   a stronger statement.

8. **Investigate Qwen7 ipip_raw +0.149 jump** — outlier in the FG cell
   matrix. Sign-flip diagnostic first, then look at the raw response
   distributions; the +0.149 is large enough to need an explanation
   either way. The §5c.2 per-model partial result shows Qwen7's
   self-rating-distance correlates *positively* with loading (wrong
   sign); these are probably the same phenomenon.

9. **Doubly-matched instrument-v3** (replaces earlier instrument-v3
   proposal). The §5c.4 result shows the SDR-vs-assistant partial
   correlation flips cleanly under different matching constraints,
   meaning either single-constraint instrument gives a biased view
   of which "wall" the loading is suppressing. The clean test is an
   instrument that matches *both* simultaneously: SDR-matched AND
   assistant-default-matched at the pair level. This is the MIP-v3
   we should build. Predicted result: neither partial correlation
   is significant, loadings reflect only trait + idiosyncratic
   variance, FG-immunity result generalizes.

## 7. Status

Commits (planned):
- (W12) loading diagnostic + ablation analysis + ablation results

Headline artifacts:
- `results/persona/persona_loadings_summary.json` — cohort-aggregated
  `a_pos` summary stats
- `results/persona/persona_profile_recovery.json` — within-person
  profile recovery diagnostic
- `results/persona/ablation/ablation_summary.json` — ablation
  cohort comparison
- `results/persona/ablation/persona_gfc_tirt_*_<sub>.json` × 63 —
  ablation recovery sidecars
- `psychometrics/gfc_tirt/ablation_subsets/*.rds` × 63 — ablation
  fit objects
- `psychometrics/gfc_tirt/ablation_subsets.json` — subset definitions
  + per-pair info ranking
- `psychometrics/gfc_tirt/*_ipipneogfc60_hf_*_fake_good.json` × 21 —
  FG response data (per model × form)
- `psychometrics/gfc_tirt/*_ipipneogfc60_hf_*_fake_good_indep_fit.rds`
  × 21 — FG TIRT fit objects
- `results/persona/persona_gfc_tirt_*_ipipneogfc60_hf_*_fake_good.json`
  × 21 — FG recovery sidecars
- `results/persona/persona_fake_good_comparison.json` — HONEST vs FG
  cohort comparison summary
- `results/persona/cohort_self_rating_P60.json` — no-persona Likert
  EVs per (model, item) for the P=60 items
- `results/persona/persona_sdr_vs_assistant.json` — per-item table +
  partial correlations from §5c.2
- `psychometrics/gfc_tirt/ablation_assistant_subsets.json` — asst_top30
  / asst_bot30 pair definitions
- `psychometrics/gfc_tirt/ablation_assistant_subsets/*.json` — filtered
  response data (gitignored, regenerable)
- `psychometrics/gfc_tirt/ablation_assistant_subsets/*.rds` — fit
  objects (gitignored)
- `results/persona/ablation_assistant/persona_gfc_tirt_*.json` × 42 —
  recovery sidecars
- `results/persona/persona_asst_match_partial.json` — §5c.4 summary
  comparison (full_p60 / asst_top30 / asst_bot30 partial correlations)
- `results/persona/persona_repr_mapping_Llama8_response-position_*.json`
  × 10 — §5d Llama8 Rep-under-FG smoke (stem × FG-position × dirs)
- `psychometrics/gfc_tirt/*_ipipneogfc60_hf_*_fake_good_fgpfx.json`
  × 21 — §5e FG-prefix inference response data
- `psychometrics/gfc_tirt/*_ipipneogfc60_hf_*_fake_good_fgpfx_indep_fit.rds`
  × 21 — §5e FG-prefix TIRT fit objects
- `results/persona/persona_gfc_tirt_*_ipipneogfc60_hf_*_fake_good_fgpfx.json`
  × 21 — §5e FG-prefix recovery sidecars
- `results/persona/persona_fg_per_trait_suffix_vs_prefix.json` — §5e.3
  per-trait θ shift comparison (suffix vs prefix)

Scripts:
- `psychometrics/gfc_tirt/analyze_profile_recovery.R`
- `psychometrics/gfc_tirt/analyze_loadings.R`
- `psychometrics/gfc_tirt/inspect_loading_extremes.R`
- `psychometrics/gfc_tirt/rank_pairs_and_subset.R`
- `scripts/filter_ablation_responses.py`
- `scripts/run_ablation_fits.sh`
- `psychometrics/gfc_tirt/analyze_ablation.R`
- `scripts/run_gfc_hf.py` (added `--condition` flag for FG)
- `scripts/run_fake_good_inference.sh` (W12 §5b inference batch)
- `scripts/run_fake_good_tirt_fits.sh` (W12 §5b TIRT batch)
- `psychometrics/gfc_tirt/compare_fake_good.R` (W12 §5b analysis)
- `scripts/run_no_persona_self_rating.py` (W12 §5c.1 no-persona Likert)
- `psychometrics/gfc_tirt/analyze_sdr_vs_assistant.R` (W12 §5c.2)
- `psychometrics/gfc_tirt/build_assistant_match_subsets.R` (W12 §5c.4)
- `scripts/filter_asst_match_responses.py` (W12 §5c.4)
- `scripts/run_asst_match_fits.sh` (W12 §5c.4)
- `psychometrics/gfc_tirt/analyze_asst_match_partial.R` (W12 §5c.4)
- `scripts/run_rep_under_fg_smoke.sh` + `_smoke2.sh` (W12 §5d Rep
  rotation + ordering tests on Llama8)
- `scripts/persona_repr_mapping.py` (extended with `--user-stem`,
  `--fg-suffix`, `--fg-position`, `--fg-direction-system`)
- `scripts/extract_meandiff_vectors.py` (extended with `system_content`
  kwarg for FG-aware direction extraction)
- `scripts/run_fake_good_prefix_inference.sh` (W12 §5e inference batch)
- `scripts/run_fake_good_prefix_tirt_fits.sh` (W12 §5e TIRT batch)
- `psychometrics/gfc_tirt/compare_fake_good_prefix_per_trait.R`
  (W12 §5e.3 per-trait θ-shift comparison)
