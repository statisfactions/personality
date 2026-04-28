# Okada A/C sign flip — root-cause investigation

**Date:** 2026-04-28
**Author:** ECB
**Predecessor:** [`okada_pooled_replication.md`](okada_pooled_replication.md)

## TL;DR

**There is a bug.** Inference scripts randomize L/R presentation per prompt
(`swapped` field in JSON output), but every R Stan-driver feeds raw
`response_argmax` straight into Stan as if responses were always in
instrument-canonical L/R coordinates. With ~50.5% of rows swapped, this
scrambles half the data into the wrong sign. The {A, C} sign flip Haiku
showed across every "scoring variant" was the same bug — every variant
shared the same buggy data-prep pipeline.

After fixing the prep step (un-swap responses before Stan):

| Trait | Haiku honest, BUGGY | Haiku honest, FIXED |
|-------|--------------------:|--------------------:|
| A | **−0.654** | **+0.542** |
| C | **−0.591** | **+0.191** |
| E | +0.448 | +0.844 |
| N | +0.738 | +0.814 |
| O | +0.495 | +0.548 |
| mean \|r\| | 0.585 | **0.588** |

All five traits now recover with the **correct sign**. Mean magnitude is
unchanged — the signal was already there, the prep code just mis-mapped
half the rows. E and N jump substantially; A and O improve modestly; C is
weakest in magnitude (~0.19) but signed correctly.

This single fix resolves the headline finding from the prior report.
Hypotheses §7 ({A, C} flip from "persona role-play", "Okada item content",
"need bigger Anthropic model") were chasing a phantom.

## The bug

### What inference does (correct)

`scripts/run_gfc_anthropic.py` and `scripts/run_gfc_ollama.py` randomize
which statement of each pair appears as LEFT vs RIGHT in the prompt
(default behavior, controlled by `--no-randomize`). For each prompt they
record:

```json
{
  "block": 4,
  "swapped": false,                          // or true
  "left_trait":  "A", "left_keying":  "+",   // displayed (post-swap)
  "right_trait": "O", "right_keying": "+",   // displayed (post-swap)
  "response_argmax": "1"                      // 1=displayed-LEFT endorsed
}
```

Crucially, `left_trait` / `right_trait` reflect the **displayed** L/R
(post-swap), and `response_argmax` is in those displayed coordinates.
Within a single row, this is internally consistent.

### What the Stan drivers do (buggy)

Every Stan driver builds `stmt_df` from the **instrument-canonical** L/R
in `instruments/okada_gfc30.json` (i.e., `pairs$left`, `pairs$right`):

```r
stmt_df <- tibble(
  block = rep(seq_len(P), each = 2),
  side  = rep(c("L", "R"), times = P),
  trait = c(rbind(pairs$left$trait, pairs$right$trait)),  # canonical
  key   = c(rbind(pairs$left$keying, pairs$right$keying)) # canonical
)
L_idx <- stmt_df$stmt_index[stmt_df$side == "L"]   # canonical-left ids
R_idx <- stmt_df$stmt_index[stmt_df$side == "R"]   # canonical-right ids
```

Then it pivots responses into a wide y matrix indexed by `block`:

```r
results_df <- results_df %>%
  mutate(response = as.integer(response_argmax))   # <-- raw, no un-swap
wide <- results_df %>%
  pivot_wider(names_from = block, values_from = response, ...)
```

Stan then evaluates `eta = (a[R[p]]·θ_{R} − a[L[p]]·θ_{L}) / √2` against
`y[i, p]`, treating `y[i,p]=1` as "canonical-LEFT endorsed". But for the
~50% of (persona × block) cells with `swapped=TRUE`, `y=1` actually means
the **canonical-RIGHT** statement was endorsed. The L/R label gets
inverted per random row.

### Effect on recovery

Half the rows enter the likelihood with η flipped. For traits where
position-bias is small and content-signal is large, the likelihood finds a
reflected posterior mode that matches the average sign of the corrupted
data. Empirically that produces **systematic sign flips on the traits with
the strongest content-signal mismatch** — A and C in our data — while
leaving traits where the noise averages near-symmetrically (E, N, O)
mostly correct. With the bug in place, the latent rotation finds a higher
likelihood at the {A−, C−} basin; "every scoring variant landed in the
same basin" because every variant shared the broken data-prep step.

## What was checked

I went through the full pipeline. Findings:

### 1. Inference prompt template — **clean**

`run_gfc_anthropic.py` builds prompts from `GFC_INSTRUCTION` +
`GFC_ITEM_TEMPLATE` exactly as Okada specifies (1=LEFT much more, …,
7=RIGHT much more). Persona/honest/fake-good/bare/respondent preambles
match Okada Appendix F.1/F.2 verbatim. No off-by-one in scale labels.

### 2. Inference response parsing — **clean**

`parse_response()` regexes the first `[1-7]` from the model output. We
verified 100% valid responses on Haiku across all 4 conditions.
`generated_text` is preserved alongside for audit. The
displayed→stored coordinate handling at `administer_one()` is correct:
when `swapped=True`, `actual_left = pair["right"]` and `actual_right =
pair["left"]`, so the JSON's `left_trait` / `right_trait` truly reflect
the displayed pair.

### 3. Instrument file `okada_gfc30.json` — **clean**

Spot-checked traits/keying/text against Okada Table 3. 30 pairs, 12 items
per Big Five trait, 3 per trait-pair combination. Item text matches the
public-domain IPIP markers. No mis-keying.

### 4. Synthetic personas / ground truth — **clean**

`instruments/synthetic_personas.json` has 400 personas with `z_scores.A`,
`.C`, `.E`, `.N`, `.O`. The first 50 (s1..s50) are what the fits use.
Ground-truth covariance approximately matches van der Linden Σ. No sign
inversion vs. Big Five convention.

### 5. Stan model `tirt_okada_indep.stan` — **clean**

Matches Okada Appendix D: `θ ~ N(0, I_5)`, `a_pos ~ HalfNormal(0, 0.5)`,
`κ ~ N(0, 1.5)`, signed loadings `a = g · a_pos`, `η = (μ_R − μ_L)/√2`,
`y ~ ordered_logistic(η, κ)`. The keying convention is correct: a high-θ
on a "+"-keyed left statement → high μ_L → low η → low y → "LEFT
endorsed". Sanity-checked the math.

### 6. Simple scoring — **clean and informative**

`fit_tirt_synthetic.R` (and the per-model sibling scripts) compute simple
scores using `right_trait`/`left_trait` and `right_keying`/`left_keying`
from the JSON — i.e., post-swap displayed metadata. This is internally
consistent with `response_argmax` and yields **sign-correct recovery on
Haiku honest** for all five traits (A=+0.37, C=+0.21, E=+0.73, N=+0.73,
O=+0.34). This was the smoking gun: the same data, scored properly, gives
positive A and C. So the data carry the right sign; only the Stan-bound
scoring path was wrong.

### 7. Stan-driver data prep — **bug here**

`fit_tirt_okada.R`, `fit_tirt_okada_indep.R`, `fit_tirt_okada_marker.R`,
`fit_tirt_per_model_pooled_conditions.R`, and `fit_tirt_pooled.R` all
share the same defective `mutate(response = as.integer(response_argmax))`
followed by `pivot_wider(... values_from = response)`. None of them
consult the `swapped` field. Every Stan recovery number in the prior
reports inherits this bug.

### 8. Ground-truth alignment — **clean**

`gt_aligned <- gt %>% filter(...) %>% arrange(match(persona_id,
wide$persona_id))` — persona row order in y_mat matches gt order. No
shuffling.

## Confirming counter-test

For each of the 5 models on the honest condition, I ran simple scoring
two ways: (a) using the displayed-L/R metadata in the JSON (correct), and
(b) using instrument-canonical L/R indexed by `block` and ignoring
`swapped` (the same pattern as the Stan drivers). Result:

```
                 trait    FIXED    BUGGY
Haiku honest       A     +0.373   -0.612    <-- flips negative
                   C     +0.212   -0.545    <-- flips negative
                   E     +0.734   +0.093
                   N     +0.726   +0.532
                   O     +0.340   +0.527
Phi4-mini honest   A     +0.210   -0.391    <-- flips negative
                   C     +0.066   -0.288    <-- flips negative
                   E     +0.442   +0.150
                   N     +0.118   +0.027
                   O     +0.441   +0.608
Qwen2.5-3B honest  A     +0.283   -0.181
                   C     -0.151   -0.006
                   ... (weaker signal across; bug less systematic)
```

The {A−, C−} flip is **specifically the buggy-canonical-coords artefact**
on the two models with the strongest content signal (Haiku, Phi4). Models
with weaker content signal (Gemma3, Qwen, Llama) showed mixed/noisy
patterns under both — for those models the bug is added noise rather than
a systematic flip.

## The fix

Two-line change to every Stan driver — un-swap the response before
pivoting:

```r
# Before
mutate(response = as.integer(response_argmax))

# After
mutate(response_raw = as.integer(response_argmax),
       response = ifelse(swapped, 8L - response_raw, response_raw))
```

Files patched:

- `psychometrics/gfc_tirt/fit_tirt_okada.R`
- `psychometrics/gfc_tirt/fit_tirt_okada_indep.R`
- `psychometrics/gfc_tirt/fit_tirt_okada_marker.R`
- `psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R`
- `psychometrics/gfc_tirt/fit_tirt_pooled.R`

A short verification script `psychometrics/gfc_tirt/verify_swap_fix.R`
runs the Haiku honest single-model fit both ways and prints recovery
side-by-side. Output (full Stan, 4 chains × 1000 iter):

```
=== BUGGY: ignore swapped (current behavior) ===
  A: -0.654   C: -0.591   E: +0.448   N: +0.738   O: +0.495   mean |r|: 0.585

=== FIXED: 8 - response when swapped ===
  A: +0.542   C: +0.191   E: +0.844   N: +0.814   O: +0.548   mean |r|: 0.588
```

## What changes about the prior report's conclusions

The prior report's **§3 (recovery)**, **§4b (cross-trait matrices)**, and
**§7 (open hypotheses)** are all affected. Specifically:

- The {A, C} flip on Haiku, Phi4, Qwen across all four scoring variants —
  was the bug, not a model property.
- "Pooling hurts" (§3.2): unclear whether this still holds; the pooled
  fit will need to be re-run with the fix. The heterogeneity argument may
  still apply but the magnitude difference shown was inflated by the bug.
- "{A, C} share the flip across three models" (§3 conclusion) — same root
  cause, gone after fix.
- "Persona-induced behavior change" / "block-1 confound" / "try Sonnet/Opus"
  (§7 hypotheses 1–3) — were chasing the bug. Worth still trying Sonnet
  for absolute magnitude but not for fixing the sign.

The §4 (fake-good shift) results probably need re-checking too. The
within-model fake-good−honest *difference* may have been less affected
since the bug applies symmetrically to both conditions, but the
sign-corrected anchor and the per-trait magnitudes will shift.

## Re-fit results (per-model joint H+FG, FIXED)

All 5 models re-fit with the swap-aware prep. Convergence clean
(0 Rhat > 1.05 across 741 params per fit).

| Model | Cond | A | C | E | N | O | mean \|r\| | (was, BUGGY) |
|-------|------|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **+0.85** | +0.38 | **+0.85** | **+0.85** | **+0.62** | **0.71** | (0.58) |
| Haiku 4.5 | fakegood | **+0.84** | +0.44 | **+0.83** | **+0.74** | **+0.64** | **0.70** | (0.44) |
| Gemma3-4B | honest | −0.06 | −0.15 | +0.52 | +0.04 | +0.21 | 0.20 | (0.21) |
| Gemma3-4B | fakegood | +0.08 | +0.19 | +0.57 | −0.10 | +0.03 | 0.19 | (0.31) |
| Qwen2.5-3B | honest | +0.32 | −0.20 | +0.02 | −0.10 | −0.34 | 0.20 | (0.22) |
| Qwen2.5-3B | fakegood | +0.35 | −0.04 | +0.11 | −0.16 | −0.13 | 0.16 | (0.14) |
| Phi4-mini | honest | +0.37 | +0.00 | +0.49 | +0.08 | +0.34 | 0.26 | (0.32) |
| Phi4-mini | fakegood | −0.12 | +0.13 | +0.34 | +0.06 | +0.34 | 0.20 | (0.25) |
| Llama3.2-3B | honest | −0.34 | +0.32 | +0.06 | −0.01 | −0.02 | 0.15 | (0.21) |
| Llama3.2-3B | fakegood | +0.11 | −0.05 | +0.13 | +0.05 | +0.01 | 0.07 | (0.10) |

### Headline updates

1. **Haiku honest reaches mean |r| = 0.71**, up from 0.58 buggy and now
   matching/exceeding Okada's ≥0.50 band on **four of five traits**
   (A=0.85, E=0.85, N=0.85, O=0.62). C=0.38 is the only trait below
   Okada's band, signed correctly.
2. **Sign alignment is no longer a problem on Haiku.** All five traits
   correlate positively with ground truth on both honest and fake-good.
3. **Open-source models are essentially unchanged.** Gemma3, Qwen, Phi4,
   Llama mean |r| stay in 0.15–0.26 range — the bug fix doesn't rescue
   them because their content signal is too weak. The pattern of
   sign-mixed results across traits (Gemma A−, C−; Qwen O−; Llama A−,
   E flat) is the residual signal-vs-noise issue, not a swap artifact.
4. **Phi4 honest, A= +0.37 (was −0.38).** The other "{A−, C−}" model
   from the prior report is also fixed — Phi4 now has A & C signed
   correctly (A=+0.37, C=+0.00) instead of negative.
5. **Fake-good vs honest comparison still meaningful.** Haiku
   fake-good = 0.70 (almost unchanged from honest 0.71), so the
   instruction doesn't materially degrade trait recovery.

The §4 fake-good shift table from the prior report needs to be
re-computed against the corrected θ̂. Direction-of-shift may flip on the
A and C cells. Cross-model pooled fit has not yet been re-run.

## Source artifacts

- `scripts/run_gfc_anthropic.py`, `scripts/run_gfc_ollama.py` — inference
  (unchanged; bug is downstream).
- `psychometrics/gfc_tirt/fit_tirt_*.R` — patched (see "The fix").
- `psychometrics/gfc_tirt/verify_swap_fix.R` — verification harness.
- All `*_gfc30_*.json` inference outputs are unchanged; data is fine, only
  the prep code was wrong.
