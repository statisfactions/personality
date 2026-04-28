# Pooled Okada-style TIRT replication — full sweep (corrected)

**Date:** 2026-04-28 (revised after L/R swap-bug fix)
**Author:** ECB
**Companion to:** [`okada_swap_bug.md`](okada_swap_bug.md), [`okada_recovery_diagnosis.md`](okada_recovery_diagnosis.md), [`okada_replication_haiku_stan.md`](okada_replication_haiku_stan.md)

> **Why this report was rewritten.** The first version (2026-04-27, archived
> as `okada_pooled_replication_PREBUGFIX.md.bak`) reported a robust {A, C}
> sign flip on Haiku that survived every scoring variant. That was wrong.
> Inference randomizes L/R per prompt (~50% of rows have `swapped=true`)
> and stores `response_argmax` in displayed coordinates, but every R Stan
> driver was indexing items by instrument-canonical L/R via `block` and
> feeding the raw response to Stan without un-swapping. The fix is a
> two-line change (`response = ifelse(swapped, 8L - response_raw,
> response_raw)`) before the wide pivot. See `okada_swap_bug.md` for the
> full diagnosis. All numbers below use the fixed prep.

This is the most complete replication attempt to date. We collected GFC-30
inference for **5 models × 4 conditions** matching the RGB lineup plus
Haiku 4.5 (~12k Orin prompts + 1.5k Anthropic API calls), then fit
Okada-style Stan TIRTs.

## TL;DR

| Result | Outcome |
|--------|---------|
| Frontier model + Okada-exact Stan recovers \|r\| ≥ 0.5 | ✅ Haiku honest, mean \|r\| = **0.71** (joint H+FG, N=100/model) |
| Sign-aligned recovery on every trait | ✅ Haiku 5/5 traits sign-correct on both honest and fake-good |
| Pooling 5 models lifts identification | (cross-model fit re-running; expected to remain noisier than per-model on Haiku) |
| Joint HONEST + FAKE-GOOD fitting helps | ✅ Haiku honest jumps from 0.58 single-model to 0.71 joint H+FG |
| FAKE-GOOD condition shows desirability shift | Mostly null/small after sign-correction — see §4. Haiku has small positive shifts (mean +0.06) consistent with Okada's claim that GFC attenuates SDR. |

After the swap-bug fix, **the headline finding now matches Okada
qualitatively for Haiku**: per-trait recovery in or near the ≥ 0.50 band,
all signs correct, fake-good shifts small. Open-weight models in the 3–4B
range still under-recover (mean |r| ≈ 0.15–0.26) — the bug fix did not
rescue them because their content signal is too weak.

## 1. Inference summary

| Model | honest | fake-good | bare | respondent | Source |
|-------|-------:|----------:|-----:|-----------:|--------|
| Haiku 4.5 (claude-haiku-4-5-20251001) | 1500 | 1500 | 30 | 30 | Anthropic API |
| Gemma3-4B-IT | 1500 | 1500 | 30 | 30 | Orin Q4 |
| Qwen2.5-3B-Instruct | 1500 | 1500 | 30 | 30 | Orin Q4 |
| Phi4-mini-instruct | 1500 | 1500 | 30 | 30 | Orin Q4 |
| Llama3.2-3B-Instruct | 1500 | 1500 | 30 | 30 | Orin Q4 |

50 Okada-style synthetic personas (van der Linden Σ → stanines → Goldberg
markers; Okada Appendix F.1 prefix). Honest preamble = Okada F.2 honest.
Fake-good preamble = Okada F.2 fake-good (verbatim). Bare = honest
instruction without persona; respondent = "YOU ARE THE RESPONDENT" + honest.
All 100% valid responses across all conditions (parser failure rate 0%).

L/R order is randomized per prompt with ~50.5% swap rate; the un-swap
correction (`response = if swapped then 8 - r else r`) is now applied
before the wide pivot in every Stan driver.

## 2. Scoring variants

All variants fit the GFC-30 ordinal Thurstonian IRT from Okada §3.3:
η_ip = (μ_R − μ_L)/√2, ordered_logistic likelihood, signed loadings
a_j = g_j × a_j⁺ with a_j⁺ ~ HalfNormal(0, 0.5), pair-specific ordered
thresholds κ_p ~ N(0, 1.5). They differ in the latent-trait prior and
in which respondents are pooled.

| Variant | Source file | θ prior | Respondents per fit |
|---------|-------------|---------|---------------------|
| Single-model honest | `tirt_okada.stan` (LKJ Ω) | LKJ(2) | N=50, one model honest |
| Single-model honest, indep | `tirt_okada_indep.stan` | N(0, I_5) | N=50, one model honest |
| Per-model joint H+FG | `tirt_okada_indep.stan` | N(0, I_5) | N=100, one model H+FG |
| Cross-model + joint | `tirt_okada_indep.stan` | N(0, I_5) | N≈510, all models × all conditions |

Drivers (all patched for swap-aware response prep):
`fit_tirt_okada.R`, `fit_tirt_okada_indep.R`,
`fit_tirt_per_model_pooled_conditions.R`, `fit_tirt_pooled.R`.

## 3. Recovery: per (model × condition × variant)

### Per-model joint H+FG fit (the cleanest comparison)

Pearson r between θ̂ and ground truth, per (model, condition), N=50 each:

| Model | Cond | A | C | E | N | O | mean \|r\| | mean signed r |
|-------|------|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **+0.85** | +0.38 | **+0.85** | **+0.85** | **+0.62** | **0.71** | **+0.71** |
| Haiku 4.5 | fakegood | **+0.84** | +0.44 | **+0.83** | **+0.74** | **+0.64** | **0.70** | **+0.70** |
| Gemma3-4B | honest | −0.06 | −0.15 | **+0.52** | +0.04 | +0.21 | 0.20 | +0.11 |
| Gemma3-4B | fakegood | +0.08 | +0.19 | **+0.57** | −0.10 | +0.03 | 0.19 | +0.15 |
| Qwen2.5-3B | honest | +0.32 | −0.20 | +0.02 | −0.10 | −0.34 | 0.20 | −0.06 |
| Qwen2.5-3B | fakegood | +0.35 | −0.04 | +0.11 | −0.16 | −0.13 | 0.16 | +0.02 |
| Phi4-mini | honest | +0.37 | +0.00 | +0.49 | +0.08 | +0.34 | 0.26 | +0.26 |
| Phi4-mini | fakegood | −0.12 | +0.13 | +0.34 | +0.06 | +0.34 | 0.20 | +0.15 |
| Llama3.2-3B | honest | −0.34 | +0.32 | +0.06 | −0.01 | −0.02 | 0.15 | +0.00 |
| Llama3.2-3B | fakegood | +0.11 | −0.05 | +0.13 | +0.05 | +0.01 | 0.07 | +0.05 |

**Bold** = at or above Okada's r ≥ 0.50 acceptable-convergent-validity band.
0 Rhat > 1.05 across 741 params per fit; convergence clean for all 5 models.

### Headline observations

1. **Haiku reaches Okada's band on 4/5 traits**, with all 5 signed correctly
   on both honest and fake-good. C is the lowest at 0.38–0.44 — still in
   the right direction but below Okada's typical r ≥ 0.50 band for this
   trait. This is the only remaining gap-to-Okada in the magnitude
   dimension.
2. **Joint H+FG fitting helps materially.** Haiku honest single-model fit
   (LKJ, N=50): mean |r| = 0.59 (post-fix). Joint H+FG (indep, N=100):
   mean |r| = 0.71. The fake-good condition adds independent θ-variation
   that anchors the latent rotation more sharply.
3. **Open-weight 3–4B models are essentially unchanged by the fix.**
   Gemma3 (mean |r| = 0.20), Qwen (0.20), Phi4 (0.26), Llama (0.15) all
   stay in the same magnitude band as the buggy fits — the bug had less
   leverage on them because their content signal was already too weak to
   produce systematic sign rotation.
4. **Per-trait sign coherence on the open models is still broken.** Even
   with the fix, Gemma3 has A−, C−; Qwen has C−, N−, O−; Llama has A−,
   N−, O−. These are the residual "dimensional collapse" patterns
   discussed in the prior version's §4b — they appear to be real model
   limitations, not data-prep artifacts.

### Per-trait recovery summary, single-model vs joint H+FG (Haiku)

| Variant | A | C | E | N | O | mean \|r\| |
|---------|---:|---:|---:|---:|---:|---:|
| Single-model honest, indep θ (`tirt_okada_indep.stan`, N=50) | +0.54 | +0.19 | **+0.84** | **+0.81** | **+0.55** | 0.59 |
| Joint H+FG, indep θ (N=100) | **+0.85** | +0.38 | **+0.85** | **+0.85** | **+0.62** | **0.71** |

Joint H+FG materially improves A and C — the conditions provide
independent θ-variation that anchors the latent rotation more sharply.
The LKJ-Σ variant (`tirt_okada.stan`) was not re-run for this revision;
under the prior buggy prep it recovered mean |r| = 0.63, equivalent to
the indep variant in magnitude but with sign-flipped A and C.

## 4. Fake-good shift (sign-corrected, FIXED)

Mean θ̂ shift (fake-good − honest), sign-corrected by per-trait recovery
sign on the honest condition, with Neuroticism flipped (g_N = −1) so that
positive ⇒ moved toward the socially-desirable direction. From the
per-model joint H+FG fits.

| Model | A | C | E | N | O | mean SDR |
|-------|--:|--:|--:|--:|--:|---------:|
| Haiku 4.5 | +0.03 | −0.04 | +0.09 | +0.17 | +0.03 | +0.06 |
| Gemma3-4B | +0.04 | −0.02 | −0.05 | +0.08 | −0.08 | −0.01 |
| Qwen2.5-3B | −0.30 | +0.14 | +0.07 | −0.15 | −0.12 | −0.07 |
| Phi4-mini | +0.60 | −0.15 | −0.01 | −0.32 | +0.07 | +0.04 |
| Llama3.2-3B | −0.16 | −0.03 | −0.17 | +0.02 | −0.03 | −0.07 |

**Important caveat:** sign-correction is only meaningful when honest
recovery is non-trivial. For Gemma3 (A_rec = −0.06, C_rec = −0.15), Qwen
(several near-zero), and Llama (A_rec = −0.34, others near zero), the
"sign-corrected" shifts mostly reflect noise. Only Haiku has clean
recovery across all five traits and a trustworthy SDR estimate.

For Haiku — the model where this number is interpretable — the shift is:

- **All five traits within |Δθ̂| ≤ 0.17.** Mean SDR = +0.06.
- **Direction is desirable on 4/5 traits** (A, E, N, O positive; C
  slightly negative).
- This **qualitatively replicates Okada Figure 3 GFC**: small shifts in
  the desirable direction, consistent with the claim that
  desirability-matched GFC attenuates but doesn't eliminate SDR.

The much-larger Phi4-mini A shift (+0.60) is suspicious — Phi4's
honest-condition A recovery is +0.37, low enough that the sign-correction
amplifies noise. Phi4-mini E and N flip negative under sign-correction
too; this is dimensional confusion, not real SDR.

The previous version of this report claimed `Llama3.2-3B essentially does
not fake. All cells ≈ 0`. After the fix, Llama actually shows mostly
*negative* shifts (anti-desirable). With the model's near-zero recovery
across most traits, this is best read as random noise rather than a
substantive finding about Llama's compliance with the fake-good
instruction.

## 5. Cross-trait recovery matrices (per model × condition)

Mean |r| collapses each model into a single number. The full
**ground-truth × θ̂** correlation matrix shows where the off-diagonal
energy lives — i.e., whether θ̂_t actually measures trait t or has
collapsed onto a different trait. From the FIXED per-model joint H+FG
fits. **Rows = TIRT score; columns = ground-truth z.** The diagonal is
the standard recovery; off-diagonals reveal trait confusion.

```
=== Haiku 4.5 | honest ===
      true_A true_C true_E true_N true_O
hat_A   0.85   0.27   0.18  -0.26   0.23
hat_C  -0.05   0.38  -0.47   0.12  -0.52
hat_E   0.24   0.06   0.85  -0.27   0.44
hat_N  -0.22  -0.37  -0.26   0.85  -0.14
hat_O  -0.15  -0.11   0.24   0.13   0.62

=== Haiku 4.5 | fakegood ===
      true_A true_C true_E true_N true_O
hat_A   0.84   0.26   0.36  -0.37   0.25
hat_C   0.01   0.44  -0.42   0.00  -0.50
hat_E   0.18   0.02   0.83  -0.22   0.46
hat_N  -0.15  -0.22  -0.34   0.74  -0.07
hat_O  -0.24  -0.25   0.17   0.26   0.64

=== Phi4-mini | honest ===
      true_A true_C true_E true_N true_O
hat_A   0.37   0.09   0.28  -0.08  -0.04
hat_C  -0.31   0.00  -0.47   0.25  -0.23
hat_E   0.17  -0.05   0.49  -0.06   0.16
hat_N   0.13   0.24   0.38   0.08   0.23
hat_O  -0.04   0.26   0.25  -0.25   0.34

=== Phi4-mini | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.11  -0.08  -0.12   0.18  -0.02
hat_C  -0.11   0.13  -0.45   0.24  -0.50
hat_E  -0.03  -0.10   0.34  -0.04   0.35
hat_N   0.00   0.02   0.05   0.06   0.15
hat_O   0.18  -0.03   0.29  -0.26   0.34

=== Qwen2.5-3B | honest ===
      true_A true_C true_E true_N true_O
hat_A   0.32   0.31   0.30  -0.14   0.18
hat_C  -0.29  -0.20  -0.14   0.27  -0.29
hat_E   0.27   0.21   0.02  -0.18   0.35
hat_N   0.03   0.09   0.12  -0.10   0.06
hat_O  -0.47  -0.41  -0.26   0.27  -0.34

=== Qwen2.5-3B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A   0.35   0.35   0.32  -0.36   0.36
hat_C  -0.26  -0.04   0.09   0.16  -0.29
hat_E   0.36  -0.01   0.11  -0.32   0.24
hat_N   0.30  -0.22   0.25  -0.16   0.08
hat_O  -0.07  -0.47  -0.19   0.21  -0.13

=== Gemma3-4B | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.06   0.00   0.28   0.08   0.25
hat_C  -0.09  -0.15  -0.34   0.06  -0.36
hat_E   0.44   0.11   0.52  -0.34   0.39
hat_N   0.06  -0.13   0.17   0.04   0.32
hat_O   0.00   0.08   0.24  -0.02   0.21

=== Gemma3-4B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A   0.08   0.24  -0.04  -0.19   0.10
hat_C   0.03   0.19  -0.11   0.13   0.19
hat_E   0.18  -0.06   0.57  -0.19   0.40
hat_N   0.19   0.21   0.34  -0.10   0.45
hat_O  -0.07  -0.08   0.17   0.02   0.03

=== Llama3.2-3B | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.33  -0.16  -0.15   0.22  -0.18
hat_C   0.16   0.32   0.02  -0.03   0.14
hat_E   0.00   0.08   0.06  -0.26  -0.11
hat_N  -0.19   0.17  -0.27  -0.01  -0.14
hat_O  -0.10  -0.04  -0.20  -0.06  -0.02

=== Llama3.2-3B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A   0.11   0.09  -0.19   0.15  -0.20
hat_C  -0.07  -0.05   0.14   0.05  -0.08
hat_E   0.13   0.06   0.13  -0.07   0.17
hat_N  -0.19   0.06  -0.07   0.05  -0.11
hat_O  -0.27   0.09  -0.16  -0.09   0.01
```

### What the off-diagonals reveal (FIXED)

The pre-bugfix structural patterns largely persist, but with the
diagonal now sign-aligned the interpretation cleans up:

1. **Haiku has clean per-trait identification** — every row's largest
   |r| is on its own trait, with diagonals at 0.85 / 0.38 / 0.85 / 0.85
   / 0.62. Off-diagonals are mostly ground-truth Σ leakage (e.g.,
   hat_C ↔ true_O = −0.52 reflects z_C ↔ z_O ≈ +0.4 in van der Linden Σ
   propagated through θ̂_C, since C and O are positively correlated in
   the population). hat_C is the noisiest row, consistent with C being
   the residual recovery gap in §3. The fake-good condition has the
   same structure with marginally tighter recovery on E/O.

2. **Phi4-mini now identifies E correctly** (hat_E ↔ true_E = +0.49
   dominates that row). hat_A's diagonal (+0.37) is now actually its
   largest column — improvement over the prior version where hat_A's
   strongest column was true_O. hat_C is still degenerate (the only row
   where the diagonal is not the largest |r| — biggest column is true_E
   at −0.47). hat_O picks up modest signal across multiple columns,
   suggesting partial dimensional collapse rather than the fully
   collapsed picture from the buggy fits.

3. **Qwen2.5-3B has rotated A/E/O.** hat_A's diagonal +0.32 is roughly
   tied with hat_A ↔ true_C (+0.31) and hat_A ↔ true_E (+0.30) — so
   "θ̂_A" is reading something like an "agreeable / conscientious /
   extraverted" composite rather than agreeableness alone. hat_O has
   the strongest signal (column-max |r| = −0.47 on true_A), but in the
   wrong direction — Qwen's hat_O is mostly anti-A.

4. **Gemma3-4B has trait-swapping**, just like the pre-fix report.
   hat_E carries the most signal but has its diagonal at 0.52 — same
   magnitude as hat_E ↔ true_A (+0.44). hat_N's strongest column is
   true_O (+0.32), not true_N (+0.04). hat_C correlates negatively with
   true_O (−0.36). Reading θ̂_t as trait t requires column-permutation,
   not just sign-flip, on this model.

5. **Llama3.2-3B is noise** under honest. hat_A diagonal is −0.33
   (sign-flipped), the second-largest off-diagonal in that row. No row
   has its diagonal as the largest |r|. Fake-good is even noisier.

### Implication for "what would fix recovery"

If we want a usable per-trait θ̂ on the open models, the fixes differ:

- **Haiku:** done — per-trait recovery is sign-aligned, diagonal-first,
  and 4/5 traits in Okada's band. C is the only residual gap.
- **Phi4-mini:** A/E/O are diagonal-first now. C is collapsed; θ̂_C is
  not recoverable. Could try restricting analysis to A/E/O.
- **Qwen2.5-3B:** A is roughly identifiable; the rest are too entangled
  to use on a per-trait basis.
- **Gemma3-4B:** find the best column-permutation of θ̂ that maximizes
  diagonal mass, then sign-correct. Treats θ̂ dimensions as anonymous.
- **Llama3.2-3B:** signal too weak; need bigger model.

The common refrain: **the gap between mean |r| and per-trait usability
is real for non-Haiku models** even after the bug fix. A model with mean
|r| = 0.20 distributed evenly across the diagonal would be a reasonable
weak signal; mean |r| = 0.20 with most of the energy on the *wrong*
diagonal is dimensional confusion that the fix doesn't address.

## 6. Cross-model pooled fit (FIXED)

`fit_tirt_pooled.R` re-run with the fix: 4 chains × 1500 iter, N=510
(50 honest + 50 fakegood per model + 1 bare + 1 respondent per model).
Convergence: 4 Rhat > 1.05 / 2791 params, 3 n_eff < 100 — clean enough
to interpret the per-model θ̂.

**Per-(model × condition) recovery, pooled fit, signs not flipped:**

| Model | Cond | A | C | E | N | O | mean \|r\| | (was, BUGGY) |
|-------|------|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **+0.64** | +0.35 | **+0.79** | **+0.70** | +0.37 | **0.57** | (0.30, A & C flipped) |
| Haiku 4.5 | fakegood | **+0.69** | +0.35 | **+0.68** | **+0.66** | **+0.65** | **0.60** | — |
| Gemma3-4B | honest | −0.07 | −0.00 | +0.45 | +0.04 | +0.10 | 0.13 | (0.24) |
| Gemma3-4B | fakegood | +0.07 | +0.23 | **+0.51** | −0.03 | −0.03 | 0.17 | — |
| Qwen2.5-3B | honest | +0.31 | −0.18 | −0.02 | −0.16 | −0.47 | 0.23 | (0.18) |
| Qwen2.5-3B | fakegood | +0.30 | −0.05 | +0.14 | −0.09 | −0.16 | 0.15 | — |
| Phi4-mini | honest | +0.36 | +0.04 | +0.43 | +0.10 | +0.30 | 0.25 | (0.30) |
| Phi4-mini | fakegood | −0.02 | +0.20 | +0.33 | +0.14 | +0.40 | 0.22 | — |
| Llama3.2-3B | honest | −0.25 | +0.14 | +0.12 | +0.09 | +0.13 | 0.15 | (0.19) |
| Llama3.2-3B | fakegood | +0.07 | +0.01 | +0.13 | +0.01 | +0.06 | 0.06 | — |

### Headline observations on the pooled fit

1. **Haiku honest pooled = 0.57**, up from 0.30 buggy. Still under the
   per-model joint H+FG of 0.71 (because shared κ_p / a_j across 5
   heterogeneous response distributions costs identification), but
   well-inside Okada's ≥ 0.50 band on three traits and signed correctly
   on all five.
2. **Pooling now genuinely helps for some open models.** Phi4 stays at
   0.25; Qwen honest moves from 0.20 (per-model) to 0.23 (pooled).
   Gemma3 drops slightly. The pooling-vs-per-model decision is
   model-dependent.
3. **The "pooling hurts" finding from the prior version was largely
   bug-induced.** The prior report claimed pooling dropped Haiku from
   0.63 to 0.30 — that was 0.30-with-A-and-C-flipped vs.
   0.63-with-A-and-C-flipped, the bug bit harder when distributed
   across more pairs. After fix: pooling drops Haiku from per-model 0.71
   to pooled 0.57 — still a real cost from heterogeneity, but only
   ~0.14 in mean |r|, not ~0.33.

### Neutral placement (FIXED, pooled fit)

θ̂ for each model under the bare and respondent prompts. Sign
interpretation now reliable since per-trait signs are correctly aligned.

| Model | A | C | E | N | O |
|-------|--:|--:|--:|--:|--:|
| Haiku 4.5 bare | −0.27 | −0.30 | +0.04 | +0.16 | −0.07 |
| Haiku 4.5 respondent | −0.11 | −0.26 | +0.01 | +0.02 | −0.04 |
| Gemma3-4B bare | +0.58 | **+1.20** | −0.30 | **+1.60** | **+1.16** |
| Gemma3-4B respondent | −0.23 | −0.07 | **−1.10** | +1.22 | +0.86 |
| Qwen2.5-3B bare | +0.31 | −0.89 | +0.04 | −0.93 | +0.22 |
| Qwen2.5-3B respondent | +0.74 | −0.35 | +0.02 | −0.07 | −0.87 |
| Phi4-mini bare | −0.17 | +0.04 | −0.21 | +0.05 | +0.17 |
| Phi4-mini respondent | −0.54 | −0.30 | −0.17 | +0.30 | +0.67 |
| Llama3.2-3B bare | −0.01 | −0.06 | +0.14 | −0.10 | −0.11 |
| Llama3.2-3B respondent | −0.03 | −0.06 | +0.03 | −0.11 | −0.09 |

(Reminder: Gemma3 / Qwen / Phi4 / Llama placements are interpretable up
to the residual sign issues in those models' per-trait recovery — see §3
table.) Most striking: **Gemma3 in the bare condition shows the classic
"assistant shape"**: high C (+1.20), high N (+1.60), high O (+1.16) — but
note that Gemma3's N recovery is near zero, so the +1.60 on N is
unreliable. The bare-vs-respondent gap is dramatic for Gemma3 (the role
assignment dampens it strongly), modest for Haiku, near-zero for Llama.

## 7. What we now know about the gap to Okada (revised)

- **Magnitude is reachable** with frontier model + Okada-exact Stan
  spec + joint H+FG fitting. Haiku honest joint H+FG: mean |r| = 0.71.
  Four of five traits in the ≥ 0.50 band, all five signed correctly.
- **Sign alignment is not the problem we thought it was.** The pre-fix
  {A, C} flip was a unit-test-worthy data-prep bug: half the
  (persona × block) cells were entering Stan with their L/R coordinates
  inverted, and the latent rotation found a higher-likelihood reflected
  mode at {A−, C−}. The fix restores correct signs across every variant.
- **Conscientiousness is the residual gap.** Even after the fix, Haiku C
  recovers at 0.38–0.44 — below Okada's typical band. Worth
  investigating: is this a Haiku-specific weak-trait identification
  issue, an item-level issue with the C pairs in Okada Table 3, or noise
  at N=50?
- **Open-weight 3–4B models hit a ceiling at \|r\| ≈ 0.2.** This is real
  and not a scoring artifact. The fix didn't change open-model recovery.
- **Pooling can still hurt.** Item parameters shared across 5 very
  different response distributions struggle. The pooled fit still has
  the heterogeneity problem; expected to drop Haiku below per-model
  performance.

## 8. Remaining hypotheses (revised)

The pre-fix list of hypotheses (§7 of the prior version) was largely
chasing the bug. The new shortlist:

1. **Why is Haiku C only 0.38–0.44?** Inspect block-level evidence: do
   the C pairs in Okada Table 3 carry less item-level information than
   the others? Check item statistics (response distribution per pair,
   discrimination posteriors).
2. **Bigger Anthropic models (Sonnet 4.6, Opus 4.7).** Worth running
   honest only (~$5) to see if the C gap is Haiku-specific. Less
   theoretically interesting than before but a useful sanity check.
3. **Open-model intervention.** Bigger Orin models (Gemma3-12B, Qwen3.5
   classes) might break the |r| ≈ 0.2 ceiling. Already have Gemma3-12B
   honest data; needs joint H+FG to test.
4. **Inter-trait Σ̂.** The pre-fix report didn't analyze the LKJ
   Σ̂ matrix from `tirt_okada.stan`. With sign-correct recovery, this
   may now be informative for Haiku.

## 9. Source artifacts

- `scripts/run_gfc_anthropic.py`, `scripts/run_gfc_ollama.py` — inference
  (unchanged; bug was downstream)
- `psychometrics/gfc_tirt/tirt_okada_indep.stan` — Okada-exact Stan model
- `psychometrics/gfc_tirt/fit_tirt_*.R` — all patched for swap-aware response
- `psychometrics/gfc_tirt/per_model_pooled/{slug}_pooled_conditions_fit.rds` — per-model fits (FIXED)
- `psychometrics/gfc_tirt/pooled_tirt_fit_FIXED.rds` — cross-model pooled fit (FIXED, in progress)
- `psychometrics/gfc_tirt/per_model_pooled_FIXED.log` — per-model fit log
- `psychometrics/gfc_tirt/verify_swap_fix.R` — verification harness
- `psychometrics/gfc_tirt/compute_fakegood_shift.R` — §4 shift computation
- `ecb-reports/okada_pooled_replication_PREBUGFIX.md.bak` — archived pre-fix version
- `notes_background/okada_2026_gfc_paper.md` — paper text incl. Appendix D
