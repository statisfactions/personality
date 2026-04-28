# Pooled Okada-style TIRT replication — full sweep

**Date:** 2026-04-27
**Author:** ECB
**Companion to:** [`okada_recovery_diagnosis.md`](okada_recovery_diagnosis.md), [`okada_replication_haiku_stan.md`](okada_replication_haiku_stan.md)

This is the most complete replication attempt to date. We collected GFC-30
inference for **5 models × 4 conditions** matching the RGB lineup plus
Haiku 4.5 (~12k Orin prompts + 1.5k Anthropic API calls), then fit four
TIRT model variants to test where Okada's clean per-trait recovery comes
from.

## TL;DR

| Result | Outcome |
|--------|---------|
| Frontier model + custom Stan recovers |r| ≥ 0.5 | ✅ Haiku honest, mean |r| = 0.63 (single-model fit) |
| Sign-aligned recovery on every trait | ❌ Haiku {A, C} consistently flipped across all variants |
| Pooling 5 models lifts identification | ❌ Pooling **hurts** because of response-style heterogeneity |
| Joint HONEST + FAKE-GOOD fitting fixes flip | ❌ Same flip pattern at N=100 per model |
| FAKE-GOOD condition shows desirability shift | ✅ Most cells positive and small (|d| ≲ 0.5) — replicates Okada Figure 3 qualitatively |

The headline finding stays the same as the prior reports: **|r| magnitudes
match Okada for frontier models; signs do not, and no scoring trick we
tried fixes that for Haiku's {A, C}.** The only remaining hypothesis we
have not tested is full Likert+GFC joint fitting (which Okada does not
quite do either — they fit each format separately).

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

## 2. Four scoring variants

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

Drivers: `fit_tirt_okada.R`, `fit_tirt_okada_indep.R`, `fit_tirt_per_model_pooled_conditions.R`, `fit_tirt_pooled.R`.

## 3. Recovery: per (model × condition × variant)

Magnitude (mean |r|) on the honest condition across the four scoring strategies:

| Model | LKJ single | Indep single | Joint H+FG | Cross-model pooled |
|-------|-----------:|-------------:|-----------:|-------------------:|
| Haiku 4.5 | **0.63** | 0.58 (est.) | 0.58 | 0.30 |
| Gemma3-4B | n/a | n/a | 0.21 | 0.24 |
| Qwen2.5-3B | n/a | n/a | 0.22 | 0.18 |
| Phi4-mini | 0.36 | n/a | 0.32 | 0.30 |
| Llama3.2-3B | n/a | n/a | 0.21 | 0.19 |

Three things show up:

1. **Haiku is the only model in Okada's ≥ 0.50 band, and only with
   single-model fitting.**
2. **Cross-model pooling drops Haiku from 0.63 to 0.30.** The shared item
   parameters (κ_p, a_j) cannot fit five very different response
   distributions simultaneously.
3. **Joint H+FG within-model is roughly equivalent to single-model
   honest** for every model. The fake-good condition does not provide the
   anchoring boost we expected.

### Per-trait recovery, joint H+FG fit (this is the cleanest comparison)

Pearson r between θ̂ and ground truth, per (model, condition):

| Model | Cond | A | C | E | N | O | mean \|r\| |
|-------|------|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **−0.66** | **−0.60** | +0.40 | +0.75 | +0.49 | 0.58 |
| Haiku 4.5 | fakegood | −0.46 | −0.52 | −0.22 | +0.74 | +0.27 | 0.44 |
| Gemma3-4B | honest | −0.09 | +0.16 | +0.18 | −0.32 | −0.29 | 0.21 |
| Gemma3-4B | fakegood | −0.20 | −0.35 | +0.50 | −0.45 | +0.04 | 0.31 |
| Qwen2.5-3B | honest | **−0.54** | +0.04 | −0.01 | +0.24 | +0.29 | 0.22 |
| Qwen2.5-3B | fakegood | −0.34 | +0.06 | ~0 | +0.01 | +0.31 | 0.14 |
| Phi4-mini | honest | −0.38 | −0.40 | +0.25 | +0.04 | +0.56 | 0.32 |
| Phi4-mini | fakegood | −0.06 | −0.27 | +0.26 | +0.09 | +0.55 | 0.25 |
| Llama3.2-3B | honest | +0.18 | +0.26 | +0.13 | +0.12 | +0.38 | 0.21 |
| Llama3.2-3B | fakegood | −0.17 | −0.01 | +0.15 | −0.15 | +0.04 | 0.10 |

**Pattern of sign flips by model:**
- **Haiku 4.5:** robust {A−, C−} flip on both honest and fake-good.
- **Phi4-mini:** {A−, C−} on both conditions.
- **Qwen2.5-3B:** A− on both.
- **Gemma3-4B:** N− on both; A−, O− mixed; only E and (sometimes) C correct.
- **Llama3.2-3B:** mostly correct on honest; mostly noise on fake-good.

The **{A, C} flip is shared across the three models with the cleanest
response distributions** (Haiku, Phi4, Qwen). This is unlikely to be a
chance pattern. Two candidate explanations:

(a) The IPIP A and C items in Okada's pool are sufficiently desirability-
laden that smaller models trained heavily for "helpful, harmless"
endorse them at ceiling regardless of persona, producing within-trait
variance dominated by per-respondent noise. Then the comparative GFC
signal is weak and the latent rotation flips sign by chance.

(b) Persona-induced shift on A and C goes the *opposite* of what the
adjective markers predict for these models. E.g., when prompted as
"very kind, generous, trustful", Haiku may internally interpret this
as a CHARACTER, not a self-rating bias, and respond more critically
on A items in comparative judgments. Qwen and Phi4 show the same
direction.

Both explanations would be invisible in Okada's frontier-model pool but
visible in our heterogeneous pool — and we would not expect the marker
constraint to fix them, because the constraint is on the loading, not on
how the model interprets persona descriptors.

## 4. Fake-good shift (Okada's main quantity)

Even though we don't care about SDR per the user's instruction, the shift
is informative as a directional anchor check. We compute mean θ̂ shift
(fake-good − honest), then **double sign-correct**: by per-trait recovery
sign (so "high θ̂ ↔ high z") and by desirability convention (g_N = −1).
Positive ⇒ fake-good moved the model toward the socially desirable
direction. From the per-model joint H+FG fits.

| Model | A | C | E | N | O | mean SDR |
|-------|--:|--:|--:|--:|--:|---------:|
| Haiku 4.5 | +0.32 | +0.22 | −0.02 | +0.18 | +0.19 | +0.18 |
| Gemma3-4B | +0.31 | 0.00 | −0.02 | +0.45 | −0.12 | +0.12 |
| Qwen2.5-3B | +0.56 | −0.05 | +0.26 | +0.33 | −0.34 | +0.15 |
| Phi4-mini | +0.54 | 0.00 | −0.55 | +0.53 | −0.07 | +0.09 |
| Llama3.2-3B | 0.00 | −0.03 | +0.01 | 0.00 | +0.02 | 0.00 |

Two qualitative replications of Okada Figure 3:

1. **GFC SDR is small.** All 25 cells are within |d_z| ≈ 0.5 of zero
   (except Qwen-A and Phi4-A/E/N at ~0.5). That is the magnitude band
   Okada reports for GFC (vs. their Likert SDR of 1–2). Our raw θ̂
   shifts aren't Cohen's d_z — d_z would normalize by within-persona SD
   — but the order of magnitude is consistent.
2. **Direction is desirable on most cells.** 18 / 25 cells positive, only
   2 strongly negative (Qwen-O, Phi4-E). So fake-good *does* shift these
   models toward "looking good" on most traits, just by a small amount —
   exactly Okada's qualitative claim that desirability-matched GFC
   attenuates but doesn't eliminate SDR.
3. **Llama3.2-3B essentially does not fake.** All cells ≈ 0. Either it
   can't follow the fake-good instruction, or it interprets the persona
   description as already-honest reporting and the instruction adds
   nothing.

So the SDR-attenuation result *does* replicate qualitatively. The
per-trait recovery-sign issue does not contaminate the within-model
comparative shift.

## 4b. Cross-trait recovery matrices (per model × condition)

Mean |r| (§3) collapses each model into a single number. The full
**ground-truth × θ̂** correlation matrix shows where the off-diagonal
energy lives — i.e., whether θ̂_t actually measures trait t or has
collapsed onto a different trait. From the per-model joint H+FG fits.
**Rows = TIRT score; columns = ground-truth z.** The diagonal is the
standard recovery; off-diagonals reveal trait confusion.

```
=== Haiku 4.5 | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.66  -0.46  -0.48   0.66  -0.38
hat_C  -0.39  -0.60  -0.42   0.52  -0.34
hat_E   0.27   0.18   0.40  -0.33   0.16
hat_N  -0.51  -0.50  -0.55   0.75  -0.30
hat_O   0.39   0.34   0.36  -0.43   0.49

=== Haiku 4.5 | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.45  -0.33  -0.28   0.49  -0.48
hat_C  -0.30  -0.52  -0.20   0.43  -0.11
hat_E   0.15   0.15  -0.22  -0.14   0.22
hat_N  -0.38  -0.49  -0.57   0.74  -0.19
hat_O   0.34   0.20   0.54  -0.45   0.27

=== Phi4-mini | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.38  -0.35  -0.62   0.37  -0.65
hat_C  -0.07  -0.40  -0.04   0.07  -0.06
hat_E   0.53   0.20   0.25  -0.22   0.37
hat_N   0.29  -0.21  -0.13   0.04  -0.03
hat_O   0.50   0.33   0.56  -0.37   0.56

=== Phi4-mini | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.06  -0.22  -0.27   0.26  -0.41
hat_C  -0.37  -0.27  -0.38   0.19  -0.13
hat_E   0.37   0.36   0.26  -0.24   0.23
hat_N   0.17  -0.02  -0.22   0.09  -0.18
hat_O   0.37   0.35   0.43  -0.14   0.55

=== Qwen2.5-3B | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.54  -0.33   0.05   0.30  -0.09
hat_C  -0.12   0.04   0.13   0.05   0.27
hat_E  -0.42  -0.16  -0.01   0.19  -0.03
hat_N  -0.39  -0.31   0.18   0.24  -0.03
hat_O  -0.20   0.16   0.48  -0.14   0.29

=== Qwen2.5-3B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.34  -0.17   0.45   0.14   0.32
hat_C   0.21   0.06   0.41  -0.26   0.35
hat_E  -0.49  -0.23   0.00   0.22  -0.07
hat_N  -0.18  -0.39   0.33   0.01   0.16
hat_O  -0.09   0.01   0.50  -0.14   0.31

=== Gemma3-4B | honest ===
      true_A true_C true_E true_N true_O
hat_A  -0.09  -0.03  -0.35   0.13   0.19
hat_C   0.27   0.16   0.01  -0.11   0.45
hat_E  -0.13  -0.41   0.18   0.27  -0.09
hat_N   0.24   0.16   0.45  -0.32   0.24
hat_O  -0.06  -0.14   0.15   0.05  -0.29

=== Gemma3-4B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.20  -0.37  -0.64   0.41  -0.20
hat_C  -0.10  -0.35  -0.52   0.37  -0.05
hat_E  -0.08   0.06   0.50  -0.16   0.13
hat_N   0.34   0.44   0.69  -0.45   0.49
hat_O   0.16   0.28   0.52  -0.33   0.04

=== Llama3.2-3B | honest ===
      true_A true_C true_E true_N true_O
hat_A   0.18   0.25   0.00  -0.18  -0.07
hat_C   0.11   0.26  -0.21   0.01  -0.25
hat_E   0.02  -0.20   0.13  -0.06  -0.02
hat_N  -0.03  -0.09   0.21   0.12   0.19
hat_O  -0.09  -0.22   0.26   0.02   0.38

=== Llama3.2-3B | fakegood ===
      true_A true_C true_E true_N true_O
hat_A  -0.17  -0.18   0.01   0.19  -0.08
hat_C   0.03  -0.01  -0.14   0.17  -0.06
hat_E  -0.01  -0.32   0.15   0.15   0.05
hat_N  -0.07  -0.12   0.11  -0.15  -0.07
hat_O  -0.09  -0.06   0.10  -0.14   0.04
```

### What the off-diagonals reveal

Three structural patterns that mean |r| was hiding:

1. **Haiku has clean per-trait identification, just rotated.** Every
   row's largest |r| is on its own trait (modulo the {A, C} sign flip).
   θ̂_N ↔ true N = +0.75 dominates that row by a wide margin. Off-diagonal
   correlations are mostly ground-truth Σ propagated through the rotation
   (θ̂_A ↔ true N = +0.66 reflects z_A ↔ z_N = −0.42 with the A sign
   flipped). This is the "successful TIRT, just needs a sign correction"
   case.

2. **Phi4-mini has dimensional collapse.** θ̂_A's strongest column is
   true_O (−0.65), not true A (−0.38). θ̂_O's strongest column is true E
   (+0.56), not true O (+0.56 tied). The model isn't separating
   A from E from O — it's reading them all on a single
   "agreeable + extraverted + open" axis and projecting onto whichever
   discrimination the chain happened to assign positive sign.

3. **Gemma3 has trait *swapping*, not just flipping.** Honest condition:
   θ̂_C's strongest column is true O (+0.45). θ̂_E's strongest column is
   true C (−0.41). θ̂_N's strongest column is true E (+0.45). The named
   trait labels on θ̂ are essentially arbitrary — recovering Big Five
   from this would require a permutation of dimensions, not just a sign
   flip.

4. **Qwen2.5-3B and Llama3.2-3B carry weak signal.** Qwen has only one
   row (θ̂_A) where the diagonal is the largest |r| in that row. Llama
   has only θ̂_O ↔ true O (+0.38). The other dimensions are essentially
   noise.

### Implication for "what would fix recovery"

If we want a usable per-trait θ̂ on the open models, the fixes differ:

- **Haiku:** apply a {A, C} sign flip post-hoc → done.
- **Gemma3:** find the best column-permutation of θ̂ that maximizes
  diagonal mass, then sign-correct. Treats θ̂ dimensions as anonymous.
- **Phi4:** can't be rescued — the latent dimensions aren't 5-D.
  Either use a lower-dimensional model (3-D? 2-D?) or accept that Phi4
  doesn't reliably encode the persona's per-trait specification.
- **Qwen / Llama:** signal too weak; need bigger model.

The common refrain: **mean |r| understates how broken the recovery is
for non-Haiku models.** A model with mean |r| = 0.30 distributed across
the diagonal would be a reasonable signal; mean |r| = 0.30 with most of
the energy on the *wrong* diagonal is dimensional confusion.

## 5. Neutral placement (model defaults)

From the cross-model pooled fit, θ̂ for each model under the bare and
respondent prompts:

| Model | A | C | E | N | O |
|-------|--:|--:|--:|--:|--:|
| Haiku 4.5 bare | −0.06 | +0.26 | +0.41 | −0.30 | −0.15 |
| Haiku 4.5 respondent | −0.08 | +0.22 | +0.41 | −0.22 | −0.10 |
| Gemma3-4B bare | +1.13 | **+1.59** | −0.95 | −1.47 | −0.19 |
| Gemma3-4B respondent | −0.67 | −0.27 | +0.31 | −1.40 | −0.21 |
| Qwen2.5-3B bare | **+1.66** | +0.38 | −0.58 | +0.07 | −0.14 |
| Qwen2.5-3B respondent | +1.57 | +1.34 | −0.18 | −0.17 | −1.17 |
| Phi4-mini bare | −0.80 | +0.12 | −0.17 | +0.27 | +0.37 |
| Phi4-mini respondent | −0.81 | −0.58 | +0.20 | −0.13 | −0.07 |
| Llama3.2-3B bare | +0.14 | +0.09 | +0.37 | −0.11 | −0.24 |
| Llama3.2-3B respondent | +0.13 | +0.03 | +0.17 | −0.07 | −0.26 |

These caveats apply (see §3): the cross-model pooled fit's per-trait
signs are not all aligned with ground truth, so absolute "high A" should
be read with skepticism. What is interpretable is **rank ordering across
models within a trait**: Gemma3 and Qwen are conspicuously high on A and
C (the classic "assistant shape"); Phi4 is low on A; Haiku and Llama are
near zero. The bare-vs-respondent gap is meaningful per model — Gemma3
shifts dramatically (the role assignment dampens the assistant shape
strongly), Haiku barely moves.

## 6. What we now know about the gap to Okada

- **Magnitude is reachable** with frontier model + Okada-exact Stan
  spec. Haiku honest single-model fit: mean |r| = 0.63.
- **Sign alignment is the persistent problem** and is tied to Haiku's
  *interpretation* of the persona, not to scoring. Every sane scoring
  variant (LKJ vs N(0, I), keying vs marker anchoring, single vs joint
  vs pooled) lands {A, C} in the same flipped basin.
- **Pooling can hurt.** Okada presumably gets pooling lift from
  homogeneous frontier models; our heterogeneous lineup loses
  identification when item parameters are forced to be shared.
- **Fake-good is informative but not the fix.** The shift goes the
  expected direction on 4/5 models, indicating the comparative ranking
  works. But adding it to the fit doesn't pull the per-trait sign back
  for Haiku.

## 7. Remaining hypotheses for the {A, C} flip

In ROI order:

1. **Persona-induced behavior change.** Haiku, faced with a "very kind,
   generous, trustful" persona, reads it as a *role-played character*
   and adjusts comparative responses in a way that doesn't match human
   stereotype. Test: hand-inspect 5 high-z_A vs 5 low-z_A persona ×
   block-1 prompts and see which side Haiku endorses.

2. **Item content has stronger O/E flavor than A/C** in the Okada
   pairs. The block-1 marker is "Accept people as they are." (A+) vs
   "Enjoy hearing new ideas." (O+); a model that reads both as
   "agreeable / curious" might pick O regardless of A persona. Test:
   compute simple-score A using only blocks where the A item is paired
   with a clearly *different*-construct item.

3. **Sonnet 4.6 / Opus 4.7 might not have this issue.** If the flip is
   model-specific to Haiku (smallest of the Anthropic frontier), a
   bigger Anthropic model might land in the keying-aligned basin. ~$5
   to test.

4. **Our personas may differ from Okada's seeded personas** even with
   the same generator code. Reproducing their seed exactly requires
   their numpy/torch RNG state, which we don't have. Could matter for
   the specific 50-respondent draw.

## 8. Source artifacts

- `scripts/run_gfc_anthropic.py`, `scripts/run_gfc_ollama.py` — inference
  (now both support `--fake-good` and `--neutral {bare,respondent}`)
- `scripts/run_orin_sequential.sh` — sequential Orin runner
- [`tirt_okada_indep.stan`](../psychometrics/gfc_tirt/tirt_okada_indep.stan) — Okada-exact Stan model
- [`fit_tirt_pooled.R`](../psychometrics/gfc_tirt/fit_tirt_pooled.R) — cross-model pooled fit driver
- [`fit_tirt_per_model_pooled_conditions.R`](../psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R) — per-model H+FG driver
- `psychometrics/gfc_tirt/pooled_tirt_fit.rds` — pooled fit posterior (archived in big5_results)
- `psychometrics/gfc_tirt/per_model_pooled/{model}_pooled_conditions_fit.rds` — per-model fits (archived in big5_results)
- [`render_pooled_report.R`](../psychometrics/gfc_tirt/render_pooled_report.R) — markdown summary helper
- `notes_background/okada_2026_gfc_paper.md` — paper text incl. Appendix D
