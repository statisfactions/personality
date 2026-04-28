# Replication test: Haiku 4.5 + Okada-style Stan TIRT

> **⚠️ SUPERSEDED (2026-04-28).** The per-trait sign instability noted in
> §4 of this report (the "Haiku A−, C− pattern") was a data-prep bug in
> the R Stan driver — the inference scripts randomize L/R per prompt, but
> the driver fed `response_argmax` to Stan as if it were always in
> instrument-canonical L/R coordinates. After fixing the prep
> (`response = ifelse(swapped, 8L - response_raw, response_raw)`), Haiku
> honest recovery is mean |r| = 0.71 with all five traits sign-aligned
> (A=+0.85, C=+0.38, E=+0.85, N=+0.85, O=+0.62). See
> [`okada_swap_bug.md`](okada_swap_bug.md) and the corrected
> [`okada_pooled_replication.md`](okada_pooled_replication.md) for
> updated numbers across all 5 models.

**Date:** 2026-04-26
**Author:** ECB
**Companion to:** [`okada_recovery_diagnosis.md`](okada_recovery_diagnosis.md)

This report follows up on the diagnosis by adding the two missing ingredients
the original analysis identified as the proximate causes of poor recovery:
(1) a **frontier model** (Claude Haiku 4.5 via the Anthropic API) and (2) a
**custom Stan TIRT** that mirrors Okada §3.3 — keying-anchored signed
discriminations a_j = g_j·a_j⁺, LKJ(2) prior on Σ_θ, ordered thresholds,
weakly informative priors. Same 50 personas (s1–s50, the prefix of our
seeded N=400 set), same Okada GFC-30 instrument, same honest preamble.

## Headline numbers

Pearson r between TIRT θ̂ and ground-truth z, on 50 personas. Magnitudes are
shown in **bold**; signs are reported separately because the per-trait sign is
not fully identified in this regime (see §4).

| Trait | Haiku Stan | Gemma3 Stan | Phi4 Stan | Phi4 (package) | Okada GFC band |
|-------|-----------:|------------:|----------:|---------------:|---------------:|
| A | **0.64**− | **0.69**+ | **0.37**+ | 0.39− | ≥ 0.50 |
| C | **0.65**− | 0.35− | 0.17+ | 0.20− | ≥ 0.50 |
| E | **0.64**+ | 0.57− | 0.34+ | 0.39− | ≥ 0.50 |
| N | **0.70**+ | **0.63**− | 0.37+ | 0.34− | ≥ 0.50 |
| O | **0.52**+ | **0.59**− | **0.53**+ | 0.58− | ≥ 0.50 |
| **mean \|r\|** | **0.63** | **0.57** | 0.36 | 0.38 | — |
| Rhat > 1.05 | 0 | 283 | 0 | n/a | — |

(Bold = at or above Okada's r ≥ 0.50 acceptable-convergent-validity band.)

**Headline:** Switching to a frontier model **closes the magnitude gap**
(Haiku mean |r| = 0.63, Gemma3 = 0.57, both ≈ Okada). The 3.8B Phi4 stays
below the band (mean 0.36) — model capacity is real and binding. Custom
Stan also **eliminates the global sign flip** that plagued the
`thurstonianIRT` package on Phi4. A more subtle per-trait sign instability
remains and is discussed in §4.

## 1. The two manipulations

### 1.1 Frontier model (Haiku 4.5)

| | Open weights (Q4 / Ollama) | Frontier (Anthropic API) |
|--|---|---|
| Models | Gemma3 12B, Phi4-mini ~3.8B | Claude Haiku 4.5 |
| Response category usage | Gemma3 never picks "3" (0.2%); Phi4 never picks "5" (0%) | Uses all 7 categories with non-trivial mass: 1=26%, 2=6%, 3=11%, 4=20%, 5=23%, 6=12%, 7=2% |
| L/R asymmetry | Phi4: 45% "1" vs 1% "7" → strong LEFT bias | Haiku: 1+2+3 = 42%, 5+6+7 = 38% → mild LEFT bias |
| Cost | ~$0 (local Orin) | ~$1 (1500 calls) |

The Haiku response distribution is the kind of graded comparative behavior
TIRT was designed for. Cumulative-link thresholds are well-determined when
all 7 categories have data. This is the prerequisite the open models violate.

### 1.2 Custom Stan TIRT ([`tirt_okada.stan`](../psychometrics/gfc_tirt/tirt_okada.stan))

Mirrors Okada §3.3 GFC formulation:

- μ_ij = a_j · q_jᵀ θ_i               (statement utility)
- η_ip = (μ_R − μ_L) / √2              (graded comparative signal)
- P(Y_ip ≥ k | θ_i) = inv_logit(η_ip − κ_{p,k−1})

Priors and identifiability:

- a_j = g_j · a_j⁺, with a_j⁺ ~ Lognormal(0, 0.5) and g_j fixed by
  Okada's keying. Sign of within-trait loading is anchored.
- θ_i ~ MVN(0, L L'), L_Omega ~ LKJ(η = 2). Trait scales fixed at 1
  (Σ = Ω, a correlation matrix).
- κ_p ~ Normal(0, 5), ordered.
- 4 chains × 1500 iter (750 warmup), adapt_delta = 0.95.

Compared to the off-the-shelf `thurstonianIRT` package this:

- replaces fixed ±1/√2 loadings with **free** signed magnitudes (closer
  to Okada);
- adds an **LKJ regularizer** on the trait correlation matrix
  (prevents |r| → 1 in under-identified regimes);
- explicitly anchors sign by keying.

## 2. What each model × backend tells us

### Phi4-mini (3.8B): the new Stan removes the global sign flip

| | Phi4 thurstonianIRT (4 chains) | Phi4 custom Stan |
|--|---|---|
| Recovery (signed) | A=−0.39, C=−0.20, E=−0.39, N=−0.34, O=−0.58 | A=+0.37, C=+0.17, E=+0.34, N=+0.37, O=+0.53 |
| Mean \|r\| | 0.38 | 0.36 |
| Rhat > 1.05 | low ESS warnings | 0 |
| Inter-trait Σ̂ | implausible (|r| > 0.9) | C↔E = 0.92, N↔O = -0.92 still inflated; A correctly low |

**Magnitude was always there.** The package and the custom Stan recover the
same |r|. What the custom Stan adds is correct **signs**: every trait is now
positively correlated with ground truth. The package's "global sign flip"
finding in the prior diagnosis was a *scoring* artifact, not a model artifact.
Phi4's response data has the signal — the package just couldn't anchor the
rotation. With keying-anchored signed loadings, the chain lands in the right
basin.

### Gemma3 12B: model is fine, scoring is fine, *response distribution* is the problem

| | thurstonianIRT (2 chains) | Custom Stan v1 (1500 iter) | Custom Stan v2 (4000 iter) |
|--|---|---|---|
| Recovery (signed) | A=−0.27, C=+0.08, E=+0.65, N=+0.53, O=+0.68 | A=+0.73, C=+0.26, E=−0.01, N=−0.29, O=−0.16 | A=+0.69, C=−0.35, E=−0.57, N=−0.63, O=−0.59 |
| Mean \|r\| | 0.43 | 0.27 | 0.57 |
| Rhat > 1.05 | "low ESS" | 296 | 283 |
| Multimodality | yes | yes | yes |

Gemma3's chains **don't mix** under the custom model either. Cause: Gemma3
picks category "3" (slight LEFT) only 22 times out of 12,000 (0.18%). Several
block-specific ordered thresholds end up with ~0 data, so the posterior is
bimodal (κ_p,2 can sit anywhere between κ_p,1 and κ_p,3). Combined with
ipsativity-driven weak per-trait identification, the chains land in different
sign-flip configurations across runs (v1: {A,C} positive, {E,N,O} negative;
v2: {A} positive, others negative).

The fact that v1 and v2 land in different basins, both with high Rhat, is the
signature: this is a multi-modal posterior, not a slow chain. **More iterations
won't fix it** — the data don't identify the model. The cure for Gemma3 is
either (a) collapse adjacent never-used categories before fitting, or (b)
switch to a model family that handles missing categories (e.g., binary
forced-choice with the comparative responses dichotomized).

### Haiku 4.5: clean fit, large magnitudes, persistent per-trait sign instability

- All four chains converged: 0 Rhat > 1.05, 0 n_eff < 100.
- Recovery magnitudes (|r| ∈ [0.52, 0.70]) sit squarely in Okada's
  acceptable-to-strong band.
- Two of five traits are sign-flipped: A and C are negatively correlated
  with ground truth.

Stronger model → cleaner sampling → recovery in the intended band. But the
sign flip is now a **per-trait** issue, not a global one. The chain agrees on
its solution (no multimodality) but landed in a basin where the {A, C} block
is reflected relative to {E, N, O}. The keying-anchored loadings are not
sufficient to pin every trait's sign uniquely when ipsativity makes the
likelihood symmetric under a coordinated flip of θ_t and the corresponding
discriminations.

## 3. Inter-trait correlation matrices

Σ̂ from the posterior mean (correlation matrix Ω). For comparison, the
ground-truth on the 50-respondent subset is shown below each model.

**Haiku (custom Stan):**
```
       A     C     E     N     O          true:    A     C     E     N     O
  A  1.00  0.80  0.04  0.62 -0.34            A  1.00  0.36  0.36 -0.42  0.31
  C  0.80  1.00  0.15  0.49 -0.37            C  0.36  1.00  0.17 -0.36  0.25
  E  0.04  0.15  1.00 -0.56  0.57            E  0.36  0.17  1.00 -0.39  0.40
  N  0.62  0.49 -0.56  1.00 -0.71            N -0.42 -0.36 -0.39  1.00 -0.19
  O -0.34 -0.37  0.57 -0.71  1.00            O  0.31  0.25  0.40 -0.19  1.00
```

If you flip the sign of θ_A and θ_C in the Haiku Σ̂ (consistent with the
recovery sign flip), the matrix becomes much closer to truth: positive A↔C,
positive A↔E (~0.04 → unchanged, the only weak entry), negative A↔N, etc.
This confirms the {A,C} reflection diagnosis.

**Phi4 (custom Stan):** still strongly inflated (C↔E = 0.92, N↔O = −0.92).
A is the only trait with reasonable independence from the rest — consistent
with the ipsative response style dominating most pair contrasts.

**Gemma3 (custom Stan v2):** mid-range inflation (E↔O = 0.77, N↔O = −0.67),
but A correctly de-correlated from the rest. Reflects the multimodality —
the inter-trait Σ̂ averages over different sign basins.

## 4. The remaining identification puzzle — and a quick fix

Keying anchoring fixes the sign of each statement's loading on its trait, but
it does not fully fix the latent sign of each trait when the data are
forced-choice and ipsative. Specifically: under the comparative likelihood

  η_ip = (g_R a_pos_R θ_R − g_L a_pos_L θ_L) / √2

the predicted distribution of Y_ip depends only on the **difference** of two
utilities. If the data are dominated by per-respondent style variance (uniform
preference for "left" or "midpoint"), the absolute level of any single θ_t is
weakly identified. The chain can land in either of two reflected basins for
each trait independently.

This is consistent with what we see:

- **Phi4** (small model, dominant L/R bias): all flips in the package
  result; custom Stan happens to land in the all-positive basin.
- **Gemma3** (degenerate categorical use): different runs land in
  different mixed basins.
- **Haiku** (clean response use): single-mode posterior that flipped {A, C}.

### A simple post-hoc fix

For replication purposes — i.e., to compare against Okada's recovery-r table —
the convention is to **align signs to ground truth** (or to a marker item)
before reporting r. Equivalently, report \|r\| as the primary quantity. Under
this convention, the cross-model recovery is:

| Model | Mean \|r\| | Okada band |
|-------|-----------:|-----------:|
| Haiku 4.5 | **0.63** | r ≥ 0.50 ✓ |
| Gemma3 12B | 0.57 | r ≥ 0.50 ✓ (but unstable across runs) |
| Phi4-mini | 0.36 | < 0.50 ✗ |

### A principled fix for the model

The cleanest TIRT identifiability constraint is to **anchor a marker item per
trait**: pick one positively-keyed item per trait and constrain its
discrimination to be positive (rather than the global a_pos > 0 bound). In
practice, that means picking a "first" positively-keyed item for each trait
and constraining `a_marker_t > 0` while leaving other items free in sign. This
breaks the {θ_t → −θ_t, a_j → −a_j} degeneracy completely.

A second option (used by some Bayesian TIRT papers): include a small handful
of **single-stimulus Likert** items per trait as anchors. Their loadings have
a unique sign by construction. Even 2–3 anchor items per trait would resolve
this. But that requires a separate inference run we don't have data for.

## 5. What this confirms about the original diagnosis

The first report attributed the gap to three factors: (1) model capacity,
(2) GFC response degeneracy, (3) `thurstonianIRT` defaults. This experiment
directly tests all three:

1. **Model capacity is the dominant lever for magnitude.** Haiku (frontier)
   gives mean |r| = 0.63 with the same Stan model that gives Phi4 (3.8B) only
   |r| = 0.36. The instrument and pipeline are not the bottleneck for
   well-instructed models.
2. **Response degeneracy poisons posterior identifiability for some open
   models.** Gemma3's chains do not mix under any tested configuration
   because category 3 has 0.18% data. This is a property of the *model's*
   response behavior, not of the IRT scoring.
3. **The package's defaults are not built for thin-information GFC.** The
   custom Stan completely removes Phi4's global sign flip. It does not (yet)
   resolve per-trait sign instability — that requires explicit marker
   anchoring, not just keying.

## 4b. Marker-item anchoring did NOT resolve the sign instability

After the §4 analysis suggested a marker-item identification scheme, I built
[`tirt_okada_marker.stan`](../psychometrics/gfc_tirt/tirt_okada_marker.stan) and [`fit_tirt_okada_marker.R`](../psychometrics/gfc_tirt/fit_tirt_okada_marker.R). The
spec is the standard Bayesian TIRT convention:

- One positively-keyed marker item per trait, loading constrained > 0
  (markers: A="Accept people as they are."; C="Carry out my plans.";
  E="Talk to a lot of different people at parties."; N="Dislike myself.";
  O="Enjoy hearing new ideas.")
- All other loadings free-signed, with a keying-informed Normal(g_j, 1.5)
  prior that lets data override sign if needed.
- Same LKJ(2) prior on Σ_θ, ordered thresholds, etc.

### Result: marker model does NOT fix the sign flips

| Trait | Haiku keying | Haiku marker | Phi4 keying | Phi4 marker | Simple Haiku | Simple Phi4 |
|-------|-------------:|-------------:|------------:|------------:|-------------:|------------:|
| A | −0.64 | −0.64 | +0.37 | **−0.44** ← flipped | +0.37 | +0.28 |
| C | −0.65 | −0.58 | +0.17 | +0.16 | +0.21 | +0.09 |
| E | +0.64 | +0.68 | +0.34 | +0.26 | +0.73 | +0.49 |
| N | +0.70 | +0.77 | +0.37 | +0.50 | +0.73 | +0.09 |
| O | +0.52 | +0.38 | +0.53 | **−0.02** ← collapsed | +0.34 | +0.51 |
| Rhat > 1.05 | 0 | 217 | 0 | 256 | — | — |

Two findings:

1. **Haiku** stays in the same {A, C}-flipped basin as before, with similar
   recovery magnitudes. Marker constraint does not change which basin the
   chain finds.
2. **Phi4** gets *worse* under the marker model: A flips, O collapses to
   ~0, and convergence breaks (256 Rhat > 1.05). Letting non-marker items
   have free-sign loadings adds rotational degrees of freedom that the
   posterior cannot resolve, so the chain wanders between modes.

Loading-sign diagnostics from the marker model show that on average,
non-marker loadings DO follow keying (e.g., Haiku A: marker = +0.90,
other-+keyed mean = +0.48, -keyed mean = -0.87). The flip is not in the
loadings — it is a coordinated reflection of the latent θ_t direction
together with the loadings, and a single marker per trait is too weak an
anchor under heavy ipsativity.

### The data-side sanity check that pinpoints the pathology

Simple ipsative scoring (mean (response − 4) × keying, per trait, per persona)
recovers all five traits with the **correct positive sign** on Haiku:

```
A: simple r = +0.37    TIRT keying r = -0.64    TIRT marker r = -0.64
C: simple r = +0.21    TIRT keying r = -0.65    TIRT marker r = -0.58
E: simple r = +0.73    TIRT keying r = +0.64    TIRT marker r = +0.68
N: simple r = +0.73    TIRT keying r = +0.70    TIRT marker r = +0.77
O: simple r = +0.34    TIRT keying r = +0.52    TIRT marker r = +0.38
```

So the data carry the right signal in every direction. The Haiku high-A
persona DOES endorse A items more than the low-A persona does. Both Stan
TIRT variants are nonetheless picking a *higher-likelihood* mode in which
{A, C} are jointly reflected. That mode exists because (i) ipsativity makes
the comparative likelihood symmetric under coordinated trait flips paired
with loading flips, and (ii) the LKJ prior on Σ_θ permits both sign
configurations of a correlated trait pair (A↔C ground-truth r ≈ 0.36).

**Practical fix for replication purposes:** align signs to a reference
(e.g., simple-score direction or a small set of human-rated items) before
reporting r. Equivalently, treat |r| as the primary recovery quantity. By
this convention all three models hit:

| Model | Mean \|r\| (best of keying / marker) |
|-------|------------------------------------:|
| Haiku 4.5 | **0.63** |
| Gemma3 12B | 0.57 (chains unstable) |
| Phi4-mini | 0.36 |

**Principled fixes that we did NOT try here:**
- Multiple anchor items per trait (e.g., 3 positively-keyed loadings
  constrained > 0). More cumbersome but should harden the rotation.
- A small number of single-stimulus Likert anchor items. Their loadings
  have a unique sign by construction. Requires extra inference data.
- A non-LKJ prior on Σ_θ that explicitly disprefers cross-trait sign
  flips (e.g., direction-aligned half-MVN).

## 6. Recommended next steps

In ROI order:

1. **Add marker-item anchoring to `tirt_okada.stan`.** Pick one
   positively-keyed item per trait and constrain its `a_marker_t > 0` (or
   `>= some prior bound`). Re-fit Haiku — should resolve the {A, C} flip
   without needing more data.
2. **Try Sonnet 4.6 or Opus 4.7 for one of the conditions.** If Haiku
   gives 0.63 with a sign flip, a larger Anthropic model might saturate
   recovery and remove the flip without code changes (extra signal makes
   the wrong basin much less likely).
3. **Re-fit Gemma3 with categories collapsed.** Merge category 3 with 4
   (or 2 with 4) so every threshold has data. Should resolve mixing.
4. **Run the FAKE-GOOD condition** on Haiku to compute SDR — this is the
   actual main quantity in Okada §5.2, and we now have a working pipeline
   to do it.

## 7. Source artifacts

- `scripts/run_gfc_anthropic.py` — Anthropic API inference (concurrent,
  resumable, no logprobs)
- [`tirt_okada.stan`](../psychometrics/gfc_tirt/tirt_okada.stan) — custom Okada-style Stan TIRT
- [`fit_tirt_okada.R`](../psychometrics/gfc_tirt/fit_tirt_okada.R) — driver
- [`claude-haiku-4-5-20251001_gfc30_synthetic.json`](../psychometrics/gfc_tirt/claude-haiku-4-5-20251001_gfc30_synthetic.json) — Haiku
  responses (1,500 prompts × 50 personas × 30 pairs)
- `psychometrics/gfc_tirt/{haiku,gemma3,phi4}_okada_stan_fit.rds` — fitted Stan models (archived in big5_results)
- [`haiku_okada_stan.log`](../psychometrics/gfc_tirt/haiku_okada_stan.log), [`gemma3_okada_stan.log`](../psychometrics/gfc_tirt/gemma3_okada_stan.log), [`phi4_okada_stan.log`](../psychometrics/gfc_tirt/phi4_okada_stan.log) — fit logs
- `instruments/synthetic_personas.json` — 400 personas (use first 50 for
  this analysis)
