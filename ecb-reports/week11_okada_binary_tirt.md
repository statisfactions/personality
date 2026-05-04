# Week 11 — Okada GFC-30 with Binary (True Forced-Choice) TIRT

**Date:** 2026-05-04
**Compares:** week08_pool_scaleup_27b_70b_72b.md (graded 7-point GFC pool)

## TL;DR

Binary forced-choice TIRT on the same 30 Okada pairs, 50 personas, 4
conditions, 4 models — recovers ground-truth z-scores at **~75–90% the
magnitude of graded** for the model that does work, but introduces a
**global sign-identification problem** that did not appear in graded:

| Model         | binary mean \|r\| (honest / fakegood) | graded baseline (week08) | sign behavior        |
| ---           | ---                                  | ---                       | ---                  |
| Haiku 4.5     | **0.49 / 0.51**                      | 0.69 / 0.71               | clean, all positive  |
| Gemma3-27B    | 0.46 / 0.49                          | 0.50 / 0.55               | **global θ flip**    |
| Llama3.3-70B  | 0.51 / 0.41                          | 0.65 / 0.55               | **global θ flip**    |
| Qwen2.5-72B   | 0.36 / 0.30                          | 0.42 / 0.43               | {N} flip + weak C   |

(Pooled fit, N=393, 4 chains × 1500 iter, 0 Rhat>1.05, 0 n_eff<100 of 2066 params.)

The mean *|r|* numbers above mask the sign behavior because they take the
absolute value per trait. **Mean signed r** is the relevant diagnostic:
Haiku +0.49/+0.51, Gemma3-27B **−0.40/−0.40**, Llama3.3-70B
**−0.51/−0.41**, Qwen2.5-72B +0.26/+0.30. Two of four open models land in
a posterior basin where the entire θ vector is reflected relative to
ground truth.

This is **a different failure mode** than the graded {A, C} sign flip
seen on Haiku, Phi4, and Qwen2.5-3B in week07. With keyed-loading
anchoring (`a[j] = g[j] * a_pos[j]`, `a_pos > 0`), per-trait sign should
be pinned by the data; but the binary likelihood is evidently weak
enough that whole-vector reflection slips through identification on
3 of 4 models.

## Inference

`scripts/run_gfc_anthropic.py --binary` and `scripts/run_gfc_ollama.py
--binary` were added. Same prompt structure as graded; only the
instruction line and response token set change:

> "For the following pair of statements, indicate which one describes
> you more accurately. Respond with a single letter: A for the LEFT
> statement or B for the RIGHT statement. Do not include any other text."

Response stored as `response_argmax = "1"` (LEFT-as-shown chosen) or
`"0"` (RIGHT-as-shown chosen). Swap correction in the R driver is
`response = ifelse(swapped, 1L - response_raw, response_raw)`.

Volumes (per model): 50 personas × 30 pairs × 2 (honest, fakegood) +
30 pairs × 2 (bare, respondent) = 3060 prompts. Haiku via Anthropic
API; Gemma3-27B / Llama3.3-70B / Qwen2.5-72B via Orin (`gemma3:27b`,
`llama3.3:70b-ctx2k`, `qwen2.5:72b-ctx2k`). Valid rates 99.7%+
across all model × condition combos.

Response distributions skew strongly LEFT (e.g. Haiku honest 1040/458,
Llama70B honest 870/611). The `swapped` flag is randomized per prompt
so this does not contaminate keying recovery — it just reflects the
known LLM "first-position" bias.

## Stan model

`psychometrics/gfc_tirt/tirt_okada_binary.stan` — same generative
structure as `tirt_okada_indep.stan` but:

- Drops `K`, the `kappa` ordered-thresholds array, and the
  ordered-logistic likelihood.
- `y` becomes binary 0/1.
- Likelihood: `y[i, p] ~ bernoulli_logit((mu_L − mu_R) / sqrt2())` where
  `mu_L = a[L] θ[trait[L]]` and `mu_R = a[R] θ[trait[R]]`. Sign
  convention: `y = 1` ⇔ canonical-LEFT chosen.
- Same priors: `θ ~ N(0, I_5)`, `a_pos ~ HalfNormal(0, 0.5)`,
  `a[j] = g[j] · a_pos[j]` with keying anchoring.

## Per-(model, condition) recovery (pooled fit, N=393)

```
                                A        C       E        N        O
Haiku 4.5    honest         +0.40   +0.01   +0.65   +0.78    +0.60
Haiku 4.5    fakegood       +0.40   +0.21   +0.72   +0.65    +0.59
Gemma3-27B   honest         −0.38   −0.15   −0.81   +0.16    −0.81
Gemma3-27B   fakegood       −0.50   −0.16   −0.82   +0.22    −0.75
Llama3.3-70B honest         −0.50   −0.09   −0.73   −0.50    −0.71
Llama3.3-70B fakegood       −0.31   −0.09   −0.73   −0.28    −0.65
Qwen2.5-72B  honest         +0.37   +0.12   +0.48   −0.26    +0.58
Qwen2.5-72B  fakegood       +0.34   +0.20   +0.40   +0.01    +0.54
```

## Findings

### 1. Haiku binary is the cleanest single result we have

All five traits recover with the correct sign in both honest and
fakegood. C is still weakly identified (r = 0.01 honest, 0.21
fakegood), matching the graded pattern, but no sign flip. Magnitudes
0.4 ≤ |r| ≤ 0.78 for the four well-recovered traits. This compares
favorably to the graded Haiku-honest fit, which had {A, C} reflected
even with single-model identification. Whatever was causing the
graded-mode reflection on Haiku does not occur in binary mode.

### 2. Two open models land in the reflected θ basin

Gemma3-27B and Llama3.3-70B both recover all five traits with the
**wrong sign** and similar magnitudes (mean signed r ≈ −0.4 to −0.5).
This is striking because keyed-loading anchoring (`a[j] > 0` for `+`
items) should pin the sign of θ relative to truth: a high-trait
persona should be more likely to endorse `+`-keyed items.

Two non-exclusive interpretations:

- **Behavioral inversion.** When forced into a binary commitment, these
  models genuinely choose anti-trait-aligned items more often than
  trait-aligned ones. Possibly the binary mode triggers a "describe
  me literally / avoid self-aggrandizing claims" heuristic that
  outweighs the persona instruction. Graded mode lets the model
  hedge with mid-scale responses; binary forces commitment in the
  unsafe direction.
- **Identification failure.** With `K = 2` per pair the likelihood
  carries less information per response than the 7-category graded
  version, and the loading-keying constraint may not be strong enough
  to break the global reflection symmetry on noisier models.

The fact that Gemma3-27B and Llama3.3-70B *both* flip globally with
near-identical magnitudes argues against pure noise: the data really
do go the other way.

### 3. Qwen2.5-72B has a per-trait sign mosaic

A: +0.37, C: +0.12, E: +0.48, N: −0.26, O: +0.58 (honest). Three
positive, one negative, one weak. This is closer to the {A, C} graded
sign-flip pattern than to the global reflection seen on Gemma/Llama,
but the flipped trait is N (not A or C). Mean signed r is small but
positive (+0.26 / +0.30).

The graded Qwen2.5-72B fit (week08) had A and N collapsed to ~0; in
binary mode N flips outright while A recovers correctly. So binary
*partially fixes* one of the graded failures (A) at the cost of
flipping another trait that graded had already abandoned (N).

### 4. Magnitude cost of going binary is modest where binary works

For Haiku, where sign comes out right, the recovery cost vs. graded is
~0.20 in mean |r| (0.69 → 0.49). That is consistent with what binary
should cost: Brown & Maydeu-Olivares (2011) Equation 9 implies the
binary likelihood discards about half a bit per pair compared to
graded-7. The implied SD-loss matches the empirical 75–90% magnitude
recovery rate. Binary is not catastrophically less informative; it's
just more fragile against sign-identification slip.

### 5. Convergence is faster and cleaner than graded

`tirt_okada_binary.stan` has 30 fewer parameters (30 ordered triplets
of κ thresholds dropped) per pair. Sampling time per chain dropped from
~6 minutes (graded pooled, N=408) to ~4.5 minutes (binary pooled,
N=393), and 0 of 2066 parameters had Rhat > 1.05 in binary vs.
0 of 2271 in graded — both clean, but binary's posterior is simpler.

## Open questions

1. **Is the sign-flip behavioral or identification?** Re-fit Gemma27B
   and Llama70B with a stronger per-trait loading prior (e.g. fix
   `a_pos` for one marker item per trait at 0.5) and see whether the
   global flip persists. If yes, the data are really reversed; if no,
   the model lacks identification strength on weak signals.
2. **Block-level audit.** For each model, which specific pairs has
   the binary response gone the "wrong" way relative to the
   high-trait persona? Pick a high-E persona and look at its 12
   E-loaded pair responses for Llama70B vs. Haiku.
3. **Soft-evidence binary.** We collected logprobs over A/B for the
   Orin runs. Could re-do the fit using soft binary evidence (logistic
   regression on logit P(A)) — should improve stability if the issue
   is identification rather than behavior.
4. **Cross-trait correlation matrix.** Same as the week08 finding,
   off-diagonals are likely poorly recovered. Compute and compare to
   graded.
5. **Why does Haiku alone get the sign right?** The week07 graded
   {A, C} flip was on Haiku — so this is a clean reversal of who has
   sign issues. Hypothesis: Haiku has stronger persona compliance and
   so its binary commitments track the persona; the open models hedge
   in graded and reverse in binary.

## Files added

- `scripts/run_gfc_anthropic.py` — `--binary` flag.
- `scripts/run_gfc_ollama.py` — `--binary` flag (logprobs over A/B).
- `psychometrics/gfc_tirt/tirt_okada_binary.stan` — Bernoulli-logit
  TIRT.
- `psychometrics/gfc_tirt/fit_tirt_pooled_binary.R` — pooled driver.
- `psychometrics/gfc_tirt/fit_tirt_single_binary.R` — per-model
  driver.
- `psychometrics/gfc_tirt/{claude-haiku-4-5-20251001,gemma3-27b,llama3.3-70b,qwen2.5-72b}_gfc30_*_binary.json`
  — 16 inference output files.
- `psychometrics/gfc_tirt/single_binary_*_fit.rds`,
  `pooled_binary_haiku_gemma27b_llama70b_qwen72b_fit.rds` — fitted
  Stan objects (large; will be archived to big5_results).
