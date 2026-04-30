# Gemma3-27B added to Okada GFC + TIRT lineup

**Date:** 2026-04-28
**Author:** ECB
**Companion to:** [`week07_okada_pooled_replication.md`](week07_okada_pooled_replication.md), [`week07_okada_swap_bug.md`](week07_okada_swap_bug.md)

## TL;DR

Added **Gemma3-27B-IT (Q4 on Orin)** to the lineup as a controlled scale-up
from Gemma3-4B, plus fit a 2-model pooled TIRT (Haiku 4.5 + Gemma3-27B)
to test whether pooling with one strong anchor model is enough to lift
identification on the new entrant.

| Result | Outcome |
|---|---|
| 27B inference is tractable on Orin | ✅ ~0.7–0.8 prompts/sec, full sweep (3060 prompts) in ~95 min wall |
| Size lifts Gemma in single-model fit | ✅ mean \|r\| 0.196 (4B) → 0.323 (27B) honest; E doubles, O doubles, N goes from noise to 0.27 |
| 27B uses the full 7-cat range | ✅ All 7 categories used (4B skips cat 5) — better fit for shared κ |
| Gemma's {A, C} traits become identifiable | ❌ Single-model: A=0.05, C=-0.01 — essentially random |
| Pooling rescues A on Gemma | ✅ Pooled with Haiku: A jumps 0.05 → 0.43, C still ≈0 |
| Net cost to Haiku from pooling | Small: 0.71 → 0.64 mean \|r\|, no sign flips |

The headline: **27B alone reaches 0.32 mean \|r\| but cannot identify A or
C from its own data.** Pooling with a single strong anchor (Haiku) rescues
A almost completely. C remains stuck near zero on Gemma at any size we've
tested — looks like a real content-signal limit, not a TIRT identification
artifact.

## 1. Inference

Pulled `gemma3:27b` (~17 GB Q4_K_M) onto the Orin server. Ran the
canonical 4-condition sweep:

| Condition | Prompts | Wall time | Throughput |
|---|---:|---:|---:|
| honest (50 personas × 30 pairs) | 1500 | ~33 min | 0.75/sec |
| fake-good (50 personas × 30 pairs) | 1500 | ~31 min | 0.81/sec |
| neutral bare | 30 | < 1 min | — |
| neutral respondent | 30 | < 1 min | — |
| **Total** | **3060** | **~65 min sampling + 30 min Stan + setup** | — |

Parser: 3060/3060 valid responses. `generated_text` starts with
`response_argmax` in 100% of rows. Swap balance 50/50 (744 swapped on
honest, 760 on fake-good). All blocks evenly covered (50 personas × 30
blocks per condition).

Output files (in `psychometrics/gfc_tirt/`):

```
gemma3-27b_gfc30_synthetic.json            # honest, 1500 rows
gemma3-27b_gfc30_synthetic-fakegood.json   # fake-good, 1500 rows
gemma3-27b_gfc30_neutral-bare.json         # 30 rows
gemma3-27b_gfc30_neutral-respondent.json   # 30 rows
```

### Response-style: 27B uses all 7 categories

| Cat | Gemma3-4B (honest) | Gemma3-27B (honest) |
|---:|---:|---:|
| 1 | 202 | 189 |
| 2 | 41 | 65 |
| 3 | 679 | 88 |
| 4 | 18 | 135 |
| 5 | **0** | 68 |
| 6 | 67 | 748 |
| 7 | 493 | 207 |

The 4B variant skips category 5 entirely and concentrates on {1, 3, 7} — a
classic compressed-Likert response style. The 27B variant uses all 7
categories, with the modal response shifted to 6 (away from cat-7 ceiling).
This is exactly the response-style smoothing that makes pooled κ thresholds
work better — the same threshold structure can fit Haiku and 27B without
forcing one into the other's grid.

## 2. Single-model TIRT fit (joint honest + fake-good, N=100)

Driver: `psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R`
(added a cache check so existing models aren't refit). Stan model is
`tirt_okada_indep.stan` (Okada Appendix D priors: a_j+ ~ HN(0, 0.5),
κ ~ N(0, 1.5), θ ~ N(0, I_5)). 4 chains × 1000 iter (333 warmup).
Convergence: 0 Rhat > 1.05 of 741 params.

Per-trait recovery vs synthetic ground truth (Pearson r), comparing
the two Gemma3 sizes plus Haiku as anchor:

| Model | Cond | A | C | E | N | O | mean \|r\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **0.854** | 0.378 | 0.853 | 0.847 | 0.620 | **0.710** |
| Haiku 4.5 | fakegood | 0.842 | 0.440 | 0.833 | 0.741 | 0.643 | 0.700 |
| Gemma3-27B | honest | 0.045 | -0.006 | **0.817** | 0.266 | 0.479 | 0.323 |
| Gemma3-27B | fakegood | 0.040 | -0.279 | 0.701 | 0.273 | 0.636 | 0.386 |
| Gemma3-4B | honest | -0.056 | -0.154 | 0.520 | 0.041 | 0.210 | 0.196 |
| Gemma3-4B | fakegood | 0.080 | 0.188 | 0.566 | -0.103 | 0.029 | 0.193 |

Δ(27B − 4B) by trait, honest condition:

| | A | C | E | N | O | mean \|r\| |
|---|---:|---:|---:|---:|---:|---:|
| Δ | +0.10 | +0.15 | **+0.30** | **+0.22** | **+0.27** | +0.13 |

**Size-effect interpretation.** E, N, and O each gain 0.20–0.30 in
recovery. E nearly doubles (0.52 → 0.82), approaching Haiku's 0.85. O more
than doubles (0.21 → 0.48). N goes from indistinguishable from zero to a
modest 0.27. **A and C do not benefit** — both stay essentially zero.
This is not the {A, C} sign flip we saw on Haiku before the swap fix
(commit `a007852`); it's a clean failure to identify the latent direction
at all.

## 3. Two-model pooled fit (Haiku + Gemma3-27B)

Driver: `psychometrics/gfc_tirt/fit_tirt_pooled_haiku_gemma27b.R` (new).
Same Stan model. 4 chains × 3000 iter (1000 warmup). N=204 rows
(50 honest + 50 fakegood + 1 bare + 1 respondent per model).
Convergence: **0 Rhat > 1.05, 0 n_eff < 100** of 1261 params.

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.810 | 0.210 | 0.860 | 0.808 | 0.507 | 0.639 | -0.07 |
| Haiku 4.5 | fakegood | 0.811 | 0.275 | 0.786 | 0.718 | 0.459 | 0.610 | -0.09 |
| Gemma3-27B | honest | **0.434** | 0.090 | 0.802 | 0.245 | 0.497 | **0.414** | **+0.09** |
| Gemma3-27B | fakegood | 0.340 | -0.289 | 0.653 | 0.222 | 0.651 | 0.431 | +0.05 |

The most striking number is **Gemma3-27B's A, which jumps from 0.045 (single) → 0.434 (pooled)** — a roughly 10× lift. The signal was always
in Gemma's data; what was missing was a way to anchor the latent direction.
Haiku's strong A signal (r=0.85 on its own) provides that anchor through
the shared a_j+ discriminations and shared κ thresholds.

C also recovers a small positive sign (-0.006 → 0.090) but remains weak.
N and O are essentially unchanged. The pooling cost on Haiku is real but
modest: A holds, E holds, N holds; C drops from 0.378 to 0.210 and O drops
from 0.620 to 0.507. No sign flips.

### Mean \|r\| across the four (model × condition) rows

| | mean of mean \|r\| |
|---|---:|
| Single-model fits | 0.530 |
| Pooled (2-model) fit | 0.524 |

Cash-neutral on average, but qualitatively much better: Gemma's A is
identifiable, where it wasn't before.

## 4. Why pooling rescued A but not C on Gemma

**A** improves dramatically (0.045 → 0.434) because Gemma3-27B carries
*some* A-related variance that, on its own, the model couldn't disambiguate
from rotational noise — the off-diagonal in the trait latent space. With
Haiku anchoring the A direction (loadings, thresholds), Gemma's A
projection finds a stable basis. The fact that the lift is so large
suggests the latent A signal in Gemma's responses is genuinely there —
just unidentified single-model.

**C** does not improve in any meaningful way (-0.006 → 0.090). Two ways
to read this: either C-related GFC items elicit no between-persona
variance from Gemma at the 27B scale (a content-signal failure), or
Gemma's C signal is sufficiently mismatched in shape from Haiku's that
the shared a_j+/κ machinery cannot fit both at once. The fact that
**C honest is fine on Haiku alone** (single-model 0.38) but the pooled
fit drops Haiku's C to 0.21 favors the second reading: shared discriminations
cannot accommodate both.

**A useful test** would be to run the pooled fit with relaxed (per-model)
discriminations, keeping only thresholds shared. That would localize whether
the limitation is Gemma's content or the shared-a_j constraint.

## 5. Neutral placement

Pooled-fit θ̂ for the no-persona conditions (in pooled latent space —
sign convention from Haiku's anchoring):

| Model | Condition | A | C | E | N | O |
|---|---|---:|---:|---:|---:|---:|
| Haiku 4.5 | bare | -0.28 | -0.07 | 0.20 | 0.02 | -0.31 |
| Haiku 4.5 | respondent | -0.12 | 0.04 | 0.06 | -0.15 | -0.10 |
| Gemma3-27B | bare | -0.41 | -0.13 | 0.10 | 0.28 | -0.01 |
| Gemma3-27B | respondent | **0.74** | 0.06 | -0.25 | **0.97** | -0.65 |

Haiku is near-zero in both modes — its "default" personality on GFC is
roughly average. Gemma3-27B is also near-zero in `bare` mode, but the
`respondent` instruction pushes it to extreme positive A and N (≈ +1 SD)
and somewhat negative O. This is unusual — most models we've tested place
neutrally regardless of framing. Worth a closer look in a follow-up,
particularly because A and N moving together is suggestive of a
"compliant/anxious respondent" voicing rather than a single trait shift.

## 6. Diagnosis for the audit asked at start

Before fitting, audited inference + scoring code for both Gemma sizes
to rule out a measurement bug:

| Check | Gemma3-4B | Gemma3-27B |
|---|---|---|
| Rows | 1500/1500 | 1500/1500 |
| Swap balance T/F | 772/728 | 745/755 |
| Block coverage min/max | 50/50 | 50/50 |
| `generated_text` starts with `response_argmax` | 1500/1500 | 1500/1500 |
| Categories used | {1,2,3,4,6,7} | {1..7} |
| Stan Rhat > 1.05 | 0/741 | 0/741 |
| Swap-fix applied (`response = if swapped 8 - r else r`) | yes | yes |

**No bugs identified.** The 4B → 27B comparison is therefore a clean
size-effect read.

## 7. Open hypotheses / follow-ups

1. **Per-model discriminations, shared thresholds.** Run a variant of
   the pooled Stan model where a_j+ varies by model but κ_p is shared.
   Tests whether Gemma's C failure is a content-signal limit or a
   shared-discrimination constraint.
2. **Pooled fit with all "good responders".** Add Phi4-mini honest
   (which has the best mean \|r\| of the 3B class) to the Haiku +
   Gemma3-27B pool. Three anchors should further tighten the latent
   geometry. (Caveat: Phi4 skips category 5, like 4B does — would need
   to verify the κ thresholds can absorb that.)
3. **Gemma's A-and-N respondent shift.** Investigate why
   `respondent` framing pushes Gemma3-27B to extreme A and N. Is this
   a roleplay artifact ("I am a survey participant" → compliant +
   anxious)? Compare prompts directly.
4. **Try Gemma3 4B with the larger model's responses as a prior.** Use
   the pooled fit's posterior κ and a_j+ as fixed point estimates,
   then refit 4B's θ alone against them. Tests whether 4B has *any*
   recoverable A signal once the geometry is fixed.
5. **Gemma 4 once the Orin Ollama is updated.** Today's pull failed
   with `412: requires newer Ollama version`. Worth pinging the server
   admin — gemma4:26b (MoE, 4B active) would be a meaningful step
   beyond gemma3:27b on capability while running faster.

## 8. Files added / modified this report

```
psychometrics/gfc_tirt/
  gemma3-27b_gfc30_synthetic.json                # NEW
  gemma3-27b_gfc30_synthetic-fakegood.json       # NEW
  gemma3-27b_gfc30_neutral-bare.json             # NEW
  gemma3-27b_gfc30_neutral-respondent.json       # NEW
  fit_tirt_per_model_pooled_conditions.R         # +Gemma3-27B in MODELS, +cache check
  fit_tirt_pooled_haiku_gemma27b.R               # NEW
  per_model_pooled/gemma3-27b_pooled_conditions_fit.rds  # NEW (archived)
  pooled_haiku_gemma27b_fit_3000.rds             # NEW (archived)
```
