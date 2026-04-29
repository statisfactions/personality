# Qwen2.5-72B added to Okada GFC + TIRT lineup; 4-model pool

**Date:** 2026-04-30
**Author:** ECB
**Companion to:** [`week09_llama33_70b_replication.md`](week09_llama33_70b_replication.md), [`week08_gemma3_27b_replication.md`](week08_gemma3_27b_replication.md), [`week07_okada_pooled_replication.md`](week07_okada_pooled_replication.md)

## TL;DR

Added **Qwen2.5-72B (Q4_K_M, 47 GB) on Orin** as the second 70B-class
model in the lineup, completing the §6.3 follow-up from week09. Three
results worth foregrounding:

1. **Pooling rescued, not broke, Qwen's collapsed dimensions.** Qwen's
   single-model honest fit had **A = 0.02 and N = 0.14** (essentially
   uncorrelated with ground truth) while E/C/O recovered cleanly.
   Adding Qwen to the 3-model pool with Haiku + Gemma3-27B + Llama3.3-70B
   *increased* its recovery: **A 0.02 → 0.42, N 0.14 → 0.33**, mean |r|
   0.48 → 0.61. The keying-aligned anchors pulled Qwen's weak axes onto
   the correct latent direction.
2. **Heterogeneous category usage didn't fight shared κ thresholds.**
   Qwen's response distribution is bimodal-with-skips (cats 1/4/5
   carry ~83% of mass, cat 7 ~15%, cats 2/3/6 nearly absent at <2%
   combined). The week07 5-model pool failed because of κ heterogeneity,
   so this looked like a stress test. It wasn't: the 4-model pool
   converged cleanly (0/2271 Rhat>1.05) and other models held within
   ±0.07 of single-model recovery. Suggests the original failure was
   specifically *skipped middle-of-scale categories* (Phi4 no-cat-5,
   Gemma3-4B no-cat-3) rather than category subsets in general.
3. **Qwen2.5-72B does not fake-good.** Mean EV is 3.84 honest vs 3.83
   fake-good — a numerical zero. Joins Llama3.2-3B as the second model
   in the lineup that either can't or won't shift in the desirability
   direction. The other six models all show |Δ| ≥ 0.1 on at least
   three traits.

| Result | Outcome |
|---|---|
| Pull + run 72B on the 64 GB Orin (47 GB on disk) | ✅ — same `num_ctx=2048` Modelfile workaround as Llama3.3-70B |
| 47 GB pull through nginx proxy | Required two retries via streaming endpoint to avoid 504 timeouts |
| Inference throughput | 0.6 prompts/sec (matches Llama70B), 1500-prompt persona run in ~44 min |
| Single-model recovery | Partial — strong on E/C/O, collapsed on A/N |
| 4-model pool convergence | ✅ 0 Rhat>1.05 of 2271 params |
| Pool effect on Qwen | **+0.13 mean \|r\|** — anchor effect lifts collapsed dimensions |
| Pool effect on Haiku / Llama70B | -0.07 / -0.02 mean \|r\| (within noise) |
| Inter-trait correlation recovered | ❌ off-diagonals still poor for all 4 (RMSD 0.40–0.47) |

## 1. Inference setup

### Pull and memory

`qwen2.5:72b` is 47 GB Q4_K_M on disk — slightly larger than Llama3.3-70B
(43 GB). Same memory math as week09: default 128K context blows the
64 GB Orin budget; reduced-context Modelfile variant fixes it.

```bash
curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:72b-ctx2k","from":"qwen2.5:72b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create
```

The 47 GB pull itself was fragile through the nginx proxy. The
non-streaming `/api/pull` request hit a 504 Gateway Timeout after
~10 minutes (proxy gives up before pull finishes). Switching to
`stream:true` and sending `-N` (no buffering) on curl let the proxy see
periodic progress bytes, but the first stream still dropped at ~28 / 47 GB.
Ollama caches partial blob downloads server-side, so a second streaming
retry resumed from where it left off and completed cleanly. **Lesson
for future >40 GB pulls: assume one retry will be needed.**

### 4-condition sweep

| Condition | Prompts | Wall time | Throughput | Parser-valid |
|---|---:|---:|---:|---:|
| honest (50 personas × 30 pairs) | 1500 | ~44 min | 0.57/s | 1498/1500 |
| fake-good (50 personas × 30 pairs) | 1500 | ~42 min | 0.59/s | 1499/1500 |
| neutral bare | 30 | ~45 sec | — | 30/30 |
| neutral respondent | 30 | ~45 sec | — | 30/30 |

Output files (in `psychometrics/gfc_tirt/`):

```
qwen2.5-72b_gfc30_synthetic.json            # honest, 1500 rows
qwen2.5-72b_gfc30_synthetic-fakegood.json   # fake-good, 1500 rows
qwen2.5-72b_gfc30_neutral-bare.json         # 30 rows
qwen2.5-72b_gfc30_neutral-respondent.json   # 30 rows
```

The 2 dropped responses on each persona run were emitted as text the
parser couldn't map back to a 1–7 integer. Per-persona valid count
≥ 28/30 for all 50 personas — well within the threshold for fitting.

### Response style: bimodal-with-skips, effectively 5 categories

Qwen2.5-72B honest distribution (n=1498):

| Cat | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count | 433 | 4 | 20 | 484 | 325 | 3 | 229 |
| % | 28.9% | 0.3% | 1.3% | 32.3% | 21.7% | 0.2% | 15.3% |

Categories 2 and 6 are nearly absent (~0.5% combined), and cat 3 is
near-skipped (1.3%). Effectively a 5-category response model on
{1, 3, 4, 5, 7}. Compare:

| Cat | Qwen2.5-72B | Llama3.3-70B (week09) | Gemma3-27B (week08) |
|---:|---:|---:|---:|
| 1 | 28.9% | 15.3% | 12.6% |
| 2 | **0.3%** | **0.0%** | 4.3% |
| 3 | 1.3% | 7.2% | 5.9% |
| 4 | 32.3% | 14.7% | 9.0% |
| 5 | 21.7% | **41.6%** | 4.5% |
| 6 | **0.2%** | 5.4% | **49.9%** |
| 7 | 15.3% | 15.8% | 13.8% |

Three different "shapes" across the three pooled non-Haiku models.
Llama70B and Gemma27B each skip one category; Qwen72B skips two-and-a-half.
This was the heterogeneity that *should* have broken pooling, given the
week07 result. It didn't (see §3).

## 2. Single-model TIRT fit (joint honest + fake-good, N=98)

Driver: `psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R`
(now lists Qwen2.5-72B in `MODELS`). Stan model: `tirt_okada_indep.stan`.
4 chains × 1000 iter (333 warmup). Convergence: 0 Rhat>1.05 of 731.

| Model | Cond | A | C | E | N | O | mean \|r\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.854 | 0.378 | 0.853 | 0.847 | 0.620 | 0.710 |
| Llama3.3-70B | honest | 0.809 | 0.335 | 0.867 | 0.493 | 0.784 | 0.658 |
| **Qwen2.5-72B** | **honest** | **0.018** | **0.530** | **0.853** | **0.137** | **0.863** | **0.480** |
| Qwen2.5-72B | fakegood | -0.116 | 0.499 | 0.838 | 0.194 | 0.894 | 0.508 |
| Gemma3-27B | honest | 0.045 | -0.006 | 0.817 | 0.266 | 0.479 | 0.323 |
| Phi4-mini | honest | 0.372 | 0.002 | 0.492 | 0.078 | 0.340 | 0.257 |
| Gemma3-4B | honest | -0.056 | -0.154 | 0.520 | 0.041 | 0.210 | 0.196 |
| Qwen2.5-3B | honest | 0.318 | -0.197 | 0.016 | -0.104 | -0.339 | 0.195 |
| Llama3.2-3B | honest | -0.335 | 0.317 | 0.060 | -0.007 | -0.022 | 0.148 |

Qwen2.5-72B sits in a notably different position from Llama3.3-70B at
the same scale. **E (0.85), C (0.53), and O (0.86) recover at or above
the Okada r ≥ 0.50 band**, but **A (0.02) and N (0.14) collapse to
≈ 0** — neither correlated with ground truth nor sign-flipped. This is
not the "{A, C} reflection" pattern seen on Haiku, Phi4, and Qwen2.5-3B
(which moves both A and C to the wrong sign). Qwen72B's A and N
dimensions just aren't being recovered at all.

C is interestingly *higher* on Qwen72B (0.53) than on Haiku (0.38) or
Llama70B (0.34) — the largest C among the four pooled models.

### Fake-good is identical to honest

Mean EV is 3.84 (honest) vs 3.83 (fake-good) — a numerical zero shift.
Per-trait recovery under fake-good is also nearly identical to honest
(±0.07 across all five traits). For comparison, Haiku and Llama70B both
show shifted EV under fake-good and per-trait recovery changes of ≥ 0.1
on at least two traits.

This puts Qwen2.5-72B in the same "doesn't fake-good" camp as
Llama3.2-3B from week07. Llama3.2-3B was hypothesized to be too small
to follow the meta-instruction; Qwen2.5-72B at 72B parameters
clearly isn't a capability ceiling. It's a different failure mode —
the model holds its persona-derived response stable even under explicit
fake-good instruction.

## 3. Four-model pooled fit (Haiku + Gemma3-27B + Llama3.3-70B + Qwen2.5-72B)

Driver: `psychometrics/gfc_tirt/fit_tirt_pooled_haiku_gemma27b_llama70b_qwen72b.R`
(new, mirrors the 3-model driver). Same Stan model.
4 chains × 1500 iter (500 warmup). N=408 rows
(50 honest + 50 fake-good + 1 bare + 1 respondent per model, with 49
honest + 49 fake-good for Qwen due to the 2 dropped parses).
Convergence: **0 Rhat>1.05, 0 n_eff<100** of 2271 params.

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.856 | 0.320 | 0.836 | 0.740 | 0.457 | 0.642 | -0.07 |
| Haiku 4.5 | fakegood | 0.851 | 0.397 | 0.806 | 0.617 | 0.529 | 0.640 | -0.06 |
| Llama3.3-70B | honest | 0.803 | 0.333 | 0.848 | 0.424 | 0.785 | 0.638 | -0.02 |
| Llama3.3-70B | fakegood | 0.817 | 0.298 | 0.887 | 0.655 | 0.827 | 0.697 | -0.01 |
| Gemma3-27B | honest | 0.423 | 0.186 | 0.770 | 0.138 | 0.544 | 0.412 | +0.09 |
| Gemma3-27B | fakegood | 0.398 | -0.257 | 0.590 | 0.189 | 0.650 | 0.417 | +0.03 |
| **Qwen2.5-72B** | **honest** | **0.424** | **0.559** | **0.878** | **0.330** | **0.865** | **0.611** | **+0.13** |
| **Qwen2.5-72B** | **fakegood** | **0.291** | **0.535** | **0.878** | **0.523** | **0.880** | **0.621** | **+0.11** |

Picture under pooling:

* **Qwen2.5-72B**: the headline change. A jumps 0.018 → 0.424 (single
  → pool, honest) — the largest single anchor-effect we've seen in the
  lineup. N also recovers from 0.137 → 0.330. C, E, O are essentially
  unchanged (already strong). Fake-good gains a comparable jump on
  N (0.194 → 0.523).
* **Haiku 4.5**: -0.07 honest, -0.06 fake-good. Adding Qwen as a fourth
  anchor cost Haiku slightly more than the 3-model pool (-0.05 in
  week09), which is consistent with the cost-sharing pattern: each
  added anchor reduces Haiku's recovery a small amount as Stan
  accommodates the new κ regime.
* **Llama3.3-70B**: -0.02 honest, -0.01 fake-good. Robust to the Qwen
  addition.
* **Gemma3-27B**: +0.09 honest. The anchor effect on Gemma's A persists
  through to the 4-model pool (now A=0.42 vs 0.05 single-model), and
  fake-good O even gets a bonus (0.65 vs 0.50).

The story is: **two pooled models win, two lose, none catastrophically.**
Qwen and Gemma both gain from being adjacent to keying-aligned anchors
(Haiku, Llama70B). Haiku and Llama70B pay a small cost to bring the
weaker models along.

### Why didn't heterogeneity break this pool?

The week07 5-model pool failed because Phi4-mini never used cat 5 and
Gemma3-4B never used cat 3 — categories *in the middle of the
endorsement range*, where most informative threshold movement happens.
Shared κ couldn't simultaneously be "high enough for Phi4 to skip cat 5"
and "low enough for Llama-style models to use cat 5".

Qwen2.5-72B skips cats 2 and 6 (the *adjacent-to-extreme* categories,
not middle ones). Stan can place κ thresholds that bracket cat 2
tightly between κ_1 and κ_3 — Qwen's likelihood penalizes that bracket
since it's never used, but the threshold values that satisfy the
smooth-7 models also satisfy a 5-category subset model, because TIRT
parameterizes underlying *utility* and lets the data reveal which
category each utility maps to. The cost is paid in `a_pos` (slightly
inflated to push Qwen utilities into the 1/4/5/7 buckets) but doesn't
break the cross-model pooling structure.

This is a useful finding for designing future pools: **adjacent-to-extreme
category skipping is tolerable for TIRT pooling; middle-category skipping
isn't.**

## 4. Inter-trait correlation: scored vs true

Same analysis as week09 §4, now with all four pooled models.

### True correlation among the 50 target personas

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | 0.36 | 0.36 | -0.42 | 0.31 |
| C | 0.36 | 1.00 | 0.17 | -0.36 | 0.25 |
| E | 0.36 | 0.17 | 1.00 | -0.39 | 0.40 |
| N | -0.42 | -0.36 | -0.39 | 1.00 | -0.19 |
| O | 0.31 | 0.25 | 0.40 | -0.19 | 1.00 |

### Recovered correlations from the 4-model pooled fit (honest condition)

**Haiku 4.5** (N=50; nearly identical to week09 — pool change had small
effect on Haiku's joint structure):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.07 | 0.15 | -0.05 | -0.09 |
| C | -0.07 | 1.00 | -0.51 | 0.11 | -0.31 |
| E | 0.15 | -0.51 | 1.00 | -0.21 | 0.28 |
| N | -0.05 | 0.11 | -0.21 | 1.00 | -0.01 |
| O | -0.09 | -0.31 | 0.28 | -0.01 | 1.00 |

**Llama3.3-70B** (N=50; also nearly identical to week09):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | 0.03 | 0.00 | 0.00 | 0.08 |
| C | 0.03 | 1.00 | -0.51 | 0.46 | -0.50 |
| E | 0.00 | -0.51 | 1.00 | -0.39 | 0.52 |
| N | 0.00 | 0.46 | -0.39 | 1.00 | -0.06 |
| O | 0.08 | -0.50 | 0.52 | -0.06 | 1.00 |

**Gemma3-27B** (N=50; ditto):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.39 | 0.32 | 0.21 | 0.17 |
| C | -0.39 | 1.00 | -0.26 | -0.05 | -0.14 |
| E | 0.32 | -0.26 | 1.00 | 0.00 | 0.22 |
| N | 0.21 | -0.05 | 0.00 | 1.00 | 0.18 |
| O | 0.17 | -0.14 | 0.22 | 0.18 | 1.00 |

**Qwen2.5-72B** (N=49):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.15 | -0.15 | 0.13 | -0.28 |
| C | -0.15 | 1.00 | -0.09 | 0.17 | -0.30 |
| E | -0.15 | -0.09 | 1.00 | -0.10 | **0.60** |
| N | 0.13 | 0.17 | -0.10 | 1.00 | 0.16 |
| O | -0.28 | -0.30 | 0.60 | 0.16 | 1.00 |

Qwen's joint geometry has the same flat-A row as Llama70B's (per-trait
A is poor or middling, so the row is uninformative). The strongest
cell is E–O = +0.60, which is also the strongest cell on Llama70B
(+0.52) and is the only off-diagonal where the pooled fit consistently
*exceeds* the true value (+0.40). Reading: **GFC items load E and O
together more strongly than the personas were generated with.** Could
be a genuine response pattern of how 70B-class models conceptualize
extraversion + openness, or could be a feature of the GFC pair
construction (the 30 pairs over-represent E/O contrasts in cross-trait
blocks). Worth a follow-up.

A–O = -0.28 is unusual — true is +0.31 (positively signed), and Qwen
flips it. C–O = -0.30 is similar (true +0.25). Qwen's C and O
dimensions are anti-correlated with A in θ̂-space, where they should
all be modestly positively related as part of the alpha factor.

### Summary distance from truth

| Model | RMSD off-diag | max \|deviation\| | mean off-diag (recovered) | mean off-diag (true) |
|---|---:|---:|---:|---:|
| Haiku 4.5 | 0.398 | 0.677 | -0.070 | +0.049 |
| Gemma3-27B | 0.415 | 0.746 | +0.025 | +0.049 |
| Llama3.3-70B | 0.466 | 0.816 | -0.036 | +0.049 |
| Qwen2.5-72B | 0.454 | 0.593 | -0.000 | +0.049 |

Qwen has the *smallest* max deviation (0.59 vs 0.68–0.82 for the other
three) but middle-of-pack RMSD. Mean off-diagonal is essentially zero
for Qwen — recovered structure has no overall positive correlation
across traits, where truth has a small +0.05 pull. None of the four
models reproduces the alpha-cluster (A, C, low N) or beta-cluster (E, O)
structure. **Per-trait recovery still does not imply joint structure
recovery, even at 70B scale.**

## 5. Neutral placement on the latent scale

Pooled-fit θ̂ for the no-persona conditions, in the 4-model pooled
latent space:

| Model | Condition | A | C | E | N | O | max \|θ̂\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | bare | -0.44 | 0.06 | 0.37 | 0.24 | -0.24 | 0.44 |
| Haiku 4.5 | respondent | -0.17 | 0.15 | 0.15 | 0.19 | -0.05 | 0.19 |
| Gemma3-27B | bare | -0.46 | -0.10 | 0.13 | 0.43 | 0.20 | 0.46 |
| Gemma3-27B | respondent | **0.90** | 0.40 | -0.20 | **0.91** | -0.26 | **0.91** |
| Llama3.3-70B | bare | -0.30 | 0.46 | 0.09 | -0.13 | 0.14 | 0.46 |
| Llama3.3-70B | respondent | -0.58 | -0.23 | 0.10 | 0.06 | -0.21 | 0.58 |
| **Qwen2.5-72B** | **bare** | 0.09 | 0.22 | 0.14 | 0.00 | 0.09 | **0.22** |
| **Qwen2.5-72B** | **respondent** | 0.12 | 0.24 | -0.01 | -0.20 | 0.28 | **0.28** |

**Qwen2.5-72B is the most centered model in the lineup**, by a
meaningful margin: max |θ̂| ≤ 0.28 across both neutral conditions, vs
0.46–0.58 for the other three pooled models (and Gemma's 0.91 outlier
on `respondent`). All five trait shifts fall in [-0.20, +0.28] —
notably tighter than Llama3.3-70B (which week09 already called out as
"well-centered" with max 0.46).

Two ways to read this. Generous: Qwen is genuinely the most calibrated
default subject in the lineup. Conservative: Qwen's collapsed A and N
dimensions in the single-model fit (§2) are still being expressed in
neutral mode as small near-zero values, and the apparent "good
centering" is partly an artifact of weak per-trait recovery flattening
the latent footprint. Distinguishing these requires looking at neutral
placement *relative to* per-trait recovery quality — Qwen's bare-A
shift is +0.09 on a dimension where its single-model recovery was
~0, vs Haiku's bare-A shift is -0.44 on a dimension where Haiku's
recovery was 0.85. Larger absolute shift with stronger underlying
recovery is genuine signal; smaller shift with no recovery is noise
hiding as calibration. On this lens, Qwen's tight neutral footprint
is partly real (E, C, O are well-recovered and stay near 0) and partly
artifactual (A, N are weakly recovered and round toward 0).

The fact that Qwen does *not* show Gemma's +A / +N "compliant
respondent" voicing under the `respondent` framing is a positive sign
regardless — even if the small shift is partly artifactual on A/N, it
isn't being pulled to extreme values by the framing.

## 6. Open hypotheses / follow-ups

1. **Why does Qwen2.5-72B fail on A and N specifically?** Per-trait
   collapse on these two dimensions while E/C/O recover is unusual —
   most weak models flip C, not A. Inspect Qwen's responses on A-loaded
   GFC blocks (those with A+ markers in either the L or R slot) to see
   whether the failure mode is consistent endorsement of one side, or
   genuinely random response. If it's consistent endorsement, then
   Qwen has a "default agreeableness" floor that doesn't move with
   personas, similar to the way some models had ceiling effects on
   Likert.
2. **Why doesn't Qwen2.5-72B fake-good?** Llama3.2-3B's hypothesis was
   capacity-related; Qwen2.5-72B at 72B parameters is clearly not
   capacity-limited. Possibilities: (a) RLHF-trained refusal of
   strategic self-presentation; (b) the persona instruction is taking
   precedence over the meta-instruction; (c) Qwen's GFC responses are
   already at a kind of fake-good ceiling under the persona alone, so
   the fake-good prompt has no headroom. Worth checking by comparing
   Qwen's bare `θ̂` (which we have) against its honest persona θ̂
   distribution.
3. **Block-level inspection of E–O over-correlation.** Both 70B-class
   models (Llama3.3, Qwen2.5) have E–O recovered correlation ≥ +0.50
   versus true +0.40. This is on the right side of truth, but
   suspiciously high for two models with otherwise different recovery
   patterns. Could be a structural feature of the Okada GFC pair set:
   how many of the 30 pairs have E and O on opposite sides? If E vs O
   choices dominate, then E and O become co-determined in TIRT.
4. **Run on Sonnet 4.6 / Opus 4.7 to fill out the Anthropic frontier.**
   Haiku 4.5 is in the lineup; the larger Anthropic models are the
   obvious next test of whether the keying-aligned basin extends with
   scale. Cost is ~$5–10 per model for the full 4-condition sweep.
5. **Consider adding LKJ θ prior to recover joint structure.** §4
   continues to find poor inter-trait recovery. The Stan
   `tirt_okada_indep.stan` uses N(0, I_5) on θ (Okada Appendix D
   exact); switching to LKJ would let the model recover correlated θ.
   Earlier per-trait LKJ-vs-N(0,I) comparisons found similar mean |r|
   on the diagonal, but didn't compare off-diagonal recovery — that's
   now the relevant metric.

## 7. Files added / modified

```
psychometrics/gfc_tirt/
  qwen2.5-72b_gfc30_synthetic.json                          # NEW (1500 rows, 1498 valid)
  qwen2.5-72b_gfc30_synthetic-fakegood.json                 # NEW (1500 rows, 1499 valid)
  qwen2.5-72b_gfc30_neutral-bare.json                       # NEW (30 rows)
  qwen2.5-72b_gfc30_neutral-respondent.json                 # NEW (30 rows)
  fit_tirt_per_model_pooled_conditions.R                    # +Qwen2.5-72B in MODELS
  fit_tirt_pooled_haiku_gemma27b_llama70b_qwen72b.R         # NEW (4-model driver)
  analyze_pooled_haiku_gemma27b_llama70b_qwen72b.R          # NEW (post-hoc cor + neutrals)
  per_model_pooled/qwen2.5-72b_pooled_conditions_fit.rds    # NEW (archived)
  pooled_haiku_gemma27b_llama70b_qwen72b_fit.rds            # NEW (archived, 117 MB)
  pooled_haiku_gemma27b_llama70b_qwen72b_summary.rds        # NEW (small, committable)
```

Inference command (for the record):

```bash
# Modelfile variant with reduced context (one-time setup)
curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:72b-ctx2k","from":"qwen2.5:72b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create

# 4-condition sweep
export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\"")
for cond in bare respondent; do
  python3 scripts/run_gfc_ollama.py --remote --model qwen2.5:72b-ctx2k \
    --neutral $cond --output psychometrics/gfc_tirt/qwen2.5-72b_gfc30_neutral-${cond}.json
done
python3 scripts/run_gfc_ollama.py --remote --model qwen2.5:72b-ctx2k \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 \
  --output psychometrics/gfc_tirt/qwen2.5-72b_gfc30_synthetic.json --checkpoint-every 100
python3 scripts/run_gfc_ollama.py --remote --model qwen2.5:72b-ctx2k \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 --fake-good \
  --output psychometrics/gfc_tirt/qwen2.5-72b_gfc30_synthetic-fakegood.json --checkpoint-every 100
```
