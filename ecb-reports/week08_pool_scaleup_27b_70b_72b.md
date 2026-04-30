# Scaling the Okada GFC + TIRT lineup: Gemma3-27B, Llama3.3-70B, Qwen2.5-72B; 2-, 3-, and 4-model pools

**Dates:** 2026-04-28 through 2026-04-30
**Author:** ECB
**Companion to:** [`week07_okada_pooled_replication.md`](week07_okada_pooled_replication.md), [`week07_okada_swap_bug.md`](week07_okada_swap_bug.md)

## TL;DR

This week extended the Okada GFC + TIRT replication along the *scale*
axis on the Orin server: 27B (Gemma3), 70B (Llama3.3), and 72B (Qwen2.5)
all collected, single-model fit, and progressively pooled. Five results
worth foregrounding:

1. **Llama3.3-70B is the first open-weight model to crack mean |r| > 0.5.**
   Single-model honest recovery: A=0.81, C=0.34, E=0.87, N=0.49, O=0.78
   (mean |r| = 0.658). All five traits positive — no {A, C} sign flip.
   Holds up under pooling.
2. **Pooling is mostly a transfer of identification, not raw signal.**
   Adding Gemma3-27B to a Haiku-only pool lifted Gemma's A from 0.05 →
   0.43 (≈10× anchor effect) at -0.07 cost to Haiku. Adding Qwen2.5-72B
   to the 3-model pool lifted Qwen's collapsed A and N (0.02 → 0.42,
   0.14 → 0.33) at small cost to Haiku and Llama70B. This is mostly
   re-anchoring the latent direction — the signal was already in the
   weak model's data, just unidentifiable single-model.
3. **Heterogeneous category usage matters less than category-skip
   *position*.** The week07 5-model pool failed because Phi4-mini and
   Gemma3-4B skipped *middle-of-scale* categories (cat 5 / cat 3
   respectively). Qwen2.5-72B's adjacent-to-extreme skipping (cats 2
   and 6, with a 1/3/4/5/7 effective subset) did *not* fight shared κ
   thresholds in the 4-model pool. Lesson: TIRT pooling tolerates
   category subsets, but not skipped middle categories.
4. **Per-trait recovery does not imply joint structure recovery.** The
   recovered inter-trait correlation matrix is far from the true one
   for all four pooled models (RMSD on off-diagonals ≈ 0.40–0.47).
   Multiple sign disagreements per cell. Llama70B's C dimension
   appears rotated (C–E=-0.51, C–N=+0.46, C–O=-0.50 vs true +0.17,
   -0.36, +0.25). Both 70B-class models over-correlate E and O
   (+0.52 / +0.60 vs true +0.40), suggesting a possible structural
   feature of the Okada GFC pair set.
5. **Two models in the lineup do not fake-good at all.** Llama3.2-3B
   (week07) and Qwen2.5-72B both show ≈0 EV shift between honest and
   fake-good (3.84 vs 3.83 on Qwen). 72B parameters refutes the
   "too small to follow meta-instruction" explanation for Llama3.2-3B.

| Result | Outcome |
|---|---|
| 27B / 70B / 72B inference on the 64 GB Orin | ✅ — 70B/72B require `num_ctx=2048` Modelfile variant; default 128K context blows memory budget |
| Gemma3-4B → 27B size effect on single-model fit | ✅ mean \|r\| 0.20 → 0.32 honest; E/N/O each gain +0.20–0.30; A and C stay ≈0 |
| Llama3.3-70B clears mean \|r\| > 0.5 single-model | ✅ all five traits positive on honest |
| Qwen2.5-72B partial single-model recovery | ✅ E/C/O at 0.5–0.9; A/N collapse to ≈0 (different failure mode from {A,C} flip) |
| Pool convergence (2-, 3-, 4-model) | ✅ all clean (0 Rhat>1.05) |
| Anchor effect on weak models | ✅ Gemma3-27B A: 0.05 → 0.43 in 2-model pool; Qwen72B A: 0.02 → 0.42 in 4-model pool |
| Cost to anchor models from pooling | Modest: Haiku -0.05 / -0.07 mean \|r\|; Llama70B -0.02 |
| True inter-trait correlation recovered | ❌ off-diagonals poorly recovered for all 4 (RMSD 0.40–0.47) |

## 1. Inference setup

### Pulls and memory

| Model | Disk size (Q4_K_M) | num_ctx workaround? | Notes |
|---|---:|---|---|
| `gemma3:27b` | 17 GB | No | Standard pull; runs at full default context |
| `llama3.3:70b` | 43 GB | **Yes** (`-ctx2k`) | Default 128K context blew 62.8 GiB allocation |
| `qwen2.5:72b` | 47 GB | **Yes** (`-ctx2k`) | 47 GB pull dropped through nginx; 2 retries needed |

For the two 70B-class models, the default 128K KV cache pushes peak
allocation past the 64 GB Orin budget:

```
{"error":{"message":"model requires more system memory (62.8 GiB) than is available (43.5 GiB)"}}
```

Fix: bake a reduced context into a Modelfile-derived variant via
`/api/create`:

```bash
curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:72b-ctx2k","from":"qwen2.5:72b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create
```

This is needed because the OpenAI-compatible `/v1/chat/completions`
endpoint (which `run_gfc_ollama.py` uses) does not honor `num_ctx` in
the `options` field — only the native `/api/chat` does. `num_ctx=2048`
is plenty for our prompts (~150–300 tokens including persona + GFC
pair).

The 47 GB Qwen pull was fragile through the nginx proxy. The
non-streaming `/api/pull` request hit a 504 Gateway Timeout after
~10 minutes (proxy gives up before pull finishes). Switching to
`stream:true` and sending `-N` (no buffering) on curl let the proxy
see periodic progress bytes, but the first stream still dropped at
~28 / 47 GB. Ollama caches partial blob downloads server-side, so a
second streaming retry resumed and completed cleanly. **Lesson for
future >40 GB pulls: assume one retry will be needed.**

### 4-condition sweeps

All three models ran the canonical sweep: 50 honest personas + 50
fake-good personas + 1 bare neutral + 1 respondent neutral.

| Model | Honest | Fake-good | Bare | Resp. | Total wall | Throughput | Parser-valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma3-27B | 1500 | 1500 | 30 | 30 | ~65 min | 0.78/s | 3060/3060 |
| Llama3.3-70B | 1500 | 1500 | 30 | 30 | ~80 min | 0.62/s | 3060/3060 |
| Qwen2.5-72B | 1500 | 1500 | 30 | 30 | ~85 min | 0.58/s | 3058/3060 |

All output files in `psychometrics/gfc_tirt/`, named
`{slug}_gfc30_synthetic{,-fakegood}.json` and
`{slug}_gfc30_neutral-{bare,respondent}.json`.

### Response styles vary substantially

Honest-condition response distribution by model:

| Cat | Gemma3-4B | Gemma3-27B | Llama3.3-70B | Qwen2.5-72B | Haiku 4.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.5% | 12.6% | 15.3% | **28.9%** | (smooth) |
| 2 | 2.7% | 4.3% | **0.0%** | **0.3%** | (smooth) |
| 3 | **45.3%** | 5.9% | 7.2% | 1.3% | (smooth) |
| 4 | 1.2% | 9.0% | 14.7% | 32.3% | (smooth) |
| 5 | **0.0%** | 4.5% | **41.6%** | 21.7% | (smooth) |
| 6 | 4.5% | **49.9%** | 5.4% | **0.2%** | (smooth) |
| 7 | 32.9% | 13.8% | 15.8% | 15.3% | (smooth) |

Each non-Haiku model has its own "shape":
- **Gemma3-4B** skips cat 5 entirely, modes on cat 3 (compressed-Likert)
- **Gemma3-27B** uses all 7, modes on cat 6
- **Llama3.3-70B** skips cat 2, modes on cat 5
- **Qwen2.5-72B** is bimodal-with-skips: cats 1/4/5 carry ~83%, cats
  2/3/6 nearly absent (<2% combined). Effectively a 5-cat subset on
  {1, 3, 4, 5, 7}

This was the heterogeneity that *should* have broken pooling, given
the week07 5-model failure. It mostly didn't (see §3).

## 2. Single-model TIRT fits (joint honest + fake-good, N=98–100)

Driver: `psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R`
(adds Gemma3-27B, Llama3.3-70B, Qwen2.5-72B to `MODELS`; cache check so
existing models aren't refit). Stan model: `tirt_okada_indep.stan`
(Okada Appendix D priors: a_j+ ~ HN(0, 0.5), κ ~ N(0, 1.5),
θ ~ N(0, I_5)). 4 chains × 1000 iter (333 warmup). All fits converged
cleanly (0 Rhat > 1.05).

Per-trait recovery vs synthetic ground truth (Pearson r):

| Model | Cond | A | C | E | N | O | mean \|r\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | **0.854** | 0.378 | 0.853 | 0.847 | 0.620 | **0.710** |
| Haiku 4.5 | fakegood | 0.842 | 0.440 | 0.833 | 0.741 | 0.643 | 0.700 |
| **Llama3.3-70B** | **honest** | **0.809** | 0.335 | 0.867 | 0.493 | 0.784 | **0.658** |
| Llama3.3-70B | fakegood | 0.807 | 0.309 | 0.892 | 0.706 | 0.826 | 0.708 |
| **Qwen2.5-72B** | **honest** | **0.018** | **0.530** | **0.853** | **0.137** | **0.863** | **0.480** |
| Qwen2.5-72B | fakegood | -0.116 | 0.499 | 0.838 | 0.194 | 0.894 | 0.508 |
| Gemma3-27B | honest | 0.045 | -0.006 | 0.817 | 0.266 | 0.479 | 0.323 |
| Gemma3-27B | fakegood | 0.040 | -0.279 | 0.701 | 0.273 | 0.636 | 0.386 |
| Phi4-mini | honest | 0.372 | 0.002 | 0.492 | 0.078 | 0.340 | 0.257 |
| Gemma3-4B | honest | -0.056 | -0.154 | 0.520 | 0.041 | 0.210 | 0.196 |
| Qwen2.5-3B | honest | 0.318 | -0.197 | 0.016 | -0.104 | -0.339 | 0.195 |
| Llama3.2-3B | honest | -0.335 | 0.317 | 0.060 | -0.007 | -0.022 | 0.148 |

### 2a. Gemma3-4B → 27B is a clean size-effect read

Δ(27B − 4B) by trait, honest:

| | A | C | E | N | O | mean \|r\| |
|---|---:|---:|---:|---:|---:|---:|
| Δ | +0.10 | +0.15 | **+0.30** | **+0.22** | **+0.27** | +0.13 |

E nearly doubles (0.52 → 0.82, approaching Haiku's 0.85). O more than
doubles (0.21 → 0.48). N goes from indistinguishable from zero to a
modest 0.27. **A and C do not benefit** — both stay essentially zero.
This is not the {A, C} sign flip we saw on Haiku before the swap fix
(commit `a007852`); it's a clean failure to identify the latent
direction at all.

Pre-fit audit ruled out measurement bugs (1500/1500 valid, balanced
swap, all blocks covered, swap-fix applied, cat-5 absence on 4B is
genuine response style). The 4B → 27B comparison is a clean read.

### 2b. Llama3.3-70B is the first open model in the keying-aligned basin

All five traits positive on honest; four of five clear Okada's r ≥
0.50 band (only C at 0.34). Compare to every smaller open model in
the lineup, where C and/or A flip to negative. Fake-good condition
recovers slightly *better* than honest (0.708 vs 0.658), driven by
stronger N and O — same pattern as Haiku in the per-model joint fit
(larger spread under fake-good gives the latent direction a better
foothold).

### 2c. Qwen2.5-72B has a different failure mode

E (0.85), C (0.53), O (0.86) all recover at/above Okada's band. **A
(0.02) and N (0.14) collapse to ≈0** — neither correlated with truth
nor sign-flipped. This is *not* the {A, C} reflection seen on
Haiku/Phi4/Qwen2.5-3B; Qwen72B's A and N just aren't being recovered.

C is interestingly *higher* on Qwen72B (0.53) than on Haiku (0.38) or
Llama70B (0.34) — the largest C among the strong models.

### 2d. Qwen2.5-72B does not fake-good

Mean EV is 3.84 (honest) vs 3.83 (fake-good) — a numerical zero shift.
Per-trait recovery under fake-good is also nearly identical to honest
(±0.07 across all five traits). This puts Qwen2.5-72B in the same
camp as Llama3.2-3B from week07 — neither shifts in the desirability
direction under explicit instruction. The capacity-ceiling explanation
for Llama3.2-3B's no-fake-good is clearly wrong here at 72B parameters.
Possibilities: RLHF refusal of strategic self-presentation; persona
instruction taking precedence over meta-instruction; or persona
responses already at a kind of fake-good ceiling with no headroom.

## 3. Pooled fits (2-model → 3-model → 4-model)

Driver pattern (one R script per pool):

```
fit_tirt_pooled_haiku_gemma27b.R                          # 2-model
fit_tirt_pooled_haiku_gemma27b_llama70b.R                 # 3-model
fit_tirt_pooled_haiku_gemma27b_llama70b_qwen72b.R         # 4-model
```

All use the same `tirt_okada_indep.stan`, 4 chains × 1500–3000 iter.
All converged cleanly (0 Rhat > 1.05). Honest-condition per-trait
recovery is the headline, with deltas vs single-model in the rightmost
column.

### 3a. 2-model pool (Haiku + Gemma3-27B), N=204

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.810 | 0.210 | 0.860 | 0.808 | 0.507 | 0.639 | -0.07 |
| Haiku 4.5 | fakegood | 0.811 | 0.275 | 0.786 | 0.718 | 0.459 | 0.610 | -0.09 |
| **Gemma3-27B** | **honest** | **0.434** | 0.090 | 0.802 | 0.245 | 0.497 | **0.414** | **+0.09** |
| Gemma3-27B | fakegood | 0.340 | -0.289 | 0.653 | 0.222 | 0.651 | 0.431 | +0.05 |

**Gemma's A jumps from 0.045 → 0.434** — a roughly 10× lift. The signal
was always in Gemma's data; what was missing was a way to anchor the
latent direction. Haiku's strong A signal (r=0.85 on its own) provides
that anchor through the shared a_j+ discriminations and shared κ
thresholds. C also recovers a small positive sign (-0.006 → 0.090) but
remains weak. Pooling cost on Haiku is modest: A holds, E holds, N
holds; C drops from 0.378 to 0.210 and O drops from 0.620 to 0.507.
No sign flips.

### 3b. 3-model pool (+ Llama3.3-70B), N=306

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.860 | 0.351 | 0.837 | 0.760 | 0.467 | 0.655 | -0.05 |
| Haiku 4.5 | fakegood | 0.850 | 0.405 | 0.793 | 0.648 | 0.570 | 0.653 | -0.05 |
| **Llama3.3-70B** | **honest** | **0.803** | 0.307 | 0.848 | 0.382 | 0.787 | **0.626** | **-0.03** |
| Llama3.3-70B | fakegood | 0.819 | 0.271 | 0.881 | 0.668 | 0.836 | 0.695 | -0.01 |
| Gemma3-27B | honest | 0.427 | 0.199 | 0.777 | 0.135 | 0.502 | 0.408 | +0.08 |
| Gemma3-27B | fakegood | 0.401 | -0.276 | 0.611 | 0.207 | 0.662 | 0.431 | +0.05 |

Llama70B is robust to shared κ + a_j+ across three heterogeneous
response styles. Adding Llama as a third strong-signal anchor does not
further hurt Haiku (-0.05 here vs -0.07 in the 2-model pool — actually
slightly *better*). Gemma's anchor effect on A persists, confirming
the week08 finding is robust to a different anchor combination. C
goes from -0.006 to +0.199 — small, but positively signed for the
first time on Gemma.

### 3c. 4-model pool (+ Qwen2.5-72B), N=408

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.856 | 0.320 | 0.836 | 0.740 | 0.457 | 0.642 | -0.07 |
| Haiku 4.5 | fakegood | 0.851 | 0.397 | 0.806 | 0.617 | 0.529 | 0.640 | -0.06 |
| Llama3.3-70B | honest | 0.803 | 0.333 | 0.848 | 0.424 | 0.785 | 0.638 | -0.02 |
| Llama3.3-70B | fakegood | 0.817 | 0.298 | 0.887 | 0.655 | 0.827 | 0.697 | -0.01 |
| Gemma3-27B | honest | 0.423 | 0.186 | 0.770 | 0.138 | 0.544 | 0.412 | +0.09 |
| Gemma3-27B | fakegood | 0.398 | -0.257 | 0.590 | 0.189 | 0.650 | 0.417 | +0.03 |
| **Qwen2.5-72B** | **honest** | **0.424** | 0.559 | 0.878 | **0.330** | 0.865 | **0.611** | **+0.13** |
| **Qwen2.5-72B** | **fakegood** | 0.291 | 0.535 | 0.878 | 0.523 | 0.880 | 0.621 | +0.11 |

**Qwen2.5-72B's collapsed dimensions get rescued.** A: 0.018 → 0.424
(largest single anchor effect we've seen). N: 0.137 → 0.330. Mean |r|
0.480 → 0.611. C, E, O are essentially unchanged (already strong).
Fake-good gains a comparable jump on N (0.194 → 0.523). Two pooled
models win, two lose, none catastrophically: Qwen and Gemma both gain
from being adjacent to keying-aligned anchors; Haiku and Llama70B pay
a small cost to bring the weaker models along.

### 3d. Why didn't the 4-model heterogeneity break this pool?

The week07 5-model pool failed because Phi4-mini never used cat 5 and
Gemma3-4B never used cat 3 — categories *in the middle of the
endorsement range*, where most informative threshold movement happens.
Shared κ couldn't simultaneously be "high enough for Phi4 to skip
cat 5" and "low enough for Llama-style models to use cat 5".

Qwen2.5-72B skips cats 2 and 6 (the *adjacent-to-extreme* categories,
not middle ones). Stan can place κ thresholds that bracket cat 2
tightly between κ_1 and κ_3 — Qwen's likelihood penalizes that bracket
since it's never used, but the threshold values that satisfy the
smooth-7 models also satisfy a 5-category subset model, because TIRT
parameterizes underlying *utility* and lets the data reveal which
category each utility maps to. The cost is paid in `a_pos` (slightly
inflated to push Qwen utilities into the 1/4/5/7 buckets) but doesn't
break the cross-model pooling structure.

This is a useful design lesson for future pools: **adjacent-to-extreme
category skipping is tolerable for TIRT pooling; middle-category
skipping isn't.**

### 3e. Why did pooling rescue A but not C on Gemma?

**A** improves dramatically (0.045 → 0.434) because Gemma3-27B carries
*some* A-related variance that, on its own, the model couldn't
disambiguate from rotational noise — the off-diagonal in the trait
latent space. With Haiku anchoring the A direction (loadings,
thresholds), Gemma's A projection finds a stable basis. The fact that
the lift is so large suggests the latent A signal in Gemma's responses
is genuinely there — just unidentified single-model.

**C** does not improve in any meaningful way (-0.006 → 0.090 in the
2-model pool; +0.186 in the 4-model). Two ways to read this: either
C-related GFC items elicit no between-persona variance from Gemma at
the 27B scale (a content-signal failure), or Gemma's C signal is
sufficiently mismatched in shape from Haiku's that the shared a_j+/κ
machinery cannot fit both at once. The fact that **C honest is fine
on Haiku alone** (single-model 0.38) but the pooled fit drops Haiku's
C to 0.21 favors the second reading: shared discriminations cannot
accommodate both.

A useful test would be to run the pooled fit with relaxed (per-model)
discriminations, keeping only thresholds shared. That would localize
whether the limitation is Gemma's content or the shared-a_j+
constraint.

## 4. Inter-trait correlation: scored vs true (4-model pool)

The Okada and earlier RGB reports focus on *per-trait* recovery (one
correlation per trait between θ̂ and ground-truth). But the original
Serapio-García et al. (2023) work also examined the **inter-trait
correlation matrix** of recovered scores: when the model places a
persona high on A, does it also place them high on C, low on N, etc.,
the way human respondents do? This is a separate test from per-trait
recovery — a model could pass the per-trait test (each θ̂ correlates
with the right ground truth) while completely scrambling the joint
geometry.

### True correlation among the 50 target personas

Personas are drawn from a multivariate normal with the van der Linden
(2010) "Big Five intercorrelation" Σ. The realized 50-persona
correlation matrix:

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | 0.36 | 0.36 | -0.42 | 0.31 |
| C | 0.36 | 1.00 | 0.17 | -0.36 | 0.25 |
| E | 0.36 | 0.17 | 1.00 | -0.39 | 0.40 |
| N | -0.42 | -0.36 | -0.39 | 1.00 | -0.19 |
| O | 0.31 | 0.25 | 0.40 | -0.19 | 1.00 |

The familiar pattern: a positive "alpha factor" cluster (A, C, low N),
a positive "beta factor" cluster (E, O), and N negatively related to
everything.

### Recovered correlations from the 4-model pooled fit (honest condition)

**Haiku 4.5** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.07 | 0.15 | -0.05 | -0.09 |
| C | -0.07 | 1.00 | -0.51 | 0.11 | -0.31 |
| E | 0.15 | -0.51 | 1.00 | -0.21 | 0.28 |
| N | -0.05 | 0.11 | -0.21 | 1.00 | -0.01 |
| O | -0.09 | -0.31 | 0.28 | -0.01 | 1.00 |

Most off-diagonals are near zero (Haiku's per-trait recovery is good,
so the rotational structure between traits is largely uninformative).
Notable: C–E=-0.51 (true +0.17) and C–O=-0.31 (true +0.25) — Haiku's
C dimension is anti-correlated with E and O where it should be
modestly positive.

**Gemma3-27B** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.39 | 0.32 | 0.21 | 0.17 |
| C | -0.39 | 1.00 | -0.26 | -0.05 | -0.14 |
| E | 0.32 | -0.26 | 1.00 | 0.00 | 0.22 |
| N | 0.21 | -0.05 | 0.00 | 1.00 | 0.18 |
| O | 0.17 | -0.14 | 0.22 | 0.18 | 1.00 |

A–C is *negative* (true +0.36). A–N is positive (true -0.42 — strong
sign disagreement). The whole alpha-cluster structure is broken.

**Llama3.3-70B** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | 0.03 | 0.00 | 0.00 | 0.08 |
| C | 0.03 | 1.00 | -0.51 | 0.46 | -0.50 |
| E | 0.00 | -0.51 | 1.00 | -0.39 | 0.52 |
| N | 0.00 | 0.46 | -0.39 | 1.00 | -0.06 |
| O | 0.08 | -0.50 | 0.52 | -0.06 | 1.00 |

A's row is essentially flat (Llama's A is good per-trait but
uncorrelated with anything else in θ̂-space). E–O=+0.52 is in the right
direction (true +0.40). E–N=-0.39 is also right (true -0.39, exact).
But C is inverted relative to the other traits: C–E=-0.51 (true
+0.17), C–N=+0.46 (true -0.36), C–O=-0.50 (true +0.25). It's as
though the C axis was flipped — except per-trait C-recovery is
positive (+0.31), so the issue isn't a simple sign convention.

**Qwen2.5-72B** (N=49):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.15 | -0.15 | 0.13 | -0.28 |
| C | -0.15 | 1.00 | -0.09 | 0.17 | -0.30 |
| E | -0.15 | -0.09 | 1.00 | -0.10 | **0.60** |
| N | 0.13 | 0.17 | -0.10 | 1.00 | 0.16 |
| O | -0.28 | -0.30 | 0.60 | 0.16 | 1.00 |

Qwen's joint geometry has the same flat-A row as Llama70B's. The
strongest cell is E–O=+0.60, which is also the strongest cell on
Llama70B (+0.52) and is the only off-diagonal where the pooled fit
consistently *exceeds* the true value (+0.40). Reading: **GFC items
load E and O together more strongly than the personas were generated
with.** Could be a genuine response pattern of how 70B-class models
conceptualize extraversion + openness, or could be a feature of the
GFC pair construction (the 30 pairs over-represent E/O contrasts in
cross-trait blocks). Worth a follow-up.

### Summary distance from truth

| Model | RMSD off-diag | max \|deviation\| | mean off-diag (recovered) | mean off-diag (true) |
|---|---:|---:|---:|---:|
| Haiku 4.5 | 0.398 | 0.677 | -0.070 | +0.049 |
| Gemma3-27B | 0.415 | 0.746 | +0.025 | +0.049 |
| Llama3.3-70B | 0.466 | 0.816 | -0.036 | +0.049 |
| Qwen2.5-72B | 0.454 | 0.593 | -0.000 | +0.049 |

All four models pull the typical Big Five intercorrelation toward zero
on average — the average off-diagonal is near zero rather than the
mildly positive +0.05 of truth. Per-cell deviations of 0.6–0.8 are
common. Qwen has the *smallest* max deviation but middle-of-pack
RMSD. None of the four models reproduces the alpha-cluster (A, C, low
N) or beta-cluster (E, O) structure.

**Per-trait recovery does not imply joint structure recovery.** Most
LLM-personality work — including ours up to this report — reports only
mean |r| or per-trait correlations. A model could "personalize" each
trait independently while losing the correlated structure that makes
Big Five a meaningful joint construct. On this evidence, none of the
four pooled models produces a plausible inter-trait covariance pattern,
even when per-trait recovery is high.

## 5. Neutral placement on the latent scale

Pooled-fit θ̂ for the no-persona conditions (4-model pool, sign
convention from Haiku's anchor):

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

Reading in 1-SD units (the θ scale is N(0, 1) by prior):

* **Haiku 4.5** is well-centered in both modes. Slight negative A
  baseline, slight positive E baseline. Behaves like a well-calibrated
  default subject.
* **Gemma3-27B** is centered in `bare`, but the `respondent`
  instruction pulls it to extreme +A (0.90) and +N (0.91 SD) with -O
  (-0.26). This is unusual — most models we've tested place neutrally
  regardless of framing. A and N moving together is suggestive of a
  "compliant + anxious survey participant" voicing rather than the
  model's own default.
* **Llama3.3-70B** is mildly +C in `bare` (0.46) and mildly -A in
  `respondent` (-0.58). No extreme shifts.
* **Qwen2.5-72B is the most centered model in the lineup**, by a
  meaningful margin: max |θ̂| ≤ 0.28 across both neutral conditions.
  Two ways to read this. Generous: Qwen is genuinely the most
  calibrated default subject. Conservative: Qwen's collapsed A and N
  dimensions in the single-model fit (§2c) are still being expressed
  in neutral mode as small near-zero values, and the apparent "good
  centering" is partly an artifact of weak per-trait recovery
  flattening the latent footprint. Distinguishing these requires
  looking at neutral placement *relative to* per-trait recovery
  quality — Qwen's bare-A shift is +0.09 on a dimension where its
  single-model recovery was ~0, vs Haiku's bare-A shift is -0.44 on a
  dimension where Haiku's recovery was 0.85.

This is the placement-on-scale check Serapio-García et al. ran in the
original PaLM2 work (Figure 4 in that paper) — they wanted models to
be calibratable but also to have a sensible "default" in the absence
of a persona. Llama3.3-70B passes this check most cleanly of the
four pooled models. Qwen2.5-72B's tight footprint is partly real
(E, C, O are well-recovered and stay near 0) and partly artifactual
(A, N collapse to near 0 anyway). Gemma3-27B's extreme respondent
shift is the most concerning behavior.

## 6. Open hypotheses / follow-ups

1. **Per-model discriminations, shared thresholds.** Run a variant of
   the pooled Stan model where a_j+ varies by model but κ_p is shared.
   Tests whether Gemma's C failure is a content-signal limit or a
   shared-discrimination constraint (§3e).
2. **Cross-anchor inter-trait correlation test.** §4 finds poor
   off-diagonal recovery across all 4 models. Worth testing whether
   this is intrinsic to the GFC + Okada-prior TIRT fit (which assumes
   independent θ_d), or whether allowing a structured θ prior (LKJ
   on the trait covariance matrix) recovers the correlation pattern.
   We tested LKJ vs N(0, I) earlier and found similar per-trait
   recovery, but didn't compare inter-trait recovery — that should
   now be the headline outcome metric for that comparison.
3. **Llama3.3-70B's rotated C axis.** C–E=-0.51 and C–O=-0.50 are
   roughly mirror-images of true (+0.17, +0.25). Inspect Llama's
   responses on C-loaded GFC blocks to see whether it's interpreting
   C in a way that systematically anti-aligns with E and O (e.g.,
   reading "orderly" as "rigid/closed-minded").
4. **Why does Qwen2.5-72B fail on A and N specifically?** Per-trait
   collapse on these two dimensions while E/C/O recover is unusual —
   most weak models flip C, not A. Inspect Qwen's responses on
   A-loaded GFC blocks (those with A+ markers in either the L or R
   slot) to see whether the failure mode is consistent endorsement of
   one side, or genuinely random response.
5. **Why doesn't Qwen2.5-72B fake-good?** 72B parameters refutes the
   capacity-ceiling hypothesis. Possibilities: (a) RLHF-trained
   refusal of strategic self-presentation; (b) the persona instruction
   taking precedence over the meta-instruction; (c) Qwen's GFC
   responses are already at a kind of fake-good ceiling under the
   persona alone, so the fake-good prompt has no headroom. Check by
   comparing Qwen's bare θ̂ against its honest persona θ̂ distribution.
6. **Block-level inspection of E–O over-correlation.** Both 70B-class
   models have E–O recovered correlation ≥ +0.50 vs true +0.40. This
   is on the right side of truth but suspiciously high. Could be a
   structural feature of the Okada GFC pair set: how many of the 30
   pairs have E and O on opposite sides? If E vs O choices dominate,
   then E and O become co-determined in TIRT.
7. **Run on Sonnet 4.6 / Opus 4.7.** Haiku 4.5 is in the lineup; the
   larger Anthropic models are the obvious next test of whether the
   keying-aligned basin extends with scale. Cost is ~$5–10 per model
   for the full 4-condition sweep.
8. **Gemma3-27B's A-and-N respondent shift.** Investigate why
   `respondent` framing pushes Gemma3-27B to extreme A and N. Compare
   prompts directly — is this a roleplay artifact ("I am a survey
   participant" → compliant + anxious)?
9. **Llama-anchored pool variant.** Llama3.3-70B's tight neutral
   placements suggest it might be a useful "default reference frame"
   for cross-model comparisons. Define a pooled fit anchored on
   Llama's neutral=0 rather than Haiku's loadings, and check whether
   Gemma's `respondent` shift is reproduced.

## 7. Files added / modified

```
psychometrics/gfc_tirt/
  # Inference outputs (all NEW)
  gemma3-27b_gfc30_synthetic.json                     # honest, 1500
  gemma3-27b_gfc30_synthetic-fakegood.json            # fakegood, 1500
  gemma3-27b_gfc30_neutral-bare.json                  # 30
  gemma3-27b_gfc30_neutral-respondent.json            # 30
  llama3.3-70b_gfc30_synthetic.json                   # honest, 1500
  llama3.3-70b_gfc30_synthetic-fakegood.json          # fakegood, 1500
  llama3.3-70b_gfc30_neutral-bare.json                # 30
  llama3.3-70b_gfc30_neutral-respondent.json          # 30
  qwen2.5-72b_gfc30_synthetic.json                    # honest, 1500 (1498 valid)
  qwen2.5-72b_gfc30_synthetic-fakegood.json           # fakegood, 1500 (1499 valid)
  qwen2.5-72b_gfc30_neutral-bare.json                 # 30
  qwen2.5-72b_gfc30_neutral-respondent.json           # 30

  # Drivers
  fit_tirt_per_model_pooled_conditions.R              # +Gemma27B/Llama70B/Qwen72B in MODELS, +cache check
  fit_tirt_pooled_haiku_gemma27b.R                    # NEW (2-model)
  fit_tirt_pooled_haiku_gemma27b_llama70b.R           # NEW (3-model)
  fit_tirt_pooled_haiku_gemma27b_llama70b_qwen72b.R   # NEW (4-model)
  analyze_pooled_haiku_gemma27b_llama70b.R            # NEW (post-hoc cor + neutrals, 3-model)
  analyze_pooled_haiku_gemma27b_llama70b_qwen72b.R    # NEW (post-hoc cor + neutrals, 4-model)

  # Fits (archived to big5_results)
  per_model_pooled/{slug}_pooled_conditions_fit.rds   # NEW × 3 models
  pooled_haiku_gemma27b_fit_3000.rds                  # NEW
  pooled_haiku_gemma27b_llama70b_fit.rds              # NEW
  pooled_haiku_gemma27b_llama70b_qwen72b_fit.rds      # NEW (117 MB)

  # Slim summaries (committable)
  pooled_haiku_gemma27b_llama70b_summary.rds          # NEW
  pooled_haiku_gemma27b_llama70b_qwen72b_summary.rds  # NEW
```

### Inference commands (for the record)

```bash
# Pull + Modelfile variant for 70B-class models (one-time setup)
curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.3:70b-ctx2k","from":"llama3.3:70b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create

curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:72b-ctx2k","from":"qwen2.5:72b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create

# 4-condition sweep (substitute model slug as needed)
export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\"")
MODEL=qwen2.5:72b-ctx2k
SLUG=qwen2.5-72b
for cond in bare respondent; do
  python3 scripts/run_gfc_ollama.py --remote --model "$MODEL" \
    --neutral $cond \
    --output psychometrics/gfc_tirt/${SLUG}_gfc30_neutral-${cond}.json
done
python3 scripts/run_gfc_ollama.py --remote --model "$MODEL" \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 \
  --output psychometrics/gfc_tirt/${SLUG}_gfc30_synthetic.json --checkpoint-every 100
python3 scripts/run_gfc_ollama.py --remote --model "$MODEL" \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 --fake-good \
  --output psychometrics/gfc_tirt/${SLUG}_gfc30_synthetic-fakegood.json --checkpoint-every 100
```
