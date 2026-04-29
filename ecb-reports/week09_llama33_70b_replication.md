# Llama3.3-70B added to Okada GFC + TIRT lineup; 3-model pool

**Date:** 2026-04-28
**Author:** ECB
**Companion to:** [`week08_gemma3_27b_replication.md`](week08_gemma3_27b_replication.md), [`week07_okada_pooled_replication.md`](week07_okada_pooled_replication.md)

## TL;DR

Added **Llama3.3-70B (Q4_K_M, 43 GB) on Orin** to the lineup and ran a
three-model pool (Haiku 4.5 + Gemma3-27B + Llama3.3-70B). Three results
worth foregrounding:

1. **Llama3.3-70B is the first open-weight model to land in Haiku's
   keying-aligned basin.** Single-model honest recovery: A=0.81, C=0.34,
   E=0.87, N=0.49, O=0.78 (mean |r| = 0.658). All five traits positive,
   no {A, C} sign flip. Holds up under pooling (mean |r| 0.626).
2. **Per-trait recovery does not imply joint structure recovery.** The
   recovered inter-trait correlation matrix is far from the true one for
   all three pooled models — RMSD on off-diagonals ≈ 0.39–0.47, with
   sign disagreements on multiple cells. Llama3.3-70B in particular has
   a rotated C dimension (C–E=-0.53 vs true +0.17, C–N=+0.42 vs true
   -0.36, C–O=-0.54 vs true +0.25).
3. **Llama3.3-70B's no-persona placements are well-centered**, unlike
   Gemma3-27B's `respondent` framing which pulls θ̂ to extreme +A and
   +N. Llama bare: max |θ̂| = 0.46 (C). Llama respondent: max 0.41 (A,
   weakly negative). This is the placement-on-scale check Serapio-García
   et al. ran in the original work, and Llama behaves the way a
   "well-calibrated" subject should.

| Result | Outcome |
|---|---|
| Pull + run 70B on the 64 GB Orin (43 GB on disk) | ✅ — required `num_ctx=2048` Modelfile variant; default 128K blew memory (62.8 GiB requested) |
| Inference throughput | 0.6 prompts/sec, 1500-prompt persona run in ~40 min |
| Single-model recovery without sign flip | ✅ all five traits positive on honest |
| 3-model pool convergence | ✅ 0 Rhat>1.05 of 1771 params, 0 n_eff<100 |
| Pool cost on Llama and Haiku | Modest: -0.03 and -0.05 mean \|r\| |
| Pool gain on Gemma3-27B | +0.08 mean \|r\| (carries over from the 2-model anchor effect) |
| True inter-trait correlation recovered | ❌ off-diagonals poorly recovered for all 3 models |

## 1. Inference setup

### Pull and memory

`llama3.3:70b` is 43 GB Q4_K_M on disk. The 64 GB Orin (confirmed via
`/api/ps` on already-loaded `qwen3.5:35b` reporting `size_vram=34.6 GB`)
should fit, but the default 128K-context KV cache pushes peak
allocation to 62.8 GiB — over budget:

```
{"error":{"message":"model requires more system memory (62.8 GiB) than is available (43.5 GiB)"}}
```

Fix: bake a reduced context into a Modelfile-derived variant via
`/api/create`:

```bash
curl -X POST .../api/create -d '{
  "model":"llama3.3:70b-ctx2k",
  "from":"llama3.3:70b",
  "parameters":{"num_ctx":2048},
  "stream":false
}'
```

Note: this is needed because the OpenAI-compatible `/v1/chat/completions`
endpoint (which `run_gfc_ollama.py` uses) does not honor `num_ctx` in
the `options` field — only the native `/api/chat` does. Custom variant
sidesteps that.

`num_ctx=2048` is plenty for our prompts (~150–300 tokens including
persona + GFC pair).

### 4-condition sweep

| Condition | Prompts | Wall time | Throughput | Parser-valid |
|---|---:|---:|---:|---:|
| honest (50 personas × 30 pairs) | 1500 | ~40 min | 0.62/s | 1500/1500 |
| fake-good (50 personas × 30 pairs) | 1500 | ~40 min | 0.63/s | 1500/1500 |
| neutral bare | 30 | ~50 sec | — | 30/30 |
| neutral respondent | 30 | ~50 sec | — | 30/30 |

Output files (in `psychometrics/gfc_tirt/`):

```
llama3.3-70b_gfc30_synthetic.json            # honest, 1500 rows
llama3.3-70b_gfc30_synthetic-fakegood.json   # fake-good, 1500 rows
llama3.3-70b_gfc30_neutral-bare.json         # 30 rows
llama3.3-70b_gfc30_neutral-respondent.json   # 30 rows
```

### Response style: skips category 2

| Cat | Llama3.3-70B (honest) | Gemma3-27B (honest) | Haiku 4.5 (honest) |
|---:|---:|---:|---:|
| 1 | 229 | 189 | (smooth) |
| 2 | **0** | 65 | (smooth) |
| 3 | 108 | 88 | (smooth) |
| 4 | 221 | 135 | (smooth) |
| 5 | **624** | 68 | (smooth) |
| 6 | 81 | 748 | (smooth) |
| 7 | 237 | 207 | (smooth) |

Llama3.3-70B uses 6 of 7 categories, with mode 5 (vs Gemma's mode 6
and Haiku's smooth distribution). Skipping cat 2 is the same response
style we saw on Llama-2 BFI Likert in earlier work — appears stable
across the Llama family. This is a heterogeneity risk for pooled κ
thresholds, but in practice the pool fit cleanly (see §3).

## 2. Single-model TIRT fit (joint honest + fake-good, N=100)

Driver: `psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R`
(now lists Llama3.3-70B in `MODELS`). Stan model: `tirt_okada_indep.stan`
(Okada Appendix D priors). 4 chains × 1000 iter (333 warmup).
Convergence: 0 Rhat>1.05 of 741 params.

| Model | Cond | A | C | E | N | O | mean \|r\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.854 | 0.378 | 0.853 | 0.847 | 0.620 | **0.710** |
| **Llama3.3-70B** | **honest** | **0.809** | **0.335** | **0.867** | **0.493** | **0.784** | **0.658** |
| Llama3.3-70B | fakegood | 0.807 | 0.309 | 0.892 | 0.706 | 0.826 | 0.708 |
| Gemma3-27B | honest | 0.045 | -0.006 | 0.817 | 0.266 | 0.479 | 0.323 |
| Phi4-mini | honest | 0.372 | 0.002 | 0.492 | 0.078 | 0.340 | 0.257 |
| Gemma3-4B | honest | -0.056 | -0.154 | 0.520 | 0.041 | 0.210 | 0.196 |
| Qwen2.5-3B | honest | 0.318 | -0.197 | 0.016 | -0.104 | -0.339 | 0.195 |
| Llama3.2-3B | honest | -0.335 | 0.317 | 0.060 | -0.007 | -0.022 | 0.148 |

**Llama3.3-70B is the first open-weight model in the lineup to clear
mean |r| > 0.5.** All five traits are positive; on honest, four of five
are above Okada's r ≥ 0.50 band (only C is below at 0.34, but still
positively signed — no flip). Compare to Phi4-mini and Qwen2.5-3B,
which have C and other traits flipping to negative on honest.

The fake-good condition recovers slightly *better* than honest on
Llama3.3-70B (0.708 vs 0.658), driven by stronger N and O. We saw the
same pattern on Haiku in the per-model joint fit — the larger spread
under fake-good gives the latent direction a better foothold.

## 3. Three-model pooled fit (Haiku + Gemma3-27B + Llama3.3-70B)

Driver: `psychometrics/gfc_tirt/fit_tirt_pooled_haiku_gemma27b_llama70b.R`
(new, mirrors `fit_tirt_pooled_haiku_gemma27b.R`). Same Stan model.
4 chains × 1500 iter (500 warmup). N=306 rows
(50 honest + 50 fake-good + 1 bare + 1 respondent per model).
Convergence: **0 Rhat>1.05, 0 n_eff<100** of 1771 params.

| Model | Cond | A | C | E | N | O | mean \|r\| | Δ from single |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | honest | 0.860 | 0.351 | 0.837 | 0.760 | 0.467 | 0.655 | -0.05 |
| Haiku 4.5 | fakegood | 0.850 | 0.405 | 0.793 | 0.648 | 0.570 | 0.653 | -0.05 |
| **Llama3.3-70B** | **honest** | **0.803** | **0.307** | **0.848** | **0.382** | **0.787** | **0.626** | **-0.03** |
| Llama3.3-70B | fakegood | 0.819 | 0.271 | 0.881 | 0.668 | 0.836 | 0.695 | -0.01 |
| Gemma3-27B | honest | 0.427 | 0.199 | 0.777 | 0.135 | 0.502 | 0.408 | +0.08 |
| Gemma3-27B | fakegood | 0.401 | -0.276 | 0.611 | 0.207 | 0.662 | 0.431 | +0.05 |

The pool-vs-single picture:

* **Llama3.3-70B**: -0.03 honest (within noise), -0.01 fakegood. Robust
  against shared κ + a_j+ across three heterogeneous response styles.
* **Haiku 4.5**: -0.05 (was -0.07 with just Gemma anchoring). Adding
  Llama as a third strong-signal anchor does not further hurt Haiku.
* **Gemma3-27B**: A jumps from 0.045 (single) → 0.427 (pooled), confirming
  the week08 anchor-effect finding is robust to a different anchor
  combination. C goes from -0.006 to +0.199 — small, but positively
  signed for the first time on Gemma.

## 4. Inter-trait correlation: scored vs true

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

The familiar pattern: a positive "alpha factor" cluster (A, C, low N), a
positive "beta factor" cluster (E, O), and N negatively related to
everything.

### Recovered correlations from the pooled fit (honest condition only)

**Haiku 4.5** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.04 | 0.14 | -0.07 | -0.07 |
| C | -0.04 | 1.00 | **-0.52** | 0.05 | -0.32 |
| E | 0.14 | -0.52 | 1.00 | -0.14 | 0.28 |
| N | -0.07 | 0.05 | -0.14 | 1.00 | -0.01 |
| O | -0.07 | -0.32 | 0.28 | -0.01 | 1.00 |

Most off-diagonals are near zero (Haiku's per-trait recovery is good,
so the rotational structure between traits is largely uninformative).
Notable: C–E=-0.52 (true +0.17) and C–O=-0.32 (true +0.25) — Haiku's
C dimension is anti-correlated with E and O where it should be
modestly positive.

**Gemma3-27B** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | -0.40 | 0.29 | 0.27 | 0.10 |
| C | -0.40 | 1.00 | -0.30 | -0.11 | -0.14 |
| E | 0.29 | -0.30 | 1.00 | 0.08 | 0.19 |
| N | 0.27 | -0.11 | 0.08 | 1.00 | 0.28 |
| O | 0.10 | -0.14 | 0.19 | 0.28 | 1.00 |

A–C is *negative* (true +0.36). A–N is positive (true -0.42 — strong
sign disagreement). The whole alpha-cluster structure is broken.

**Llama3.3-70B** (N=50):

|   | A | C | E | N | O |
|---|---:|---:|---:|---:|---:|
| A | 1.00 | 0.04 | -0.02 | -0.01 | 0.07 |
| C | 0.04 | 1.00 | **-0.53** | **0.42** | **-0.54** |
| E | -0.02 | -0.53 | 1.00 | -0.32 | 0.51 |
| N | -0.01 | 0.42 | -0.32 | 1.00 | 0.03 |
| O | 0.07 | -0.54 | 0.51 | 0.03 | 1.00 |

A's row is essentially flat (Llama's A is good per-trait but
uncorrelated with anything else in θ̂-space). E–O=+0.51 is in the right
direction (true +0.40). E–N=-0.32 is also in the right direction (true
-0.39). But C is inverted relative to the other traits: C–E=-0.53
(true +0.17), C–N=+0.42 (true -0.36), C–O=-0.54 (true +0.25). It's as
though the C axis was flipped — except per-trait C-recovery is
positive (+0.31), so the issue isn't a simple sign convention.

### Summary distance from truth

| Model | RMSD off-diag | max \|deviation\| | mean off-diag (recovered) | mean off-diag (true) |
|---|---:|---:|---:|---:|
| Haiku 4.5 | 0.39 | 0.69 | -0.07 | +0.05 |
| Gemma3-27B | 0.45 | 0.76 | +0.03 | +0.05 |
| Llama3.3-70B | 0.47 | 0.79 | -0.03 | +0.05 |

All three models pull the typical Big Five intercorrelation toward
zero on average — the average off-diagonal is near zero rather than
the mildly positive +0.05 of truth. Per-cell deviations of 0.7–0.8
are not rare. **Per-trait recovery does not imply joint structure
recovery.**

This matters because most LLM-personality work — including ours up to
this report — reports only mean |r| or per-trait correlations. A
model could "personalize" each trait independently while losing the
correlated structure that makes Big Five a meaningful joint construct.
On this evidence, none of our three top models produces a plausible
inter-trait covariance pattern, even when per-trait recovery is high.

## 5. Neutral placement on the latent scale

Pooled-fit θ̂ for the no-persona conditions, in the pooled latent
space (sign convention from Haiku's anchor):

| Model | Condition | A | C | E | N | O | max \|θ̂\| |
|---|---|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | bare | -0.34 | 0.07 | 0.26 | 0.22 | -0.26 | 0.34 |
| Haiku 4.5 | respondent | -0.11 | 0.19 | 0.09 | 0.10 | -0.04 | 0.19 |
| Gemma3-27B | bare | -0.46 | -0.07 | 0.08 | 0.47 | 0.15 | 0.47 |
| Gemma3-27B | respondent | **0.84** | 0.38 | -0.29 | **1.04** | -0.45 | **1.04** |
| **Llama3.3-70B** | **bare** | -0.20 | **0.46** | 0.02 | -0.14 | 0.19 | 0.46 |
| **Llama3.3-70B** | **respondent** | -0.41 | -0.09 | 0.04 | 0.02 | -0.08 | 0.41 |

Reading in 1-SD units (the θ scale is N(0, 1) by prior):

* **Haiku 4.5** is well-centered in both modes. Slight negative A
  baseline, slight positive E baseline. Nothing extreme. Behaves like
  a well-calibrated default subject.
* **Gemma3-27B** is also well-centered in `bare` (max +0.47 on N), but
  the `respondent` instruction pulls it to extreme +A (0.84) and
  extreme +N (1.04 SD) with -O (-0.45). This was already noted in
  week08 — the pattern is suggestive of a "compliant + anxious survey
  participant" voicing rather than the model's own default, and it
  generalizes here in the 3-model pool.
* **Llama3.3-70B** is the most "neutral" of the three — both `bare`
  and `respondent` placements have max |θ̂| ≤ 0.46. Its `bare` baseline
  is mildly +C (0.46), and `respondent` is mildly -A (-0.41). No
  extreme shifts on any trait.

This is the placement-on-scale check Serapio-García et al. ran in the
original PaLM2 work (Figure 4 in that paper) — they wanted models to
be calibratable but also to have a sensible "default" in the absence
of a persona. **Llama3.3-70B passes this check most cleanly of the
three pooled models.** Gemma3-27B's extreme respondent shift is the
most concerning, since it suggests the framing alone produces
substantial trait movement beyond the persona's intended effect.

## 6. Open hypotheses / follow-ups

1. **Cross-anchor inter-trait correlation test.** The off-diagonal
   recovery in §4 is poor across all 3 models. Worth testing whether
   this is intrinsic to the GFC + Okada-prior TIRT fit (which assumes
   independent θ_d), or whether allowing a structured θ prior (e.g.,
   LKJ on the trait covariance matrix) recovers the correlation
   pattern. We tested LKJ vs N(0, I) earlier and found similar
   per-trait recovery, but didn't compare inter-trait recovery — that
   should now be the headline outcome metric for that comparison.
2. **Llama3.3-70B's rotated C axis.** C–E=-0.53 and C–O=-0.54 are
   roughly mirror-images of true (+0.17, +0.25). Inspect Llama's
   responses on C-loaded GFC blocks to see whether it's interpreting C
   in a way that systematically anti-aligns with E and O (e.g., reading
   "orderly" as "rigid/closed-minded").
3. **Run on more 70B-class models.** The week07 pool result (mean |r|
   plateau on open models around 0.20–0.35) suggested a frontier
   plateau. Llama3.3-70B breaks that plateau — but is it scale or
   alignment that matters? Candidates that fit on the 64 GB Orin:
   `qwen2.5:72b` (47 GB) is the obvious next step.
4. **No-persona placement as a calibration screen.** Llama3.3-70B's
   tight neutral placements suggest it might be a useful "default
   reference frame" for cross-model comparisons. Worth defining a
   pooled fit anchored on Llama's neutral=0 rather than Haiku's
   loadings, and checking whether Gemma's `respondent` shift is
   reproduced or moves around.

## 7. Files added / modified

```
psychometrics/gfc_tirt/
  llama3.3-70b_gfc30_synthetic.json                          # NEW (1500 rows)
  llama3.3-70b_gfc30_synthetic-fakegood.json                 # NEW (1500 rows)
  llama3.3-70b_gfc30_neutral-bare.json                       # NEW (30 rows)
  llama3.3-70b_gfc30_neutral-respondent.json                 # NEW (30 rows)
  fit_tirt_per_model_pooled_conditions.R                     # +Llama3.3-70B in MODELS
  fit_tirt_pooled_haiku_gemma27b_llama70b.R                  # NEW (3-model driver)
  analyze_pooled_haiku_gemma27b_llama70b.R                   # NEW (post-hoc cor + neutrals)
  per_model_pooled/llama3.3-70b_pooled_conditions_fit.rds    # NEW (archived)
  pooled_haiku_gemma27b_llama70b_fit.rds                     # NEW (archived)
  pooled_haiku_gemma27b_llama70b_summary.rds                 # NEW (small, committable)
```

Inference command (for the record):

```bash
# Modelfile variant with reduced context (one-time setup)
curl -X POST -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.3:70b-ctx2k","from":"llama3.3:70b","parameters":{"num_ctx":2048},"stream":false}' \
  https://apollo.quocanmeomeo.io.vn/api/create

# 4-condition sweep
export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\"")
for cond in bare respondent; do
  python3 scripts/run_gfc_ollama.py --remote --model llama3.3:70b-ctx2k \
    --neutral $cond --output psychometrics/gfc_tirt/llama3.3-70b_gfc30_neutral-${cond}.json
done
python3 scripts/run_gfc_ollama.py --remote --model llama3.3:70b-ctx2k \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 \
  --output psychometrics/gfc_tirt/llama3.3-70b_gfc30_synthetic.json --checkpoint-every 100
python3 scripts/run_gfc_ollama.py --remote --model llama3.3:70b-ctx2k \
  --synthetic-personas instruments/synthetic_personas.json --max-personas 50 --fake-good \
  --output psychometrics/gfc_tirt/llama3.3-70b_gfc30_synthetic-fakegood.json --checkpoint-every 100
```
