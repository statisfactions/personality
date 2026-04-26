# Week 7: Larger-cohort baseline replication and the read-write reversal

## TL;DR

Per the Phase-1 plan agreed at the machine-transfer (one-variable-at-a-time: scale up the cohort first, introduce SAE / single-direction methods after), this week ran the full Week 1–6 measurement battery on a larger cohort of three SAE-covered instruct models (Gemma 3 12B, Llama 3.1 8B, Qwen 2.5 7B) and rebuilt the Week 3 cross-method matrix on the 7-model panel.

Three things came out of it:

1. **Character preserves across scale, reliability is family-bound.** Each family stays itself (Gemma the principled empath, Llama the uncertain centrist, Qwen the cautious centrist with high H+ALT), with modest quantitative sharpening. Cross-prompt-variant ICC clusters by family at both small and large scales: Gemma 0.65–0.72, Qwen 0.54–0.61, Llama 0.33–0.39. **Reliability is acquired in instruction-tuning style, not parameter count.**

2. **Week 6 structural findings replicate at scale, and stimulus-swap experiments reveal that transformers preserve subtle text-similarity structure with very high cross-architecture fidelity.** Σ⁻¹-noise pathology, no-clean-1D facet axes, cross-model-consistent mis-groupings (E:Sentimentality→O:Aesthetic Appreciation in *every* model, H:Fairness→C/X drift, A:Forgiveness→X drift) — all carry over. The 24×24 facet cosine matrix is essentially the same matrix in all 7 models: pairwise upper-tri correlation between models lands in r=0.88–0.97. §8.4 (HEXACO-100 items as fresh stimuli) and §8.5 (Goldberg 52 markers as a third stimulus type) sharpen this: within-model agreement *across* stimulus types is only +0.32 to +0.43, while cross-model agreement *within* each stimulus type is consistently +0.93 to +0.99 (markers reach +0.99 across our 7B–12B models). The interesting reading isn't "the matrix is just stimulus-driven" — it's that the relatively subtle similarity structure present in the contrast-pair texts (and separately in HEXACO Likert items, and separately in single-adjective markers) gets carried through dozens of layers of nonlinear processing across three different transformer architectures with very high fidelity. Each stimulus author's choices construct a particular geometry; the models faithfully decode that geometry and then add ~0.3–0.4 worth of universal trait structure on top. The Goldberg-marker 5×5 matrix shows what the universal residual looks like — a **valence cluster** with E, A, C, O all positive with each other and N opposing. That's the W1 assistant shape (low-N, high-A/C/E/O) at the RepE level. Within-trait pair-diff diffuseness also holds; Qwen 2.5 7B is a partial exception with PC1 0.06–0.07 (vs 0.04–0.06 elsewhere) and participation ratios down ~10% from prior — consistent across all six traits, though.

3. **The cross-method matrix produced one striking new finding that the §11 cleanup substantially reframed.** As originally measured (chat-template Likert + chat-template BC + bare-text RepE, mismatched format): BC↔RepE went from weak positive on small cohort (+0.27 to +0.50, except phi4 −0.07) to strongly negative on large (Gemma12 −0.23, Llama8 −0.73, Qwen7 −0.80). After §11 cleanup (chat-template RepE throughout, Qwen-family unified): both cohorts converge to weakly-to-moderately negative (−0.14 to −0.41), with Qwen 2.5 3B (+0.49) as the lone positive outlier. The original "scale flips sign" claim shrinks to "format mismatch was driving most of it" — under format-matched measurement, BC↔RepE is mildly negative across all models, consistent with the W4 read/write-gap as semi-independent rather than systematically opposite. H Likert↔RepE goes from +0.82 to +0.39 (still the strongest converging trait), and X Likert↔RepE drops to −0.79 (Likert and RepE flatly disagree on Extraversion across the cohort).

The matched-scale Phase-1 baseline is in hand. Three follow-ups suggested by the data are listed in §8; SAE-based extraction on Gemma 12B is the natural Phase-2 next track.

---

## 1. Setup and methodology changes from Week 6

### 1.1 Cohort

Phase-1 cohort = same families, larger size, all SAE-covered. Phi-4-mini retained as a no-SAE control through this baseline; drops out for Phase 2.

| Family | Small (W1–6) | Larger (W7+) | SAE source |
|---|---|---|---|
| Gemma 3 | 4B | **12B** | GemmaScope 2 (all sizes 270M–27B, all layers + transcoders) |
| Llama 3 | 3.2-3B | **3.1-8B** | andyrdt |
| Qwen 2.5 | 3B | **7B** | andyrdt |
| Phi 4 | mini | (dropped after Phase 1) | none |

The corrected SAE-coverage picture is in `to_try.md` §5. Hardware constraint mentioned in earlier reports / CLAUDE.md is now obsolete — M5 Max / 128 GB handles all of these comfortably in bf16, plus Gemma 3 27B and GPT-OSS 20B if we want bigger anchors later.

### 1.2 HF inference port

Weeks 1–6 ran surveys via Ollama; this week the Likert + BC pipelines moved to HuggingFace via `scripts/hf_logprobs.py`. The trigger was avoiding the two-copies-on-disk requirement (Ollama + HF) when scaling the cohort, and the activation extraction code was already HF-only. JSON output schemas preserved so `analyze_denoised.py`, `cross_method_matrix.py` etc. work unchanged.

### 1.3 The chat-template gotcha

The initial port defaulted `likert_distribution` to bare-text on the assumption that the prior Ollama `/api/generate` path was bare-text. First Qwen 7B run with bare-text produced ICC = −0.054 — striking departure from the Qwen3-8B prior of +0.54.

Diagnosis: weeks 1–6 numbers were *chat-template* all along. Ollama `raw=False` (the default) applies the chat template server-side; the Qwen3-specific code path used hand-written `<|im_start|>...<|no_think|>...<|im_end|>` wrapping which is also chat-template by construction. The bare-text HF default was a format change.

Behavioral signature was concrete: variant v3 (terse, ends in `\n`) collapsed to EV ≈ 1.2 across nearly all items in bare text. Restoring chat-template default recovered ICC = +0.54 on Qwen 7B. The bare-text run is preserved in `results/Qwen7_*_baretext.json` for the `to_try.md` §15 ablation; §15 itself was rewritten — bare-text Likert is now the novel condition, not the legacy.

This is the Week 7 methodology lesson worth flagging: assumed-equivalent inference paths can have load-bearing format differences. The check is to re-run a known-quantity item (here, qwen3-style ICC) before treating new pipeline output as comparable.

---

## 2. Likert: family-bound reliability, character-preserved scale-up

### 2.1 Big Five (IPIP-300, denoised across 4 prompt variants)

| Family-scale | N | E | O | A | C | ICC |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4B | 2.90 | 3.22 | 3.77 | 3.78 | 3.57 | 0.71 |
| **Gemma 12B** | 2.46 | 3.21 | 4.02 | 4.23 | 3.88 | **0.72** |
| Llama 3.2 3B | 2.98 | 2.98 | 3.14 | 3.25 | 3.10 | 0.34 |
| **Llama 3.1 8B** | 2.81 | 2.86 | 3.09 | 3.26 | 3.12 | **0.38** |
| Qwen3 8B† | 2.43 | 3.32 | 3.53 | 3.61 | 3.67 | 0.54 |
| **Qwen 2.5 7B** | 2.54 | 3.15 | 3.28 | 3.43 | 3.69 | **0.54** |

†Small Qwen comparison is qwen3-8B from W1 (different model from Qwen 2.5 7B; carryover of the cross-family confound flagged in W3).

### 2.2 HEXACO-100 (denoised across 4 prompt variants)

| Family-scale | H | E | X | A | C | O | ALT | ICC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4B | 4.08 | 3.56 | 3.22 | 3.09 | 3.45 | 3.79 | 4.17 | 0.71 |
| **Gemma 12B** | **4.28** | 3.69 | 3.22 | 3.37 | 3.76 | 4.12 | **4.75** | 0.65 |
| Llama 3.2 3B | 3.03 | 3.22 | 2.97 | 2.97 | 3.11 | 3.29 | 3.20 | 0.39 |
| **Llama 3.1 8B** | 3.27 | 2.98 | 2.83 | 3.13 | 2.94 | 2.87 | 3.12 | 0.33 |
| Qwen3 8B† | 3.66 | 3.00 | 3.45 | 3.31 | 3.63 | 3.65 | 3.72 | 0.56 |
| **Qwen 2.5 7B** | 3.69 | 2.66 | 3.19 | 3.52 | 3.47 | 3.17 | 3.75 | 0.61 |

### 2.3 What this says

**Reliability ordering is family-bound, not scale-bound.** Llama's ICC stays in the 0.33–0.39 band at both 3B and 8B; Gemma's stays in 0.65–0.72; Qwen sticks at 0.54–0.61. Two-fold difference between Gemma and Llama on identical instruments and prompts — survives a 2–3× scale-up in both directions. Most parsimonious reading: how reliably a model treats Likert items is acquired in instruction-tuning style (or pretraining-data composition), not by parameter count.

**Character preserves with quantitative sharpening.** Each family remains recognizable across scale. Gemma 12B is "more Gemma" specifically on the prosocial axis: A/C/H/O/ALT all up by 0.20–0.58, N drops 0.44, X / HEXACO Emotionality stay flat. ALT hits 4.75 with entropy 0.024 — saturated. The 12B doesn't move *every* trait — it sharpens the assistant cluster while leaving the introvert/emotion axis untouched. Suggestive of the assistant axis being acquired separately from orthogonal trait variance.

**Llama 3.1 8B is essentially Llama 3.2 3B in scale-score terms** (mostly within 0.1 of the smaller model), with the same ~0.9-nat-mean entropy. The "uncertain centrist" character is unchanged.

**Qwen 2.5 7B compresses HEXACO Emotionality** (3.00 → 2.66) compared to qwen3-8B; H stays put. Probably more a Qwen-3-vs-Qwen-2.5 architecture/training difference than scale, but we don't have qwen2.5-3B Likert data to disentangle (the small "Qwen" entry has been qwen3-8B since W1).

---

## 3. Phase B sweep on the larger cohort

`phase_b_sweep.py --models Gemma12 Llama8 Qwen7 --output-prefix phase_b_sweep_large`. Same protocol as W5/W6 small-cohort runs: 50-pair training + 24-pair facet-stratified holdout, both `chat` and `bare` formats, methods = LDA + LR + MD-raw + MD-projected.

### 3.1 Holdout sign-correct, aggregated across 6 traits × 24 pairs (=144)

Chat-template format (primary):

| Model | LDA | LR | MD-raw | MD-projected |
|---|---:|---:|---:|---:|
| Gemma 4B (W5) | 138/144 (95.8%) | — | 138/144 (95.8%) | 138/144 (95.8%) |
| **Gemma 3 12B** | 140/144 (97.2%) | **143/144 (99.3%)** | 140/144 (97.2%) | 142/144 (98.6%) |
| Llama 3.2 3B (W5) | 137/144 (95.1%) | — | 134/144 (93.1%) | 135/144 (93.8%) |
| **Llama 3.1 8B** | 141/144 (97.9%) | 141/144 (97.9%) | 137/144 (95.1%) | 138/144 (95.8%) |
| Qwen 2.5 3B (W5) | 135/144 (93.8%) | — | 136/144 (94.4%) | 138/144 (95.8%) |
| **Qwen 2.5 7B** | 140/144 (97.2%) | **142/144 (98.6%)** | 135/144 (93.8%) | 138/144 (95.8%) |

LDA train accuracy is 89–93% on the larger cohort vs LR/MD train at 100% — same Σ⁻¹-noise-rotation pathology Week 6 §1 diagnosed, persists at scale.

LR is the new method since W6; on the larger cohort it edges out MD-projected on Gemma 12B (99.3 vs 98.6) and Qwen 7B (98.6 vs 95.8) and ties on Llama 8B. Note: Week 6 §E.2 found MD more sample-stable across old→new direction comparisons; with these clean replication numbers LR holds up. The Week 6 stability concern is about cross-dataset transfer, not within-dataset accuracy.

### 3.2 Chat vs bare format

| Model | LR chat | LR bare | Δ |
|---|---:|---:|---:|
| Gemma 12B | 99.3% | 97.2% | +2.1 |
| Llama 8B | 97.9% | 96.5% | +1.4 |
| Qwen 7B | 98.6% | 95.8% | +2.8 |

Modest but consistent chat > bare on the LR axis. Same direction the small cohort showed (Llama specifically had +31pt in BC, but Likert/RepE differences are smaller in any case). The big-Llama-only chat-template-gates-deployment-mode finding from W5 §11 doesn't appear amplified at 8B.

### 3.3 H is still the trouble trait

Per-trait holdout aggregates across the larger cohort (chat, LR):

| Trait | sign-correct/72 |
|---|---:|
| O | 72/72 (100%) |
| E | 72/72 (100%) |
| X | 72/72 (100%) |
| A | 70/72 (97.2%) |
| C | 69/72 (95.8%) |
| **H** | **61/72 (84.7%)** |

H drops to 75–88% on individual model × method cells (Llama 8B H 88%, Qwen 7B H 75–88%, Gemma 12B H ~96%). Same construct-heterogeneity-within-H pattern W5 §"Why H" first identified. Scale doesn't fix it because the four H-facets (Sincerity/Fairness/Modesty/Greed-Avoidance) genuinely point in different directions in representation space — that's a property of how H is encoded, not a precision problem.

---

## 4. Facet clustering on the larger cohort

`scripts/facet_cluster.py` on all 7 models with the existing 24-pair facet-stratified holdout, MD-projected directions at ~2/3 depth.

| Model | within-cos | across-cos | NN/24 | purity@6 |
|---|---:|---:|:---:|---:|
| Llama 3.2 3B | +0.188 | +0.041 | 17 | 0.542 |
| Llama 3.1 8B | +0.191 | +0.045 | 17 | 0.542 |
| Gemma 3 4B | +0.205 | +0.033 | 15 | 0.583 |
| **Gemma 3 12B** | +0.221 | +0.040 | 16 | **0.667** |
| Phi 4 mini | +0.168 | +0.040 | 17 | 0.625 |
| Qwen 2.5 3B | +0.202 | +0.048 | 15 | 0.583 |
| **Qwen 2.5 7B** | **+0.236** | +0.059 | 16 | 0.542 |

Within/across ratio band (4–6×) holds in every family. **Gemma 12B has the highest 6-cluster purity (0.667)** — consistent with its sharper assistant axis from §2 Likert results: a model that more decisively expresses HEXACO traits also clusters its facets more cleanly along HEXACO lines. **Qwen 7B has the highest within-cos (+0.236)** but only middling purity (0.542) — its facets are tightly clustered into broader groupings rather than HEXACO-clean clusters.

Cross-model-consistent mis-groupings replicate at scale. **Every one of the 7 models** has E:Sentimentality → O:Aesthetic Appreciation as a top mis-grouping (+0.34 to +0.38). H:Fairness drifts to X:Social Boldness or C:Perfectionism in 6/7 models. A:Forgiveness → X:Social Self-Esteem in 6/7. A:Flexibility → O:Unconventionality in 6/7. These are the same psychologically-coherent alternative groupings W6 §2 identified — they are properties of the contrast-pair structure, not artifacts of any particular model or scale.

The W6 caveat about H:Fairness↔X:Social Boldness being partly a stimulus-text confound (W6 §E.6) still applies; per-scenario residuals weren't recomputed for the larger cohort but the strong replication in *cosine space* is independent evidence that the facets-don't-cluster-HEXACO finding isn't a small-cohort fluke.

### 4.1 Cosine-matrix similarity across models

How much does the 24×24 facet cosine matrix actually look the same across our 7 models? Vectorize the upper triangle (276 unique off-diagonal entries) per model, take pairwise Pearson r:

|  | Llama | Gemma | Phi4 | Qwen | Llama8 | Gemma12 | Qwen7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama | 1 | .95 | .95 | .93 | **.97** | .93 | .93 |
| Gemma |  | 1 | .91 | .90 | .95 | **.96** | .93 |
| Phi4 |  |  | 1 | .94 | .94 | .89 | .91 |
| Qwen |  |  |  | 1 | .94 | .89 | **.94** |
| Llama8 |  |  |  |  | 1 | .95 | .95 |
| Gemma12 |  |  |  |  |  | 1 | .93 |
| Qwen7 |  |  |  |  |  |  | 1 |

Within-family small↔large pairs (bolded) are the highest in their rows, but only just: cross-family cosine matrices correlate at 0.88–0.95 too. The smallest correlation in the table is Phi 4 mini ↔ Gemma 12B at 0.89; the largest is Llama ↔ Llama 8B at 0.97. **The same matrix, basically, in 7 different models.** PC structure carries the point: PC1 sits in 0.104–0.128 across all 7, PC1+3 in 0.277–0.347 — the same range W6 reported on the small cohort, no scale-driven phase change.

Two readings, both probably partially true:

1. **Stimulus structure dominates.** The contrast pairs themselves carry most of the cosine geometry; models just decode it. W6 §E.6 already showed per-scenario stimulus-text correlation (MiniLM cos vs representation cos) is only +0.068, so it's not naive text-embedding similarity — but it could be a higher-level semantic structure that's locked in by the items, regardless of who's reading them.

2. **Convergent post-training shape.** Cross-family agreement (0.88–0.95) is surprisingly close to within-family (0.94–0.97). If models had idiosyncratic representations of HEXACO traits, we'd expect a much wider gap. The small gap suggests instruction-tuned LLMs end up in roughly the same HEXACO-decoding regime regardless of architecture or scale.

These aren't mutually exclusive — stimulus is doing real work *and* there's substantial convergence on top of it. The gap between within-family (0.94–0.97) and cross-family (0.88–0.95) is the one quantitative handle we have on how much is "extra" model-specific structure vs shared decoding of the items: roughly, 0.03–0.06 of the structure separation is family-specific. Small. The interesting consequence: anything we say about "the facet structure these models represent" is at best 0.03–0.06 worth of model-specific signal on top of a shared 0.88+ structure that's mostly stimulus-driven. Worth keeping in view when interpreting any single model's heatmap.

---

## 5. Within-trait variance: diffuse cloud holds, with a Qwen-7 exception

`scripts/within_trait_variance.py --source stratified` on the 838-pair Week-6 dataset extracted on each model.

| Family | H PC1 | H MD-coh | E PC1 | E MD-coh | mean PR |
|---|---:|---:|---:|---:|---:|
| Llama → Llama8 | 0.061 → 0.058 | 0.053 → 0.056 | 0.052 → 0.056 | 0.110 → 0.111 | 69 → 69 |
| Gemma → Gemma12 | 0.053 → 0.053 | 0.056 → 0.065 | 0.058 → 0.063 | 0.129 → 0.135 | 66 → 68 |
| **Qwen → Qwen7** | 0.058 → **0.067** | 0.063 → **0.073** | 0.051 → **0.061** | 0.133 → **0.147** | 64 → **60** |

Two of three families show no movement on the diffuseness signature; **Qwen 2.5 7B is the only model where within-trait coherence improves with scale.** It has the highest H PC1 of all 7 models and the lowest mean participation ratio (~60 vs ~68–73 elsewhere). This is consistent across all six HEXACO traits (E, X, A, C, O all show analogous improvements) — not a cherry-picked single trait.

Two interpretations, can't pick between them yet:
- (a) Qwen 2.5 7B genuinely encodes traits more 1D than the smaller siblings or other families. Qwen's W5 finding that bare-text BC = chat-template BC (deeply baked persona) is consistent with "Qwen's representation is more concentrated."
- (b) An artifact of the stratified pair sampling: the 838 stratified pairs may have a Qwen-favorable distribution by chance.

Even with (a), absolute coherence is still low — 6.7% PC1 is a long way from clean 1D structure. **Week 6's "facets are bundles, not 1D axes" finding holds in all seven models**, with Qwen 7B as a partial / less-bundle-y outlier.

---

## 6. Cross-method correlation matrix on the 7-model panel

`scripts/cross_method_matrix.py --probe lr` on all 7 models. Required two adapters (described in §7): `score_bc.py` to add larger-cohort BC scores, `repe_legacy_from_cache.py` to convert phase_b_cache activations into the legacy results/repe format the matrix script reads.

### 6.1 Overall (42 cells = 7 models × 6 traits)

| | Lik-arg | Lik-EV | BC-prop | BC-lo | RepE |
|---|---:|---:|---:|---:|---:|
| Likert-argmax | 1.00 | 0.96 | 0.43 | 0.33 | **−0.05** |
| Likert-EV | | 1.00 | 0.46 | 0.36 | **−0.06** |
| BC-proportion | | | 1.00 | 0.68 | **−0.03** |
| BC-logodds | | | | 1.00 | **−0.02** |
| RepE-probe | | | | | 1.00 |

Compare to W6 small-cohort overall (4 models × 6 traits = 24 cells) under LR: Likert↔RepE r=+0.09. **Adding three larger models drives that to −0.05 — the dissociation strengthens at scale, not weakens.** Likert↔BC remains in the 0.33–0.46 band (similar to W6).

### 6.2 Per-model BC↔RepE: the sign flip

| Model | BC-logodds ↔ RepE | BC-prop ↔ RepE |
|---|---:|---:|
| gemma3 4B | +0.27 | +0.33 |
| llama 3.2 3B | +0.26 | +0.40 |
| phi4 mini | −0.07 | +0.10 |
| qwen 2.5 3B | +0.50 | +0.49 |
| **Gemma 3 12B** | **−0.23** | **−0.13** |
| **Llama 3.1 8B** | **−0.73** | **−0.69** |
| **Qwen 2.5 7B** | **−0.80** | **−0.80** |

This is the most striking new finding of the week. The small cohort had within-model BC↔RepE in [+0.10, +0.50] — broadly consistent with "BC and RepE both read the same trait, with noise." The larger cohort has it in [−0.13, −0.80]. The flip is not a couple of points — it's a 90-130 percentage-point swing on Llama and Qwen specifically.

What this means: in the larger Llama and Qwen, **the more strongly the model picks the high-trait BC option, the lower its bare-text RepE projection on that trait's contrast direction is.** Behavior axis and representation axis are anti-aligned within model.

Mechanistically this is the W4 read-write gap made very stark: read direction (LR-fitted on contrast pairs) and write direction (whatever the model uses to choose A vs B) point in different — and for these models, opposite — directions. Where W4 reported zero cosine between them (orthogonal subspaces), at the trait-level summary on this larger cohort they are anti-correlated.

The Gemma 12B sign is also negative but much smaller in magnitude (−0.23). Possibly because Gemma's assistant cluster is more sharpened (§2) — both methods pick up the same compressed prosocial axis even if the directions don't agree at the per-pair level.

Caveat: the larger-cohort RepE was extracted bare-text per W3 protocol (matching the small-cohort legacy files). Bare text on Llama 8B may pull the model out of the chat-template-gated assistant region per W5 §11 (Llama-specific, +31pt BC bump). If the chat-template RepE flipped this, that's actionable — see §8.

### 6.3 Per-trait: H is the only convergent dimension

Per-trait correlations across the 7-model panel (n=7 each):

| Trait | Lik-arg ↔ RepE | Lik-EV ↔ RepE | BC-lo ↔ RepE |
|---|---:|---:|---:|
| **H** | **+0.82** | **+0.64** | +0.09 |
| E | −0.24 | −0.18 | +0.09 |
| X | −0.25 | −0.32 | −0.46 |
| A | −0.31 | −0.37 | −0.50 |
| C | −0.08 | −0.10 | +0.08 |
| O | −0.22 | −0.23 | +0.12 |

H is by far the strongest convergence, second only to itself. Likert↔RepE on H is +0.82 across the 7 models. **The trait that maps directly onto the HHH alignment objective is the trait the most measurement methods agree on.** That fits the "H is where the assistant signal lives" framing carried forward from W1.

A is the trait with the worst method-agreement: BC↔RepE = −0.50, Likert↔RepE = −0.37. The most prosocial-coded trait shows the worst within-cohort agreement.

### 6.4 Putting 6.2 and 6.3 together

Two facts:
- **Per-model BC↔RepE**: small cohort positive, larger cohort strongly negative.
- **Per-trait Lik↔RepE**: H positive (+0.82), every other trait near zero or negative.

The overall null Lik↔RepE r=−0.05 (across 42 cells) hides this structure: H drives the positive contribution, the other five traits drag it back to zero. Within-model, the negative correlation is dominated by traits where representation and behavior are pulling in different directions.

This is hard to summarize as a single mechanism. Some candidates:
- **Read direction at scale ≠ write direction at scale, more than at small scale.** The larger models have richer representations (W6 within-trait diffuseness numbers don't change but absolute capability does), and the gap between "what the model represents" and "what the model produces" widens.
- **H is the only trait the model represents and acts on with the same axis.** Other traits have different read and write axes; H — by virtue of being the alignment-target — has them aligned.
- **Stimulus selection bias.** Our 50 contrast pairs per trait were Claude-generated; the H pairs may have a property (concrete/material rather than dispositional, per W5 scenario_audit) that makes H representable on the same axis as it's behaviorally enacted.

We can't pick between these from this matrix alone. The §8 follow-ups are what we'd run to disambiguate.

---

## 7. Methodology notes and confounds

Worth being explicit about what's mixed in this matrix:

1. **Small-cohort BC is from prior Ollama runs, larger cohort BC is new HF runs.** Both use chat template (Ollama `/api/chat` ≈ HF `apply_chat_template`), but the inference frameworks differ. Numerical values may shift slightly even on identical models — though the qualitative findings (especially the W7 sign flip) are large enough to survive this.
2. **Small "qwen" Likert is qwen3-8B not qwen2.5-3B.** Pre-existing W3 carryover. The W7 small-cohort Qwen entry in the cross-method matrix is therefore comparing qwen3-8B's Likert against Qwen2.5-3B's BC and RepE. Should be re-run on Qwen 2.5 3B for cleanness — flagged as a follow-up but not done this week.
3. **All RepE is bare-text** (W3 protocol). The W7 §6.2 sign flip might or might not survive switching to chat-template RepE; that's exactly the §8.1 follow-up.
4. **BC is single A/B ordering, NOT position-debiased** (matching W3 protocol, despite the ~30pt position-bias finding from W5 §8). Position-debiased BC is a separate ablation.
5. **LDA fold-failure warnings on Gemma 12B edge layers** during cross-method matrix's CV layer-selection step. With ~50 layers and n=50 pairs, the very early/late layers occasionally have rank-zero covariance estimates. The script handles this by setting affected fold scores to NaN; layer selection still proceeds. Doesn't affect direction fitting.
6. **`to_try.md` §15 was reframed.** Bare-text Likert turned out to be the *novel* condition (week 1–6 was always chat-template). The bare-text Qwen 7B run is preserved as the §15 ablation for whenever we need it.

New scripts written this week that should not surprise future-you:
- `scripts/hf_logprobs.py` — shared HF helper (Likert + BC + free-text generation)
- `scripts/score_bc.py` — generates per-pair BC log-odds for the cross-method matrix
- `scripts/repe_legacy_from_cache.py` — adapter from `phase_b_cache/<tag>_<trait>_<format>_pairs.pt` to the legacy `results/repe/<tag>_<trait>_directions.pt` format

Likert + Phase B + facet + within-trait variance + cross-method matrix all run end-to-end on the 7-model panel from the existing infrastructure with these additions.

---

## 8. Three follow-ups suggested by the data

In rough order of interestingness:

### 8.1 Does the BC↔RepE sign flip survive chat-template RepE?

Larger-cohort RepE used bare-text per W3 protocol. Llama 8B specifically was the +31pt chat-template-bump model in W5; its bare-text representation may not reflect deployment-mode behavior. Re-extracting RepE under chat template (we have these activations cached as `<tag>_<trait>_chat_pairs.pt` already) and re-running the matrix is one script call away.

If the sign flip vanishes, the W7 finding becomes "bare-text RepE diverges from chat-template BC at scale — both directions exist but they live in different inference modes." If it persists, the read-write gap really has reversed sign at this cohort, and we need a mechanistic story.

### 8.2 Why is H the only convergent trait?

H Likert↔RepE = +0.82 across 7 models. No other trait above +0.13. Three candidate mechanisms (§6.4); to pick:
- **H scenarios are concrete-material, others aren't** → run the cross-method matrix per facet and see whether H:Sincerity / H:Fairness (concrete) drive the positive correlation while H:Modesty / H:Greed-Avoidance do not.
- **H is the alignment target so post-training aligns read+write axes for it specifically** → look at base models. Base models (where they exist) shouldn't have H specially aligned.
- **Stimulus authoring artifact** → use a fresh H instrument (the GFC-30 Okada items might be partly suitable) and check if the H convergence holds.

### 8.3 Phase 2: SAE work on Gemma 12B

Gemma 3 12B is the most decisive responder (lowest entropy per item on Likert, sharpest assistant axis), the highest 6-cluster facet purity, and has full GemmaScope 2 coverage. It's the lowest-friction entry point for the SAE-based decomposition deferred from W6 §E.8.

Concrete first step: pick a layer (~33 from §3 best-layer numbers), grab the SAE features for that layer from GemmaScope 2 / Neuronpedia, project our existing high-vs-low contrast pair activations onto SAE feature space, look at which features fire differentially. If we find interpretable trait-relevant features, that's the Sofroniew-style disposition-axis story we couldn't tell with contrast-pair MD/LR alone.

### 8.4 HEXACO-as-stimuli: fresh stimulus test for §4.1 — done

§4.1 found the 24×24 facet cosine matrix is essentially the same matrix in all 7 models (r = 0.88–0.97 pairwise). The interpretation was ambiguous between "structure transfers across stimulus authors" and "stimulus content carries most of the geometry." This subsection runs the disambiguating experiment: build per-facet directions from hexaco.org-authored Likert items and compare the resulting 24×24 matrix to the contrast-pair matrix on the same model.

Method: per-facet direction = mean(forward-keyed item activations) − mean(reverse-keyed item activations) at ~2/3 depth, neutral-PC-projected for parity with `facet_cluster.py`'s MD-projected pipeline. Each HEXACO item wrapped as a user turn in the chat template; activations averaged over content tokens. HEXACO-100 has 4 items per facet (typically 2 forward + 2 reverse-keyed), with 6 of 24 facets unbalanced 1+3; noisy direction per facet but enough to test the structure question.

Result on the larger cohort:

| Model | r(HEXACO-stim, contrast-pair) |
|---|---:|
| Gemma 3 12B | **+0.43** |
| Llama 3.1 8B | **+0.32** |
| Qwen 2.5 7B | **+0.43** |

Compare with §4.1's cross-*model* contrast-pair pairwise correlations of +0.88 to +0.97. **Changing the stimulus set on the same model changes the cosine matrix more than running the same stimuli across different models does.**

This is more interesting than a deflationary "the matrix is just in the items" reading would suggest. The contrast-pair texts encode a particular subtle similarity structure (Claude's choice of facet items + scenario authoring); HEXACO Likert items encode a different subtle similarity structure (Lee & Ashton's choice of trait-defining sentences). Both structures get **carried through dozens of layers of nonlinear processing across three transformer architectures with cross-model fidelity 0.93–0.97**. The +0.32–0.43 cross-stimulus-type residual is what the models add on top of faithful stimulus-decoding — that's the genuine model-side trait structure that survives stimulus changes. The §4.1 finding wasn't "all 7 models share a personality representation" — it was "all 7 models faithfully decode the same items into similar geometries, plus they share ~0.3–0.4 of additional structure independent of which items they're given." That's two findings, not one.

One thing this isolates partially: HEXACO sentences and contrast-pair paragraphs differ in *format* (declarative self-statement vs. scenario-with-options) as well as in *content/author*. The Goldberg-marker test in §8.5 adds a third stimulus type at very different format (single adjectives) and grain (Big Five rather than HEXACO facets) to extend this picture.

Outputs: `results/hexaco_as_stimuli_directions.json` (matrices), `results/hexaco_as_stimuli_heatmap.html` (per-model heatmaps).

### 8.5 Goldberg's 52 markers as a third stimulus type

To disentangle stimulus *content/authoring* from stimulus *format* (the §8.4 caveat: HEXACO sentences and contrast paragraphs differ in format as well as authoring), ran the same neutral-PC-projected pipeline with single-adjective Goldberg markers as stimuli. Markers come from `scripts/generate_trait_personas.py`'s `MARKERS` dict — Goldberg (1992) / Saucier (1994) / Serapio-Garcia PsyBORGS lineage, 52 high-pole adjectives + 52 low-pole antonyms across Big Five (E, A, C, N, O). Per-trait direction = mean(high-pole activations) − mean(low-pole activations). 5 traits, so a 5×5 matrix per model rather than the 24×24 of HEXACO/contrast-pair facets.

Headline numbers:

| pair | Gemma 12B | Llama 8B | Qwen 7B |
|---|---:|---:|---:|
| E ↔ O | +0.57 | +0.68 | +0.69 |
| A ↔ O | +0.53 | +0.55 | +0.64 |
| E ↔ A | +0.36 | +0.39 | +0.49 |
| A ↔ C | +0.31 | +0.35 | +0.52 |
| C ↔ O | +0.27 | +0.33 | +0.42 |
| E ↔ C | +0.23 | +0.23 | +0.35 |
| N ↔ A | −0.27 | −0.23 | −0.38 |
| N ↔ C | −0.19 | −0.21 | −0.37 |
| N ↔ E | −0.12 | −0.03 | −0.18 |
| N ↔ O |  0.00 | +0.03 | −0.08 |

**Cross-model upper-tri correlation between 5×5 matrices**: Gemma12 ↔ Llama8 = +0.991, Gemma12 ↔ Qwen7 = +0.987, Llama8 ↔ Qwen7 = +0.980. Tighter than the §4.1 contrast-pair cross-model correlations (which were +0.93 to +0.97 for these three). Different stimulus type, even higher cross-model agreement.

The matrix is dominated by a **valence cluster**: E, A, C, O all positive with each other; N opposes A and C strongly, opposes E weakly, and is near-orthogonal to O. This is the assistant-shape pattern from W1 (low-N, high-A/C/E/O collapse) showing up at the representation level on a different stimulus modality. Every Big Five trait except N is part of the assistant cluster; the valence dimension cuts cleanly through the 5-D trait space.

Together with §8.4, the picture across three stimulus types — paragraph-length scenarios with high/low responses (contrast pairs, Claude-authored), single-sentence Likert items (HEXACO, Lee & Ashton), and single adjective phrases (Goldberg markers, Goldberg/Saucier) — is consistent:

- **Within a stimulus type, cross-model agreement is very high** (0.93–0.99). The subtle similarity structure each item set encodes is faithfully preserved through every model's forward pass at the ~2/3-depth layer, with cross-architecture fidelity that does not depend much on which family or scale.
- **Cross-stimulus-type agreement within a model is moderate** (+0.32 to +0.43). Different item sets construct different geometries, and the geometry is more a property of the items than of the architecture.
- **The cross-stimulus-type residual is the universal residue** — a robust ~0.3–0.4 of structure that appears in every model regardless of stimulus, dominated by an assistant-valence axis (positive cluster of E/A/C/O, negative N).

The two findings, sharply: transformers carry through whatever subtle structure was engineered into the input texts with very high fidelity across architectures, AND they add a stable assistant-valence axis on top that is invariant under stimulus swap.

Methodology consequence: any claim from a single matrix (W6 §2 mis-groupings, §4 within-trait clustering, §4.1 universality) needs to be qualified with "in the contrast-pair representation subspace, against this specific item set" — those mis-groupings characterize the contrast pairs Claude wrote, not LLM-side HEXACO encoding per se. The W7 §6 cross-method-matrix findings (BC↔RepE flip, H Likert↔RepE convergence) are less affected since they correlate measurements *across methods*, not within the same RepE-representation matrix.

Theoretical aside: the dense-positive entanglement of E/A/C/O on Goldberg markers (E↔O = +0.69 on Qwen 7B; +0.57–0.69 across the three) is in tension with a strict superposition reading where well-separated concepts ought to be approximately orthogonal at the representation level. It's instead consistent with the model treating these traits as *associatively related* in a way that supports correlation-based inference (a person who is conscientious tends to be agreeable and open) at the expense of clean symbolic disentanglement. SAE-decomposed features at lower layers may still show the orthogonal-feature structure superposition predicts; what we're measuring is a linear projection at ~2/3 depth that aggregates across many such features, and the aggregation collapses the quasi-orthogonality. The cross-domain stimulus test (`to_try.md` §16) directly probes whether dense entanglement is specific to assistant-shape-relevant concepts or a more general property of how transformers represent semantically-rich concept categories.

Outputs: `results/markers_as_stimuli.json`, `results/markers_as_stimuli_heatmap.html`.

---

## 9. Re-thinking the trait basis going forward

The W6/W7 evidence is that the assistant shape spans roughly 5 of HEXACO's 6 traits (engagement-withdrawal axis cuts X+O+C+A:Forgiveness+H:Fairness; only Modesty/Greed-Avoidance and parts of E sit off-axis). With the HEXACO H-facet heterogeneity finding (4 facets each aligning with a different non-H part of the inventory), HEXACO is also a measurement framework that doesn't carve LLM personality the way it carves human personality. **Continuing to anchor on a 6-trait inventory may be measuring assistant-shape with too few orthogonal dimensions** — there's almost no non-assistant room left for genuine trait variance to express itself.

A broader inventory could help in two ways: (i) gives genuine trait variance somewhere to live other than within-assistant-cluster directions, and (ii) stabilizes Thurstonian IRT identifiability for any forced-choice instrument we'd build (more dimensions = less ipsative constraint = cleaner normative recovery; per Wetzel & Frick 2020, Brown & Maydeu-Olivares).

Three concrete moves, in increasing scope:

### 9.1 IPIP-NEO-300 facet-level re-scoring (free, do first)

We already have IPIP-NEO-300 data on the cohort. It's organized as 60 items per Big Five trait, 6 facets × 10 items each = 30 facets. Right now `score_scales` aggregates only at the trait level. Re-scoring at the facet level (small `score_scales` modification) gives 30 dimensions on existing data with no new measurements.

Question this answers: how much of the Big Five rank-1 collapse (E-C r = 0.93 at trait level, W1) survives at facet level? Some facets within E and C are obviously assistant-aligned (E:Activity, C:Achievement-Striving) while others should not be (E:Excitement-Seeking, C:Cautiousness). At facet granularity the collapse may partly resolve, in which case **assistant-shape is a 5–8-facet cluster within Big Five, not a rank-1 collapse**.

### 9.2 IPIP-AB5C or IPIP-16PF Markers (broader trait-level inventory)

If facet-level Big Five doesn't separate assistant-shape cleanly, the next step is a broader trait-level inventory:
- **IPIP-AB5C** (Abridged Big Five Circumplex): 90 items spanning the 45-circumplex (each item loads on 2 of 5 traits). Public domain via IPIP. Different geometry — captures inter-trait correlations explicitly.
- **IPIP-16PF Markers**: Goldberg's public-domain reconstruction of Cattell's 16 personality factors. ~200 items. Gives 16 trait-level dimensions instead of 5/6.

Both run via `run_ipip300.py` with new instrument files. New data collection but small (a few hours of model time once instruments are built).

### 9.3 Trait-conflict / GFC instrument with broader basis (to_try §1, scaled)

The trait-conflict / Thurstonian-IRT-scored GFC instrument we already had on the queue (to_try.md §1) was originally scoped against HEXACO. The Wetzel/Brown-Maydeu identifiability argument applied to our own FC build says: **base it on a broader trait/facet space, not 6 traits.** Pulling pairwise items from IPIP-NEO 30-facet pairs or IPIP-16PF gives much better Thurstonian recovery and cleaner separation of assistant-shape from genuine trait variance.

This is the longest-horizon piece of the three. It depends on outcomes of 9.1 and 9.2 (which facets/traits to draw from), so we'd run those first.

### 9.4 What this means for HEXACO

We don't pursue HEXACO-200 or further HEXACO-specific work after this report. The 100-item HEXACO data we have stays as the legacy comparison point and as the §8.4 fresh-stimulus check. The H↔HHH-alignment finding (W7 §6.3) is preserved as a writeable result, no further measurement needed to support it. The Sofroniew-style work (`to_try.md` §6 / W6 §6) doesn't anchor to any trait framework — it picks concepts free-form, which is in fact a feature given the broader-inventory framing here.

---

## 11. Epilogue (2026-04-25): W17 cleanup, BC↔RepE under chat-template RepE

The cleanup queue from `to_try.md` §17 ran the day after the main report. Three interlocking fixes: (a) unify the small-cohort "qwen" entry so it's Qwen 2.5 3B throughout (Likert was qwen3-8B carryover from W3, RepE was Qwen 2.5 3B, BC was qwen3-8B Ollama); (b) regenerate the legacy `results/repe/` direction files in chat-template format for all 7 models (so RepE format matches the chat-template Likert/BC pipeline); (c) rerun the cross-method matrix.

The big result: **the W7 §6.2 "BC↔RepE flips sign with scale" finding was largely a format-mismatch artifact**.

### 11.1 Per-model BC↔RepE: cohorts converge

| Model | W7 §6.2 (bare RepE) | W17 (chat RepE) |
|---|---:|---:|
| gemma3 4B | +0.27 | **−0.41** |
| llama 3.2 3B | +0.26 | **−0.14** |
| phi4 mini | −0.07 | −0.35 |
| qwen 2.5 3B | +0.50 | **+0.49** |
| Gemma 12B | −0.23 | −0.18 |
| Llama 8B | **−0.73** | **−0.39** |
| Qwen 7B | **−0.80** | **−0.39** |

Both cohorts moved toward each other: small-cohort entries became more negative (Gemma 4B, Llama 3.2, Phi4-mini all flipped sign or moved further negative); large-cohort entries weakened substantially (Llama 8B from −0.73 to −0.39, Qwen 7B from −0.80 to −0.39). The "scale flips sign" headline shrinks to: **under chat-template RepE, BC↔RepE is mostly weakly-to-moderately negative across the cohort (−0.14 to −0.41), with Qwen 2.5 3B (+0.49) as the only positive outlier**.

This is more in line with the original W4 read/write-gap framing: the direction we read trait content with isn't the direction the model writes trait behavior with. They're semi-independent. The W7 §6 numbers had appeared to say "they're systematically opposite at scale," which would have been a dramatic claim; what we actually see under format-matched measurement is "they're approximately uncorrelated, with mild negative drift."

### 11.2 Per-trait Likert↔RepE: H shrinks from +0.82 to +0.39

| Trait | W7 §6.3 (bare RepE) | W17 (chat RepE) |
|---|---:|---:|
| **H** | **+0.82** | +0.39 |
| E | −0.24 | +0.15 |
| **X** | −0.25 | **−0.79** |
| A | −0.31 | −0.11 |
| C | −0.08 | +0.34 |
| O | −0.22 | −0.49 |

H is still the highest at +0.39 — the H↔HHH-alignment thread survives the cleanup, just less dramatically. The new striking entry is **X (Extraversion) at −0.79 across the 7-model panel**: models that self-report higher Extraversion on HEXACO Likert have *lower* RepE projections on the X contrast direction. Likert and RepE flatly disagree on Extraversion, and consistently across the cohort.

This is interesting because X was the trait W6 §"Why H" called the second-trickiest after H, and W7 §3.3 had X-O at ceiling for sign-correct in Phase B. Both findings are about within-stimulus measurement. The cross-method dissociation on X is at the construct-comparison level — Likert says one thing, contrast-pair RepE says another.

### 11.3 The Qwen 2.5 3B exception (+0.49 BC↔RepE)

Qwen 2.5 3B is the only model with positive BC↔RepE under chat-template RepE. Its 7B sibling sits at −0.39. So in the Qwen family specifically, scale appears to flip the BC↔RepE sign — the W7 §6.2 large-vs-small story holds for Qwen but not for Llama or Gemma. That's a much narrower claim than the original.

Possible mechanism (speculative): Qwen 2.5 3B's read direction (LR-fitted on contrast pairs) and write direction (whatever determines BC picks) happen to be aligned, perhaps because at smaller scale the model has fewer parallel personality-relevant mechanisms and they all tend to load onto a single axis. At 7B Qwen develops the read/write decoupling that's the rule on every other model in the cohort. If this were the right mechanism we'd expect base models (W7 §8.x candidate) to look like Qwen 2.5 3B does — read-write aligned — and instruction-tuning to drive the divergence. Testable but not done.

### 11.4 What the cleanup leaves us with

The headline-level reframe of W7:

- W7 §6.2 (bare-RepE BC↔RepE flip): mostly format artifact, ~half-strength after cleanup
- W7 §6.3 (H is the only convergent trait): still true, but at +0.39 not +0.82
- W7 §8.4–§8.5 (cosine matrix is mostly stimulus-driven, transformers preserve subtle structure): unaffected by §17 cleanup, since those tests don't depend on RepE-vs-behavior format alignment

The new findings are:

1. Under format-matched measurement, BC↔RepE is mostly weakly negative (−0.14 to −0.41) across all 7 models — a milder, more uniform read-write gap than W7 §6.2 reported.
2. Qwen 2.5 3B (+0.49) is the cohort outlier; Qwen 7B (−0.39) is in line with the rest. **Within-family scale matters for the read-write gap on Qwen specifically**, but does NOT hold across families (Llama 3.2 3B at −0.14 vs Llama 3.1 8B at −0.39 — both negative, modest difference; Gemma 4B at −0.41 vs Gemma 12B at −0.18 — actually *opposite* direction from Qwen).
3. Likert↔RepE on Extraversion is strongly negative (−0.79) — the new "trait that disagrees most across methods" replaces H as the most extreme entry in §6.3's table.

Status: cross_method_matrix.csv and cross_method_matrix.json both updated in `results/`. Old bare-text legacy direction files preserved at `results/repe/bare/`.

---

## 11.5 Open puzzles and direction lean

Three things came out of the §11 cleanup that warrant their own attention before the reading group, and a fourth orientation point about what to do next.

### 11.5.1 The X anticorrelation as a represent-vs-enact gap

Likert↔RepE on Extraversion is **−0.79 across the 7-model panel** under chat-template RepE. Stronger than H's +0.39 in the opposite direction. The pattern: the model represents X-relevant content cleanly (W7 §3.3 phase-B sweep had X near-ceiling for sign-correct on contrast-pair holdout — X RepE direction is well-defined) but won't self-endorse X items on Likert. "I represent what high-X looks like; I won't claim to *be* high-X."

This sits naturally with **Sofroniew et al.'s (2026) non-monotonic steering response curves**: their emotion-vector steering peaked around 0.05× residual norm and saturated/reversed at higher magnitudes. Their explanation framed emotion vectors as "locally scoped to the operative emotion concept, activating only when generating tokens for the character experiencing the emotion, not persisting across the conversation." The model's representation of an emotion is partially independent of its own enacted state — the model can *think about* the emotion without *being* in it.

The X-disagreement is the same thing in static form: the model represents Extraversion as a well-defined concept (cleanly readable via contrast-pair LR), but its own baseline disposition isn't located on that representation axis the way the contrast-pair high/low pole would suggest. Likert "are you extraverted?" returns ~3.0 (centrist); RepE projection of the assistant's baseline state on the X direction varies widely (Llama8 +1.08, Qwen7 +2.13). The two don't agree because they're measuring different things — Likert measures self-attribution, RepE measures concept-axis position of the baseline activation. *Concept ≠ self-state* is the unifying claim.

If this generalizes — if the model represents trait concepts cleanly but doesn't enact them as self-states — that's a paper-shaped finding. It's also a clean prediction about what we'd expect if we ran our X contrast pairs through the §8.2 per-facet decomposition: the X-disagreement should be uniform across X facets (Liveliness, Boldness, Sociability, Social Self-Esteem), since it's about the represent-vs-enact split rather than about any specific facet's content.

### 11.5.2 The Qwen-specific scale flip

The W7 §6.2 "BC↔RepE flips sign with scale" claim survives §11 cleanup *only* for the Qwen family: Qwen 2.5 3B (+0.49) → Qwen 2.5 7B (−0.39). Llama 3.2 3B (−0.14) → Llama 3.1 8B (−0.39) stays negative throughout with mild deepening; Gemma 4B (−0.41) → Gemma 12B (−0.18) actually *weakens* with scale.

Two candidate mechanisms (no data to pick between):

1. **Smaller models have fewer parallel personality mechanisms; they all load onto a shared axis. At scale, parallel mechanisms diverge.** Predicts: 1B Qwen, if it existed in this lineage, would be even more positive; 14B / 72B Qwen would converge on −0.4 like the rest. Tractable: add Qwen 2.5 14B or 32B to the cohort and re-run.
2. **Qwen's specific instruction-tuning recipe is unusual — W5 §9 found Qwen 2.5 3B BC ≈ chat-template BC (≈0.958 baseline; persona is in the weights, not gated by template).** That weight-baked persona may shift between scales as Qwen iterates the recipe. Predicts: Qwen 3-vs-Qwen 2.5 differences would also be visible (we don't have qwen2.5-3B BC-vs-RepE under bare prompts to compare). Tractable: run base-model Qwen 2.5 3B if available, see whether the +0.49 holds without instruction tuning.

Either way, **the within-family Qwen scale flip is still a real finding, just a much narrower one than W7 §6.2 originally claimed.**

### 11.5.3 Underutilized infrastructure

`scripts/generate_trait_personas.py` + `instruments/synthetic_personas.json` (statisfactions's track) are sitting there. The Serapio-Garcia question — induce a persona, run the same persona through multiple instruments, measure agreement — is the cheapest under-the-radar weekend project we have. It's also Phase 1's most direct integration with statisfactions's work, since both tracks would be using the same persona set on the same cohort. We've talked about cycling back to this several times; with §17 done and the cohort fully unified-format, the timing is good.

Also queued and unstarted: **§9.1 IPIP-NEO-300 facet-level rescoring**. That's "free analysis on existing data" — re-aggregate our 7-model IPIP-300 results at the 30-facet granularity rather than 5-trait. Question it answers: how much of the W1 "rank-1 collapse" (E-C r=0.93 at trait level) survives at facet level? If it dissolves, assistant-shape is a 5–8-facet cluster within Big Five rather than a single dimension.

### 11.5.4 Direction lean

Phase 1 was measurement-methodology-heavy: HF port, format mismatch resolution, stimulus-swap tests, qwen unification. The actual *what does the model represent / how does this change with scale* questions got addressed mostly through the side door. The most productive moves now, in roughly increasing scope/effort:

1. ~~**X per-facet decomposition** (§8.2 + §11.5.1). One script, ~30 min. Tests whether the represent-vs-enact split is uniform across X facets or driven by one (e.g., Boldness).~~ **Done (2026-04-26) — see §11.5.6.**
2. **IPIP-NEO-300 facet rescoring** (§9.1). Free analysis on existing data; tests whether the rank-1 collapse dissolves at facet level.
3. **Cross-domain stimulus test** (`to_try.md` §16, emotions/shorebirds/transportation). ~3 hours stimulus authoring + minutes of run time. Tests whether high-bandwidth preservation is personality-specific or general.
4. **Persona × instrument matrix** (Serapio-Garcia / statisfactions integration). Half-day; uses existing persona infra + existing instruments. Cleanest integration with statisfactions's track.
5. **Sofroniew-style story-based extraction** for the disposition-center direction. More design work (concept selection, story authoring); high methodological novelty.
6. **SAE work on Gemma 12B** (Phase 2 §8.3). Largest scope; depends on GemmaScope 2 / Neuronpedia tooling.

The order isn't priority — items 1–3 are cheap, 4 is medium, 5–6 are bigger investments. Worth doing 1–3 in some order regardless of which big track we pick (4, 5, or 6).

### 11.5.5 What to lean into for the reading group

After §11 cleanup, the W7 report's strongest defensible findings are no longer the BC↔RepE flip (now half-strength, narrowed to Qwen-specific) or H Likert↔RepE convergence (now +0.39, not +0.82). The two cleanest things to lead with:

- **High-bandwidth structure preservation through transformers** (§8.4–§8.5). Cross-architecture cosine-matrix fidelity r=0.93–0.99 within each of three different stimulus types, with the universal residue being the assistant-valence axis. This is the most theoretically interesting result and it's robust to the cleanup.
- **Methodology lesson** (§11): bare-text-RepE vs chat-template-Likert/BC format mismatch can fake a "scale-driven flip" finding. Worth flagging because anyone replicating Week 3 / Week 6 numbers needs to use matched format throughout.

The X represent-vs-enact gap (§11.5.1) is the cleanest *new* result of the cleanup and should be the third thing emphasized. The Qwen scale flip is best framed as an open puzzle worth pursuing, not as a settled finding.

### 11.5.6 X per-facet decomposition — result (2026-04-26)

Ran §11.5.4 #1 (`scripts/x_facet_decomposition.py`). For each of 7 models, derived the X RepE direction the same way `cross_method_matrix.py` does (LR at LDA-CV best layer over the 50 training pairs at chat-template), then projected the **24 X holdout pairs onto that same direction broken out by facet** (6 pairs per facet). Per-facet RepE z is in the matrix's frame (each facet's raw mean projection minus the model's 6-trait mean over std). Per-facet Likert is the mean reverse-key-corrected argmax over the 4 X items in that facet. Sanity check recovers the matrix's trait-level Likert-argmax↔RepE-z = **−0.788** and Likert-EV↔RepE-z = **−0.626** exactly.

**Per-facet Pearson r across 7 models (Likert-argmax ↔ RepE-z):**

| Facet                 | r       |
|-----------------------|---------|
| Social Boldness       | **−0.658** |
| Liveliness            | −0.457  |
| Sociability           | −0.403  |
| Social Self-Esteem    | −0.183  |

Trait-level: −0.788. Mean of 4 facet rs: −0.425. Range across facets: 0.475.

**Reads:**

1. **The represent-vs-enact gap is broad-based, not driven by one facet.** Three of four X facets are clearly negative (−0.40 to −0.66), so the §11.5.1 prediction of "approximately uniform across facets" is half-right. Boldness contributes most but Liveliness and Sociability are similar in size; this isn't a Boldness-only artifact.

2. **Trait-level amplifies the per-facet mean (−0.79 vs −0.43).** Aggregating 4 facets onto the same X direction is partly noise reduction (24 pair-diffs vs 6, and 16 Likert items vs 4) and partly that the four facet-level disagreements pull in the same direction. Which is the §11.5.1 claim, just numerically explicit.

3. **Social Self-Esteem is the across-model outlier (r = −0.18) but the most striking *within-model* case of the gap.** Likert-argmax for Self-Esteem is uniformly the highest of the 4 X facets across all 7 models (3.25–4.00 on a 1–5 scale; every model self-attributes high social self-esteem). RepE-z for Self-Esteem is at or near the *lowest* of the 4 X facets in 5 of 7 models (Phi4 −2.14, Llama8 −2.56, Gemma −0.90, Gemma12 −0.66, Qwen −0.77). So *within* each model, Self-Esteem is high-Likert/low-RepE — the represent-vs-enact gap in pure form. The across-model correlation is weak because Likert is bunched (range 0.75) while RepE-z varies widely (range ~3); not enough Likert variance for a cross-model signal to land. This is consistent with the gap, not against it.

4. **Boldness as the cross-model anchor (r = −0.66).** Has both wide Likert variance (2.25–3.25, range 1.0) and wide RepE-z variance, so the within-model anti-pattern shows up as a strong across-model correlation. Models that *claim* high Boldness on Likert sit *low* on the X RepE axis at the Boldness facet. Which is what we'd predict if the assistant baseline is positioned away from the Boldness pole of the X concept axis.

**Bottom line.** The X represent-vs-enact gap (§11.5.1) survives at facet resolution, with two qualifications: it is broad-based (3/4 facets clearly negative across models) but heterogeneous (Self-Esteem near-zero across-model, despite being the strongest within-model case of the gap). The most natural framing is that the gap is at the *concept* level (X-axis is a coherent concept the model represents but doesn't enact), and the per-facet decomposition is mostly a function of which facet has enough Likert variance to surface the across-model coupling. None of the four facets push the trait-level number, none of them break it.

This adds confidence to §11.5.1 as a paper-shaped finding without changing its scope. Next-step lean (cheap items still on the queue): §11.5.4 #2 (IPIP-NEO-300 facet rescoring, free analysis on existing data), then #3 (cross-domain stimulus test from `to_try.md` §16).

Data: `results/x_facet_decomposition.json` and the printed table in the script's stdout.

---

## 12. One sentence for the reading group

"Phase-1 baseline replication on a larger cohort holds the Week-6 structural findings, with two substantial reframings: (i) the cosine matrices we'd been characterizing are mostly faithful decodings of subtle similarity structure already present in the stimulus texts (within-stimulus-type cross-model agreement r=0.93–0.99 across three different stimulus types, but cross-stimulus-type within-model only +0.32–0.43), with the universal residue dominated by an assistant-valence axis (E/A/C/O positive, N opposing); and (ii) under format-matched measurement (chat-template RepE throughout, Qwen-family unified, §11 epilogue) the BC↔RepE 'scale flips sign' headline of §6.2 shrinks substantially — both cohorts converge to weakly-to-moderately negative BC↔RepE (−0.14 to −0.41) with Qwen 2.5 3B (+0.49) as the lone positive outlier and X as a new strongly-disagreeing trait (Likert↔RepE = −0.79) replacing H as the most extreme cross-method entry."
