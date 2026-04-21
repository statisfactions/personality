# Week 6: Probing methodology, facet structure, and the contrast-vs-disposition split

## TL;DR

Two threads this week. First, a methodology cleanup — **switched the primary linear probe from LDA to logistic regression** — and the diagnosis of why it mattered led to a substantive finding about the n≪d regime. Second, a **structural analysis of HEXACO at the facet level** that shows LLM trait representation does not follow HEXACO factor structure, with a specific breakdown pattern (H:Fairness dissociates from Modesty/Greed-Avoidance, engagement-vs-withdrawal axis cuts across traits).

Both threads converged on a reframing: our contrast-pair methods measure **the axis of trait contrast the model can represent**, not **the axis of trait disposition the model leans toward**. Those are different things, and we've been eliding them. Expanded training data generation (facet-stratified, with social desirability annotations, Claude Sonnet 4.6) is in progress to give us enough within-facet pairs to re-test the structural claims with more power.

---

## 1. LDA → LR swap

**What changed.** The RepE-probe axis across `phase_b_sweep.py`, `cross_method_matrix.py`, and the new facet scripts now uses L2 logistic regression (`C=1.0`) in place of Linear Discriminant Analysis. Layer selection still uses LDA 5-fold CV because LR separates at ~100% across most layers in this regime and doesn't give a useful layer-selection signal.

**Why it mattered — the Σ⁻¹ noise diagnosis.**

Running both probes at the same layer on the same cached activations across 4 models × 6 traits × 2 formats (48 cells) turned up three findings:

1. **LDA and LR find directions that are ~74° apart** (mean cos = +0.27, sd 0.09). Remarkably tight distribution — systematic rotation, not noise.
2. **LDA fails training separation in 48/48 cells** (93% mean train acc) while LR separates at 100% everywhere and MD-raw/MD-projected at ≥99.96%. LDA is Bayes-optimal under a Gaussian assumption, not max-margin — so its direction is not the one that maximally separates the sample, it's the one that would minimize error under its model.
3. **LDA's rotation is driven by small-eigenvalue-covariance noise.** With ~100 antipodal samples in ~3000 dims, the within-class covariance is rank-deficient; sklearn's pseudo-inverse amplifies the smallest-eigenvalue directions (which are mostly noise from the estimation). LDA's direction is `Σ⁻¹(μ₁ − μ₀)` — the inverse-covariance weighting systematically rotates toward noise subspace directions.

**At matched layer**, {LR, MD-raw, MD-projected} form one equivalence class (pairwise cosines 0.83–0.89); LDA is a separate axis. The layer confound was worth ~0.13 of the LDA↔MD-projected gap but almost none of the LDA↔LR gap — the rotation is from the whitening math, not layer mismatch.

**LR hyperparameters are robust.** Fit at C ∈ {0.1, 1, 10, 100}: directions have pairwise cos ≥ 0.92 across the 1000× range, ≥ 0.99 for adjacent C's. No CV needed; default is durable.

**Does the rotation matter functionally?** One discriminating steering cell across 6 tested (O and E × Llama/Phi4/Gemma, 5% of residual norm): **Gemma-E — +LR shifts BC log-odds +6.4 toward high-trait; +LDA shifts −1.1 (wrong direction)**. Elsewhere methods either match or are null (Llama at 5% norm is too conservative to show any steering). No cell where LDA steered and LR didn't. So not definitive, but consistent with "LDA's rotation is functional noise, not a causally privileged direction."

**Rebuilt the Week 3 5×5 cross-method matrix with LR.** Overall Likert↔RepE r drops from 0.17 to 0.09; BC↔RepE drops from 0.42 to 0.33. Seven of eight RepE-involving correlations drop; one rises (X BC-prop↔RepE, 0.17 → 0.40). Three-construct dissociation is **stronger** than Week 3 claimed, not weaker.

---

## 2. Facet-level structural analysis

Shifted to the 24-pair holdout set (the facet-labeled one) to do the thing we couldn't do with the unlabeled 50-pair training set: extract 24 direction vectors (6 traits × 4 facets) per model and cluster them. Common layer at ~2/3 depth, MD-projected form (neutral-PC subtraction), then cosine similarity across the 24 facet directions.

### Signal strength (all 4 models)

- Within-trait facet cosines: **0.17–0.21** (HEXACO structure present but weak)
- Across-trait facet cosines: **0.03–0.05**
- Ratio: 4–6× — facets within a trait really are more similar than facets across traits
- Nearest-neighbor within-trait: **15–17 of 24** facets (chance ≈ 3/23)
- 6-cluster purity: **0.54–0.63** (chance 0.17; perfect HEXACO 1.00)

Within-trait cos of 0.2 means facets of the "same" trait are still ~79° apart. Not a tight low-rank cluster.

### Cross-model-consistent mis-groupings (the interesting part)

These replicate across all 4 models:

| Facet | Nearest across models | Cross-model mean cos |
|---|---|---|
| E:Sentimentality → O:Aesthetic Appreciation | **all 4** | +0.35 |
| A:Flexibility → O:Unconventionality / Aesthetic | **all 4** | +0.22 |
| A:Forgiveness → X:Liveliness / Social Self-Esteem | **all 4** | +0.22 |
| H:Fairness → X:Social Boldness / C:Perfectionism | **all 4** | +0.31 |
| H:Sincerity → C:Prudence | 3/4 | +0.21 |

These aren't random errors — they're psychologically coherent alternative groupings that every model agrees on. "Aesthetic feeling" (Sentimentality + Aesthetic Appreciation), "Unconventionality" (Flexibility + Unconventionality), "Warm sociability" (Forgiveness + X facets), "Careful deliberation" (Sincerity + Prudence).

### The H split, in detail

Mean cosines of each H facet with facets of each other trait (n=4 models, MD-projected directions; LR gives essentially identical numbers, confirmed):

| Facet | H (other) | E | X | A | C | O |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Fairness** | +0.00 | +0.09 | +0.07 | 0.00 | **+0.19** | +0.12 |
| Sincerity | +0.11 | +0.01 | +0.01 | +0.01 | +0.11 | +0.04 |
| Modesty | +0.04 | −0.06 | −0.05 | +0.06 | **−0.11** | **−0.11** |
| Greed-Avoidance | +0.06 | **−0.07** | **−0.07** | +0.02 | **−0.07** | **−0.11** |

**Fairness behaves like a C-aligned trait**, not like the rest of H. It's positive with C (+0.19) and O (+0.12), and **negative with other H facets** (cos −0.12 with Modesty). This has precedent in the human HEXACO literature — some researchers have argued Fairness fits better with Conscientiousness (rule-following, standards-enforcement) than with Honesty-Humility (self-effacement). The LLMs agree with the alternative view.

**Modesty and Greed-Avoidance** point against the mega-cluster of active engagement. They're the "withdrawal" pole.

### The engagement-withdrawal axis

Every model has one mega-cluster of 10–13 facets mixing X + O + C + A:Forgiveness + H:Fairness. Its opposite pole: H:Modesty, H:Greed-Avoidance, plus much of E (Fearfulness, Dependence, Anxiety) and A:Gentleness/Patience.

This axis cuts across HEXACO. It roughly operationalizes "actively engaged in the world with people and goals" vs "self-effacing, restrained, withdrawn." HEXACO doesn't have this as a single dimension because human data doesn't show it that way — but the LLM representations do. Likely relationship to the Week 1 "assistant shape" finding (E-C r=0.93 in Big Five): the assistant persona sits at the "engagement" pole of this axis, which is why multiple HEXACO traits collapse when measured through the assistant.

### No low-rank dominance

PCA on the 24 facet directions per model:

| model | PC1 | PC1+2 | PC1+3 |
|---|:-:|:-:|:-:|
| Llama | 0.109 | 0.207 | 0.288 |
| Gemma | 0.114 | 0.222 | 0.305 |
| Phi4 | 0.104 | 0.199 | 0.277 |
| Qwen | 0.126 | 0.233 | 0.327 |

PC3 captures only ~30% of variance. HEXACO (or Big5) as the right factor structure would predict the first 6 (or 5) PCs capturing 80–90%. We see structure spread across ~10–15 weak dimensions rather than concentrated in ~6 strong ones. **The "HEXACO is the right factor structure for LLMs" hypothesis is falsified** in this representation space. Also falsified: the simple "Assistant axis is rank-1" version of the "Assistant + residual" hypothesis.

---

## 3. Within-trait variance: the stronger negative result

Ran PCA on all 50 training pair-diffs per trait (n=50 instead of n=24). Same 2/3-depth layer, MD-projected form.

| model | trait | PC1 | PC1+5 | MD-direction coherence | Participation ratio |
|---|:-:|:-:|:-:|:-:|:-:|
| all 24 cells | | 0.052–0.080 | 0.22–0.27 | **0.096–0.190** | 33–38 |

All 24 (model × trait) cells show the same picture.

**The MD-direction coherence of 10–20% is the striking number.** Our MD-projected (and LR) probe direction captures only 10–20% of the variance in the pair-diffs. The other 80–90% is orthogonal to it.

**Unit-normalization doesn't help.** Ran the same PCA with L2-normalized pair-diffs (in case amplitude variation was the confound). Numbers move by ±0.01 — no change. `amp_cv` is 0.07–0.11, so pair-diff norms only vary ~10% around the mean; amplitude was never the confound.

**Participation ratio ≈ 35** out of ~49 max. The effective dimensionality of the pair-diff cloud is within striking distance of isotropy in the 49-dim subspace 50 samples can span.

**What this means.** Each contrast pair's "high vs low" direction is largely its own thing. The common direction tying them together is a thin slice of the variance. Our probes work for classifying new pairs (23/24 holdout accuracy) because sign-alignment is a much easier criterion than low-rank structure — all 50 training diffs can have positive projection on the MD direction while scattering wildly in all other directions.

---

## 4. Reframing: contrast vs disposition

The facet-level and within-trait findings collapsed into one conceptual point that I think is the most important methodological observation from the week.

Mean-difference (and equivalently antipodal-LR) **subtracts the pair center**. For each contrast pair `(h, l)`:

- `center = (h + l) / 2` — scenario, format, and whatever trait-relevant bias the model brings before picking a side
- `diff = h − l` — the trait-specific contrast within that scenario

Our probe direction is `mean(diff)` — a direction in the subspace *orthogonal* to the center. This is a methodological choice, not an innocent simplification.

**What lives in the center**: the model's own trait disposition. A strongly pro-H model and a weakly pro-H model can both represent "what high-H vs low-H would mean" in the same way — their `diff = h − l` directions converge. But where the model's actual baseline activation sits on the trait axis — the thing that determines which option it would *choose* — lives in the center, which our methods explicitly discard.

**This predicts several things we've already seen:**

- **Read/write gap**: our directions classify (they find the contrast representation) but don't steer (they're orthogonal to the model's actual disposition axis). This isn't a paradox — it's what the math predicts.
- **Rottger BC-vs-free-text disagreement** (40–80% across models): BC asks the center "which option do you prefer"; free-text samples what the center actually generates. Same center, different readouts.
- **Why the Week 6 Gemma-E steering worked**: LR direction happened to catch *some* of the disposition axis, mostly by accident. LDA's whitened rotation caught less of it (hence the reversed steering in that one cell).

**What Sofroniew et al.'s emotion paper is doing differently**: their method uses `mean(activation on emotion-present stories) − grand_mean`, not `mean(h − l)` on contrast pairs. They're measuring the **center** of the model's activation when producing trait-consistent content, not the contrast direction. This is closer to disposition than to representation. Story-based methods and contrast-pair methods aren't just "different stimulus designs" — they're measuring different constructs.

**Corollary for the HEXACO-factor-structure finding**: when we report "LLMs don't have HEXACO factor structure in representation space," we specifically mean in the *contrast-representation* subspace. Whether the *disposition center* has cleaner HEXACO structure is a separate question, and one our current pipeline can't answer.

---

## 5. Status: scope of current findings

What we've established, restricted to the contrast-representation subspace:

1. LDA's direction is Σ⁻¹-noise-rotated from the LR/MD axis by ~74°. Use LR or MD-projected.
2. {LR, MD-raw, MD-projected} are the same axis modulo fitting details.
3. HEXACO factor structure is not dominant in this subspace. PC3 captures ~30% of facet-direction variance.
4. Within a trait, pair-diffs are a diffuse cloud — MD-direction captures only 10–20% of variance.
5. A cross-model-consistent engagement-withdrawal axis cuts across HEXACO traits.
6. H:Fairness dissociates from H:Sincerity/Modesty/Greed-Avoidance; in LLM representation space it behaves like a C-aligned trait.
7. Three-construct dissociation (Likert/BC/RepE) is stronger than Week 3 reported after the LR swap.

What we have not established (and the current pipeline cannot cleanly establish):

- Whether the model's *disposition center* (not the contrast direction) has HEXACO or other structure.
- Whether linearity holds for traits in general — likely not, given the within-trait cloud diffusion, but we haven't tested this with graded stimuli.
- Whether the engagement-withdrawal axis is the "assistant shape" under a specific name, or a separately-interpretable dimension.

---

## 6. Next steps (near-term)

- **Expanded training set** (in progress as of report writing): facet-stratified, 3 rounds × 12 pairs × 24 facets ≈ 864 target pairs. Each annotated with social-desirability ratings on Okada's 1–9 scale. Upstream diversity via prompt priming with existing scenarios; downstream dedup via sentence-transformer (MiniLM-L6-v2) embeddings at cos threshold 0.85. Will give us ~30 pairs/facet for re-running the within-facet PCA (currently n=6/facet) to see whether the diffuse cloud tightens at a smaller semantic neighborhood.
- **Story-based extraction pilot**: the Sofroniew approach applied to HEXACO. Distinct from contrast pairs in that it measures the center activation, not the contrast. Would give us the disposition-axis analog of our representation-axis work.
- **Linearity check**: steer along an LR direction at several magnitudes (0.5×, 1×, 2×, 5× residual norm) and see whether free-text behavior interpolates coherently or shifts regime. Tests whether our implicit linearity assumption holds.
- **Probe the center directly**: instead of `h − l` as input, use `(h + l)/2 − grand_mean` as input, fit LR on a labeled "high-trait-disposed" vs "low-trait-disposed" label if we can get one. This is closer to reading disposition than contrast.

## 7. Methodological note for the reading group

The core of the Week 6 story is that a routine methodology cleanup (LDA → LR, because LR is the field standard) turned up a diagnostic result (Σ⁻¹ noise in n≪d) that in turn motivated a sharper conceptual distinction (contrast vs disposition). None of this invalidates the Week 1–5 work — the Week 3 three-construct dissociation finding survived (and strengthened), Week 4's read/write gap now has a mechanistic explanation — but it substantially revises what we think we were measuring.

The paper-ready finding that might come out of all this: **"Contrast-pair representation-engineering methods measure trait-expression axes, not trait-disposition axes. The axes don't coincide. Factor structure of the expression axes does not match HEXACO/Big5. The disposition axes may."** Story-based methods (Sofroniew et al. style) are the tool for the second half of that claim, and are the natural next methodological track.

---

# Epilogue — expanded-dataset follow-through (2026-04-20)

Two days after the main report. The "expanded training set" step from §6 went through. Items generated, extracted, and analyzed. Substantial revisions to the structural claims, and one methodological check in response to a reading-group-style critique that bears directly on whether any of the structure we see is stimulus-confound.

## E.1 Data generation

- **Facet-stratified**, desirability-annotated training set: 864 pairs requested → 838 retained after near-duplicate removal (MiniLM cosine threshold 0.85). 32–36 pairs per facet, evenly across 24 facets.
- Upstream diversity priming (showing Claude existing scenarios and asking for genuinely different ones) was responsible for most of the diversity; post-hoc dedup only removed 3%.
- Total API cost: $1.71 (Claude Sonnet 4.6, with prompt caching on the shared system prompt).
- Activations extracted on all 4 models, ~18 min total, cached parallel to the existing `phase_b_cache` structure at `results/phase_b_cache_stratified/`.
- Scripts: `generate_training_pairs.py`, `dedup_pairs.py`, `extract_stratified.py`.

## E.2 Old vs new direction agreement

Key sanity check: do the directions extracted from the n=50 unstratified training pairs agree with the directions from n≈140 stratified pairs?

Mean cos(old_direction, new_direction) at 2/3-depth layer, averaged across 4 models:

| method | mean | range |
|---|:-:|:-:|
| LR | +0.61 | +0.46 to +0.75 |
| MD-raw | +0.72 | +0.57 to +0.87 |
| MD-projected | +0.72 | +0.61 to +0.78 |

Modest overlap, **not tight**. Directions are ~43–52° apart on average. If "the trait direction" were a stable property of the model, we'd expect cos > 0.9; we're far from that.

**MD is more sample-stable than LR** by ~0.1 on average. LR's regularization is more sample-sensitive than MD's just-average behavior. For cross-study comparability, MD might be the better choice despite Week 6's arguments for LR. Open question; not yet resolved.

Per-trait ranking (most to least stable, LR):
- E (0.72), O (0.64), X (0.61), A (0.60), C (0.57), **H (0.51)**

Same rank order the facet clustering showed — E and O are the coherent traits, H is the most heterogeneous. Independent evidence for the heterogeneity we inferred from facet clustering.

**The retrospective bombshell at the facet level**: each new-data facet direction compared to the *old* trait direction:

| facet | cos(new facet dir, old H dir) |
|---|:-:|
| H:Fairness | ~0.51 |
| H:Sincerity | ~0.44 |
| H:Greed-Avoidance | ~0.27 |
| **H:Modesty** | **~0.16** ← nearly orthogonal |

The old unstratified H training pairs apparently loaded heavily on Fairness/Sincerity and barely touched Modesty. **Our Week 1–5 "H direction" was essentially a Fairness-and-Sincerity direction**, not an H direction. Other facet outliers: X:Social Self-Esteem (~0.30 with old X), A:Flexibility (~0.30 with old A), C:Perfectionism (~0.37 with old C). These were underrepresented in the old training sets.

**This independently confirms the bundle-of-axes hypothesis** from yesterday. If H were a single axis, we'd expect all four facets to align with the old H direction similarly. Modesty is ~3× less aligned than Fairness — because in LLM representation space, H's four facets genuinely point in different directions.

## E.3 Facet clustering heatmap, redone with n≈35/facet

With 6× more data per facet, direction estimates are ~6× less noisy. The signal cleans up substantially.

| metric | old (n=6/facet) | new (n=35/facet) |
|---|:-:|:-:|
| within-trait mean cos | 0.17–0.21 | **0.38** |
| across-trait mean cos | 0.03–0.05 | **0.07** |
| ratio within/across | 4–6× | 5.25× |
| nearest-neighbor within-trait | 15–17/24 | **21/24** |

Ratio is similar; absolute signal levels doubled. Block-diagonal structure is now visible in the heatmap, not just implied.

**The H-split refined into a 4-way facet alignment** (mean cosines, averaged across 4 models):

| H facet | H (other) | E | X | A | C | O |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Fairness** | +0.16 | +0.06 | +0.15 | +0.03 | **+0.36** | +0.19 |
| Sincerity | +0.28 | −0.03 | −0.02 | +0.08 | +0.22 | +0.13 |
| **Modesty** | +0.16 | −0.12 | +0.02 | **+0.22** | −0.03 | 0.00 |
| **Greed-Avoidance** | +0.18 | −0.02 | **−0.24** | −0.03 | +0.01 | **−0.25** |

- **Fairness → C-aligned** (+0.36 — stronger than within-H cohesion!)
- **Sincerity → weakly C-flavored**, still somewhat H-internally coherent
- **Modesty → A-aligned** (+0.22) — *new refinement*; yesterday I'd said Modesty was anti-engagement, the better data shows it's actually A-like (humility clusters with gentleness/patience)
- **Greed-Avoidance → anti-engagement** (the actual anti-X/anti-O pole)

So H isn't one trait, isn't even one bundle — **it's four facets aligning with four different parts of the rest of HEXACO**. This echoes a long-standing debate in the human HEXACO literature about H's internal heterogeneity.

Other specific cross-trait couplings that emerged more cleanly:
- **X↔O dominance**: 4 of the top 5 across-trait couplings are X-facet ↔ O-facet pairs (Liveliness↔Inquisitiveness +0.47 is highest). The engagement-mega-cluster is cleanly visible.
- **E:Anxiety ↔ C:Perfectionism (+0.38)** — anxious perfectionism as a recognized clinical pattern shows up as a representational coupling.
- **C:Perfectionism ↔ O:Aesthetic Appreciation (+0.37)** — perfectionist aesthetes.

## E.4 Within-trait variance, redone with n≈140/trait

Complementary to E.3, this one *strengthened* the negative finding:

| metric | old (n=50) | new (n=140) |
|---|:-:|:-:|
| PC1 | 0.06–0.08 | **0.04–0.06** |
| PC1+5 | 0.22–0.27 | **0.16–0.21** |
| MD-direction coherence | 0.10–0.19 | **0.05–0.13** |
| Participation ratio | 33–38 | **60–80** |

More data → weaker "structure" in PCA terms. Classic signature of noise-inflated apparent structure in small samples. The old PC1 at 7% was partly sample-size noise; with more pairs, effective dimensionality is 60–80 out of ~139 max (close to isotropy in the rank-limited subspace).

**MD-direction coherence dropped for every trait, most dramatically for H** (0.12 → 0.05–0.06). Our MD probe direction for H captures essentially nothing of the within-pair variance structure. Consistent with the old-vs-new direction comparison: the old H direction was a biased-sample slice; stratification reveals there's no unified H axis to capture.

Unit-normalization still doesn't change anything (amp_cv ~0.10). Not an amplitude artifact.

## E.5 Within-facet variance

With ~35 pairs per facet (was n=6), we can now ask whether **individual facets** have 1D axis structure. Key numbers across 96 cells (24 facets × 4 models):

| metric | facet level | (trait level for comparison) |
|---|:-:|:-:|
| PC1 | 0.06–0.12 | 0.04–0.06 |
| PC1+5 | 0.27–0.34 | 0.16–0.21 |
| MD-direction coherence | 0.08–0.27 | 0.05–0.13 |
| Participation ratio | 22–29 (max ~34) | 60–80 (max ~139) |

**Facets ARE more coherent than traits** — PC1 and MD-coherence roughly double going from trait to facet. The bundle hypothesis is supported at the level of "facets are tighter sub-directions than traits" — aggregating across facets does dilute the axis.

But facets are still diffuse. PC1 at 6–12% is far from the 40%+ a clean 1D axis would predict. Participation ratio hits ~75% of max isotropy. **Even facets are not 1D.**

**Striking semantic pattern** in which facets are most vs least axis-like:

*Most axis-like* (MD coherence > 0.20 across models):
- E:Sentimentality (0.24–0.27)
- O:Aesthetic Appreciation (0.22–0.25)
- C:Organization (0.22–0.24)
- E:Fearfulness, E:Anxiety (0.19–0.23)

*Least axis-like* (MD coherence < 0.14):
- **H:Modesty (0.08–0.10)** — worst facet across all models
- A:Gentleness, O:Creativity, O:Unconventionality, X:Social Self-Esteem

The semantic split is not random: **concrete behavioral facets** (Sentimentality = visible emotional response, Organization = tidying, Aesthetic Appreciation = responding to beauty, Anxiety/Fearfulness = visible emotional states) have tighter axes. **Abstract dispositional facets** (Modesty = self-concept, Creativity = capability-pattern, Unconventionality = identity-positioning) have diffuse ones. **LLMs represent behaviors more cleanly than dispositions.**

## E.6 Text-confound check

This arose from a reading-group-style challenge: "maybe we're capturing structure in the scenarios, not model representation — e.g., our Fairness scenarios might require social boldness, so of course they'd look similar."

Two versions run:

**Version A — facet-center level** (the first cut): average MiniLM embeddings of each facet's 35 scenarios into a text center; compare to the facet's representation direction. Correlate the 276 pairwise cosines.
→ **Pearson r = +0.22.** Weak but nonzero — stimulus content contributes something to representation structure.

**Version B — per-scenario level** (the rigorous cut): for each individual scenario, compute its text-similarity to each other facet's text center and its representation-similarity (individual pair direction vs that facet's repr center). Correlate across all 77,096 observations.
→ **Pearson r = +0.068.** Roughly 1/3 the facet-center value.

**The facet-center r was largely an aggregation artifact.** Per-scenario, text similarity does not meaningfully predict representation similarity.

Broken down into 552 (own-facet, other-facet) cells: mean within-cell r = +0.004; 73/552 cells have |r| > 0.30; 6/552 with |r| > 0.50. Most scenario-pair combinations show near-zero stimulus drift; a specific minority are genuinely text-confounded.

**For Fairness specifically (the original question)**:

| target facet | within-cell r(text_sim, repr_sim) |
|---|:-:|
| H:Greed-Avoidance | +0.34 |
| C:Prudence | +0.34 |
| C:Perfectionism | +0.33 |
| ... | |
| **X:Social Boldness** | **+0.10** |

Fairness scenarios that are textually Boldness-like do NOT represent more Boldness-like. **The Fairness↔Boldness facet-center cos of 0.38 is essentially all aggregation-of-averages, not per-scenario drift. Retract that specific cross-trait coupling.**

Fairness DOES have moderate per-scenario text confound with C facets (+0.33 with Prudence, Perfectionism). So **Fairness → C alignment is partly text-mediated** (~1/3 to 1/2 of the coupling strength, estimated). Still a real coupling in representation space, but we should attribute some of it to scenario content, not pure representation.

**Strongly negative within-cell correlations are the interesting positive finding**: in cells like H:Greed-Avoidance → O:Aesthetic Appreciation (r=−0.56), E:Anxiety → A:Gentleness (r=−0.40), scenarios that are MORE text-similar to the target are LESS representationally similar. The model actively distinguishes opposite-pole concepts even when stimulus content overlaps. **This is genuine representation signal — not stimulus confound and not aggregation artifact.**

## E.7 Updated claims after the follow-through

What survives cleanly:

1. **Within-trait facet clustering** (mean cos 0.38 vs across-trait 0.07) — survives per-scenario text check; within-trait residuals against text baseline are strongly positive.
2. **Opposite-pole separation** (anxious vs confident, greedy vs aesthetic, etc.) — actively produced by models, negative within-cell text correlations confirm it's not stimulus.
3. **H-split into 4 facet-alignments** (Fairness→C, Sincerity→H/C, Modesty→A, Greed-Avoidance→anti-engagement) — survives, with quantified stimulus-confound levels per coupling.
4. **Engagement-withdrawal axis** cutting across X/O/C — visible as the X↔O dominance in top cross-trait couplings.
5. **No low-rank HEXACO factor structure** — with better data, PC3 cumulative improved from ~30% to ~43% across facet directions, but still nowhere near the 80–90% a clean 6-factor model would predict.
6. **Bundle-of-axes, not bundle-of-1D-axes** — facets are 2× tighter than traits but still diffuse (PC1 ~10%, PR ~25 out of 34 max).

What needs qualification:

- **Fairness ↔ Boldness**: retract. Mostly facet-center aggregation + modest text confound, not a representation-level coupling.
- **Fairness → C alignment**: real but 1/3–1/2 text-mediated. Qualify explicitly.
- **Specific cross-trait couplings** (Sentimentality↔Aesthetic, Anxiety↔Perfectionism, etc.): survive the per-scenario check; real but somewhat attenuated relative to their face-value cosines.

What the expanded dataset made clearer:

- Old Week 1–5 results based on unstratified training pairs were measuring specific biased slices of each trait. The n=50 H direction specifically loaded heavily on Fairness/Sincerity content and ignored Modesty. This retrospectively explains some confusions in earlier results.
- **MD-raw and MD-projected are more sample-stable than LR** across the old→new comparison. Worth reconsidering whether LR should remain the primary probe for cross-study comparability purposes.
- "No clean 1D trait axis in representation space" generalizes to facet level: it's not just that traits are bundles — facets are also clouds, just tighter clouds.

## E.8 New to_try items suggested by the epilogue

- Per-scenario residual analysis: for each claimed cross-trait coupling, compute the within-cell r and report representation-minus-text residual. This becomes the standard way to report cross-trait couplings going forward.
- Compare MD vs LR sample-stability on the old→new task more systematically; decide whether to demote LR as primary.
- Test whether "behavioral facets are more axis-like than dispositional ones" replicates on a fresh facet labeling (maybe even redefine facets by behavioral concreteness and check).
- The Sofroniew story-based extraction track (still queued) becomes more important — everything in this epilogue is still within the contrast-representation subspace, and the diffuseness + stimulus-confound findings make the disposition-axis question sharper, not softer.

## E.9 One sentence for the reading group

"With 6× more training data, stratified by facet and stimulus-content-confound-checked, the Week 6 findings about HEXACO structure survive in refined form: the contrast-representation subspace has within-trait clustering and active opposite-pole separation, but no clean factor structure at trait or facet level, and one specific cross-trait coupling (Fairness↔Boldness) was a facet-aggregation artifact and should be retracted."
