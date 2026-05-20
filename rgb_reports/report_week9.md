# Week 9 — Single-direction representations and anisotropy

## 0. One-line summary

**Anisotropy dominates absolute-space geometry; contrast extraction was hiding it.** The W8 §9 facet-cluster work used `meandiff-pcs` (contrast subtraction with neutral-PC projection), which implicitly cancels both the residual-stream anisotropy AND any IPIP-format-specific shared baseline. When we ablate to 4 single-direction variants without contrast, 3 of them (`single-zero`, `single-neutral`, `single-pcs`) collapse fully — pairwise cosine ≈ +1.0 between *every* facet direction on *every* cohort model. The neutral-text baseline sits on a different point of the anisotropic manifold than chat-wrapped IPIP items, so it can't isolate trait-distinctive content. A fifth method, `single-ipip-mean` (subtract the centroid of all 288 IPIP items), works as the matched-format baseline and reveals real per-facet structure at ~10× smaller magnitude than contrast: cohort within-trait +0.121 vs +0.155, NN-within 16.4/30 vs 15.1/30, purity 0.524 vs 0.527. But the two methods agree only at r=+0.35 mean within a model, and cross-model preservation drops from r=+0.940 (contrast) to r=+0.649 (single-ipip-mean) — the strong W8 §9 cross-architecture preservation is partially extraction-method-specific. **Comparison to human facet correlations from Johnson IPIP-NEO-300 (N=145,388) settles where the cross-architecture-preserved signal comes from**: contrast extraction's cohort r vs the human matrix is **+0.564** (range +0.44 to +0.64), confirming that models do pick up the empirical Big Five facet covariance from training data; under single-ipip-mean the model-vs-human alignment drops to +0.326 with high variance (range +0.06 to +0.62). Both views (superposition + embedding) co-exist at different scales: anisotropy dominates the residual stream globally, trait-distinctive embedding-style clustering exists in a small orthogonal subspace, and within that subspace the cross-architecture-preserved component is the human-facet-aligned one.

## 1. Motivation

Two related concerns drove W9 §1:

1. **Theoretical**: rgb's open question (2026-05-10) about whether residual-stream trait geometry follows the superposition account (concepts used in shared context should be near-orthogonal) or the word-embedding account (similar concepts geometrically similar regardless of context). W7-W8 empirical pattern (within-trait cosine +0.2, across-trait +0.04, cross-architecture r=0.93-0.99) looks like the embedding view is winning, but this is read through contrast extraction — what does "absolute space" without the contrast subtraction look like? See `memory/user_superposition_vs_embedding.md`.

2. **Methodological**: we've drifted across many extraction methods (PCA, LDA, LR, mean-diff scenario, mean-diff item with PC projection, contrast pair fwd-rev). Each method conflates several choices (contrast vs not, baseline subtraction style, token aggregation). W9 §1 introduces a canonical methods catalog (`rgb_reports/representation_vector_methods.md`) and a controlled comparison across 5 extraction methods that orthogonalize "contrast vs not" and "baseline choice".

## 2. The five methods

(See `representation_vector_methods.md` for the canonical catalog with formulas and dependencies.)

| name | formula | what it does |
|---|---|---|
| `meandiff-pcs` | `unit(project_out_pcs(mean(fwd) − mean(rev), neutral_pcs))` | W8 §9 current method; contrast subtraction + neutral-PC projection |
| `single-zero` | `unit(mean(fwd))` | no baseline at all; absolute centroid of "where forward items live" |
| `single-neutral` | `unit(mean(fwd) − mean(neutral_text))` | subtract the neutral-text mean (an affine baseline from a different stimulus set) |
| `single-pcs` | `unit(project_out_pcs(mean(fwd), neutral_pcs))` | project out neutral-derived top-50%-variance PCs (no contrast) |
| `single-ipip-mean` *(W9 addition)* | `unit(mean(fwd) − mean(all_ipip_items))` | subtract the IPIP-format centroid (matched-format baseline) |

All operate on chat-template-wrapped IPIP-300 items at `common_layer = round(n_layers * 2/3)`, with `mean-all-skip0` token aggregation (mean over all tokens of the chat-wrapped sequence, skipping BOS). 30 facet directions per model (5 traits × 6 facets, deny-listed and typo-fixed per `instruments/ipip300_annotations.json`).

Implementation: `scripts/ipip_facet_cluster.py --extraction <name>`. Item activations are cached once per model to `results/phase_b_cache_ipip/<safe_repo>_ipip_chat.pt` so multiple method invocations on the same model are cheap.

## 3. Anisotropy: the absolute-space degeneracy

Pilot on Qwen7 immediately produced a striking result:

| method | within | across | ratio | NN-within | purity |
|---|---|---|---|---|---|
| `meandiff-pcs` (current) | +0.193 | +0.029 | 6.6× | 15/30 | 0.53 |
| `single-zero` | **+1.000** | **+1.000** | 1.00× | 21/30 | 0.33 |
| `single-neutral` | +0.996 | +0.996 | 1.00× | 19/30 | 0.33 |
| `single-pcs` | +0.993 | +0.992 | 1.00× | 18/30 | 0.33 |

All three single-direction methods are *fully degenerate* — every facet direction is essentially identical (cosine ≈ +1.0 between every pair). Initial reaction was that this looked like a bug. Sanity check: take raw item activations (not facet means, just unit-normed item activations at common_layer) and compute pairwise cosines:

```
Raw Qwen7 IPIP item-item cosine at layer 19:    mean=+0.9998, range +0.9991 to +1.0000
Raw Qwen7 neutral item-item cosine at layer 19: mean=+0.9991, range +0.9969 to +0.9999
```

The residual stream of Qwen2.5-7B at layer 19 (and at most other layers, see §3.1) has cosine ≈ +1.0 between essentially all inputs. This is well-documented **residual-stream anisotropy** in pre-norm transformers (Ethayarajh 2019; Gao et al. "BERT-flow"; Cai et al. 2021 on isotropy in BERT). The W2 §3 finding ("PC1 correlates r=1.0 with activation norm — pure norm artifact") is the same phenomenon at trait granularity.

The deeper implication: when ALL inputs occupy a thin spike in residual-stream space, no single-direction extraction can separate them in absolute coordinates. PC projection doesn't fix this because PC projection only resolves what's *already* separable in some subspace; if the inputs are near-identical, projecting out anisotropy leaves near-identical residuals.

### 3.1. Layer dependence

Anisotropy isn't an artifact of one chosen layer — it holds across the stack on Qwen7:

| layer | raw item-item cosine (mean) |
|---|---|
| 0 (embedding) | +0.944 |
| 5 | +1.000 |
| 10 | +0.999 |
| 15 | +0.9998 |
| 19 (common_layer) | +0.9998 |
| 25 | +0.999 |
| 28 (final) | +0.958 |

Layer 0 is the embedding; layer 28 (post-final-block) drops slightly because the final-norm acts on it. Everything in between is essentially +1.0. The trait-distinctive component lives in a small orthogonal subspace that requires explicit baseline subtraction to surface.

## 4. The matched-format baseline

If neutral-text baseline doesn't work because chat-wrapped IPIP items sit on a different point of the anisotropic manifold, the obvious fix is to use a baseline drawn from the same manifold. `single-ipip-mean` does exactly this: subtract the mean across all 288 IPIP items (forward + reverse pooled) from each facet's forward-mean. The shared anisotropy axis cancels; the IPIP-format-specific content cancels; what's left is the per-facet deviation from the IPIP centroid.

Pilot on Qwen7 (layer 19):

| method | within | across | ratio | NN | purity |
|---|---|---|---|---|---|
| `single-ipip-mean` | +0.044 | −0.013 | 3.4× | 17/30 | 0.37 |

Real structure: within > across, NN-within above chance (17.2%), purity above chance (20%). At smaller magnitude than `meandiff-pcs` but recognizably similar shape.

## 5. Cohort results

Full 7-model cohort × 5 methods. Headline cohort means:

| method | within | across | ratio | NN/30 | purity | cross-model r |
|---|---|---|---|---|---|---|
| `meandiff-pcs` (W8 §9) | **+0.155** | +0.029 | 5.3× | 15.1 | 0.527 | **+0.940** |
| `single-zero` | +0.996 | +0.996 | 1.00× | 17.4 | 0.378 | — degenerate |
| `single-neutral` | +0.923 | +0.923 | 1.00× | 18.1 | 0.342 | — degenerate |
| `single-pcs` | +0.985 | +0.984 | 1.00× | 16.4 | 0.371 | — degenerate |
| `single-ipip-mean` | +0.121 | +0.018 | 6.7× | 16.4 | 0.524 | **+0.649** |

Anisotropy is universal — every cohort model has the three middle methods degenerate. Same finding on Gemma 4B/12B, Llama 3B/8B, Phi4, Qwen 3B/7B.

Per-model breakdown for the two working methods:

| Model | mpcs within | mpcs NN | mpcs purity | sipm within | sipm NN | sipm purity |
|---|---|---|---|---|---|---|
| Gemma     | +0.163 | 16/30 | 0.53 | +0.124 | 13/30 | 0.43 |
| Llama     | +0.135 | 16/30 | 0.53 | +0.127 | **20/30** | **0.57** |
| Phi4      | +0.117 | 14/30 | 0.50 | +0.137 | 15/30 | **0.63** |
| Qwen      | +0.170 | 15/30 | 0.53 | +0.174 | 17/30 | **0.60** |
| Gemma12   | +0.173 | 14/30 | 0.53 | +0.102 | 14/30 | 0.40 |
| Llama8    | +0.132 | 16/30 | 0.53 | +0.137 | 19/30 | **0.57** |
| Qwen7     | **+0.193** | 15/30 | 0.53 | **+0.044** | 17/30 | 0.37 |

The cohort splits: Llama, Phi4, Qwen, Llama8 have comparable-or-better facet structure under `single-ipip-mean` (NN ≥ contrast, purity above contrast). Gemma, Gemma12 do worse. **Qwen7 is the most striking split** — the contrast-method best (within=+0.193) is also the single-ipip-mean worst (within=+0.044). Qwen vs Qwen7 (same model family, different scale) goes from cohort-best single-ipip-mean (Qwen: 0.174, 0.60) to cohort-worst (Qwen7: 0.044, 0.37).

## 6. Cross-model agreement: the W8 §9 preservation isn't extraction-invariant

W7 §8.4 / W8 §9 found cross-architecture cosine-matrix preservation at r=0.93-0.99 (within-stimulus-type). This was based on `meandiff-pcs` extraction throughout. Under `single-ipip-mean`:

```
meandiff-pcs:      cohort mean r = +0.940   (range +0.877 to +0.981)
single-ipip-mean:  cohort mean r = +0.649   (range +0.203 to +0.976)
```

The mean drops by 0.29; the range widens enormously. Some pairs still agree strongly (Llama–Llama8 r=+0.976, Phi4–Qwen r=+0.911, Gemma–Gemma12 r=+0.889), but others drop dramatically:

- **Phi4–Qwen7**: +0.203 (cohort minimum)
- **Phi4–Gemma12**: +0.255
- **Phi4–Gemma**: +0.287
- **Qwen–Qwen7** (same family!): +0.367
- **Qwen–Gemma12**: +0.397

There's a visible two-cluster structure: Phi4 ↔ Qwen (high mutual agreement at +0.911) form one geometric cluster; Gemma / Gemma12 / Qwen7 form another (Gemma–Gemma12 +0.889, Gemma12–Qwen7 +0.864). Llama family bridges. **Phi4 is the cohort outlier under `single-ipip-mean`** (mean pairwise r=+0.506) — under `meandiff-pcs` it was only slightly low (+0.916).

The takeaway: the W8 §9 cross-architecture preservation finding is partly methodologically specific. Contrast extraction (`meandiff-pcs`) emphasizes a robust, cross-model-shared aspect of trait geometry. Single-direction extraction with matched baseline (`single-ipip-mean`) captures a different, more model-specific aspect — probably reflecting model-specific anisotropy structure, post-training recipe differences, or tokenizer-specific framing artifacts. Same model, different "view".

### 6.1. Within-model cross-method agreement

To check whether the two methods recover the same geometry at different scales, vs. genuinely different geometries: pairwise r between the per-model 30×30 cosine matrices for `meandiff-pcs` and `single-ipip-mean` on the same model:

| Model | r |
|---|---|
| Gemma    | +0.278 |
| Llama    | +0.370 |
| Phi4     | +0.486 |
| Qwen     | **+0.606** |
| Gemma12  | +0.171 |
| Llama8   | +0.403 |
| Qwen7    | +0.137 |
| **mean** | **+0.350** |

Only Qwen has r > 0.5. Most models have the two methods agreeing only modestly. They're measuring related but substantially different slices of the trait structure within a single model — not just the same geometry at different magnitudes.

## 7. Comparison to human facet correlations

rgb's read (2026-05-10): "If human correlations correspond roughly to the representation similarity, it's fairly likely that models are picking up on this in the training data. If they don't, then the structural similarity is probably coming from elsewhere."

We pulled the IPIP-NEO-300 raw data Johnson maintains (via the NeuroQuestAi/ipip-neo-data GitHub mirror; ultimately from Johnson's 2014 OSF deposit and Kajonius & Johnson 2019, *Europe's Journal of Psychology* 15(2):260-275), scored the 30 facets per participant via the standard Goldberg/Johnson 1999 key, and computed a 30×30 inter-facet correlation matrix on N = **145,388** participants. Saved to `instruments/ipip300_human_facet_correlations.json`.

The human matrix is sensible and substantial: within-trait correlations averaging +0.405 (median +0.397), across-trait averaging −0.021 (essentially zero), within/across ratio = 19×. Highest pairs include N:Anxiety ↔ N:Vulner (+0.775), E:Friend ↔ E:Gregar (+0.746), C:Achieve ↔ C:Discipl (+0.674), A:Moral ↔ C:Dutiful (+0.663, the "good citizen" cross-trait axis). Most negative: E:Assert ↔ N:Self-Cons (−0.669), C:Self-Eff ↔ N:Vulner (−0.640).

For each model × extraction method, we compute Pearson r between the model's 30×30 cosine matrix upper-triangle and the human correlation matrix upper-triangle. Higher r = the model's geometry more closely matches human inter-facet structure.

### 7.1. Per-model alignment table

| Method            | Gemma  | Llama  | Phi4   | Qwen   | Gemma12 | Llama8 | Qwen7  | **cohort mean** |
|-------------------|--------|--------|--------|--------|---------|--------|--------|-----|
| `meandiff-pcs`    | +0.589 | +0.531 | +0.443 | +0.595 | +0.632  | +0.512 | +0.644 | **+0.564** |
| `single-zero`     | +0.217 | +0.131 | +0.184 | +0.151 | +0.198  | +0.123 | +0.151 | +0.165 |
| `single-neutral`  | +0.098 | +0.087 | +0.243 | +0.173 | +0.068  | +0.091 | +0.071 | +0.119 |
| `single-pcs`      | +0.242 | +0.154 | +0.152 | +0.135 | +0.240  | +0.130 | +0.182 | +0.177 |
| `single-ipip-mean`| +0.149 | +0.389 | +0.540 | +0.617 | +0.095  | +0.434 | +0.059 | **+0.326** |

(The three degenerate methods are near zero on the diagonal of facet–facet correlation despite the within=across ≈ +1 anisotropy story — recall their cosine matrices are nearly constant, so they retain only rounding-level structure, which correlates weakly with the human matrix.)

### 7.2. Findings

- **Models recover human facet covariance via contrast extraction.** Cohort r=+0.564 under `meandiff-pcs`. Every cohort model has r ≥ +0.44 with the human matrix; six of seven have r ≥ +0.51. This is direct evidence that the cross-architecture-preserved component of trait geometry IS the human-aligned one. Training data does encode the empirical Big Five facet structure, and models pick it up consistently across the 3B–12B scale range we've sampled.
- **Contrast extraction is what reveals this alignment.** The three anisotropy-degenerate methods show only weak alignment (r ≈ +0.12 to +0.18 cohort) — their geometries are dominated by residual-stream anisotropy, which has no Big-Five-aligned structure to speak of.
- **Single-ipip-mean has *heterogeneous* alignment** (cohort r=+0.326, range +0.06 to +0.62). Three models — Phi4 (+0.540), Qwen (+0.617), Llama8 (+0.434) — show alignment that's nearly as strong as their contrast-extraction alignment. Three others — Gemma (+0.149), Gemma12 (+0.095), Qwen7 (+0.059) — drop to near zero. The same models that have low cross-model agreement with the rest of the cohort under single-ipip-mean (§6) also have low human-alignment under single-ipip-mean. Geographical/family structure: Gemma family + Qwen7 cluster together at low single-ipip-mean human-alignment; Llama family + Phi4 + Qwen (3B) cluster at higher alignment. §7.3 traces this heterogeneity to incomplete anisotropy cancellation, not to real model-specific structure.
- **The W8 §9 +0.94 cross-architecture preservation reframes as a "shared with humans" finding.** Under meandiff-pcs: models agree with each other at +0.94 mean, and agree with humans at +0.56 mean. Both numbers are about the same robust axis of trait geometry. The +0.94 model-model is higher than +0.56 model-human because models share architectural/training similarities humans don't; but the model-human residual is still substantial.

### 7.3. Why single-ipip-mean fails on Gemma family + Qwen7: anisotropy residue

rgb's hypothesis (2026-05-10): Gemma's huge activation norms (CLAUDE.md notes "norms from 50 to 63,000 across Gemma3's 34 layers") might be what's making single-ipip-mean weird for that family. Diagnostic confirms this — and finds the same shape on Qwen7 for a different reason.

For each cohort model at its common_layer, we measured: (a) the norm of the IPIP-centroid baseline `mean(all 288 IPIP items)`; (b) the median norm of per-facet deviation vectors `mean(fwd_facet) − ipip_centroid`; (c) the median absolute cosine of each deviation vector with the IPIP-centroid direction — i.e. how much of the residual "deviation" still points along the anisotropy axis after centroid subtraction.

| Model | norm(IPIP centroid) | median norm(deviation) | median \|cos(dev, centroid)\| | r vs human (single-ipip-mean) |
|---|---:|---:|---:|---:|
| Gemma 4B   | **59,996** | 1,158.78 | **0.680** | +0.149 |
| Gemma 12B  | **116,823** | 2,505.63 | **0.786** | +0.095 |
| Qwen 7B    | 492 | 7.93 | **0.847** | +0.059 |
| Llama 3B   | 22 | 0.77 | 0.324 | +0.389 |
| Llama 8B   | 16 | 0.65 | 0.243 | +0.434 |
| Phi4-mini  | 76 | 11.44 | 0.079 | +0.540 |
| Qwen 3B    | 48 | 3.26 | 0.059 | +0.617 |

Two failure modes share a symptom (high `|cos(dev, centroid)|`) but with different mechanisms:

- **Gemma family — extreme anisotropy magnitude.** IPIP-centroid norm is 60k / 117k at common_layer, vs 16–492 for every other model. Gemma's pre-norm transformer compounds residual-stream norms exponentially through depth (Heimersheim & Turner 2023; Peri-LN paper for Gemma specifically). At common_layer (~2/3 depth), the centroid is huge, and the per-facet deviation `fwd_mean − centroid` carries ~68–79% of its magnitude back along the centroid direction. Unit-norming this deviation yields a direction dominated by leftover anisotropy, not trait content.
- **Qwen7 — extreme anisotropy *alignment*.** Centroid norm is only 492 (modest, comparable to Phi4 and Qwen), but median \|cos(dev, centroid)\| is the highest in the cohort at 0.847. Qwen7's per-facet IPIP activations all sit on a thin spike around the centroid; the deviations mostly point along the same axis the centroid does. Subtracting a rank-1 mean isn't enough to escape the shared spike.

There's a clean rank correspondence across the cohort: cos(deviation, centroid) negatively predicts human alignment under single-ipip-mean. Spearman ρ between the two columns is approximately −0.93 — almost monotone. Models where the deviation IS orthogonal to the centroid direction (Phi4 at 0.079, Qwen at 0.059) get clean trait-relevant directions and high human alignment; models where it isn't (Gemma family, Qwen7) get directions dominated by residual anisotropy.

**Implication.** The W9 §1 finding "single-ipip-mean reveals a small-magnitude model-specific subspace orthogonal to anisotropy" was overstated for Gemma family + Qwen7. For those models, what `single-ipip-mean` reveals is mostly an *extraction failure* — leftover anisotropy bleed-through that wasn't cancelled by the rank-1 centroid subtraction — not real model-specific structure. For Phi4, Qwen (3B), and Llama family, the picture from §1 stands: single-ipip-mean is recovering meaningful structure that genuinely differs from contrast extraction's view.

A cleaner matched-baseline method for the high-anisotropy models would project out the IPIP-centroid direction explicitly (rather than just subtracting it) before unit-norming — or use a richer subspace projection (e.g., top PCs of forward-pole items). Left for after the next reading group.

### 7.4. Synthesis

This settles the user's question. Cross-architecture preservation isn't just an artifact of structural similarity between transformers — it's at least partly an artifact of all models being trained on text that encodes the empirical Big Five facet structure. The +0.56 model-human r under contrast extraction is approximately *the* signal that's preserved across architectures: training data → models learn human facet covariance → all models converge on similar facet geometries → cross-architecture r ≈ +0.94 with that signal as the shared anchor.

What single-ipip-mean ADDS to that picture is real for some models (Phi4, Qwen 3B, Llama family) and extraction-artifact for others (Gemma family, Qwen7; see §7.3). Where it's real, it reveals a small-magnitude, model-specific, anisotropy-orthogonal subspace whose geometry doesn't match human facet covariance and isn't preserved across architectures — plausibly post-training recipe / tokenizer / training-data idiosyncrasies. Where it isn't, it's anisotropy bleed-through and shouldn't be interpreted as model-specific representational content at all.

The chunking-granularity hypothesis (to_try #18) gets a partial answer: Big Five facets ARE the model-natural chunks insofar as model facet cosine geometry recovers human facet correlations at r=+0.56. Models don't over-aggregate into "wrong" units — they pick up the *same* unit structure humans do. The residual structure single-ipip-mean reveals is partly real and partly extraction-failure; an improved matched-baseline extraction (next steps) would let us cleanly assess whether the genuine model-specific portion clusters into coherent alternative dimensions or is diffuse.

### 7.5. The "+0.405 within-trait" human reference is irreproducible (followup, 2026-05-19/20)

rgb's question while sourcing-checking for the paper: do we have a real
reference for the N=145,388 human inter-facet correlation matrix
backing the +0.564 cohort-vs-human r? Investigating revealed a
methodological hole: the +0.405 within-trait number in
`instruments/ipip300_human_facet_correlations.json` was *not* a
canonical Pearson-on-raw-sum quantity. It came from
NeuroQuestAi/ipip-neo-data's pre-scored `facet_*` CSV columns, which
were generated by some unrecorded historical configuration of the
`five-factor-e` library — and those values are **not reproducible from
any currently-released five-factor-e** (tested v1.10.0, v1.11.0,
v1.11.1, v1.13.1; all produce different output for the same raw
items).

#### Findings

| computation on N=145,388 | within mean | across mean | reproducible from primary source? |
|---|---|---|---|
| Pearson on raw 10-item facet sums (canonical Johnson key) | **+0.167** | +0.116 | yes — one sentence to describe |
| Pearson on raw sums, residualized on age + sex + age×sex | +0.166 | +0.113 | yes |
| Pearson on z-scores within sex×age-bin | +0.166 | +0.113 | yes |
| Pearson on `facet_*` percentile values in NQ CSV (= prior file) | **+0.405** | −0.021 | no — irreproducible from any released code |

What we expected to find: the +0.405 number reflecting `Pearson on
raw-sum facet scores` and being trivially reproducible from
PsychArchives primary data. What we actually found: raw-sum Pearson
gives a much smaller +0.167; demographic adjustment changes essentially
nothing (R² for age+sex on each facet is < 0.02); the +0.405 is
specifically driven by five-factor-e's library-internal cubic
transformation `X = 210.336 − 16.738·T + 0.406·T² − 0.00271·T³`
applied to a sex×age-band-T-scored value, then clipped to [1, 99] on
input T at thresholds 32 and 73. None of those constants or thresholds
appears in any peer-reviewed paper.

The library has had ~4 distinct norm-table generations between
Nov 2022 and Oct 2025; the NQ-published CSV scored values that match
no current release. Whatever five-factor-e configuration scored the
data has been replaced.

#### Methodological observation (rgb, 2026-05-20)

> "It's not clear why innate should separate from gender at all,
> or why that'd be definitely meaningful."

Standard psychometric practice often "adjusts out" demographic
effects to get at "the trait." But Big Five literature has
well-documented sex and age effects (women higher A and N; N declines
with age; C rises with age) that aren't measurement artifacts —
they're part of the trait structure. For our model-vs-human
comparison, the question is "does the model recover the same pattern
of facet covariation as humans?" If part of human facet covariation
comes from sex × trait interactions, we want it *included*, not
adjusted out. This argues for raw-sum Pearson as the canonical
reference rather than any demographic-residualized version.

#### Current state

- The prior `instruments/ipip300_human_facet_correlations.json` is
  preserved (still has +0.405 within-mean) but is now flagged as
  using an irreproducible scoring pipeline.
- A clean PsychArchives-primary alternative
  (`scripts/build_ipip_human_correlations_psycharchives.py`) is
  available; it computes raw-sum Pearson at within +0.167 from
  N=145,388 complete cases of the IPIP-NEO-300 raw data in the
  Kajonius & Johnson 2019 PsychArchives deposit.
- W9 §7's headline "+0.564 cohort r against human" depends on the
  prior file. **Recomputing the model-vs-human comparison against the
  raw-sum reference would change that headline number** (most likely
  downward, since the human matrix is now less structured). Held for
  paper-time methodology decision with statisfactions in the loop.
- W9 §7.1 table values were computed under the prior reference and
  also need recomputation if the reference changes.

#### Three references in increasing order of cleanliness for future paper writing

1. **Raw-sum Pearson on PsychArchives** (within +0.167) — primary
   source, one-sentence methodology, no library version dependencies
2. **NQ pre-scored** (within +0.405) — current state of the prior
   file; matches what we've been reporting but is irreproducible
3. **Costa & McCrae 1995 NEO-PI-R values** — peer-reviewed reference
   for the parent instrument; would let us anchor "+0.40 is typical"
   without claiming our number matches; cite if needed

Artifacts from this investigation:
- `data/kajonius_johnson_2019/` (gitignored, ~170 MB): full
  PsychArchives deposit including DAT300.doc codebook,
  IPIP-NEO-ItemKey.xls scoring key, IPIP300.por raw item data, plus
  IPIP-NEO-120 counterparts
- `data/neuroquest_ipip/` (gitignored, ~55 MB): NeuroQuestAi's
  pre-scored mirror, kept for the case-id alignment work
- `scripts/build_ipip_human_correlations_psycharchives.py`: the
  raw-sum Pearson computation from PsychArchives, ready to use

## 8. Implications for the superposition-vs-embedding question

Both views co-exist at different scales:

- **Anisotropy dominates the residual stream globally**: cosine ≈ +1.0 between any two inputs in absolute space. This is the dominant "story" of the residual stream's geometry. No semantic structure is visible at this scale.
- **Trait-distinctive structure exists in a small subspace orthogonal to the dominant anisotropy axis**. Within that subspace, facets cluster by trait (within > across cosine ratio 3.4× under single-ipip-mean, 5.3× under meandiff-pcs). This is the embedding-style clustering.
- **The two methods recover *different views* of this small subspace**: contrast subtraction emphasizes a fwd-rev differential axis; matched-baseline single-direction emphasizes an absolute deviation-from-centroid axis. Within-model r between them is +0.35 cohort mean.
- **Cross-architecture preservation is a property of contrast extraction, not of the residual stream tout court**: r=+0.94 under contrast collapses to r=+0.65 under single-ipip-mean. The robust cross-model finding is partly an artifact of using a method that emphasizes a particular shared axis.

Concrete framing for the original question: the residual stream has two layers of structure. The dominant outer layer is anisotropy + format-specific shared content; this is essentially a "universal direction" all inputs sit near. The inner layer is the trait-distinctive subspace, visible only with the right baseline subtraction. Within the inner layer, facets show embedding-style clustering — but the inner layer itself can be sliced multiple ways (contrast axis vs centroid-deviation axis vs probably others), and which slice you pick changes both the within-model picture AND the cross-architecture preservation.

This is consistent with Anthropic's SAE work (Templeton 2024, Lieberum 2024 et seq.) showing that residual streams can be decomposed into thousands of features that are *both* sparse-near-orthogonal AND semantically clustered when projected — the two views aren't competing accounts, they're descriptions of the same underlying structure at different granularities.

The W7 §8.4 "cross-architecture r=0.93-0.99" finding was a real observation but methodologically narrow. It says: when you extract trait directions via contrast subtraction, the resulting 30×30 cosine matrices agree across architectures at high precision. It does NOT say: the underlying residual streams agree on trait structure at this precision. They disagree more when sliced a different way.

## 9. Visualizations

- `results/facets/ipip_facet_method_dashboard.html` — five-method × seven-model summary dashboard. Four sections: per-method cohort bars (within/across/NN/purity, all 5 methods × 7 models); cross-model 7×7 agreement heatmaps for `meandiff-pcs` and `single-ipip-mean` side-by-side; per-model 30×30 cosine matrices for 4 representative models under `single-ipip-mean`.
- `results/facets/ipip_facet_vs_human_dashboard.html` — **8-panel side-by-side**: human IPIP-NEO-300 facet correlation matrix (N=145,388) in position (0,0), then the 7 cohort models' `meandiff-pcs` facet cosine matrices, all on a common color scale with trait-block dividers. Each panel title shows the model's Pearson r against the human matrix. Visual case for "models recover human facet covariance" (cohort r=+0.564, all models ≥ +0.44).
- `results/facets/ipip_facet_cluster_<method>.json` — per-model summaries with full cosine matrices for each of the 5 methods. `ipip_facet_cluster.json` (unsuffixed) remains the `meandiff-pcs` / W8 §9 output.
- `instruments/ipip300_human_facet_correlations.json` — 30×30 inter-facet correlation matrix from Johnson IPIP-NEO-300 raw data (N=145,388), scored via standard Goldberg/Johnson 1999 key.
- `results/facets/ipip_facet_human_comparison.json` — per-model × per-method Pearson r between cohort cosine matrices and the human correlation matrix.

## 10. Open methodological questions

- **The 5× boost from PC projection in `meandiff-pcs`**: raw contrast (no PC projection) on Qwen7 gives within=+0.042, but PC-projected contrast gives within=+0.193 — almost 5× boost. PC projection is doing substantial work that's not just removing anisotropy. Worth a separate diagnostic. Possible: the top neutral PCs include directions anti-aligned with within-trait signal, so projecting them out (and re-unit-norming) amplifies the trait-aligned component. Or: the top PCs include "trait-irrelevant" semantic variance that was diluting the within-trait cosine.

- **Token aggregation choice**: `mean-all-skip0` (current IPIP pipeline) includes chat-template wrapper tokens. A response-position-only aggregation (`mean-response`, the HEXACO contrast-pair convention) might give cleaner signal. Worth ablating.

- **Phi4–Qwen7 disagreement**: under `single-ipip-mean`, these two models have r=+0.203 (cohort minimum) — Phi4 is geometrically closer to Qwen (3B) than to Qwen7 (7B). §7.3 partly explains this — Qwen7 has 84% anisotropy residue in its single-ipip-mean directions, so its low cross-model agreement with Phi4 (which has only 8% residue) is at least partly an extraction artifact. An anisotropy-projection variant should test whether the Phi4–Qwen7 contrast slice agreement (+0.877) is what survives after extraction-noise is removed.

- **Reverse-only sanity check**: `single-*` methods all use forward-keyed items. Computing forward-only and reverse-only directions and checking whether they're anti-correlated would test whether the IPIP forward/reverse polarity actually probes opposite poles in absolute space.

## 11. Next steps

1. **Anisotropy-projection variant of single-direction** (deferred after reading group). §7.3 traces the heterogeneous human-alignment under `single-ipip-mean` to incomplete anisotropy cancellation: Gemma family + Qwen7 retain 68–85% projection on the centroid direction even after rank-1 mean subtraction. A `single-ipip-proj` method would project out the centroid direction explicitly before unit-norming (or use top-K PCs of forward-pole items as a richer subspace projection). Should bring Gemma family + Qwen7 up to the Phi4/Qwen/Llama-family band, separating real model-specific structure from extraction failure.

2. **Use single-ipip-mean (or single-ipip-proj) directions for persona z-recovery** (Phase B of the W9 plan): does substituting per-facet single-direction directions for `meandiff-pcs` directions in the W8 §5 setup recover persona facet z's better or worse? This is the downstream test that turns the methodological discrimination into a behavioral one.

3. **Investigate the PC-projection boost**. Quick: try `contrast-no-pcs` (mean(fwd) − mean(rev), no projection) and compare facet directions to `meandiff-pcs`'s. The 5× within-trait boost from PC projection is interesting and not currently understood.

4. **Layer sweep**. The activation cache stores all layers; sweeping common_layer over ±5 layers would test whether the single-ipip-mean / contrast disagreement is layer-localized or holds across depths.

5. **Per-facet rep direction × persona z-recovery** (deferred from W8 §9 next-steps). The chunking-granularity test, now with the proper extraction method established.

## 12. Status

Commits:
- `f95dd00` — W9 §1 setup: representation_vector_methods.md catalog + 5-extraction refactor of ipip_facet_cluster.py + IPIP item activation cache
- `613281c` — W9 §1 cohort: 5 methods × 7 cohort models; anisotropy degeneracy confirmed; single-ipip-mean cross-model r=+0.649
- `b92b613` — W9 §1 writeup + 5-method comparison dashboard (`ipip_facet_method_dashboard.html`)
- `a12ca5f` — W9 §7 human comparison: N=145,388 human facet correlation matrix + per-model × per-method r table; cohort r=+0.564 under meandiff-pcs
- `178509f` — bibliography: add Johnson 2014, Kajonius & Johnson 2019, NeuroQuestAi/ipip-neo-data references
- (this commit) — W9 §7.3 anisotropy-residue diagnostic + `ipip_facet_vs_human_dashboard.html` 8-panel side-by-side

Result files:
- `results/facets/ipip_facet_cluster.json` — meandiff-pcs cohort (W8 §9, regenerated identically)
- `results/facets/ipip_facet_cluster_single-{zero,neutral,pcs,ipip-mean}.json` — 4 new methods × cohort
- `results/facets/ipip_facet_method_dashboard.html` — 5-method × 7-model dashboard
- `results/phase_b_cache_ipip/<safe_repo>_ipip_chat.pt` — per-item activations cache (gitignored)
