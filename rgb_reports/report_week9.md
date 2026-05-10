# Week 9 — Single-direction representations and anisotropy

## 0. One-line summary

**Anisotropy dominates absolute-space geometry; contrast extraction was hiding it.** The W8 §9 facet-cluster work used `meandiff-pcs` (contrast subtraction with neutral-PC projection), which implicitly cancels both the residual-stream anisotropy AND any IPIP-format-specific shared baseline. When we ablate to 4 single-direction variants without contrast, 3 of them (`single-zero`, `single-neutral`, `single-pcs`) collapse fully — pairwise cosine ≈ +1.0 between *every* facet direction on *every* cohort model. The neutral-text baseline sits on a different point of the anisotropic manifold than chat-wrapped IPIP items, so it can't isolate trait-distinctive content. A fifth method, `single-ipip-mean` (subtract the centroid of all 288 IPIP items), works as the matched-format baseline and reveals real per-facet structure at ~10× smaller magnitude than contrast: cohort within-trait +0.121 vs +0.155, NN-within 16.4/30 vs 15.1/30, purity 0.524 vs 0.527. But the two methods agree only at r=+0.35 mean within a model, and cross-model preservation drops from r=+0.940 (contrast) to r=+0.649 (single-ipip-mean) — the strong W8 §9 cross-architecture preservation is partially extraction-method-specific. Both views (superposition + embedding) co-exist at different scales: anisotropy dominates the residual stream globally, trait-distinctive embedding-style clustering exists in a small orthogonal subspace.

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

## 7. Implications for the superposition-vs-embedding question

Both views co-exist at different scales:

- **Anisotropy dominates the residual stream globally**: cosine ≈ +1.0 between any two inputs in absolute space. This is the dominant "story" of the residual stream's geometry. No semantic structure is visible at this scale.
- **Trait-distinctive structure exists in a small subspace orthogonal to the dominant anisotropy axis**. Within that subspace, facets cluster by trait (within > across cosine ratio 3.4× under single-ipip-mean, 5.3× under meandiff-pcs). This is the embedding-style clustering.
- **The two methods recover *different views* of this small subspace**: contrast subtraction emphasizes a fwd-rev differential axis; matched-baseline single-direction emphasizes an absolute deviation-from-centroid axis. Within-model r between them is +0.35 cohort mean.
- **Cross-architecture preservation is a property of contrast extraction, not of the residual stream tout court**: r=+0.94 under contrast collapses to r=+0.65 under single-ipip-mean. The robust cross-model finding is partly an artifact of using a method that emphasizes a particular shared axis.

Concrete framing for the original question: the residual stream has two layers of structure. The dominant outer layer is anisotropy + format-specific shared content; this is essentially a "universal direction" all inputs sit near. The inner layer is the trait-distinctive subspace, visible only with the right baseline subtraction. Within the inner layer, facets show embedding-style clustering — but the inner layer itself can be sliced multiple ways (contrast axis vs centroid-deviation axis vs probably others), and which slice you pick changes both the within-model picture AND the cross-architecture preservation.

This is consistent with Anthropic's SAE work (Templeton 2024, Lieberum 2024 et seq.) showing that residual streams can be decomposed into thousands of features that are *both* sparse-near-orthogonal AND semantically clustered when projected — the two views aren't competing accounts, they're descriptions of the same underlying structure at different granularities.

The W7 §8.4 "cross-architecture r=0.93-0.99" finding was a real observation but methodologically narrow. It says: when you extract trait directions via contrast subtraction, the resulting 30×30 cosine matrices agree across architectures at high precision. It does NOT say: the underlying residual streams agree on trait structure at this precision. They disagree more when sliced a different way.

## 8. Visualizations

- `results/facets/ipip_facet_method_dashboard.html` — single-page dashboard for the group meeting. Four sections: per-method cohort bars (within/across/NN/purity, all 5 methods × 7 models); cross-model 7×7 agreement heatmaps for `meandiff-pcs` and `single-ipip-mean` side-by-side; per-model 30×30 cosine matrices for 4 representative models under `single-ipip-mean`.
- `results/facets/ipip_facet_cluster_<method>.json` — per-model summaries with full cosine matrices for each of the 5 methods. `ipip_facet_cluster.json` (unsuffixed) remains the `meandiff-pcs` / W8 §9 output.

## 9. Open methodological questions

- **The 5× boost from PC projection in `meandiff-pcs`**: raw contrast (no PC projection) on Qwen7 gives within=+0.042, but PC-projected contrast gives within=+0.193 — almost 5× boost. PC projection is doing substantial work that's not just removing anisotropy. Worth a separate diagnostic. Possible: the top neutral PCs include directions anti-aligned with within-trait signal, so projecting them out (and re-unit-norming) amplifies the trait-aligned component. Or: the top PCs include "trait-irrelevant" semantic variance that was diluting the within-trait cosine.

- **Token aggregation choice**: `mean-all-skip0` (current IPIP pipeline) includes chat-template wrapper tokens. A response-position-only aggregation (`mean-response`, the HEXACO contrast-pair convention) might give cleaner signal. Worth ablating.

- **Phi4–Qwen7 disagreement**: under `single-ipip-mean`, these two models have r=+0.203 (cohort minimum) — Phi4 is geometrically closer to Qwen (3B) than to Qwen7 (7B). What's specifically different about Phi4 and Qwen7's residual structure that makes them disagree on the absolute centroid-deviation slice but agree on the contrast slice (r=+0.877)? Possible: Phi4 has been the W7/W8 cohort outlier on the rep readout side throughout; this is another facet of "Phi4's residual stream encodes trait info differently."

- **Reverse-only sanity check**: `single-*` methods all use forward-keyed items. Computing forward-only and reverse-only directions and checking whether they're anti-correlated would test whether the IPIP forward/reverse polarity actually probes opposite poles in absolute space.

## 10. Next steps

1. **Use single-ipip-mean directions for persona z-recovery** (Phase B of the W9 plan): does substituting per-facet `single-ipip-mean` directions for `meandiff-pcs` directions in the W8 §5 setup recover persona facet z's better or worse? This is the downstream test that turns the methodological discrimination into a behavioral one.

2. **Investigate the PC-projection boost**. Quick: try `contrast-no-pcs` (mean(fwd) − mean(rev), no projection) and compare facet directions to `meandiff-pcs`'s. The 5× within-trait boost from PC projection is interesting and not currently understood.

3. **Layer sweep**. The activation cache stores all layers; sweeping common_layer over ±5 layers would test whether the single-ipip-mean / contrast disagreement is layer-localized or holds across depths.

4. **Per-facet rep direction × persona z-recovery** (deferred from W8 §9 next-steps). The chunking-granularity test, now with the proper extraction method established.

## 11. Status

Commits:
- `f95dd00` — W9 §1 setup: representation_vector_methods.md catalog + 5-extraction refactor of ipip_facet_cluster.py + IPIP item activation cache
- `613281c` — W9 §1 cohort: 5 methods × 7 cohort models; anisotropy degeneracy confirmed; single-ipip-mean cross-model r=+0.649

Result files:
- `results/facets/ipip_facet_cluster.json` — meandiff-pcs cohort (W8 §9, regenerated identically)
- `results/facets/ipip_facet_cluster_single-{zero,neutral,pcs,ipip-mean}.json` — 4 new methods × cohort
- `results/facets/ipip_facet_method_dashboard.html` — 5-method × 7-model dashboard
- `results/phase_b_cache_ipip/<safe_repo>_ipip_chat.pt` — per-item activations cache (gitignored)
