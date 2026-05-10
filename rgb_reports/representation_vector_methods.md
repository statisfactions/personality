# Representation-vector extraction methods

Catalog of every direction-extraction method we've used to recover trait or facet "directions" from residual-stream activations. Each method has a stable name, a concrete formula, the tokens/text it operates on, and a pointer to the script that implements it. New methods get a row; old methods are kept (not deleted) so reports can cite a method by name and the reader can resolve it here.

Last updated: 2026-05-10.

## Vocabulary

- **Activation `a(text, layer, pos)`**: model hidden state at the given layer for the given input text at token position `pos`. Always read from the residual stream (post-block output, pre-final-norm).
- **Aggregated activation**: a function of `a` that produces a single hidden-dim vector per input. Depends on token aggregation choice (see below).
- **Direction `d`**: a unit-norm hidden-dim vector intended to recover a trait/facet/concept axis. Constructed by combining aggregated activations across stimuli.
- **Common layer**: depth at which we read; for IPIP/HEXACO facet work this is `round(n_layers * 2/3)` per Sofroniew et al. and W5 §4.

## Token aggregation conventions

Different scripts pull the representation from different token spans. **This is methodological heterogeneity that has accumulated and must be tracked per-method**, not assumed to be one convention.

| Tag | What it does | Implementation |
|---|---|---|
| `mean-all-skip0` | Mean of hidden states over all token positions in the input, skipping position 0 (BOS). | `hidden_states_for_text(text, split_prefix=None, chat_template=...)` in `extract_meandiff_vectors.py:90`, `start = 1 if n_total > 1 else 0` branch. |
| `mean-response` | Mean over response-position tokens only (everything after a `split_prefix`). | `hidden_states_for_text(text, split_prefix=<prefix>, chat_template=True)`, divergence-detection branch. |
| `last-token` | Final non-padding token activation. | Not currently used in cohort work; appears in some W2-W3 RepE scripts. |
| `period-token` | Activation at the period token specifically; format-invariant. | `scripts/markers_as_stimuli.py`, W7. |

## Stimulus types

| Tag | Examples | Wrapping |
|---|---|---|
| `marker-adj` | "extraverted", "energetic" (Goldberg-list adjectives) | bare or chat-wrapped depending on script |
| `ipip-item` | "I love large parties." (IPIP-NEO-300 first-person items) | chat-template user turn |
| `hexaco-pair` | (high, low) sentence pairs from `contrast_pairs.json` | scenario + completion |
| `neutral-text` | Generic non-personality text used to estimate residual-stream baseline | bare text, mean over tokens |

## Methods table

For each method: extraction formula, what it operates on, where it's implemented, what cached data it depends on, and where its outputs land. Methods marked **(active)** are currently in use; others are kept for historical reference.

### `pca-pc1` — PCA on contrast pair activations (legacy)

- **Formula**: PC1 of `[ph_h - pl_h]` over contrast pairs at layer.
- **Stimulus**: `hexaco-pair`, 50 pairs/trait.
- **Token aggregation**: `mean-response` via `split_prefix`.
- **Layer**: per-trait swept; common layer = ~middle of stack.
- **Implementation**: original W2 RepE pipeline.
- **Status**: superseded; W2 §3 found PC1 correlates r=1.0 with activation norm — pure norm artifact, not trait signal.
- **Result file pattern**: legacy `results/repe/<safe_repo>_<trait>_directions.pt`.

### `lda` — Linear Discriminant Analysis on contrast pairs (legacy)

- **Formula**: Σ⁻¹(μ_high − μ_low) where Σ is pooled within-class covariance.
- **Stimulus**: `hexaco-pair`.
- **Token aggregation**: `mean-response`.
- **Status**: 100% in-distribution classification but doesn't steer behavior (W4 finding); cosine ~0 with steering directions. Σ⁻¹ amplifies low-variance noise (W6 diagnosis).
- **Implementation**: `scripts/extract_trait_vectors.py`.

### `lr` — Logistic regression as LDA control (legacy)

- **Formula**: weights from L2-regularized logistic regression on (high, low) labels.
- **Status**: confirmed Σ⁻¹-noise was the LDA failure mode (W6); LR cleaner but still not steering.
- **Implementation**: `scripts/extract_trait_vectors.py --method lr`.

### `meandiff-scenario` — Mean-difference at scenario position

- **Formula**: `mean(a_high) − mean(a_low)` at common layer over contrast pairs, where `a` is aggregated via `mean-response`.
- **Stimulus**: `hexaco-pair`.
- **Token aggregation**: `mean-response`.
- **Implementation**: `scripts/extract_meandiff_vectors.py`.
- **Status**: W5 baseline.

### `meandiff-pcs` *(active)* — Mean-difference + neutral-PC projection at facet granularity

- **Formula**: `unit(project_out_pcs(mean(a_fwd) − mean(a_rev), neutral_pcs))` per facet, where `neutral_pcs` are the top-50%-variance PCs of neutral-text activations at common layer.
- **Stimulus** (HEXACO): `hexaco-pair` items grouped by facet (4 facets × 6 traits = 24 facet directions). Token aggregation: `mean-response`.
- **Stimulus** (IPIP): `ipip-item` grouped by facet (6 facets × 5 traits = 30 facet directions; 10 items per facet, with deny-list and typo fixes per `instruments/ipip300_annotations.json`). Token aggregation: `mean-all-skip0` ⚠ (different from HEXACO).
- **Layer**: `round(n_layers * 2/3)`.
- **Implementation**: `scripts/facet_cluster.py` (HEXACO) and `scripts/ipip_facet_cluster.py` (IPIP).
- **Output**: `results/facets/facet_cluster.json`, `results/facets/ipip_facet_cluster.json` (with no `--extraction` flag, defaults to this method).
- **Headline finding** (W8 §9): IPIP within-trait cosine +0.153, across-trait +0.028, ratio 5.5×, NN-within 50% (~3× chance), 5-cluster purity 0.527. HEXACO within +0.202, across +0.044, ratio 4.6×, NN 67%, purity 0.583. Cross-architecture cosine-matrix r=+0.939 IPIP, +0.933 HEXACO.
- **Bakes in**: contrast subtraction (assumes fwd-rev symmetry) AND neutral-PC projection (removes the residual-stream "introspection prompt" subspace). The W9 single-direction work decomposes these into separable choices.

### `single-zero` *(W9)* — Forward-only, no baseline

- **Formula**: `unit(mean(a_fwd))` per facet. No subtraction. Direct centroid of "where the trait-high items live" in absolute residual-stream space.
- **Stimulus** (IPIP): `ipip-item`, forward-keyed items only (typically 5-7 per facet after deny-listing).
- **Token aggregation**: `mean-all-skip0` (chat-wrapped, mean over all tokens skipping BOS).
- **Layer**: `round(n_layers * 2/3)`.
- **Implementation**: `scripts/ipip_facet_cluster.py --extraction single-zero` (W9).
- **Theoretical use**: probes whether trait directions in absolute space carry trait-distinctive signal or are dominated by the residual stream's overall mean. Predicts: high pairwise cosine across all directions (if residual mean dominates), or selective trait-relevant signal (if the residual stream genuinely encodes trait information directionally).

### `single-neutral` *(W9)* — Forward-only, neutral-mean baseline

- **Formula**: `unit(mean(a_fwd) − mean(a_neutral))` per facet, where `a_neutral` is the mean over neutral-text activations at the same layer.
- **Stimulus** (IPIP): `ipip-item`, forward-keyed only. Neutral baseline from the cached `<safe_repo>_neutral_chat.pt` file (same as the PC-projection baseline source).
- **Token aggregation**: `mean-all-skip0`.
- **Implementation**: `scripts/ipip_facet_cluster.py --extraction single-neutral` (W9).
- **Theoretical use**: subtracts the affine "this is an introspection prompt" baseline without invoking PC structure. Cleaner alternative to PC projection if the relevant baseline is mean-rank-1 rather than higher-rank.

### `single-pcs` *(W9)* — Forward-only, neutral-PC projection

- **Formula**: `unit(project_out_pcs(mean(a_fwd), neutral_pcs))` per facet.
- **Stimulus** (IPIP): same as `single-neutral`.
- **Token aggregation**: `mean-all-skip0`.
- **Implementation**: `scripts/ipip_facet_cluster.py --extraction single-pcs` (W9).
- **Theoretical use**: matches current `meandiff-pcs` PC handling but drops the contrast subtraction. Isolates "is the contrast doing work?" from "is PC projection doing work?" — current method blends both choices.

### `single-ipip-mean` *(W9)* — Forward-only, IPIP-centroid baseline

- **Formula**: `unit(mean(a_fwd) − mean(a_all_ipip_items))` per facet, where the baseline is the mean across all 288 fwd+rev IPIP items at the same layer.
- **Stimulus** (IPIP): `ipip-item`, forward-keyed only for the per-facet mean; full 288-item pool for the baseline.
- **Token aggregation**: `mean-all-skip0`.
- **Implementation**: `scripts/ipip_facet_cluster.py --extraction single-ipip-mean` (W9).
- **Theoretical use**: the matched-format baseline. Neutral-text mean (used by `single-neutral`) sits on a different point of the anisotropic manifold than chat-wrapped IPIP items, so it can't isolate trait-distinctive structure. Subtracting the IPIP-format centroid captures both anisotropy AND the IPIP-specific "first-person introspection statement" baseline in one operation. Added W9 §1 after pilot showed `single-zero`/`single-neutral`/`single-pcs` are anisotropy-degenerate.

## Caches

| Cache | Path | Format | Used by |
|---|---|---|---|
| HEXACO contrast pairs | `results/phase_b_cache/<safe_repo>_<trait>_chat_pairs.pt` | dict with `ph_h`, `pl_h` (n_pairs, n_layers+1, hidden) | `facet_cluster.py`, RepE legacy |
| HEXACO contrast pairs (stratified) | `results/phase_b_cache_stratified/<safe_repo>_<trait>_chat_pairs.pt` | same shape | W6 stratified-extraction work |
| Neutral baseline | `results/phase_b_cache/<safe_repo>_neutral_chat.pt` | tensor (n_neutral_items, n_layers+1, hidden) | both the PC-projection neutral and the W9 single-neutral baseline |
| IPIP item activations *(W9)* | `results/phase_b_cache_ipip/<safe_repo>_ipip_chat.pt` | dict with `acts` (n_items, n_layers+1, hidden), `meta` (per-item trait/facet/pole/text) | `ipip_facet_cluster.py` for all 4 W9 extraction methods |

## Conventions for adding a new method

1. Pick a stable kebab-case name (e.g., `lr-stratified`, `period-pole`).
2. Add a row to the methods table here with the same fields as above.
3. Implement as a `--extraction <name>` flag in the relevant script, OR as a separate script if the data flow is sufficiently different.
4. Default output filename includes the method tag: `<analysis>_<extraction>.json`.
5. Cite the method by name in the report; don't re-describe the formula in the report (point at this doc instead).

## Open methodological notes

- **Token aggregation heterogeneity**: HEXACO contrast-pair extraction uses `mean-response` (response-position only), IPIP item extraction uses `mean-all-skip0` (chat-wrapped, all tokens). The W8 §9 cross-instrument comparison silently mixes these. A future controlled comparison would re-extract IPIP with `mean-response` (treating the item as a completion, with split_prefix as the chat prompt prefix) to see if the choice matters for facet geometry.
- **Forward-only vs reverse-only**: the W9 `single-*` methods use forward-keyed items only. A symmetric variant could use reverse-keyed items as a sanity check (forward and reverse should be roughly anti-aligned if the items genuinely probe opposite poles).
- **Layer sweep**: all current cohort work reads at common_layer (~2/3 depth). Layer choice is held fixed across methods; could be varied per method if motivated.

## W9 §1 finding — anisotropy dominates absolute geometry

Pilot on Qwen7 showed that raw item-item cosine at common_layer = +0.9998 mean (range +0.999 to +1.000) across all 288 IPIP items, AND across all 300 neutral-text items independently. The W2 §3 finding ("PC1 correlates r=1.0 with activation norm — pure norm artifact") is the same phenomenon at trait granularity: pre-norm transformers' residual stream is heavily anisotropic, with a dominant shared direction that swamps any per-item content signal in absolute space.

Implications for the methods:

- **`single-zero`, `single-neutral`, `single-pcs` are all degenerate on IPIP items**: pairwise cosines between facet directions all come out at ≈ +1.0. The neutral-text baseline doesn't sit on the same point of the anisotropic manifold as chat-wrapped IPIP items, so subtracting the neutral mean (or projecting out neutral-derived PCs) doesn't remove the IPIP-format-specific shared component. PC projection only resolves what was already separable.
- **`single-ipip-mean` does work** because it captures both anisotropy AND the IPIP-format-specific baseline in one matched subtraction. Pilot (Qwen7 layer 19): within +0.044, across −0.013, ratio 3.4×, NN-within 17/30. Real per-facet structure but at small magnitude (~10× smaller than `meandiff-pcs`'s within of +0.193).
- **`meandiff-pcs` works because contrast subtraction inherently cancels the shared baseline**: anisotropy + IPIP-format shared content + any other common factor between fwd and rev pools cancels in `mean(fwd) − mean(rev)`. PC projection then does an additional ≈5× boost to within-trait cosine (0.042 → 0.193 from raw contrast to PC-projected contrast) — currently unexplained; worth a side investigation.
- **Theoretical takeaway for the superposition-vs-embedding question**: anisotropy dominates the residual stream globally (cosine ≈ +1.0 between any two inputs in absolute space). Trait-distinctive structure exists in a small subspace orthogonal to the dominant anisotropy axis. Within that subspace, facets cluster by trait — embedding-style geometry holds at the *small* magnitude level. Both views co-exist at different scales rather than competing.
