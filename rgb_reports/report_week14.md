# Week 14 — Is the adjective Big Five real? + reconciling adjective geometry with IPIP facet recovery

## 0. One-line summary

Follow-up to W13 §3.11 (the single-adjective track on the 525-PDA). Three
independent over-extraction diagnostics — rotation stability, bass-ackwards
lineage, and a respondent bootstrap — run C&C-style on the human 525-PDA and the
LLM cohort, plus a varimax-convention (Kaiser/SPSS) fix that bites exactly the weak
factors. Verdict: **adjective geometry supports ≤5 dimensions with Openness already
shaky, and the model collapses that to a ~2-factor evaluative core**; the human's
data-driven 6th factor (placidity, *not* HEXACO Honesty-Humility) fails resampling
(recovered 3% of the time). §2 reconciles that "very different factor structure" with
the W9/W13 r≈0.56 IPIP facet-geometry match: it's a **metric** difference — under the
relational (matrix-correlation) lens the adjective geometry matches humans at r≈0.56
too (= facets = encoder baseline), so the Big-Five relational structure *is* present;
the "2-factor core" is what the variance-weighted *factor-extraction* lens sees,
because the model's variance concentrates evaluatively. Thin overlay = low-variance-
but-present, not absent.

## 1. Over-extraction: is the adjective Big Five even real?

C&C's supplement argues (their Figs 2, 6–8) that the human adjective Big Five is
already an *over-extraction* — five components only loosely relate to the unrotated
hierarchy, and the bass-ackwards tree shows 2–3 robust dimensions with the rest
"splitting off" the small ones. We ran their two diagnostics on the cohort and
added a third (`scripts/adjective_overextraction.py`, `adjective_factor_bootstrap.py`).

**Varimax convention first (h/t rgb — C&C supp. pp.8–9).** SPSS varimax
*Kaiser-normalizes* (row-scale loadings by communality before rotating); R/Python
defaults often don't, and C&C matched SPSS via Weide & Beauducel (2019). Our
`varimax()` was the un-normalized variant despite its docstring — the exact knob
they flagged. Added `normalize=` (Kaiser default in this analysis). It is
**irrelevant for the human Big Five (robust factors)** but **decisive for the model
(weak factors)**: model k=5 mean rotation-congruence jumps 0.68→0.83 under Kaiser,
and — importantly — the model's evaluation axis cleanly **splits into two separate
factors (Awful and Wonderful) instead of one merged Wonderful**, *strengthening* the
intensity-over-valence read (W13 §3.11). This is exactly C&C's "the knob bites the
weakly-defined factor." All three diagnostics below use Kaiser.

1. **Rotation-stability profile** (`overextraction_profile.png`; their Figs 6–8 as a
   per-component heatmap, k=2→10). Congruence of each *unrotated* component to its
   best varimax match. The flat mean curve is confounded by general-factor strength,
   so we index by variance rank. **Human:** a dominant general factor (PC1 = 0.159,
   ~2× PC2) that varimax *must* split → low PC1 cell; Big Five well-preserved in the
   mid-band; tail fabricated. **Model:** a *flat top* (0.073/0.061 — **no dominant
   general factor** in cosine geometry, itself notable), so PC1/PC2 are rotation-rigid
   (0.99) and the red over-extraction tail grows cleanly. Both land on ~2–4 supported
   dimensions.

2. **Bass-ackwards trees, k=1→10** (`bassackwards_tree.png`; Goldberg 2006, C&C Fig 2
   extended; nodes coloured by k=2 lineage, dashed edge = polarity inversion).
   **Human** = two clean colour blocks — a Neuroticism trunk and an Extraversion
   trunk trackable to k=10 (max-corr descendant r ≈ 0.96–1.0 the whole way), the
   Big Five fanning out by successive bipolar splits (O last), then appearance/
   placidity specifics. **Model** = the colours *tangle*: the evaluative core (PC1
   neg-eval, PC2 pos-eval) is rotation-rigid through k≤5, then the lineage fractures
   — trunks sign-flip and jump tracks (PC1 neg-eval flips onto Energetic at r=−0.78
   around k=7; PC2 pos-eval flips onto Aggressive at k=3). This *reconciles* "PC1/PC2
   are 0.99-stable" (profile) with "you can't trace them down the tree": the top is
   stable, the deep hierarchy reshuffles. The first model split is the headline —
   Disgusting → {Disgusting, Wonderful} at r=−0.25, i.e. positive and negative
   evaluation come apart **near-orthogonally, not bipolar** (the Wonderful≈Awful
   finding from W13 §3.11, now as lineage).

3. **Respondent bootstrap** (`factor6_bootstrap.png`; 500 resamples of the 496
   complete-case 525-PDA respondents, refit k=6 Kaiser varimax, match back). The
   data-driven human 6th factor is **placidity** (Relaxed/Calm/Peaceful) — *not*
   HEXACO's Honesty-Humility (the 8 HH markers present scatter across A and the
   disagreeable-N pole; HH never coheres). But it is **not a real factor**: it
   recovers at congruence ≥.90 in only **3% of resamples** (mean 0.50, p05 0.04),
   versus 77–90% for the Big-Five F1–F4. So the honest read of "525-PDA has a
   different 6th than HEXACO" is **it has no stable 6th at all** — neither placidity
   nor HH survives resampling. **Openness is the soft spot even among the human Big
   Five** (F5 recovers 65% at k=6, box drops to ~0.6 at k=5) — the same O-fragility
   that, in the model, becomes O's *total absence* (W13 §3.11 congruence 0.08).

Two asides. (i) The 525-PDA carries a **non-dispositional appearance cluster**
(Slim, Slender, Gorgeous, Glamorous, Stylish) because Saucier sampled person-
*descriptors*, not just traits; it surfaces as an over-extracted factor (human k=7;
the model pulls "Skinny+Slim" out as early as k=5) — part of *why* the 6th-and-beyond
factors fail resampling. (ii) Three independent methods — rotation algebra,
bass-ackwards lineage, respondent resampling — now agree: **adjective geometry
supports ≤5 dimensions with Openness already shaky; the model collapses that to a
~2-factor evaluative core.** ⚠ The earlier W13 `factor_ladder` / `factor_congruence`
used un-normalized varimax; human-side numbers are safe (robust factors) but the
model's own factor extraction there should be re-checked under Kaiser.

## 2. Reconciling adjective factor structure with IPIP facet recovery

The puzzle (rgb): if the model's adjective *factor structure* is so different from
the human's (2-factor evaluative core, no Openness, no stable 6th — §1, W13 §3.11),
why did the W9 / W13 §3.9 **IPIP facet geometry** nonetheless recover the human facet
correlation matrix *fairly* well (model r≈0.56; encoder baseline r≈0.69)? Not
incompatible — but it had to be laid out. `scripts/reconcile_facet_adjective.py`.

I chased two wrong answers first, which is worth recording because each rules out a
tempting story:

- **Not a coarse 5-block artifact.** Decomposing the 30×30 human facet matrix into
  within-trait/across-trait block means + residual: the cohort still tracks the
  *residual fine structure* at r=0.53 (vs r_full 0.59), and reproduces it both
  within-trait (0.49) and across-trait (0.54). So the facet match is genuine
  relational detail, not just "same-trait facets cluster."
- **Not "Big Five hiding under the evaluative axis."** Removing the top 1–3 eigen-
  components from the adjective geometry and re-factoring the residual makes Big-Five
  congruence *worse* (0.36 → 0.22 → 0.22), not better. The Big Five is not a clean
  rank-5 block sitting beneath the evaluative axis.

**The actual resolution is the metric.** The two analyses use different similarity
lenses on the *same* representation:

- **Relational** (matrix-correlation of all pairwise similarities to the human
  matrix — what the facet r is). Under this metric the **adjective** geometry matches
  the human adjective matrix at **r=0.56 (cohort)** — essentially identical to the
  **facet** r=0.59, and right at the **encoder baseline** (bge: facet 0.69, adjective
  0.58). Per-model, facet_r and adj_r track tightly (Phi4 lowest on both; Qwen7 /
  Qwen32 / Gemma12 highest on both). So there is **no relational contradiction** — the
  model reproduces the human pairwise structure equally for single words and for
  behavioral items, at the encoder-baseline level. The Big-Five *relational* structure
  is present.
- **Dimensional** (Tucker congruence of extracted varimax factors to the human Big
  Five — what "2-factor core" came from). Here recovery is uneven: N/E/A/C ≈ 0.42–0.52,
  **O ≈ 0.10**. Factor extraction is variance-weighted, and the model's variance
  concentrates on the evaluative axis, so it surfaces evaluation first and the Big
  Five (lower-variance) loses to it under rotation/over-extraction.

So **"thin Big-Five overlay" means low-variance-but-present, not absent.** The same
fact reads as "matches humans at r≈0.56" through the relational lens and as
"2-factor evaluative core, O absent" through the dimensional lens — see
`reconcile_facet_adjective.png` (panel A relational, high + stimulus-invariant +
encoder-level; panel B dimensional, O collapses). This also explains why removing the
top PCs didn't help: the Big-Five structure is diffuse across the relational geometry,
not concentrated in eigenvectors 3–7, so factor extraction can't cleanly isolate it
even after deflation.

This folds straight into the §3.9 thesis: model ≈ encoder ≈ baseline for *both*
stimulus types, so what the model reproduces is the **lexical-semantic correlation
structure** of the items/words (relationally), which is not the same as *having the
Big Five as latent dimensions*. The "impressive" facet picture and the "different
factor structure" are one finding under two metrics.

**Caveat.** The two pipelines aren't identical — facet geometry uses
`meandiff-itempc1` contrast directions over 10-item facets, the adjective geometry
uses raw centered cosine over single words, and they compare to different human
matrices (item-response-derived vs adjective-rating-derived). But both land at ~0.56
relational / encoder-level, so the metric reconciliation holds across the pipeline
difference rather than depending on it.
