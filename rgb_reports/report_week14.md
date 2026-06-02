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
but-present, not absent. §3 localizes where the two (close) matrices actually
disagree: agreement rises with evaluative intensity, so the model matches humans on
charged words (N/A/E/C carry valence) and diverges on the one valence-neutral trait,
**Openness** (mean row-corr 0.39 vs N 0.60), plus the non-dispositional tail
(Tall/Fat/Employed) — which is *why* O won't factor.

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

## 3. Where the close matrices disagree: Openness and the non-dispositional tail

rgb's follow-up: if the two matrices are *close* (r≈0.56) yet *factor* very
differently, the disagreement can't be uniform — it must localize to specific
adjectives, presumably the ones that define or evade the model's factors. Correct,
and the localization is sharp (`scripts/adjective_divergence.py`). Per adjective,
take the correlation of its row in the human matrix vs the cohort-mean model matrix
(relational agreement), and characterize it.

The intuition was half right, with an instructive inversion. The disagreement is
**not** at the model's *strong*-factor adjectives — it's the opposite:

- **Highest agreement** (row-corr ~0.78–0.81): evaluatively-charged *dispositional*
  words — Cruel, Loving, Abusive, Helpful, Obnoxious, Bitter. Agreement rises with
  the model's evaluative-intensity axis (|PC1|) at **r = 0.58** across all adjectives.
- **Lowest agreement** (row-corr ≤ 0.1): evaluatively-*flat* words — Openness items
  (Predictable, Dependent) and the **non-dispositional tail** (Tall, Fat, Chubby,
  Middle-aged, Employed, Masculine, Left-handed, Short) — the same descriptor cluster
  the §1 appearance aside flagged ("Slim is weird").
- **By trait:** mean row-corr is **N 0.60 > A 0.56 > C 0.54 > E 0.51 ≫ O 0.39**.

The mechanism this exposes: the model **gets N/A/E/C "for free" because those traits
carry valence** (cruel/loving/anxious/reliable are evaluatively loaded), so the
model's evaluative geometry incidentally reproduces the human trait structure for
them. **Openness is the one evaluatively-neutral trait** (intelligent/creative/curious
aren't strongly good-or-bad), so the model — which organizes by evaluation — has no
handle on it and places O words diffusely. That is *why O won't factor*, stated at
the adjective level: O is orthogonal to the model's organizing axis.

This also closes the §2 loop on "how can close matrices factor differently": they're
close on the high-variance *valenced* bulk (shared) and differ on the low-variance
*valence-neutral* residual (Openness + non-dispositional descriptors). Factor
analysis of the human matrix surfaces that residual as the 5th (O) factor; the model
has no coherent residual there, so its 5th-and-beyond factors are evaluative
sub-splits instead. Small matrix difference, large factor difference, localized to
the one trait that doesn't carry valence. (Note: for our *decoder* adjective geometry
it is O specifically that breaks — N recovers best — whereas C&C's encoders had N
weak too; the §3.9 decoder-vs-encoder distinction again. Cohort-mean M; row-corr is
one localization metric.) Figure: `adjective_divergence.png`.

## 4. The evaluative core is two near-orthogonal valence poles (framing fix)

This **refines the W13 §3.11 capstone**, which glossed the evaluative core as
"intensity first, sign second." Looking at the actual factor geometry
(`scripts/adjective_pc_grid.py`), that overstates it — there is no intensity
*factor*. What there is:

**Varimax gives two valence POLES, not intensity + sign.** Rotating the model's
top-2 adjective components (Kaiser) yields a negative-evaluation factor (Disgusting,
Awful, Terrible) and a positive-evaluation factor (Wonderful, Excellent, Amazing) —
a clean pos/neg split — and the unrotated PCs are essentially the same axes (varimax
barely rotates them). Neither rotation produces a "both-poles-load-high" intensity
axis; a true intensity axis exists only as the 45° rotation (pos+neg)/√2 that
*neither* PCA nor varimax selects. That is why §3.11's intensity result had to come
from **RSA presence-templates** (which read that diagonal directly) — factor analysis
structurally can't show it.

**The +0.41 Wonderful≈Awful is a cross-loading, not a shared factor.** In the
2-factor solution Wonderful loads (+0.21 neg, +0.65 pos) and Awful (+0.65 neg,
+0.08 pos): each extreme word carries a small positive loading on the *other* pole.
That shared-extremity leakage — not a common dimension — is the whole +0.41. The
grid (`adjective_pc_grid.png`, adjectives binned by PC1×PC2) makes it visible: the
neg-eval pole (top-left) and pos-eval pole (mid-left) **stack in the high-PC2
columns** rather than sitting in opposite corners, and the (extreme-neg ×
extreme-pos) corner is essentially empty — "Unusual" is its lone, apt resident
(nothing is both maximally good and maximally bad).

**The two poles are the human's single bipolar valence axis, split.** Human
unrotated PC1 (the general desirability factor) maps onto BOTH model poles —
congruence −0.83 to mPC1 and |0.42| to mPC2. One human valence axis → two model
axes. And that shared evaluative axis is most of the story: it carries **~70% of the
relational match** — remove the shared PC1 from both matrices and the adjective r
falls from 0.556 to 0.302. The remaining 0.30 is the genuine (thin) trait residual.
⚠ **The −0.83 is a cohort-mean fact, not a per-model one (W15 §3 / `adjective_bootstrap.py`).**
Averaging the 12 cosine matrices first manufactures a cleaner shared evaluative axis
than most individual models carry: the per-model |human-PC1 · model-PC1| congruence
runs **0.01, 0.33, 0.39, 0.49, 0.66, 0.66, 0.79, 0.81, 0.82, 0.84, 0.89, 0.90**
(median 0.73, cohort-mean 0.83). So the big Gemmas/Qwens genuinely share the human
evaluative axis; a couple of models barely do. The cohort-mean value is robust to
the *word* sample, though — an 80%-adjective subsample CI is [0.752, 0.885], and it
is not carried by the extreme pejoratives (drop them → 0.885). (Don't bootstrap the
12 models for a CI: N is tiny and they're family-correlated; the per-model spread is
the honest summary, and adjective-resampling is the right stability tool.)

**The poles wear trait clothing, but it's eval underneath.** Signed against the
human Big Five, mPC1 is +N / −A / −C (the negativity-vs-warmth/competence bundle;
visually the Agreeableness axis) and mPC2 is +E / −N (admirable-vs-dysregulated; the
Neuroticism axis). But project the human evaluative axis out of both and *every*
trait correlation collapses to ≤|0.32|: the model isn't representing A or N as such,
it's representing two flavors of good/bad that *look* like A and N only because
those are the most valenced human factors. It also splits "bad" two ways —
pejorative/moral (Disgusting, Mean: high PC1) vs dysregulation/distress (Anxious,
Impulsive: low PC2).

**Corrected framing:** the model's evaluative core is **two near-orthogonal valence
poles** (vs the human's one bipolar axis), bound by shared-extremity cross-loadings,
which is why opposite-valence extremes read as neighbors. "Intensity over valence"
named the right *phenomenon* (RSA presence≈valence; Wonderful≈Awful) but the wrong
mechanism-picture — there is no intensity dimension, only un-anti-correlated poles.

## 5. Adjective-resampling: the model-side stability twin (§1 generalized)

The §1 respondent bootstrap rejected the human 6th factor (placidity recovered ≥.90
in 3% of refits). The model has one "respondent", so the symmetric stability test
resamples the dimension we have many of — **adjectives** (rgb). Subsample 80% of the
523 adjectives (no replacement — with-replacement duplicates words and fabricates
perfect-correlation pairs), refit varimax, match each full-set factor to its refit.
`scripts/adjective_bootstrap.py`, `adjective_bootstrap.png`.

**Method validation — the human reproduces the §1 verdict on the *other* axis.** Under
adjective resampling, the human Big Five hold (F1–F5 mean congruence .88–.99) and the
data-driven 6th (placidity) is the fragile one (mean .73, ≥.90 in 41%). Two
independent resampling axes — respondents (§1) and words (here) — give the same
answer, so adjective-resampling is a sound model-side stability tool.

**Model — no factor as robust as the human Big Five; only the evaluative ones hold.**
The cohort-mean model's six varimax factors recover at mean .83–.93 (≥.90 in 40–76%):
the sturdiest are the evaluative factors (~75%), and the warmth factor is its
"placidity" (.83, 40%). None reach the human Big-Five F1–F4 robustness (.86–.98).
So the model's adjective organization is *soft* — a continuous evaluative gradient
more than a set of discrete stable factors, with only the good/bad axes approaching
realness. This is the §3.11 "Big Five is not the LLM's scheme" result re-derived as a
stability statement, and it pins down which structure is *real* vs over-extracted.

**Two robustness axes, two tools (the §4 −0.83 caveat above generalizes).** "Is X a
cohort-mean averaging artifact?" → per-model point estimates (no bootstrap; the spread
is the answer). "Is X robust to the word sample?" → adjective subsample. Bootstrapping
the 12 correlated models conflates these and is statistically weak (small N,
non-independent units); we use neither model-bootstrap.
