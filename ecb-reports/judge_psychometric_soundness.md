# JUDGE channel as measurement: a psychometric soundness argument

**Author:** ecb · **Date:** 2026-07-23 · **Reports:**
`psychometrics/judge/reports/01–05` (knitted HTML) · **Data:** rgb JUDGE **full raw
7-category distributions** (`judge_dists_full`, all 12 cohort models, complete 2026-07-20),
523² conditional-endorsement, `B`/`Hent` recomputed from `dists`. This supersedes the
earlier EV-only release; all numbers below are the full-distribution 12-model reanalysis.

## 0. What this is and why

rgb's JUDGE (`tom_likely`) channel elicits each model's **implicit personality theory**:
for every ordered adjective pair *(i, j)*, *"how strongly would a person described as
**i** also be **j**?"*, read as an expected 1–7 rating from the full token log-prob
distribution. It is the "write" side of rgb's read/write story (W15–W16) and the readout
that best reconstructs human personality covariance — the geometric case is theirs.

This writeup asks the orthogonal **measurement** question: *treated as an instrument, is
the JUDGE channel psychometrically sound?* Per the [[validity-argument program]], I frame
it as a **Kane-style validity argument** with the **model as the unit** — here specialized
to **models as raters** of a common 273k-pair judgment task. The three Kane inference
links organize the evidence; the JUDGE channel speaks primarily to the **structural
fidelity** construct, and adds a genuinely new angle the ecb/rgb program lacked: the
**inter-model generalizability** of a semantic-structural measurement.

One caution is load-bearing throughout (and shaped the whole effort): **correspondence to
humans is not, by itself, validity.** Internal structure can show the instrument is
coherent and reproducible; it cannot alone adjudicate *what* it measures (personality vs
learned lexical/evaluative co-occurrence). Human criterion evidence is a separate, weaker
inference, deliberately deferred.

## 1. Scoring inference — from raw judgments to a defensible score

*Warrant: the observed score reflects the intended response, free of construct-irrelevant
scoring artifacts.* Two threats dominate.

**Response styles are rater properties, not constants (Report I).** Across the 12 raters,
leniency (mean 3.6–4.8), differentiation (SD 0.46–1.95) and decisiveness (entropy) each
vary ~4× and are only loosely coupled — three separate scoring nuisances. Two models
(`Llama3.2-3B`, `FalconMamba`) are **range-restricted / near-non-responsive** — `Llama3.2`
effectively uses only {4,5}; `FalconMamba` sits near the uniform entropy ceiling (1.66 nats).
The sensible-looking part — an inverted-U of per-judgment entropy in similarity (certain at
the poles, graded in the middle) — is present for discriminating raters and *absent* for
these two, which is how we know their flatness is guessing, not compression. **New with the
raw distributions (Report I §6):** two moments (`B`, `Hent`) summarize 9 of the 12 raters
well but *fail* for `FalconMamba`, `Phi4` and `Aya` — and *differently*: `FalconMamba` is
diffuse-with-a-digit-prior (78% off-mode but 90% unimodal), `Phi4` broad-unimodal (56%
off-mode, 93% unimodal), `Aya` genuinely bimodal (~15% empty-middle mixtures). This corrects
the earlier medoid-based "faithful for 10/12" claim, which erred by reading faithfulness off
strict bimodality alone (not, as first supposed, because the medoid sample under-represented
spread — it tracked off-mode mass fine). Notably `FalconMamba`'s graded near-uniform mass —
exactly what EV+entropy discard — carries the cohort's **strongest** human-match (EV r≈0.73
vs the 525-PDA structure): the flattest model was the one EV-only storage under-sold most.
**Scoring implication:** cross-model *magnitude* comparison is unsafe; center raters (or use
consistency-type coefficients) before any pooling — and for `FalconMamba`/`Aya`/`Phi4`, a
soft-evidence (full-distribution) treatment is warranted over the EV.

**Asymmetry is real, not negligible (Report II).** `B[i,j] ≠ B[j,i]`. Measured against the
mean-inclusive sum-of-squares the asymmetry looks like 0.2–3.5% — but that denominator is
dominated by the grand mean and the figure is nearly meaningless. Against the **centered**
variance (all that survives the double-centering every downstream use applies) it is
**11–26%, median ~21%** of the meaningful signal. So symmetrizing is *not* a free
rounding-off. It is nonetheless the right scoring choice for a *similarity*, on three
grounds that replace the bad "it's tiny" argument: (a) the antisymmetric part is ~2× less
reliable across raters (r̄ 0.40 vs 0.76); (b) ~half of it is a **prevalence gradient**
(a Helmholtz–Hodge potential ≈ base-rate marginal) that double-centering strips anyway,
and that gradient carries most of its reliability; (c) directionality is a different
construct — not a similarity — so it does not belong in a symmetric proximity regardless
of size. **The scored object is therefore the symmetrized, double-centered `B`; the
directional curl is retained as a separate, flagged channel — sizeable but low-reliability
implicit-personality-theory directionality, not error.**

**What the asymmetry *is*: a Tversky contrast model (Report VI).** Reading the same
decomposition through Tversky (1977) turns two of Report II's descriptive results into
model fits, and amends both. JUDGE's prompt fixes subject/referent roles, so `B[i,j]` is a
contrast-model similarity `θf(A∩B) − αf(A−B) − βf(B−A)`; with an additive feature measure
the antisymmetric part is *exactly* `(β−α)(f_i − f_j)`, a pure gradient with zero curl. So
the Helmholtz–Hodge split is not a chart choice — the **gradient share (mean 50%, consensus
57%) is the contrast model's fit** and the curl is precisely what it cannot produce.

- *Amendment 1 — the gradient is not "the trivial Bayes direction."* It is the model's
  **prominence** term, and prominence is not a base rate. Net of a 6-df natural spline in
  prevalence (R² = 0.72 on its own), the evaluative halo adds **ΔR² = 0.12, t = −19**, in
  every one of the 12 raters, with a **negative** sign: *at equal base rate an undesirable
  trait is the stronger referent* — the standard negativity/diagnosticity asymmetry of
  person perception. Lexical frequency (wordfreq Zipf, 5-unit range) adds **nothing**
  (ΔR² = 0.000, p = 0.51). A triple dissociation: `f` is a feature-salience measure with a
  prevalence component and an independent negativity component, not a probability and not
  word availability. Direction is as Tversky predicts (specific ⇒ general; sign correct for
  76% of all pairs and **97%** of top-decile-prominence-gap pairs), and the size of the
  asymmetry depends on the prominence gap (r = 0.62) but **not** on overlap once the gap is
  partialled (r = −0.01), so additivity is not rejected. Double-centering still correctly
  removes this term for the similarity — but it removes a **construct**, not an artefact.
- *Amendment 2 — the curl is not "faint and idiosyncratic."* Report II correlated it
  **cell by cell** (r̄ 0.34–0.40) — the wrong resolution for a context-dependence effect,
  which shifts weights at category level. At the resolution the theory predicts, the curl
  is **low-rank and strongly shared**: the dominant skew-symmetric circulation plane carries
  47% of curl variance and replicates across raters at **subspace congruence 0.73 against a
  permutation null of 0.05**; projected onto the 5-factor basis, cross-rater agreement is
  **0.72 vs a random-basis null of 0.30 (p < 0.005)**. Its shape is interpretable — an
  evaluative × (misfortune vs malevolence) circulation, with **negative affect as the
  construct-level net source** (anchoring on F3 implies antagonism/incompetence far more
  than the reverse) — and a third of the 5×5 residual is itself a second-order prominence
  potential at trait-factor level.
- *A useful negative.* Tversky & Hutchinson's (1986) nearest-neighbour centrality — the
  test for focal structure no Euclidean space can host — comes out **clean**: C = 2.31,
  max Nᵢ = 9 (a 3-D embedding suffices). **JUDGE is non-metric in its asymmetry, not in its
  symmetric part**, which supports rather than undercuts the geometric treatments. The two
  raters with anomalous centrality (`Qwen2.5-7B` 9.5, `Llama3.2-3B` 8.3) are the
  range-restricted ones, so centrality doubles as a cheap flat-rater diagnostic.
- *Rater-level parameters.* The scale-free asymmetry weight varies ~1.7× across raters
  (0.36–0.64, unrelated to size or family), and a contrast-model **over-identification
  test** — does the same `f` govern the asymmetry and the symmetric marginal? — holds for
  the differentiated raters (`Qwen2.5-32B` 0.84, `Gemma-4-31B` 0.83) and **collapses for
  `Llama3.2-3B` (0.01) and `Aya` (0.16)**, a theory-derived rater-fit statistic that agrees
  with, rather than duplicates, the Report I response-style verdicts.

The measurement verdict is unchanged (symmetrize). What changes is the **status of what we
set aside**: a second construct with its own reliability, its own theory, and its own
model-level parameters — carried forward as such, not filed as nuisance.

## 2. Generalization inference — from one score to the universe score

*Warrant: the score generalizes across the sampled conditions — here, across raters/models.*

Treating the 12 models as raters in a crossed pair × rater G-study (Report III):

- Variance components: **pair (true score) 61%, rater/leniency 11%, residual 29%.** There
  is a large shared signal every rater tracks.
- **A single LLM is a moderately reliable rater** (ICC(2,1) = 0.61); **the 12-rater
  ensemble is excellent** (ICC(2,k) = 0.95). The dependable measurement object is the
  **cohort consensus**, not any one model — the psychometric formalization of rgb's
  wisdom-of-crowds. ~0.07 of single-rater reliability is pure leniency confound
  (consistency 0.68 > agreement 0.61), removable by centering.
- **D-study:** ~3 raters reach absolute G ≥ 0.8, ~6 reach 0.9; a reactive follow-up
  (Report V) shows the structural consensus reproduces from even **~3 raters at r > 0.95**.
  The ensemble's value is real but saturates well below 12, and model identity within the
  crowd is largely fungible.
- **Where the unreliability lives (corrected on the canonical data).** An earlier
  (12-model, EV-only) pass reported a "contested middle"; the full-distribution data show the
  opposite — between-rater disagreement is **highest at the low-similarity, confident end**
  (raters contest which traits are *dissimilar*) and *falls* with per-judgment entropy
  (raters converge on hedged mid-scale judgments). The unreliability is still
  construct-located (definitional negatives are theory-laden), just at the opposite pole.

**A correction worth recording (Reports III & V).** The intuitive move — down-weight the
flat, non-responsive raters — is **unsupported by the data**. On the marginal-free
structural metric the flat raters *agree with the consensus above average* (`FalconMamba`
loo 0.92, `Llama3.2-3B` 0.89); the genuine low-agreement outliers are the high-variance
**idiosyncratic** models (`Gemma-4-31B` 0.76, `Aya` 0.83). Rater quality is
**multidimensional** — information, between-rater agreement, and within-rater consistency
rank the raters differently and are ~orthogonal — which is exactly why a single competence
weight can't help. Competence-weighting and dropping the two idiosyncratic raters each buy
~0.003 in split-half stability over the flat mean; dropping the two flat raters *lowers* it
(mostly the cost of a smaller ensemble). Trimming either extreme nudges single-rater ICC by
*opposite* variance mechanisms (dropping flat raters raises pair variance 0.91→1.10;
dropping idiosyncratic raters cuts residual 0.42→0.35) while leaving the ensemble untouched.
**The flat unweighted mean of ≥6 discriminating raters is near-optimal.** Simplicity costs
nothing.

## 3. Extrapolation inference — from universe score to construct

*Warrant: the universe score reflects the target construct (personality-descriptive
structure), not something else.* This is where the caution bites hardest, so the claims
are deliberately bounded (Report IV).

**Established (internal-structure + structural-invariance evidence):**
- The consensus JUDGE similarity is **coherent and low-rank** — one dominant factor plus a
  short ladder of differentiated trait factors.
- That structure is **reproducible across raters** (mean Tucker congruence to consensus
  0.84; 0.85 again on the general-factor-removed residual). It is a property of the
  *measurement*, not one model — the strongest validity-relevant claim the internal
  evidence supports.
- The differentiated factors **resemble** Agreeableness (antagonism), Neuroticism,
  Openness, and a Conscientiousness/Extraversion blend.

**The central threat: an evaluative halo.** PC1 accounts for ~53% of common variance
(PC1/PC2 = 3.76) — a good/competent ↔ bad/incompetent axis pervading every judgment. It
is double-edged: a coherent, reproducible dimension *and* a **discriminant-validity
threat**, because much of the apparent "trait structure" is one evaluative dimension.
Differentiated structure does survive partialling it out (and still replicates across
raters), but any trait-specific JUDGE score should be taken **net of the general factor**.

**Not established, and not claimable from this evidence:** that the models "have"
personalities, or that the structure reflects anything beyond learned lexical/evaluative
co-occurrence (the deflationary account predicts the halo equally well); and — pointedly —
that the structure matches *human* personality structure. The last is criterion evidence,
handled separately and with care: **a human match would support but not constitute
validity, and a mismatch would not disconfirm a coherent, reliable measurement of the
models' own implicit personality theory.**

## 4. Verdict

As a measurement of the models' implicit personality theory, the JUDGE instrument is:

- **Scoring:** soundly reducible to a symmetrized, double-centered, rater-centered
  similarity — once response-style and (sizeable, ~21%) asymmetry threats are handled
  explicitly rather than waved away.
- **Generalization:** a *single* model is a moderate instrument (ICC 0.61); the **ensemble
  is dependable** (0.95) and robust — unweighted, ≥6 raters, no trimming needed.
- **Extrapolation:** internally coherent, low-dimensional, and **structurally reliable
  across raters**, resembling the Big Five — with an **evaluative halo** that is
  simultaneously its strongest factor and its main threat to discriminant validity, and
  with the "is it personality or lexical co-occurrence?" question left open on purpose.

## 5. Next steps

1. **Pass 2 — human criterion (scoped, caveated).** Use the in-repo 700-respondent human
   525-PDA matrix as *one* criterion strand. Blocked on a careful adjective alignment: the
   human labels are UPPERCASE + 8-char-truncated (`ACCOMPLI`), so a prefix/fuzzy match with
   collision auditing is needed. Frame strictly per §0.
2. **Directional (curl) construct — upgraded by Report VI.** Net of prevalence the
   asymmetry is a *diagnosticity* channel that is low-rank and **strongly** reproducible at
   plane/construct resolution (0.72–0.73 vs nulls of 0.05–0.30), not the low-reliability
   residual Report II described. Its dominant plane and the construct-level flow (negative
   affect as net source) are the natural bridge to rgb's person-perception cycles (W16 §5).
6. **Collect the 523 self-pairs (Report VI §8).** The `dists` diagonal is NaN, but
   Tversky's `s(a,a) = θ·f(A)` is the *direct* measurement of prominence and the only route
   to identifying α and β separately (we currently recover only `(β−α)f`). 523 prompts per
   model in the verbatim `tom_likely` frame; four cohort models are on the Orin. Also yields
   a **minimality** test (metric non-metricity that no symmetrizing can remove) and a free
   response-process check. Cheapest high-value collection on the track.
3. **Halo-controlled trait scores.** Provide general-factor-partialled JUDGE trait scales
   for use as a method in the model-level MTMM.
4. **Tie into the Kane MTMM.** JUDGE joins Likert / GFC-TIRT / rgb-axis as a method
   speaking to **structural fidelity**; the inter-model generalizability result is a new
   reliability warrant the program can reuse.
5. **Soft-evidence treatment (now unblocked; full cohort in hand).** The full 7-way
   distributions (all 12 models) enable distributional similarity (e.g. earth-mover distance
   between judgment distributions) as an alternative to EV-cosine, and genuine confidence
   weighting from real distributions — most valuable for `FalconMamba`/`Aya`/`Phi4`, whose EV
   the two moments misread. Bridges ecb's earlier soft-evidence IRT work. `FalconMamba`'s
   distribution-level human-match (EV r≈0.73) is the strongest single argument for it.
