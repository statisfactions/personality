# JUDGE channel as measurement: a psychometric soundness argument

**Author:** ecb · **Date:** 2026-07-22 · **Reports:**
`psychometrics/judge/reports/01–05` (knitted HTML) · **Data:** rgb JUDGE **full raw
7-category distributions** (`judge_dists_full` batch 1: 9 of the 12-model cohort;
`FalconMamba`, `Gemma-4`, `Qwen-32` pending), 523² conditional-endorsement, `B`/`Hent`
recomputed from `dists`. This supersedes the earlier EV-only 12-model release; all numbers
below are the 9-model reanalysis.

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

**Response styles are rater properties, not constants (Report I).** Across the 9 raters,
leniency (mean 3.6–4.8), differentiation (SD 0.46–1.95) and decisiveness (entropy) each
vary ~4× and are only loosely coupled — three separate scoring nuisances. One model
(`Llama3.2-3B`) is **range-restricted / near-non-responsive** — it effectively uses only
categories {4,5} and is essentially never decisive (0.002% of judgments below entropy 0.1).
The sensible-looking part — an inverted-U of per-judgment entropy in similarity (certain at
the poles, graded in the middle) — is present for discriminating raters and *absent* for
the flat one, which is how we know its flatness is guessing, not compression. **New with the
raw distributions (Report I §6):** two moments (`B`, `Hent`) summarize 7 of the 9 raters
well but *fail* for `Phi4` (56% of cells carry ≥0.25 off-mode mass — broad-unimodal
distributions the EV mis-summarizes) and `Aya` (genuinely bimodal: ~15% empty-middle "no OR
yes" mixtures). This corrects the earlier medoid-based "faithful for 10/12" claim, which
erred by reading faithfulness off strict bimodality alone. **Scoring implication:**
cross-model *magnitude* comparison is unsafe; center raters (or use consistency-type
coefficients) before any pooling — and for `Aya`/`Phi4`, a soft-evidence (full-distribution)
treatment is warranted over the EV.

**Asymmetry is real, not negligible (Report II).** `B[i,j] ≠ B[j,i]`. Measured against the
mean-inclusive sum-of-squares the asymmetry looks like 0.2–3.5% — but that denominator is
dominated by the grand mean and the figure is nearly meaningless. Against the **centered**
variance (all that survives the double-centering every downstream use applies) it is
**18–26%, median ~20%** of the meaningful signal. So symmetrizing is *not* a free
rounding-off. It is nonetheless the right scoring choice for a *similarity*, on three
grounds that replace the bad "it's tiny" argument: (a) the antisymmetric part is ~2× less
reliable across raters (r̄ 0.43 vs 0.77); (b) ~half of it is a **prevalence gradient**
(a Helmholtz–Hodge potential ≈ base-rate marginal) that double-centering strips anyway,
and that gradient carries most of its reliability; (c) directionality is a different
construct — not a similarity — so it does not belong in a symmetric proximity regardless
of size. **The scored object is therefore the symmetrized, double-centered `B`; the
directional curl is retained as a separate, flagged channel — sizeable but low-reliability
implicit-personality-theory directionality, not error.**

## 2. Generalization inference — from one score to the universe score

*Warrant: the score generalizes across the sampled conditions — here, across raters/models.*

Treating the 9 models as raters in a crossed pair × rater G-study (Report III):

- Variance components: **pair (true score) 62%, rater/leniency 10%, residual 28%.** There
  is a large shared signal every rater tracks.
- **A single LLM is a moderately reliable rater** (ICC(2,1) = 0.62); **the 9-rater
  ensemble is excellent** (ICC(2,k) = 0.94). The dependable measurement object is the
  **cohort consensus**, not any one model — the psychometric formalization of rgb's
  wisdom-of-crowds. ~0.07 of single-rater reliability is pure leniency confound
  (consistency 0.69 > agreement 0.62), removable by centering.
- **D-study:** ~3 raters reach absolute G ≥ 0.8, ~6 reach 0.9; a reactive follow-up
  (Report V) shows the structural consensus reproduces from even **2–4 raters at r > 0.95**.
  The ensemble's value is real but saturates well below 9, and model identity within the
  crowd is largely fungible.
- **Where the unreliability lives (corrected on the canonical data).** An earlier
  (12-model, EV-only) pass reported a "contested middle"; the full 9-model data show the
  opposite — between-rater disagreement is **highest at the low-similarity, confident end**
  (raters contest which traits are *dissimilar*) and *falls* with per-judgment entropy
  (raters converge on hedged mid-scale judgments). The unreliability is still
  construct-located (definitional negatives are theory-laden), just at the opposite pole.

**A correction worth recording (Reports III & V).** The intuitive move — down-weight the
flat, non-responsive rater — is **unsupported by the data**. On the marginal-free
structural metric the flat `Llama3.2-3B` *agrees with the consensus at the cohort mean*
(loo 0.89, above five of eight others); the genuine low-agreement outliers are the
high-variance **idiosyncratic** models (`Aya` 0.84, `Gemma3-4B` 0.85). (In the full 12-model
cohort `Gemma-4-31B` was the extreme outlier; it is pending in batch 2.) Rater quality is
**multidimensional** — information, between-rater agreement, and within-rater consistency
rank the raters differently and are ~orthogonal — which is exactly why a single competence
weight can't help. Competence-weighting and dropping the flat rater each buy ~0.003 in
split-half stability over the flat mean; dropping the two idiosyncratic raters *lowers* it
(mostly the cost of a smaller ensemble). Dropping the flat rater raises single-rater ICC by
adding true-score variance (pair var 1.06→1.26); dropping the idiosyncratic pair barely
moves it (cutting signal and residual together) — the ensemble is untouched throughout.
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
  0.86; 0.87 again on the general-factor-removed residual). It is a property of the
  *measurement*, not one model — the strongest validity-relevant claim the internal
  evidence supports.
- The differentiated factors **resemble** Agreeableness (antagonism), Neuroticism,
  Openness, and a Conscientiousness/Extraversion blend.

**The central threat: an evaluative halo.** PC1 accounts for ~55% of common variance
(PC1/PC2 = 3.8) — a good/competent ↔ bad/incompetent axis pervading every judgment. It
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
  similarity — once response-style and (sizeable, ~20%) asymmetry threats are handled
  explicitly rather than waved away.
- **Generalization:** a *single* model is a moderate instrument (ICC 0.62); the **ensemble
  is dependable** (0.94) and robust — unweighted, ≥6 raters, no trimming needed.
- **Extrapolation:** internally coherent, low-dimensional, and **structurally reliable
  across raters**, resembling the Big Five — with an **evaluative halo** that is
  simultaneously its strongest factor and its main threat to discriminant validity, and
  with the "is it personality or lexical co-occurrence?" question left open on purpose.

## 5. Next steps

1. **Pass 2 — human criterion (scoped, caveated).** Use the in-repo 700-respondent human
   525-PDA matrix as *one* criterion strand. Blocked on a careful adjective alignment: the
   human labels are UPPERCASE + 8-char-truncated (`ACCOMPLI`), so a prefix/fuzzy match with
   collision auditing is needed. Frame strictly per §0.
2. **Directional (curl) construct.** The ~20% asymmetry, net of prevalence, is a
   low-reliability but non-zero implicit-personality-theory *directionality* — worth its
   own small study, and the natural bridge to rgb's person-perception cycles (W16 §5).
3. **Halo-controlled trait scores.** Provide general-factor-partialled JUDGE trait scales
   for use as a method in the model-level MTMM.
4. **Tie into the Kane MTMM.** JUDGE joins Likert / GFC-TIRT / rgb-axis as a method
   speaking to **structural fidelity**; the inter-model generalizability result is a new
   reliability warrant the program can reuse.
5. **Soft-evidence treatment (now unblocked).** The full 7-way distributions (batch 1)
   enable distributional similarity (e.g. earth-mover distance between judgment
   distributions) as an alternative to EV-cosine, and genuine confidence weighting from real
   distributions — most valuable for `Aya`/`Phi4`, whose EV the two moments misread. Bridges
   ecb's earlier soft-evidence IRT work. Completes when batch 2 (`FalconMamba`, `Gemma-4`,
   `Qwen-32`) lands.
