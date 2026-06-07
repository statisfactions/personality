# Week 16 — How far back does the merge go?

**Dates:** 2026-06-03 →
**Calendar:** ~cal-week 11–12 (labels run ~2–3 ahead; see `overview.md`).

Reading-group question (rgb): W15 showed LLMs *represent* the evaluative antonyms
Wonderful ≈ Awful as merged but *judge* them apart, and the merge is in current
transformer encoders too. So — how much further back does it stretch? Do
pre-transformer LMs (LSTMs, HMMs, CRFs) experience it? This week answers the
empirical end (static word vectors) and reasons through the rest.

---

## §1 — The regress: the merge bottoms out at the distributional hypothesis

**Design** (`scripts/adjective_regress.py`, fig
`results/adjectives/regress/regress.png`). Take the same 26 pole-spanning
adjectives from W15 (pos-eval Wonderful/Amazing/Excellent/Great; neg-eval
Awful/Disgusting/Terrible/Bad; warmth, antagonism, distress, intellect, neutral).
Compute one similarity matrix per *stratum* of a model-class regress, newest →
oldest, and read the same statistic on each, **z-scored within the matrix** (the
claim is purely relational — "are the eval-antonyms positioned like synonyms, on
this model's own scale"):

- **synonym-z** — within-pole eval similarity (Wonderful~Amazing, Awful~Terrible).
- **antonym-z** — pos-eval × neg-eval (Wonderful~Awful). **The merge test.**
- **crossdim-z** — eval × intellect/distress (a different content dimension; floor).

Merge ⇒ antonym-z > 0 and ≈ synonym-z. Split ⇒ antonym-z < 0. For single words
the sentence-transformer average-word-embedding *is* the static word vector
(GloVe/komninos) or the encoder's word encoding (bge/mpnet); bare lowercased word
— the canonical word-similarity setup and the honest form for GloVe.

**Result.**

| stratum | objective / era | synonym-z | **antonym-z** | crossdim-z | verdict |
|---|---|---:|---:|---:|---|
| LLM-repr (Qwen2.5-7B) | autoregressive transformer, residual stream | 3.32 | **+1.58** | −0.79 | MERGED |
| komninos | dependency-window word2vec, 2016 | 3.03 | **+1.18** | −0.13 | MERGED |
| glove.6B | static count-predict, Wikipedia 6B, 2014 | 3.05 | **+0.71** | −0.45 | MERGED |
| mpnet-base | masked-LM + NLI encoder, 2021 | 3.79 | **+0.51** | −0.69 | MERGED |
| glove.840B | static count-predict, web 840B, 2014 | 3.01 | **+0.24** | −0.45 | MERGED |
| bge-large | retrieval-contrastive encoder, 2023 | 3.61 | **+0.16** | −0.45 | MERGED |
| **LLM-judge (Qwen2.5-7B)** | rating task (the WRITE side) | 2.98 | **−0.49** | −0.12 | split |
| **human** | 525-PDA self-report (N≈525) | 2.38 | **−0.53** | −0.01 | split |

**Three things read straight off the table.**

1. **Every distributional-geometry model merges the antonyms** — antonym-z is
   positive from 2014 static GloVe through 2023 encoders to the LLM's resting
   representation. The merge is **already complete in pre-transformer,
   pre-contextual static word vectors**. It is not a transformer phenomenon, a
   depth phenomenon, or a scale phenomenon. It is a property of learning meaning
   from co-occurrence — the distributional hypothesis itself. (Antonyms-are-
   distributional-neighbours is the oldest documented failure mode of vector-space
   semantics; the entire counter-fitting / antonym-aware-embedding literature
   exists to undo it.)

2. **Only judgment and human self-report split them** — antonym-z is negative for
   exactly the two strata that involve an explicit per-word decision, and the two
   land on top of each other (−0.49 vs −0.53). The LLM's *write* side reproduces
   the human antonym value almost exactly. Everything in between — every readout
   that is a *geometry of co-occurrence* rather than a *decision* — merges.

3. **The whole geometry agrees except one cell.** synonym-z is uniformly high
   (~2.4–3.8) and crossdim-z uniformly at/below zero *everywhere*, human included.
   Synonyms group, different dimensions separate — universally. The **only** block
   whose sign flips between the distributional models and the (human + judgment)
   pair is the eval-antonym block. The single thing the regress disagrees about is
   exactly the W15 read/write cell. The merge is one coherent error sitting on top
   of an otherwise correct geometry.

**A nuance worth stating (and not over-reading).** Merge *magnitude* is **not**
era-monotonic: the LLM representation merges hardest (+1.58), a 2016
dependency-word2vec (komninos, +1.18) merges harder than a 2023 retrieval encoder
(bge, +0.16). So nothing about modernity predicts the *strength* of the merge; the
distributional objective predicts its *sign*. The encoder gradient (bge +0.16 <
mpnet +0.51) is suggestive — retrieval-contrastive training, which explicitly
pushes semantically-distinct items apart, nudges the antonyms partway back toward
splitting without ever flipping the sign — but on 16 block cells over 26 words this
ordering is noisy; the robust claim is the sign, not the ranking.

## §2 — Below static vectors: LSTMs, HMMs, CRFs

The empirical regress stops at static word vectors because that is the oldest
class with an off-the-shelf word geometry. The rest is reasoning from what those
classes *are*, and it all points the same way — the merge gets *worse*, not
better, as the models get simpler, because they have *less* machinery to ever
represent the symbolic distinction.

- **LSTM language models / ELMo.** A contextual LSTM LM is still trained on the
  distributional objective (predict the next/context word), so its word-type
  geometry inherits the antonym proximity directly — ELMo's nearest-neighbour
  lists show the same good/bad adjacency the static vectors do. Contextualization
  gives it a *route* to the distinction (an LSTM can in principle use a negation in
  the context), but its resting type-level geometry merges, exactly like the LLM's
  residual stream does. We expect an LSTM to sit with the encoders: merged at the
  type level, with a weak context-dependent escape hatch. (We did not run ELMo —
  it needs the AllenNLP/TF-Hub stack — but the prediction is unambiguous.)

- **Class-based / HMM language models (Brown clusters, 1992).** These merge the
  antonyms **harder** than any continuous model, and in a qualitatively different
  way. A class-based n-gram LM induces discrete word classes from bigram
  distributions; *good* and *bad* share their distributional slot almost perfectly
  (same intensifiers, same syntactic position, same emphatic templates), so they
  are assigned to the **same induced class** — not merely placed nearby in a
  continuous space, but rendered *identical* for the model's purposes. The merge is
  total. An HMM POS-tagger is the limiting case: it collapses both to "adjective"
  and is done.

- **CRFs.** A category error for this question, and the exception that proves the
  rule. A CRF is a *supervised, discriminative* sequence labeler; it induces no
  unsupervised word-similarity geometry at all. It "merges" or "splits" nothing on
  its own — it has whatever distinctions its hand-engineered features and labels
  give it. Insofar as those features are distributional (Brown-cluster IDs, word
  embeddings), it inherits the merge from *them*; insofar as a human supplied a
  sentiment label, it has the split *because a human wrote it in*. The CRF has no
  associative substrate of its own to merge with — which is precisely why it is the
  wrong place to look, and why the interesting object is always a model that learns
  its geometry from co-occurrence.

So the regress bottoms out not at an architecture but at an **objective**: any
model that *learns meaning from co-occurrence* merges the evaluative antonyms, and
the simpler/older the model, the more total the merge. Architecture, depth, scale,
and recency are all orthogonal to it.

## §3 — What this does to the contribution

The reframe is the point. The geometry that Wulff & Mata (2025) and Milano et al.
(2025) recover from item text — and that our embedding baseline (W13 §3.9)
confirmed the LLM does not beat — is the **old, inherited** part. The
distributional merge of evaluative antonyms is ~30 years old and falls out of
PPMI-SVD with no learning at all.

What is **new in LLMs is the write side.** A GloVe vector is *all read, no write* —
it physically cannot un-merge the antonyms, because it has no behavioral channel to
deploy a symbolic distinction through. The LLM has one, and uses it: its judgment
lands on the human antonym value while its own representation sits at the far
merged end of the entire regress. **Within one model, the read and write sides
occupy opposite extremes of a 30-year regress** — the most-merged representation in
the table and a human-matching split judgment.

This is the behavioral teeth on the symbolic-vs-associative frame
(`project_symbolic_vs_associative`, W7) and on "representation isn't intention"
(`user_representation_not_intention`): the associative substrate is as old as
distributional semantics; the symbolic override is the thing tuning installs and
the thing our read/write instruments measure. The deviation from the embedding
baseline — not the match to it — is what is about *this kind of model*, and the
deviation is *behavioral*.

## §4 — Methods aside: should the paired and unpaired denoise rules be unified?

Tangential to the regress, but it came up while documenting the method-of-record.
We run two anisotropy-denoise rules: **paired** facet/persona extraction uses
`meandiff-itempc1` (always project the item top-1 PC out of `mean(fwd) − mean(rev)`),
**unpaired** adjective geometry uses `adaptive_denoise` (remove centered-PC1 *only*
when its inverse participation ratio flags a concentrated spike — IPR < 10). The
question (rgb): in the longer run, would unifying to a single `meandiff-adaptive`
rule — gate the paired PC1 removal on the same IPR test — be cleaner, betting it
barely moves the numbers?

Tested it (`ipip_facet_cluster.py --extraction meandiff-adaptive`, a tagged-output
diagnostic built from the cached acts — canonical path untouched; cohort r vs the
human facet matrix). Two things, one expected, one not.

**The IPR gate fires for most models in the *facet* cloud, not just Gemma.** REMOVE
(IPR<10): all Gemmas (1.0–2.1), both Llamas (4.9–5.2), Qwen7 (2.3), Qwen32 (3.6),
Aya (4.5). KEEP (IPR>10): **Phi4 (31.6), Qwen-3B (25.1), FalconMamba (208.7)** — the
low-anisotropy models. So unlike the *adjective* cloud (where only Gemma's massive
activation makes centered-PC1 a spike), the IPIP-item cloud has a concentrated top
PC for most models — the shared chat-template/sentence-structure direction. So
`meandiff-adaptive` is **bit-identical to `meandiff-itempc1` for 9/12 models**, and
differs only for the three low-anisotropy ones.

**The bet holds, but the prediction was backwards.** Cohort r 0.561 → 0.551 (Δ
−0.010); 9/12 models exactly identical. I had predicted the gate would *help* the
low-anisotropy models by sparing them an over-projection. The opposite: keeping PC1
made all three **worse** — FalconMamba 0.507 → 0.427 (−0.080), Qwen-3B 0.573 → 0.541
(−0.032), Phi4 0.404 → 0.401 (−0.003). Removing item-PC1 is beneficial *even when
PC1 is "distributed"* (high IPR); it still carries the shared item-structure
nuisance that hurts facet geometry.

**Conclusion: don't unify — the split is principled, and now has a number.** The
gate's "keep distributed PC1" rule is correct for the *unpaired* pipeline (there a
distributed PC1 might *be* the trait signal, so it must be protected) but *wrong*
for the *paired* pipeline, because the paired trait signal is contrast-defined and
lives in `fwd − rev` — PC1 is *never* the signal, so always-remove is both safe and
beneficial, and the gate's protection only costs you nuisance-removal you wanted.
This is the exact corollary of the W15/W16 anisotropy story: *because* the contrast
separates signal from the dominant axis for free, you can always strip that axis.
`meandiff-itempc1` (fixed) for paired, `adaptive_denoise` for unpaired, stays
canonical; unifying would cost −0.010 cohort and −0.08 on the SSM. The
`meandiff-adaptive` method is left in place as a diagnostic only.

## §5 — Read/write at full adjective resolution + the instrument (IN PROGRESS)

Terse capture of an in-flight thread (scripts run, numbers reproducible; full
consolidation + the human-comparison test pending — see Open). Provisional;
flagged where speculative.

**Full 523² read vs write.** Built the full behavioral matrix (`adjective_judge_full.py`)
to compare the representation (READ, `adjective_geom` cosine) against the judgment
(WRITE) at full resolution, against an unbiased distribution rather than the
26-word probe set. Report in **PERCENTILES, not z** — the judgment EV distribution
is floored and heavily skewed (semantic: 86% of pairs at the scale floor, median
digit-entropy exactly 0), so z misleads. Result (`adjective_readwrite_full.py`,
percentile transform): the representation ranks the eval-antonyms (Wonderful~Awful)
at the **98th percentile** of similarity, the judgment at the **~10th** — an ~88-pt
read/write split. **Robust**: replicates across the semantic and the dispositional
(`tom_likely`) instruments, and cross-family (Qwen7 88 pts, Gemma12 64 pts). The
merge was, if anything, *understated* by the eval-heavy probe set (its synonym-dense
baseline raised the bar). NOTE: an earlier z-based pass surfaced *morphological*
twins (Self-assured~Self-conscious) as the top read-merge; that was a z tail-inflation
artifact (shared subword tokens → extreme cosine). Under percentile the top read-merge
pairs are clean valence antonyms; the warp is high-rank (top mode ~5%) with the
evaluative axis as its largest single mode — a broad form-vs-meaning warp, not a
clean low-rank shear.

**Judgment is single-pole.** ~79% of semantic judgments put all mass on "1"
(median entropy 0); entropy tracks EV (r≈0.8) — the model only grades where it sees
similarity. The graded block is the *positive*-evaluative cloud (Wonderful, Great,
Admirable); negatives are crisp. mean-entropy (0.10) describes almost no pair.

**Instrument design (`adjective_question_form.py`).** Both default scales are
UNIPOLAR (1=different…7=same), so *opposite* and *unrelated* both floor. Swept
forms; want SYN > UNR > ANT with UNR at the middle. `sem-bi` (bipolar: 1=opposite,
4=unrelated, 7=same) separates cleanly on **capable** models (Qwen, Gemma) but
**COLLAPSES unsure models** (OLMo std→0.58, Llama→0.11; the explicit midpoint anchor
becomes a default the near-uniform model retreats into — OLMo discriminates fine on
tom-likely but mushes on sem-bi, so it's the *form*). `tom_likely` ("how likely is a
very-A person to also be B", 1=unlikely…7=likely) is **robust across the whole
cohort** — the midpoint emerges from the *semantics of probability*, not an
instruction. Chosen as the cohort instrument. Design principle: get the midpoint
from the question's meaning, not from an anchor.

**ToM vs semantic is the well-defined finding.** semantic = lexical meaning-overlap;
`tom_likely` = dispositional co-occurrence (the model's *implicit personality
theory*). They diverge where distinct-meaning traits co-occur in a person. Defined
for models that discriminate on both (Qwen, Gemma); undefined for near-uniform
(Llama). The clean, pre-registerable test (PENDING): does the ToM geometry match the
human 525-PDA *correlational* structure better than semantic or representation? —
same backbone as the W9/W13 facet comparison, the dispositional channel being the
construct the human data actually measures.

### Anatomy of the directional asymmetry (Hodge decomposition)

The both-directions `tom_likely` matrix is asymmetric (P(B|A) ≠ P(A|B)); we took it
apart. Scripts: `adjective_very_test.py`, `adjective_anchor_entropy.py`,
`adjective_asymmetry_dims.py`, `adjective_cycles.py`, `adjective_dispositional_curl.py`.

**1. Robust one-directional exclusion.** P(neg|very-pos) > P(pos|very-neg) in both
models (Qwen 2.47>1.69, Gemma 4.11>3.00). It survives **base-rate control** — positive
traits have a much higher ascription base rate (Qwen col-means: pos 3.97 vs neg 2.64),
and after subtracting it the residual gap *grows* to ~2.1–2.5 (cross-family) — and it
survives **dropping "very"** (the raw gap shrinks ~35% for Qwen, but the base-rate-
controlled gap is invariant; for Gemma even the raw gap barely moves). Precise form:
a *bad* anchor strongly suppresses the good target (well below its base rate); a *good*
anchor leaves the bad target *at* its base rate. **Bad excludes good; good tolerates
bad.**

**2. Valence, not variance (mechanism).** Two accounts for that exclusion: valence/
structural vs variance-evidence (good people carry wider behavioral variance → less
reliably ascribable; or "bad person" needs more evidence → tighter posterior). The
clean test is the model's own posterior width = the per-judgment **entropy** of
P(B|X) — the "how predictable is a person who is X" question answered *without* a
verbal probe (which would only re-measure the target word's denotation). Level-
controlled (entropy is an inverted-U in EV, and positivity bias gives good anchors
higher EV), good anchors are **not** more uncertain than bad at matched EV (Qwen
slightly *less*; Gemma flat). So variance/evidence gets **no support**; the exclusion
is structural valence.

**3. But valence is NOT the primary dimension.** Treat the asymmetry as an
antisymmetric **flow** D[A,B] = P(B|A) − P(A|B) on the trait graph. Helmholtz–Hodge:
D = gradient (a single transitive potential, curl-free, carrying all the divergence) +
curl (divergence-free cycles). Only **~40–50%** is gradient, and that gradient is
**prevalence / ascription-readiness** (universally-ascribed traits like Awake, Normal,
Employed are "sinks"; rare traits are "sources"), correlating **r²≈0.01 with valence**
— robust even on a hand-curated **411-word dispositional subset** (drops physical/
demographic/status/state/markedness words; prevalence still r≈−0.82, valence still ~0,
and the gradient fraction *falls* — dispositional traits are *more* cyclic). So the
dominant rankable axis is the trivial Bayes direction; valence is a base-rate-
controlled, **subdominant** effect that lives in the curl, which is exactly why the
base-rate control was load-bearing to see it at all.

**4. The secondary dimensions are person-perception axes.** SVD of the curl (each
degenerate singular pair = a rotational *plane*; individual axes within a plane are
not unique, so read clusters not axes) resolves into the classic social-perception
dimensions: **warmth/likeability, competence, dominance, morality (=valence),
neuroticism.** Valence is *one* co-equal thread, not the driver. The cycles are
interpretable: the **warmth×morality** plane circulates Likeable→Evil→Cold→Ethical→
Likeable (circulation +6.19), and it's a *cycle* rather than a single good/bad ranking
precisely because the model holds the off-diagonal archetypes — the **charming villain**
(likeable-but-evil) and the **cold saint** (cold-but-ethical) — as asymmetrically
conceivable. The **confidence×competence** plane clusters *Confident / Self-assured*
with *Foolish / Stupid* opposite *Modest* — i.e. the geometry encodes **Dunning–Kruger**
(unwarranted confidence sits with folly, not competence). The one-directional exclusion
reappears inside this structure as the Likeable→Evil leg (evil excludes likeable;
likeable tolerates evil).

**4b. Reading the planes — rotation caveat + an interpretive cross-model difference
(n=2, hold loosely; `adjective_curl_rotate.py`).** The SVD basis within each
degenerate plane is *arbitrary* (singular values pair exactly, e.g. 122,122,…), so
the axis-by-axis labels in §4 are only as canonical as the basis numpy happened to
return — Qwen's landed at simple structure, Gemma's did not, which is the whole
reason Gemma's raw axes looked muddier. Re-rotating each plane to varimax simple
structure: Qwen stays four clean social-perception axes (warmth, morality,
competence, the DK-confidence cluster); Gemma's valence and dominance/threat axes
*resolve* cleanly, but one axis genuinely resists — an **intensity/arousal** axis
(Valuable/Desirable/Terrific/**Evil** [impactful] ↔ Boring/Awkward/Anxious/Shy
[mild/timid]; the valence-sign mixing is the arousal *signature*, not smear). A
suggestive reading: Qwen factors the curl by **social perception** (warmth ×
competence × morality), Gemma carries an extra **affect-circumplex** axis (valence ×
arousal) — which would fit Gemma's prior affect-leaning (W9 §7.6 affect axis, the
Cheerfulness/N merge, "anthropomorphizes hardest"). **CAVEAT, load-bearing:** all of
§4–§4b is words read off rotated axes on *two* models. The archetype stories
(charming villain, cold saint, Dunning–Kruger) and the social-vs-affect split are
evocative and internally consistent, but they are **interpretive hypotheses for the
cohort to test, not established findings** — fun to tell, not to overshoot.

**5. Reframe.** This *de-centers* our own valence story: the Wonderful≈Awful read/write
split is real and specific to the evaluative pole, but it is **not "what the asymmetry
is about."** Globally the asymmetry is prevalence (primary, ~half, trivial) + a basket
of person-perception dimensions (secondary curl), with valence as one strand.

**6. Instrument caveat — "also be" ⇒ "become".** `tom_likely`'s "how likely to *also
be* B" is ambiguous between *co-occurrence* and *temporal succession*. For developmental/
state traits the model reads "become": Gemma's strongest cycle is Young→Middle-aged→
Elderly→Young — pure life-course progression, no valence. For a clean dispositional
reading, exclude developmental/state traits (as the §3 subset does) or split "is also"
from "will become."

**7. Gemma leniency — SPECULATIVE.** Gemma rates `tom_likely` higher overall (mean EV
4.29 vs 3.63) and splits antonyms more gently (judge 29th vs 10th pct) — confounded with
plain **scale calibration** (it may just use the upper half of the scale); no within-
model calibration control, no sharp hypothesis → record as *observation*, not finding.
Only the within-model direction of any asymmetry is safe; cross-model *magnitudes* ride
on the calibration confound.

**8. The dispositional channel is the human-matched readout (`adjective_human_match.py`).**
Compared each readout's symmetric structure to the human 525-PDA inter-adjective
**correlation** matrix. Correction (rgb): mean-symmetrizing only cancels the
antisymmetric part; the symmetric remainder still carries the prevalence marginals,
which a marginal-free human correlation matrix lacks — so strip additive marginals
from *both* sides (off-diagonal double-centering) before comparing. Result (Pearson of
upper-triangles, corrected): **`tom_likely` matches human best** — Qwen7 **r = 0.73**
(vs semantic 0.46, representation 0.39); Gemma12 0.69 (vs 0.68, 0.59). The
prevalence-correction is load-bearing and *reveals* the result — it lifts `tom_likely`
by +0.12 (Qwen) / +0.06 (Gemma) while barely moving representation (+0.01, already
centered): the base-rate marginals were **diluting** the match (model-side structure
with no human analog), not inflating it. Read/write flavor in the column:
**representation matches human *worst*, judgment *best*** — the resting associative
geometry is a poorer witness to human personality covariance than the dispositional
thing the model *does* when asked. Caveats: Gemma's edge over semantic is thin
(calibration); the full 523 still includes developmental/state traits (the "become"
leak), so a dispositional-subset re-run is the clean follow-up. Open cohort question:
does `tom_likely` stay the human-matched readout across all models?

**8b. Aggregation — the right comparison object is the *ensemble* (cohort partial,
8 of 12; `adjective_human_match_viz.py --model cohort`).** The human matrix
aggregates over hundreds of people, each integrating many interactions; the model
gives one judgment per pair (rgb). Averaging the 8 completed models' double-centered
matrices (an "8-judge aggregate") shows two things: (1) **wisdom of crowds** — the
aggregate matches human at **r = 0.815, above every single model** (best 0.789); and
(2) the aggregate's value distribution **normalizes to human shape** — excess kurtosis
**−0.28** vs human −0.35, against the leptokurtic *single-judge* outliers (Qwen7 +1.24,
Llama-3B +1.45) whose "washed-out mid-tones" prompted this. The 2-fold both-directions
average barely dents the spike (1.37→1.24), so it takes *many* judges, matching the
human-aggregates-over-people picture. Two honest qualifiers: (a) the per-judge
leptokurtosis is **idiosyncratic** — only Qwen7 and Llama-3B; most models already sit
near the human kurtosis (−0.45…+0.08) — so "the model is categorical about
co-occurrence" is false as a general claim; (b) the cohort aggregates over *models*
while the human matrix aggregates over *people* — a persona-aggregate *within* one
model would be the tighter analogue. Figure: `human_match_cohort.png` (the aggregate's
mids fill in and blocks sharpen vs any single model). Will firm up at the full 12.

## §6 — RLVR and model character (exploratory thread)

*Context worth flagging:* there is live external suspicion that RLVR (RL with
verifiable rewards — the reasoning-model post-training paradigm) is doing *something*
to the "character" of frontier models. Our cohort handed us a hook: across the
`tom_likely` human-match, **Gemma-4-31B is the lone true cohort outlier** (corrected
r 0.555 vs the field's 0.69–0.79, and the only model whose deviation survives a
leave-family-out check), and its bias is *partly distinct* from the Gemma-3 family
(residual corr 0.22–0.34). Gemma-4, as the newest model, is the plausible lone heavy-RL
member of an otherwise RLHF/SFT/DPO-era cohort — so: does RLVR reshape the dispositional
structure, and is that what makes Gemma-4 diverge?

**The clean within-lineage probe is the OLMo-2 ladder** (base → SFT → DPO → **RLVR**;
the Instruct endpoint *is* the RLVR stage, same model throughout). The judging stages
comply with the rating format (digit-mass ~1.0, unlike base), so a sampled `tom_likely`
across SFT/DPO/RLVR is interpretable. Result:

| stage | valence-congruence (same − opp) | entropy | digit-mass |
|---|---|---|---|
| SFT  | 1.97 | 1.41 | 0.95 |
| DPO  | 2.05 | 0.95 | 0.97 |
| **RLVR** | **2.15** | **0.90** | 1.00 |

**RLVR does not flatten valence — it slightly *sharpens* it** (the same-vs-opposite gap
*grows* through DPO and RLVR; RLVR has the largest). So the tempting "RLVR de-emphasizes
the soft social/desirability priors → flattens the human covariance" mechanism is **not
supported** in OLMo, and **Gemma-4's valence-flattening is not a generic RLVR effect.**

**What post-training *does* do, and the thread to pull:** it raises **decisiveness** —
digit-entropy collapses 1.41 → 0.90 (and format compliance climbs to 1.0) across the
stages, while the *structure* is preserved. That's a real, measurable "character" shift
— the model becomes more *confident*, not differently-organized. So if RLVR is changing
character, our handle on it is **calibration/decisiveness**, not the trait-covariance
geometry. That reframes the external question into something we can actually measure: not
"does RLVR scramble what the model thinks goes with what" (apparently no) but "does RLVR
make the model *more certain* about its dispositional judgments" (apparently yes).

*Caveats (this is a first look):* sampled, and "valence-congruence" is a coarser proxy
than the full residual-from-cohort measure; OLMo's *verifiable-reward* RLVR is one
specific implementation — it rules out *generic* RLVR-flattening, not Gemma-4's unknown
recipe; single lineage. *Levers to pull it further:* the full OLMo judgment ladder
(precise per-stage human-match + the decisiveness curve at full resolution), the incoming
**Qwen-3.6-35B-A3B** (a second, independent RLVR/reasoning judge — and an MPS-MoE speed
test), and ideally a frontier RLVR model as a ceiling. Marked exploratory; the
decisiveness finding is the part to build on.

## Open / next

- **Close the bottom of the regress empirically.** A small from-scratch PPMI-SVD
  (1990s-LSA-class) demo on a modest corpus would show the merge falling out of raw
  co-occurrence counts with *no* learned parameters — the strongest possible form
  of "it's the distributional hypothesis." Cheap; the only friction is sourcing an
  offline corpus.
- **ELMo / an LSTM LM** for the contextual-but-recurrent point, if we want the
  empirical LSTM rung rather than the predicted one.
- **Where in the residual stream the split could live but doesn't** — logit-lens /
  per-layer localization of the antonym signal (carried from W15 open items): the
  representation merges at 2/3 depth, but does a symbolic antonym signal exist at
  some layer that the judgment reads and the cosine geometry averages away?
- **Does TIRT pattern with judgment, not representation?** The forced-choice readout
  is *write*-adjacent; on the adjective antonyms it should split, not merge — a
  direct test that joins statisfactions's track to this one.
