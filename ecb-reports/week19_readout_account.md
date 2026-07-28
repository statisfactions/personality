# Week 19 — A mechanistic account of personality readouts: paper framing and decisions

**Author:** ecb (statisfactions)
**For:** rgb (lycotic)
**Date:** 2026-07-28
**Purpose:** Record the framing decisions for the first joint paper so we can act on them
without re-deriving. This is a *decisions* document, not a results document — it says what
the paper is, why, what it explicitly is *not*, the mechanistic account we're organizing
around, the criticisms that account has to survive, how two outside validity frameworks
(Kane, Lin) plug in, and a concrete outline + next steps. It restates enough of your own
W7–W18 results that you don't need anything else open to follow it. After this we can clear
context and start drafting from here.

---

## 0. The decision in one paragraph

We are writing **one empirical paper whose contribution is a mechanistic account of why the
many ways of measuring "personality" in an LLM disagree** — not a demonstration that
self-report is bad (everyone has that), not a pitch for one replacement instrument (premature
and self-defeating), and not a restatement of the persona-selection theory (that's the
interpretation, not the evidence). The object of study is the *family of readouts itself*.
The claim is that each readout is a predictable mixture of a small number of measurable
components — an inherited semantic geometry, a symbolic judgment operator, a self-presentation
policy, and a per-model response-style gain — with mixing weights set by the readout's design
and the model's style profile. The account reproduces the convergence/divergence pattern we've
documented across the 12-model cohort, dates the components to training stages, and — critically
— has to be tested on data it wasn't built from, because the account was induced post hoc and we
must not pretend otherwise. We stay focused on personality (not the broader "all LLM
psychological measurement" version, even though the account probably generalizes).

---

## 1. The problem we're responding to

The field measures "LLM personality" in at least six ways — declarative self-report
questionnaires, internal-representation probes, forced-choice IRT, free-text generation scored
for trait expression, implicit/association measures, and weight-space trait steering — and they
give contradictory answers. Some studies report clean Big Five structure and high consistency;
others report that the structure collapses under factor analysis into a single evaluative /
verbal-fluency dimension; others show trait judgments flipping with trivial prompt changes.

The dominant reaction in the methods literature (Lin, below) is to treat these contradictions as
**"measurement phantoms"** — statistical artifacts masquerading as psychological phenomena — and
to prescribe more rigorous validation to purge them. Our move is different and is the whole point
of the paper: **the contradictions are not noise to be purged, they are a decomposable signal.**
Different elicitation methods disagree because they are measuring different, identifiable things,
and once you know the components, the disagreements become predictable rather than embarrassing.

This is a claim we are unusually well-positioned to make, because we have something the phantom
literature does not: a single cohort run through many readouts *plus* mechanism probes (the
read/write work, the training-stage ladders, the raw-distribution and content-vs-hedge analyses,
the embedding baselines). Nobody else has assembled the readouts and the mechanism in one place.

---

## 2. What the paper IS — and four things it deliberately is NOT

**IS:** an explanatory, mechanistic account of the readouts, grounded in the cohort, that (a)
identifies the components each readout mixes, (b) states a mixing rule, (c) shows the rule
reproduces the observed convergence/divergence and retrodicts contradictions in the outside
literature, and (d) is confirmed on at least one out-of-sample prediction.

We rejected four tempting alternatives, and the reasons matter enough to record:

1. **NOT "self-report is the instrument that fails to measure personality."** This is a straw
   man at this point — the whole phantom literature already says it, and casting self-report as
   *the* measurement and Big Five as *the* construct reduces the paper to a tired indictment. In
   our account, self-report's collapse is one predicted cell, not the headline.

2. **NOT a pitch for a single replacement measurement.** The obvious temptation is to crown
   JUDGE, or a native "response-style" scale, as the right instrument. We decline, for the same
   reason we criticized Persona Cartography's bespoke chat questionnaire: proposing one
   instrument invites the "how much of the result is baked into your instrument's design"
   critique, and none of our candidate instruments is yet criterion-validated. Championing a
   winner is the *follow-up* paper. This paper *names what each readout measures* and thereby
   *defines the space* a good measurement would live in — which is stronger and safer.

3. **NOT a restatement of the persona-selection / "performed not possessed" theory.** That
   framing (the model runs a pretrained model of human personality and plays a character; a
   population of personas laid down in pretraining plus a selection mechanism shaped by
   post-training) is a good *interpretation*, but stated on its own it is just theory that could
   be — and largely has been — asserted elsewhere. The paper's spine is the empirical
   decomposition; persona-selection is at most a one-paragraph interpretive gloss on it.

4. **NOT the general "decomposition of all LLM psychological measurement."** The four components
   are generic (semantics, judgment, self-presentation, response style) and almost certainly also
   organize moral, values, and theory-of-mind readouts — our own ValuePortrait values rescue
   already fits the same shape. That is arguably a bigger paper, but we are **deliberately
   staying on personality** to keep the first piece focused. The cost is that a reviewer may ask
   "why personality specifically" — we accept that and treat personality as the worked case,
   noting (briefly) that the account likely generalizes without claiming it.

---

## 3. The mechanistic account

### 3.1 Four components

Every readout is generated by a weighting of four latent components. Each is anchored in results
you already have.

**A — Inherited semantic geometry.** The distributional/lexical structure of trait terms,
learned in pretraining. It is human-aligned *relationally* and is not a model-specific
achievement: our own embedding baseline (W13 §3.9) showed that models **do not exceed** a generic
sentence encoder at recovering the human IPIP-NEO facet correlation matrix — bge-large scores
+0.686, out-of-the-box MPNet +0.580 (mid-cohort), and the excess of every model over the encoder
is ≈0. The model–human facet match of r ≈ 0.564 (W9 §7) is therefore item semantics. The regress
work (W16) pushed this all the way down: the same relational structure, and specifically the
evaluative-antonym **merge** (Wonderful ≈ Awful), is already complete in 2014 static GloVe vectors
and in principle in a learning-free PPMI-SVD. There is a prior literature here we should cite as a
lineage rather than let Wulff scoop us: **Cutler & Condon (2022), "Deep Lexical Hypothesis"** — the
actual progenitor, three years ahead of Wulff and Milano, recovering the first three factors at
congruence 0.89/0.79/0.79 from encoder word-representations, and flagging N/O as the weak spot —
then **Wulff & Mata (2025, Nature Human Behaviour)** and **Milano et al. (2025)** as the
journal-visible restatements. Component A is inherited, ancient, and evaluatively merged.

**B — Symbolic judgment operator.** A per-item operation that negates, splits antonyms, and makes
discrete good ≠ bad calls, deployed when the model is asked to *judge* rather than simply *represent*.
This is the read/write gap (W15): the residual stream puts pos-eval × neg-eval **above** its own
mean (merged, antonym-z ≈ +1.58 for Qwen-7B representation), while the model's Likert-logprob
**judgment** of the same pair lands **below** the mean (split, ≈ −0.49) — right on the human
antonym value (≈ −0.53). Every model from 3B to 32B shows this as a **sign-flip, not an
attenuation**, and it is localized to the evaluative-antonym cell; the two geometries agree
everywhere else. B is what separates a judgment from a resting representation, and it is
post-training-shaped.

**C — Self-presentation policy.** The desirability-defined "assistant" character that dominates
whenever the model is asked about *itself*. Evidence: the SELF geometry (W18 §2/§2.5) matches
human block structure at 0.82 raw but crashes to **0.21** once the desirability PC1 is removed —
i.e. its human-match was almost entirely the desirability freebie. By contrast JUDGE goes 0.86 →
0.80 under the same PC1 removal (its structure is real). SELF has no model-specific self-knowledge
(the null diagonal, W17 §15.7, diagonal advantage only +0.02) and an effective dimensionality of
~4.5 — "answer-policy strata," not traits. The assistant shape underneath it is roughly rank-1
(E–C correlate at r ≈ 0.93). The structure ranking **JUDGE ≫ ENACT > REPRESENT ≫ SELF** is
monotone in how much symbolic processing the channel does and how little it is self-referential.
C is installed by post-training (RLHF).

**D — Response-style gain.** A per-model modulation of how legible any signal is: peakedness vs.
diffuseness, content vs. hedge. Gemma is razor-peaked (digit-entropy ~0.1–0.3), Llama near-uniform
(~1.4); response styles vary ~4× across the cohort. This is not trait *content* — it is a gain on
whatever content is present. Two results make it load-bearing: (i) the **content-vs-hedge**
decomposition (W18 §4/§7) shows FalconMamba has the highest total rating entropy yet only 7.5%
*content* (vs. Gemmas' 75–92%), so entropy over-credits it and the right per-rater information
weight is content share; (ii) **EV ≥ argmax** in human-match for every model (largest for Llama,
+0.095), and for the flattest raters the signal lives *entirely* in the graded mass
(FalconMamba tail-only r = 0.69; Qwen-32 tail-only 0.63 beats its own EV 0.60). D is also what the
models-as-raters reliability picks up: single-rater ICC(2,1) ≈ 0.61, cohort ICC(2,k) ≈ 0.95. Your
own one-liner — "response styles are model traits" (W17 §15.8) — is the seed of treating D as a
construct rather than a nuisance.

### 3.2 The mixing rule

Each readout's output is a weighting of {A, B, C} set by the readout's **design** — is it
self-referential? a per-item judgment? an aggregated representation? desirability-matched? — and
then scaled by the model's **D** gain.

| Readout | A (semantic) | B (symbolic) | C (self-warp) | signature result it predicts |
|---|:--:|:--:|:--:|---|
| **REPRESENT** | high | ~0 | ~0 | antonym *merge*; high-dim; PC1-removed human-match 0.44 |
| **JUDGE** | high | high | low | PC1-*robust* 0.80; ICC(2,k) 0.95; antonym *split* |
| **SELF** | content only | low | high | 0.82 → **0.21** on PC1 removal; null self-diagonal; effdim 4.5 |
| **ENACT** | high | mid | leaks when persona weak | caricature (high-contrast, blocky); effdim ~5, family-capped |
| **Likert (self)** | high | high | high | ceiling / assistant shape |
| **BC (scenario)** | high | high | mid | behavioral; H/C/O ceilings |
| **TIRT (forced-choice)** | high | high | *suppressed by design* | structurally clean; low open-model recovery (0.2–0.35 vs Haiku ≥0.5) |

The rule reproduces the whole pattern we've spent the year documenting:

- **Shared A → everyone recovers relational Big Five.** This is the inherited convergence, and it
  is why the embedding baseline matters: the agreement is in the items' semantics both models and
  encoders consume, not in anything model-specific.
- **Different B and C loadings → the divergences.** The trait-score dissociation
  (Likert ↔ representation ≈ 0.08–0.10; BC ↔ representation 0.33) is the readouts disagreeing
  because they weight B and C differently. The PC1-removal ranking is the C-loading ordering. The
  read/write merge-vs-split is exactly REPRESENT (A alone) vs JUDGE (A + B) on the one cell where
  A and B disagree.
- **D → legibility differences.** EV > argmax, the content-vs-hedge weighting, and the "certainty
  is a question-taxonomy not an entropy policy" result (W18 §4) are all D modulating the same
  underlying content.

### 3.3 Training stages date the components

The mechanism evidence assigns each component to a point in training, which is what makes the
decomposition causal rather than merely descriptive:

- **A is pretrained.** The merge is flat across Qwen (Base → Instruct) and OLMo-2
  (Base → SFT → DPO → RLVR), and regresses to 2014 GloVe.
- **B and C are post-training.** The antonym split is *built* — cumulative across OLMo's
  post-training stages, and present-at-base in Qwen where tuning mostly collapses entropy
  (confidence) rather than creating the split. The desirability collapse of SELF is the RLHF
  assistant policy.

So the divergences among readouts are manufactured at different training stages. (Caveat: this
rests on one full open ladder (OLMo) plus one two-point ladder (Qwen). It is a causal *wedge* —
proof the components can come apart in time — not a developmental *law*. The 12-model
cross-section carries generality; the ladder carries separability. We should state that split
explicitly and not over-claim a universal schedule, especially since Qwen and OLMo already differ
in *when* the split arrives.)

### 3.4 A fourth external instrument, for triangulation

The ValuePortrait rescue (W18 §7) is worth keeping in as convergent evidence: a fourth,
human-anchored, ecologically-sampled instrument we did not author recovers the same assistant
value shape (cross-model r = 0.90 once scored within-scenario), and their published "no structure
in generation" null turned out to be an instrument artifact (their scoring had ≈0 split-half
reliability). It both supports A/C (the shared prosocial/assistant axis) and demonstrates the
"reliability before validity, distribution before argmax" methodological through-line that D
embodies.

---

## 4. Why these readouts, out of a huge space

A fair challenge: of all the ways one could elicit personality from a model, why these? The
principled answer is that they form a rough **factorial over the design axes the mixing rule says
matter**:

- **associative vs symbolic** — REPRESENT vs JUDGE (same content, ± B);
- **self vs other** — SELF vs JUDGE (± C);
- **constrained → free on C** — Likert (self-declarative, high C) / BC (scenario, mid) / TIRT
  (desirability-matched forced choice, C suppressed by construction);
- **argmax vs distributional** — the logprob scoring exposes D everywhere.

They also bracket **read vs write** (representation vs behavior) and **white-box vs black-box**
(needs weights vs API-only), and they deliberately include both **field-standard** methods
(self-report, Likert) and **neglected** ones (JUDGE, representation) — so the account can both
*retrodict the existing literature* and *extend past it*.

Honest qualification, recorded here so we write it honestly later: this is a **rational
reconstruction of an accreted set.** We did not design a clean factorial; the readouts arrived
over W2–W18. The cells are therefore confounded — SELF differs from JUDGE in referent *and* format
*and* task — which is exactly why one of the confirmatory experiments (§7) has to hold format
constant and swap only the referent.

---

## 5. Qualifying criticisms the account has to survive

These are the ones I think actually threaten the contribution; we should pre-empt them in the
paper rather than wait for a reviewer.

1. **Independent identification (the load-bearing worry).** A, B, C, D are latent and are read off
   the same readouts whose divergence they explain. Only A (the embedding baseline is an
   independent anchor) and the A-pretrained / C-post-trained split (the ladder) have external
   handles. **B and D are currently inferred from the correlation structure they are meant to
   explain.** The account is only as strong as the number of components pinned by a *manipulation*
   rather than a *contrast*. This is why the variance decomposition and the referent-swap are not
   optional.

2. **Post-hoc, confounded selection.** As in §4 — the "factorial" is reconstructed and the cells
   are confounded. Attributing SELF's collapse to C (rather than to format-sensitivity) requires a
   holding-constant manipulation we have not yet run.

3. **The mixing rule is qualitative — is it falsifiable?** "High A, low C" is a cartoon; a real
   mechanistic account should predict a readout's output as a *function* of loadings. A
   four-component model with free weights can accommodate a lot. Its only current falsification
   lever is retrodiction, which is weak on its own. This is why we need at least one **forward**
   prediction — a readout or manipulation whose loadings the account specifies *before* we look.

4. **C and D may not be separable.** The assistant policy *is* low-entropy and high-desirability;
   on SELF, C and D co-occur. A skeptic folds them into one component. We need a model that is
   high-C/low-D (or the reverse) to show they are two. Worth an explicit check.

5. **N = 12, heterogeneous, mostly small open models.** The "model D profile" and the mixing rule
   are estimated on a case-study cohort. Haiku (frontier) recovers TIRT signs the open models
   cannot, hinting the rule may be scale/tuning-dependent. Generality is asserted, not
   established — state it as such.

We consciously set aside a sixth criticism ("the account is generic, not personality-specific")
by deciding in §2.4 to stay on personality; we note the generality without claiming it.

---

## 6. Two outside validity frameworks, and where they fit

Neither of these is something you've read; here is what they are and, more importantly, the
narrow job each does. **Both are scaffolding, not spine.** The account in §3 is the contribution;
if either framework starts structuring the paper, that's a smell.

**Lin (2025 "dual-validity framework"; 2026 "validity-guided workflow," now in *Behavior Research
Methods*).** Two linked papers by a psychologist (Zhicheng Lin) arguing that LLM-psychology
research produces "measurement phantoms" — statistical artifacts that look psychological but
dissolve under scrutiny — and that researchers must match the rigor of their validation to the
ambition of their claim (he distinguishes four claim-types: LLM as research tool / evaluation
target / human simulator / cognitive model), drawing construct-validity evidence from five
standard psychometric sources. Crucially, he explicitly calls for *"developing computational
analogues of psychological constructs"* instead of blindly applying human instruments — but he
supplies **no mechanism and no actual computational construct**; the papers are a checklist and a
workflow. **His role for us:** he is the citable statement of the problem (the "phantom" framing,
in the intro) and the published *license* for treating a native component like D as a real
construct (in the discussion). We are the mechanistic answer to the call he makes. He goes in the
bookends, not the skeleton — building our sections around his six-stage workflow would re-make the
mistake of parroting him.

**Kane (argument-based validity; see Kane 2013 and the *Standards for Educational and
Psychological Testing*).** The modern view that validity is not a property of a test but an
*argument* that a particular **interpretation and use** of scores is justified — and that this
argument is a chain of inferences (scoring → generalization → extrapolation → implication), each an
assailable "warrant." The single idea we use: **you must specify the intended interpretation before
you can judge validity, and validity can hold for one interpretation while failing for another on
the very same scores.** That is exactly the lever the paper needs, because "the model's
personality" is really (at least) **two** intended interpretations conflated — the model's *theory
of personality* (a competence: how traits co-occur in people) versus the model's *own disposition*
(something that governs its behavior). The account in §3 then says which readout validly serves
which interpretation: JUDGE (A + B) serves the *theory* interpretation (scoring and generalization
warrants hold; it is PC1-robust); self-report (high C) serves neither well; and only the native
response-style component (D) plausibly extrapolates to the *disposition* interpretation. The bite
is at Kane's **extrapolation** step, where the "no experiential response basis" result (below) is
the warrant failure for the disposition reading. Note this is emphatically *not* "self-report is
bad" — it is "specify your interpretation, and validity becomes readout-specific." **His role for
us:** the interpretive payoff in the discussion — the move from "here is what each readout *is*"
to "here is what each readout is *valid for*." Use only the interpretation-specification and the
extrapolation inference; do not march the full four-link chain end to end (that's what tipped us
into the straw-man version earlier). Lin does not use Kane — he leans on Messick's unitary construct
validity and Borsboom's causal theory — so bringing Kane is a genuine addition, not a duplication.

---

## 7. Paper outline

The work is emergent and post-hoc, and the structure has to own that rather than hide it. The
anti-HARKing principle is: **pose the question a priori (legitimate), label the account as
induced (transparent), and confirm on data the account did not see (the quarantined test).** The
section logic is therefore exploratory → account (induced) → confirmatory.

1. **Intro — the question.** The field measures LLM personality many ways and gets contradictory
   answers (cite the zoo + Lin's phantoms). Frame the disambiguation up front (Kane, lightly):
   "the model's personality" conflates its *theory of personality* and its *own disposition*.
   Claim: the methods measure different, decomposable things; we give a mechanistic account and
   test it. Do **not** write "we hypothesized A–D."

2. **The space of personality readouts (lit review + Table 1).** The method families the field
   uses, as prior work — this is where the literature's readouts belong (setup, not discussion).
   Close by naming the design axes they vary on, descriptively, foreshadowing the account without
   asserting it. Table 1 skeleton:

   | Method family | What it elicits | Claimed construct | Typical reported finding | Access |
   |---|---|---|---|---|
   | Self-report questionnaire (Serapio-García) | declarative Likert self-rating | the model's Big Five | high consistency *or* CFA collapse | black-box |
   | Representation / probing (RepE) | residual-stream geometry | encoded trait | structure present but "washed"; anisotropy | white-box |
   | Forced-choice IRT (Okada) | desirability-matched pairwise choice | latent trait, style-controlled | recovery on frontier, weak on open | black-box |
   | Generation / behavioral (ValuePortrait) | free-text scored for trait/value | expressed disposition | "no structure" nulls (often instrument artifact) | black-box |
   | Implicit / association | association-based latent measure | implicit tendency | predicts downstream text | black-box |
   | Weight-space steering (Persona Cartography) | LoRA amplify/suppress + chat questionnaire | steerable trait | monotone, affects safety behavior | white-box |

3. **The present program.** The 12-model cohort; the spanning subset of readouts and why it spans
   the design axes. One transparent paragraph owning the emergent nature ("this synthesizes a
   program of studies; we report the account that organizes them and test it out-of-sample").

4. **Results I — the phenomenon (descriptive).** Convergence (shared relational structure;
   inherited, per the embedding baseline) + divergences (trait-score dissociation; PC1-removal
   ranking; read/write merge/split; D modulation). Labeled exploratory.

5. **Results II — the account (induced).** Components A–D and the mixing rule, presented as
   emerging from §4. **The variance-decomposition figure is the centerpiece.** Training-stage +
   embedding baseline anchor and date the components.

6. **Results III — the test (confirmatory, out-of-sample).** The account's predictions on data it
   was not built from: the referent-swap (± C predicts collapse with format held constant);
   retrodiction of ≥3 outside-literature contradictions; and the D → downstream-behavior criterion.

7. **Discussion.** What each readout is *valid for* (Kane; the two-construct disambiguation
   resolved). The "space of good measurements" corollary; the native, tractable, useful component
   (D). Limits (N = 12; the confounds the test addresses vs. those it doesn't; the ladder as wedge
   not law). Answer Lin's call for computational analogues.

---

## 8. Next steps (and the empirical gaps the argument still needs)

Ordered by leverage. The budget we agreed is **1–2 weeks of new generation/analysis**, no more,
for this first piece.

1. **Build the variance-decomposition figure first — before writing §5.** Partition a
   model × channel × trait table into a model-style component, a channel-construct component, and
   trait. This is the account's centerpiece *and* its own falsification test: if A–D don't
   partition the variance cleanly, we learn the account is wrong before writing 5,000 words around
   it. Analysis-only; days. (It also happens to satisfy Lin's requirement to handle
   non-independent observations properly.)

2. **Draft Table 1 in parallel.** Independent of results; clarifies the design axes; the
   lit-review spine.

3. **Lock the one out-of-sample confirmatory test for §6.** Candidates, cheapest to
   highest-value:
   - *Referent-swap* (cheapest, cleanest): hold item and format constant, vary only "how much does
     X describe *you*" vs "…a person who is Y," and show the desirability collapse tracks the
     referent, not the format. Directly de-confounds criticism #2 and pins C by manipulation.
   - *Literature retrodiction*: show the components + mixing rule predict which of several outside
     contradictions you'd see (structure vs collapse vs punctuation-flip). Counts as confirmatory
     only if those cases were not used to build the components.
   - *Criterion (highest value)*: a native response-style / confidence score (D) predicting a
     consequential deployment behavior (sycophancy, refusal/hedging, or calibration) across the
     cohort, where self-report does not. This is the one genuine hole in the argument's logic —
     without it we prove the negatives (self-report fails) but never demonstrate the positive (a
     native construct does useful work). It is also the "useful" leg we keep returning to.

4. **Harden the response-process experiment** (the "no experiential basis" result — self-report
   doesn't use the model's own history even when it's in context). This is the warrant under
   Kane's extrapolation failure for the *disposition* interpretation, so it has to be clean:
   strong, sufficient in-context experience; a **positive control** showing the model *can* use
   that experience for some other task; and a dose/scale check.

5. **One training-stage localization** (cheap causal upgrade): run SELF on the OLMo
   Base → SFT → DPO → RLVR checkpoints and show the desirability collapse (0.82 → 0.21) is absent
   at base and installed post-training. Existing checkpoints, one channel.

6. **A uniform reliability table across all channels** (mostly analysis on data in hand;
   reliability is the mathematical ceiling on every validity claim we make).

**Writing order:** figure (#1) + table (#2) together → lock the test (#3) → then draft front to
back. Do not draft §5 (the account) until #1 exists and #3 is chosen; writing the account around a
figure that doesn't exist yet is how post-hoc prose slides into HARKing.

**Division of labor to settle:** #1, #4, #6 are natural ecb (psychometric) pieces; #3-referent-swap
and #3-criterion and #5 are generation-side and more natural for rgb. #2 and the outline are joint.

---

## 9. Decisions log (quick reference)

- **One paper**, empirical, first piece, from committed data + ≤1–2 weeks new work.
- **Contribution = mechanistic account of the readouts**, not an indictment of self-report, not a
  single replacement instrument, not PSM theory restated, not the general (non-personality)
  version.
- **Stay on personality** as the worked case; note but don't claim generality.
- **Components A–D + mixing rule**; variance decomposition is the centerpiece figure.
- **Emergent nature owned via** exploratory → induced-account → quarantined confirmatory-test
  structure.
- **Literature readouts go in the lit-review (Table 1)**, not the discussion.
- **Lin = problem statement + license** (bookends). **Kane = interpretation-disambiguation +
  extrapolation warrant** (discussion payoff). Both scaffolding, not spine.
- **The one real logical gap = a downstream behavioral criterion** for the native component; the
  referent-swap is the cheapest test to pin the C attribution.
- **Venue leanings** (from earlier discussion, not re-litigated here): Behavior Research Methods is
  a confirmed live home for this genre (Lin's workflow is there); Psychological Methods /
  Perspectives are alternatives depending on how provocative vs constructive we pitch it.

---

## References rgb may want (none required to follow the above)

- Kane, M. (2013). Validating the interpretations and uses of test scores. *J. Educational
  Measurement* 50(1). — the argument-based validity framework.
- *Standards for Educational and Psychological Testing* (AERA/APA/NCME, 2014) — the five sources of
  validity evidence, incl. "evidence based on response processes."
- Messick (1989); Borsboom et al. (2004) — the unitary-construct and causal-validity views Lin uses.
- Lin, Z. (2025). From Prompts to Constructs: A Dual-Validity Framework for LLM Research in
  Psychology. arXiv:2506.16697.
- Lin, Z. (2026). A validity-guided workflow for robust LLM research in psychology. *Behavior
  Research Methods* 58:216. arXiv:2507.04491.
- Cutler & Condon (2022), Deep Lexical Hypothesis (arXiv:2203.02092); Wulff & Mata (2025, *Nat.
  Hum. Behav.*); Milano et al. (2025) — the "structure is recoverable from item text alone"
  lineage that grounds component A.
