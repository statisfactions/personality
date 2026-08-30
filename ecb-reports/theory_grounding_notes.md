# Theory grounding notes — validity frameworks and the similarity literature

**Author:** ecb · **Started:** 2026-07-28
**Why this file exists:** these are the theoretical/citation-side notes behind the Week 19
readout-account framing (`week19_readout_account.md`). They were drafted into that agenda and
then pulled out to keep the agenda focused on the empirical proposal. Nothing here is settled
paper text — it's the reasoning about *what each source gives us and where it would plug in*,
kept so it doesn't have to be re-derived.

---

## 1. Lin — the "measurement phantoms" framing

**The sources.** Two linked papers by Zhicheng Lin (psychologist, Yonsei / USTC):

- Lin, Z. (2025). *From Prompts to Constructs: A Dual-Validity Framework for LLM Research in
  Psychology.* arXiv:2506.16697. A Perspective stating the thesis.
- Lin, Z. (2026). *A validity-guided workflow for robust large language model research in
  psychology.* **Behavior Research Methods 58:216** (in print). arXiv:2507.04491. The same
  argument operationalized as a six-stage workflow.

**What he argues.** LLM psychology research produces "measurement phantoms" — statistical
artifacts masquerading as psychological phenomena. His flagship examples: personality
inventories collapsing under factor analysis into a single verbal-fluency dimension; moral
preferences reversing when "Case 1/Case 2" becomes "(A)/(B)"; theory-of-mind accuracy
collapsing under trivial rephrasing; models endorsing both "I am an extrovert" and "I am an
introvert." He integrates two validity traditions — psychometrics (Cronbach & Meehl, Loevinger,
**Messick**'s five sources of construct-validity evidence: content, response processes, internal
structure, relations with other variables, consequences) and causal inference (Cook & Campbell's
internal/external/construct/statistical-conclusion validity) — and argues validation must scale
with claim ambition. He classifies four claim-types with escalating evidentiary burden: LLM as
**research tool** / **evaluation target** / **human simulator** / **cognitive model**.

**What's genuinely useful to us.**

1. He is the citable statement of the problem, published in a venue we'd target.
2. He explicitly calls for *"developing computational analogues of psychological constructs"*
   rather than uncritically applying human instruments, and in Phase 2a says outright that
   "conscientiousness" can't mean effortful self-discipline in an LLM — measuring human
   self-discipline is a category error, measuring conscientiousness-*analogous* computational
   patterns is viable. **That is a published license for treating our native model properties as
   constructs**, which de-risks the whole move.
3. He already states the ontological argument we'd otherwise have to build: an LLM might endorse
   "I worry about the future," but anxiety presupposes temporal experience, a persistent self,
   and embodied consequences — *"ontological properties the model lacks."* He cites Borsboom's
   causal theory of validity (the attribute must exist and causally produce the score).
4. Table 1 of the 2025 paper already names **response processes** ("do the mechanisms generating
   outputs align with theoretical processes?") with threats *mechanistic substitution* and
   *architectural artifacts* — so our "no experiential basis for self-report" result slots into a
   named, standard evidence category rather than being an ad-hoc argument.
5. He concedes the escape hatch we need: *"structural validity failures may indicate
   methodological mismatch rather than construct absence."*
6. His Stage 5 requires handling non-independent observations — which our variance-decomposition
   approach satisfies by construction.

**What he does not have — the gap we fill.** No mechanism and no actual computational construct.
Both papers are taxonomy plus prescription: a checklist for validating an instrument, with the
slots left empty. He also points at implicit/association-based measures as the promising
direction (citing an IAT-adapted sentiment measure correlating >0.85 with downstream generation)
but doesn't develop it.

**Where he'd plug in.** Intro (the problem) and discussion (we answer his call). **Bookends
only** — if his six stages start structuring our sections, we've become a derivative workflow
paper. Corrected note to self: an earlier draft claimed he omits response-process evidence. He
does not; it's in his Table 1. The real gap is mechanism, not coverage.

---

## 2. Kane — argument-based validity

**The source.** Kane, M. (2013), *Validating the interpretations and uses of test scores*,
Journal of Educational Measurement 50(1); and the *Standards for Educational and Psychological
Testing* (AERA/APA/NCME, 2014).

**The idea.** Validity is not a property of a test. It's an **argument** that a particular
*interpretation and use* of scores is justified — and that argument is a chain of inferences,
each an assailable warrant: **scoring** (does the response→score rule hold?) → **generalization**
(does the score hold across forms, occasions, raters?) → **extrapolation** (does it predict
non-test behavior?) → **implication** (do the decisions it supports follow?).

**The one move we need.** Kane forces you to *specify the intended interpretation before judging
validity* — and validity can hold for one interpretation while failing for another on the very
same scores. That's the lever, because **"the model's personality" conflates at least two
interpretations**:

- the model's **theory of personality** (a competence: how traits co-occur in people), and
- the model's **own disposition** (something that governs its behavior).

Our account then says which readout serves which: JUDGE supports the theory reading (its scoring
and generalization warrants hold; PC1-robust at 0.80); self-referential readouts support neither
well; and only the response-generation / native properties plausibly extrapolate to a disposition
reading. The bite is specifically at **extrapolation**, where the no-experiential-basis result is
the warrant failure for the disposition reading.

**Why Kane and not just Messick (whom Lin uses).** Messick's five sources tell you what *kinds*
of evidence to accumulate; Kane gives you an **argument structure that localizes which inference
breaks**. That's what lets us say "this is a scoring-stage failure, not a data problem" (e.g. the
TIRT {A,C} sign-flip) or "generalization holds within a channel and breaks across channels."
Lin does not cite Kane, so this is a genuine addition rather than duplication.

**Guardrail.** Use only the interpretation-specification move and the extrapolation inference.
Marching the full four-link chain end-to-end is what turned an earlier draft into a straw-man
audit of self-report. Discussion-section material, not a spine.

---

## 3. Tversky, Osgood — why representation and judgment *should* diverge

This is the grounding that lets us explain the merge/split **without any interpretability
claim**, which was the sticking point. It's psychology, not mech-interp, which is also why it
plays to our comparative advantage.

**Tversky (1977), "Features of Similarity"** (Psychological Review 84(4)). Judged similarity is
**not a metric**: it is asymmetric ("North Korea is like China" ≠ "China is like North Korea"),
violates the triangle inequality, and — the load-bearing part — **feature weighting depends on
the comparison set**. Similarity is computed over *respects*, and which respects are salient is
set by context. See also Medin, Goldstone & Gentner (1993), "Respects for similarity."

**Why it matters for us.** Our REPRESENT readout encodes two adjectives *separately* and applies
an external metric (cosine, uniform weighting over dimensions). Our JUDGE readout puts both in
one context and lets the model compute the relation. Under Tversky, these *should* diverge —
context-dependent reweighting is available to the second and unavailable to the first **by
construction**, since there is no comparison set at encoding time. Expecting representational
cosine and judged similarity to agree is the naive assumption, not the default.

This converts the merge/split from a surprising model-internal phenomenon into an expected
read/judge divergence, and it pairs with rgb's own decodability result (W18 §6): the human
structure *is* in the representation but at the wrong relative scales — ridge mapping recovers it
(0.53→0.78 raw; 0.29→0.50 beyond PC1) while rotation-only Procrustes fails (≈0.24). Reweighting
is exactly what Tversky says judgment does.

**Osgood — the semantic differential (EPA: evaluation, potency, activity).** Evaluation is the
dominant dimension of connotative meaning, robustly and cross-linguistically, established decades
before LLMs. This is independent grounding for why PC1 is evaluative in *every* channel we
measure — it's a property of meaning-space, not an artifact of RLHF or of our instruments.

**Adjacent, lower priority.** Sloman (1996) on associative vs. rule-based reasoning is the closest
thing to the "symbolic override" language, but it invites anthropomorphism and rgb may bristle —
use lightly if at all. The implicit/explicit measurement literature (Greenwald & Banaji; Nosek) is
the closest human analogue to representation-vs-judgment dissociation and comes with 25 years of
measurement modeling; double-edged, because that literature's own predictive-validity disputes
are live — but those disputes are *our* criterion question, so the engagement could be productive.
The antonym-merge itself is old news in NLP: documented from LSA/HAL in the 1990s through
word2vec/GloVe, with the counter-fitting literature (Mrkšić et al. 2016; retrofitting, Faruqui et
al. 2015) existing precisely to undo it.

---

## 4. Measurement model — reflective vs. formative

Kept here because it's ecb-side and probably shouldn't clutter a joint agenda.

The Week 19 properties are implicitly being treated with a **mixed measurement model**:

- **Persona-system and person-attribution properties → reflective.** Treated as real properties
  that causally produce the observed responses.
- **Response-generation properties (concentration, item-drivenness) → formative.** Constituted by
  their measurement rather than causing it — concentration just *is* the peakedness of the
  distribution.

**Why this needs stating deliberately.** A reflective latent-variable model asserts that the trait
*causes* the item responses — that's Borsboom's requirement, which Lin leans on. So invoking
"traits" does **not** sidestep the mechanism question; it smuggles a causal claim in through the
measurement model. Since our whole strategy is to avoid unearned mechanistic claims, we should
either (a) own the mixed model explicitly, (b) go fully formative/index and accept a weaker,
descriptive account, or (c) accept a weak causal claim (the weights cause the behavior; these are
summaries of stable tendencies) at the cost of near-vacuity. A mixed model is defensible and
arguably the honest answer, but it will draw psychometric-reviewer attention and should be argued
for, not slipped in.
