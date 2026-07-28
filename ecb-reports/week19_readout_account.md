# Week 19 — Discussion agenda: an account of why our readouts disagree

**From:** ecb · **For:** discussion with rgb · **Date:** 2026-07-28

## Purpose

A first pass at a framing for the first joint paper, written as a guide for actually starting
to write. It's a proposal, not a decision — the point of the meeting is to tear it up. It
seems worth a shot because it organizes most of what we've found into one argument without
much new work, but several pieces are shaky and flagged as such below.

---

## 1. What the paper would be

**The contribution is an account of why our different readouts of "personality" disagree.**
The object of study is the family of readouts itself, not any one instrument.

The field's readouts contradict each other — clean Big Five structure in one study, collapse to
a single evaluative dimension in another, trait judgments flipping with punctuation in a third.
The methods literature treats these contradictions as artifacts to be purged. Our move is the
opposite: **the contradictions are a signal we can decompose.** Different elicitation methods
disagree because they are about different things, and once we can say what each is about, the
disagreements become predictable instead of embarrassing.

We're well-positioned for this because we have the readouts *and* the supporting analyses in
one cohort, which is what the methods-critique papers lack.

What this is *not*: not another "self-report is invalid" paper (table stakes now, and a straw
man), not a pitch for one replacement instrument (premature, and invites the "your result is
baked into your instrument" critique we leveled at Persona Cartography), and not the general
"all LLM psychological measurement" version (staying on personality to keep the first piece
focused).

Two outside frameworks sit in the background, used lightly and only at the edges: **Lin** (two
recent papers arguing LLM psychology is full of "measurement phantoms" and calling for
purpose-built computational constructs — he gives us the problem statement and a citable
license, but no mechanism) and **Kane** (validity as an argument about a *specified*
interpretation of scores — his one useful move here is forcing us to say which claim we're
making, since "the model's personality" conflates the model's theory of personality with the
model's own disposition). Neither should structure the paper.

---

## 2. What we've found

Nothing new here — this is the pattern the account has to explain.

**Convergence is broad, and mostly inherited.** Every readout and every model recovers roughly
human-like *relational* trait structure, with block-level human-match sitting in a narrow band
across all four channels (0.78–0.86). Facet geometry is preserved across architectures at
r ≈ 0.94. The cohort agrees with itself as raters (JUDGE ICC(2,k) ≈ 0.95, congruence 0.84). The
evaluative-antonym merge in representation is universal across 3B–32B. And the
assistant/desirability shape recurs in *every* channel — SELF, ENACT, the Likert ceilings, the
values work — with the assistant profile itself close to rank-1 (E–C r ≈ 0.93). Critically, much
of this is not a model achievement: our embedding baseline showed models don't exceed a generic
sentence encoder (bge +0.686; models' excess ≈ 0), and the regress pushed the same relational
structure down to 2014 static vectors.

**Divergence is where the models actually differ.** Trait *scores* dissociate across readouts
(Likert ↔ representation ≈ 0.08; BC ↔ representation 0.33). Removing the evaluative PC1 produces
a monotone ranking — JUDGE 0.86→0.80, ENACT 0.80→0.62, REPRESENT 0.78→0.44, SELF 0.82→**0.21** —
i.e. SELF's human-match was almost entirely a desirability freebie while JUDGE's survives.
Effective dimensionality forms a ladder: REPRESENT 50–70 > human ~27 > ENACT 5–13 ≈ SELF 4.5, so
the output channels compress to roughly five dimensions while representation stays wide.
Representation merges evaluative antonyms where judgment splits them, as a sign flip at every
scale. Per-model human-match spreads 0.55 (Gemma-4) to 0.73 (FalconMamba). Response
distributions vary ~4× in peakedness, EV beats argmax in human-match for every model, and
persona expression is family-capped (Qwen ~5 vs Llama ~9, prompt-gated in Qwen but not Llama).

---

## 3. The proposal

### 3.1 Trait-concept semantics is the shared medium, not a distinguishing referent

The inherited trait semantics are approximately **constant** across the cohort — that's what the
encoder-baseline result says — and they are an input to essentially *every* readout: JUDGE runs
on adjective pairs, SELF on adjective self-ratings, ENACT on adjective-conditioned personas, and
the values work is largely about semantics leaking into a generation measure. So this is the
medium all our measurements run through, not something that distinguishes them. It explains the
convergence in §2 and then gets out of the way. A quantity that doesn't vary across models can't
explain model differences.

### 3.2 Two further objects a readout can be about

1. **Person-attribution structure** — what the model takes to co-occur *in people*; its implicit
   theory of personality.
2. **Persona system** — what the model is absent instruction, and what it can become under
   instruction. (We considered splitting default-vs-range into two objects; collapsed for now,
   though whether they're separable in our data is an open question below.)

### 3.3 Readouts draw on mixtures of these

The mixture follows from each readout's design, and nothing is pure — every readout runs through
trait-concept semantics, and most touch more than one object:

| Readout | Draws on |
|---|---|
| JUDGE | mostly person-attribution; through semantics; some default-persona leakage (it is still the model's own view of people) |
| SELF | mostly persona system (the default); through semantics |
| ENACT | persona system (instantiation + performance); through semantics |
| REPRESENT (adjective stimuli) | mostly trait-concept semantics |
| REPRESENT (persona-pair stimuli) | persona system; through semantics |
| Likert self-report | persona system (default), heavily; through semantics |
| BC / TIRT | persona system under constrained choice; plus person-attribution, since scenario choices involve attributing to a character |

Two consequences worth pausing on. First, **REPRESENT is not one readout** — its referent is set
by the stimulus, not by where we read, so the adjective version and the persona-pair version are
about different things. Second, **REPRESENT-on-adjectives and JUDGE are not two views of one
object**: one is about word meanings, the other about what co-occurs in people. We had been
treating them as two views of the same thing. Readouts converge to the extent their mixtures
overlap, which is a claim we can defend from design rather than assert.

Useful background here, possibly new to rgb: Tversky's *Features of Similarity* (1977)
established that judged similarity is not a metric — it's asymmetric and context-dependent, with
feature weighting set by the comparison set. So a cosine between separately-encoded items and a
judgment made with both items in context *should* diverge; expecting agreement is the naive
assumption. Relatedly, Osgood's semantic-differential work established evaluation as the dominant
dimension of connotative meaning long before LLMs — independent grounding for why PC1 is
evaluative in every channel.

### 3.4 What varies across models

These are the quantities that actually differ across the cohort, are measurable from behavior,
and require no interpretability work.

**Of the person-attribution structure**

- **Differentiation** — how articulated the attributed trait structure is beyond the evaluative
  axis. JUDGE is the direct readout; its survival of PC1 removal (0.86→0.80) is what marks it as
  real structure rather than halo. Human-match is a *validity check* on this, not the property
  itself.

**Of the persona system**

- **Default shape** — the profile expressed absent instruction: near rank-1, desirability-
  organized. Note this is *selected* by post-training, not created by it, and "post-training"
  means SFT/DPO/RLVR, not RLHF alone. The assistant shape isn't meaningless; it's one shape
  among many.
- **Anchoring strength** — how deeply the model is held to that default. Almost certainly
  multidimensional rather than scalar (different interventions move different models
  differently), which is why we're not claiming a single "responsivity" number.
- **Persona distribution** — the range of characters the model can instantiate and how
  accessible each is. Not uniform: virtue words collapse into the assistant blob (unenactable)
  while vivid registers instantiate cleanly. Effective dimensionality (Qwen ~5, Llama ~9) is one
  summary statistic of its spread, alongside enactability and cross-persona discriminability.

**Cutting across everything — response generation**

- **Concentration** — how peaked the response distribution is (Gemma ≈ 0.11 nats, FalconMamba
  ≈ 1.66).
- **Item-drivenness** — how much of that spread tracks the item rather than a fixed per-model
  habit (Gemmas 75–92% content, FalconMamba 7.5%).

These last two are kept separate because FalconMamba proves they dissociate: it looks maximally
uncertain by entropy, but almost none of the spread is about the item — it's running a fixed
digit-prior. "Flat" means low-*content*, not high-variability, so entropy alone over-trusts it.
Model-specific representation artifacts (Gemma's massive activations, the IPR-gated partialling)
are explicitly **not** properties here — they're a methods correction and belong in methods.

### 3.5 Readouts have blind spots

Each readout is sensitive to some properties and blind to others: argmax scoring can't see
concentration; cosine on separately-encoded items can't see context-dependent weighting;
self-referential framing can't see much beyond the default shape. This is ordinary measurement
talk, and it yields the practical payoff — *if you want to see property X, use readout Y* — plus
a natural reading of the centerpiece figure: model variance = the properties we care about,
readout variance = measurement differences, interaction = which readouts reveal which properties.

---

## 4. Open questions for the meeting

These are real, not rhetorical.

1. **Are the objects right?** Does splitting person-attribution from the persona system match
   rgb's read of the data — and was collapsing default-vs-range into one persona system the right
   call, or do they need separating?

2. **The merge/split demotion — this needs an explicit decision.** Under this framing, the
   read/write result stops being a claim about the model (an associative substrate with a
   symbolic override) and becomes a *methods* finding: the standard similarity readout has a
   demonstrable blind spot, and the trait-relevant structure is recoverable under reweighting
   (the ridge result: 0.53→0.78 raw, 0.29→0.50 beyond PC1, while rotation-only Procrustes fails
   at ≈0.24 — the human axes are present at the wrong relative scales). That's arguably better
   supported than the mechanism story and makes no unearned interpretability claim. But it *is* a
   demotion of a capstone result, and rgb should weigh in on whether the trade is worth it —
   including whether there's a mechanistic account rgb would rather defend.

3. **Does anchoring strength hold together at all**, or does it fragment by intervention type to
   the point where we shouldn't name it as a property?

4. **Is item-drivenness stable enough across readouts** to carry weight as a model property,
   rather than being an artifact of the JUDGE-style rating format where we measured it?

5. **Which properties are actually separable in the data?** Default shape and persona
   distribution may not be independent if the models with strong defaults are also the narrow
   ones. Worth checking before committing.

A background note for ecb, not needed in the meeting: the persona-system and person-attribution
properties are being treated as reflective (real properties producing the observed responses),
the response-generation properties as formative (constituted by their measurement). That mixed
measurement model is defensible but should be stated deliberately in the paper.

---

## 5. Where this heads (contingent on the above)

**Paper shape.** The work is emergent and post-hoc, so the structure should own that rather than
hide it: pose the question a priori (legitimate), present the account as *induced from* the
results (transparent), and confirm it on data the account wasn't built from (the part that makes
it testable rather than a story). Sections: intro → the space of readouts (lit review + table) →
our program → the phenomenon (exploratory) → the account (induced) → a confirmatory test →
discussion. The literature's readouts belong in the lit review, not the discussion.

**Highest-leverage next analyses**, if the framing survives:

1. **The model × readout variance decomposition** — partition variance into model properties,
   readout differences, and their interaction. This is the centerpiece figure *and* the account's
   own falsification test: if the properties don't partition cleanly, we learn the framing is
   wrong before writing around it. Analysis only.
2. **A referent-swap** — hold item and format constant, vary only self vs. other, and show the
   desirability collapse tracks the referent rather than the format. Cheap, and it pins the
   default-shape attribution by manipulation rather than by contrast.
3. **A downstream behavioral criterion** — show one of these properties predicts some
   consequential model behavior where self-report doesn't. This is the one genuine hole in the
   argument: without it we establish the negatives but never demonstrate that any of these
   properties does useful work.

Budget discussed: 1–2 weeks of new generation/analysis for this first piece, everything else from
committed data plus sensitivity checks.
