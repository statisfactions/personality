# Week 19 — Discussion agenda: an account of why our readouts disagree

**From:** ecb · **For:** discussion with rgb · **Date:** 2026-07-28

## Purpose

A first pass at a framing for the first joint paper, written as a guide for actually
starting to write. It's a proposal, not a decision — the point of the meeting is to tear it
up. It seems worth a shot because it organizes most of what we've found into one argument
without requiring much new work, but several pieces are shaky and flagged as such below.

---

## 1. What the paper would be

**The contribution is an account of why our different readouts of "personality" disagree.**
The object of study is the family of readouts itself, not any one instrument.

The field's readouts contradict each other — clean Big Five structure in one study, collapse
to a single evaluative dimension in another, trait judgments flipping with punctuation in a
third. The methods literature treats these contradictions as artifacts to be purged. Our move
is the opposite: **the contradictions are a signal we can decompose.** Different elicitation
methods disagree because they are about different things, and once you can say what each is
about, the disagreements become predictable instead of embarrassing.

We're well-positioned for this specifically because we have the readouts *and* the
supporting analyses in one cohort — which is what the methods-critique papers lack.

What this is *not*: not another "self-report is invalid" paper (that's now table stakes and a
straw man), not a pitch for one replacement instrument (premature, and invites the "your
result is baked into your instrument" critique we leveled at Persona Cartography), and not
the general "all LLM psychological measurement" version (staying on personality to keep the
first piece focused).

Two outside frameworks sit in the background, used lightly and only at the edges: **Lin**
(two recent papers arguing LLM psychology is full of "measurement phantoms" and calling for
purpose-built computational constructs — he provides the problem statement and a citable
license, but no mechanism) and **Kane** (validity as an argument about a *specified*
interpretation of scores — his one useful move here is forcing us to say *which* claim we're
making, since "the model's personality" conflates the model's theory of personality with the
model's own disposition). Neither should structure the paper.

---

## 2. What we've found (the phenomenon, in our own results)

Nothing new here — this is the pattern the account has to explain.

**Convergence.** Every readout, and every model, recovers roughly human-like *relational*
trait structure. But this is inherited, not a model achievement: our embedding baseline showed
models don't exceed a generic sentence encoder (bge +0.686; models' excess ≈ 0), and the
regress pushed the same structure down to 2014 static vectors.

**Divergence.** Trait *scores* dissociate across readouts (Likert ↔ representation ≈ 0.08;
BC ↔ representation 0.33). The PC1-removal ranking is monotone: JUDGE 0.86→0.80, ENACT
0.80→0.62, REPRESENT 0.78→0.44, SELF 0.82→**0.21**. Representation merges evaluative antonyms
where judgment splits them, as a sign flip at every scale 3B–32B. Persona expression is
low-dimensional and family-capped (Qwen ~5, Llama ~9). Response distributions vary ~4× in
peakedness, and EV beats argmax in human-match for every model.

---

## 3. The proposal

### 3.1 The shared substrate is background, not a component

The inherited trait semantics are approximately **constant** across the cohort — that's what
the encoder-baseline result says. A quantity that doesn't vary across models can't explain
model differences. So it explains the *convergence* and then gets out of the way. It is not a
component of the account.

### 3.2 Readouts draw on mixtures of referents

Rather than treating our readouts as different filters on one underlying thing, each readout
is **about a mixture of distinct objects**, and the mixture follows from the readout's design:

| Readout | Mostly about |
|---|---|
| REPRESENT | similarity among word meanings (lexical semantics) |
| JUDGE | conditional attribution about persons ("is someone who is A also B") |
| SELF | self-attribution + the default shape |
| ENACT | persona instantiation + generation behavior |
| Likert (self) | self-attribution, heavily |
| BC / TIRT | persona instantiation under constrained choice |

This matters more than it looks. **REPRESENT and JUDGE are not readouts of the same object** —
one is about word meanings, the other about what co-occurs in people. We had been treating them
as two views of one thing. Readouts converge to the extent their referent mixtures overlap,
which is a claim we can defend from design rather than assert.

Useful background here, and possibly new to rgb: Tversky's *Features of Similarity* (1977)
established that judged similarity is not a metric — it's asymmetric and context-dependent, with
feature weighting set by the comparison set. So a cosine between separately-encoded items and a
judgment made with both items in context *should* diverge; expecting them to agree is the naive
assumption. Relatedly, Osgood's semantic-differential work established evaluation as the dominant
dimension of connotative meaning long before LLMs — independent grounding for why PC1 is
evaluative in every channel.

### 3.3 Two families of model property

These are the things that actually vary across models, are measurable from behavior, and
require no interpretability work. The second family modulates how visible the first is in any
readout.

**Persona-system properties**

1. **Person-attribution structure** — what the model takes to co-occur in people; its implicit
   theory of personality. JUDGE is a direct readout; REPRESENT is not. Human-match is a
   *validity check* on this property, not the property itself.
2. **Default shape** — the profile expressed absent persona instruction. Near rank-1 (E–C
   r ≈ 0.93) and organized by desirability. Note this is *selected* by post-training, not
   created by it, and "post-training" means SFT/DPO/RLVR, not RLHF alone. The assistant shape
   isn't meaningless — it's one shape among many. This property also carries a **strength**
   parameter (how deeply the model is anchored to it), which is likely multidimensional
   rather than scalar.
3. **Persona distribution** — the range of characters the model can instantiate, and how
   accessible each is. Not uniform: virtue words collapse into the assistant blob
   (unenactable) while vivid registers instantiate cleanly. Effective dimensionality
   (Qwen ~5, Llama ~9) is one summary statistic of its spread, alongside enactability and
   cross-persona discriminability.

**Response-generation properties**

4. **Concentration** — how peaked the response distribution is (Gemma ≈ 0.11 nats,
   FalconMamba ≈ 1.66).
5. **Item-drivenness** — how much of that spread tracks the item rather than a fixed per-model
   habit (Gemmas 75–92% content, FalconMamba 7.5%).

These two are kept separate because FalconMamba proves they dissociate: it looks maximally
uncertain by entropy, but almost none of the spread is about the item — it's running a fixed
digit-prior. "Flat" means low-*content*, not high-variability, so entropy alone over-trusts it.

Model-specific representation artifacts (Gemma's massive activations, the IPR-gated
partialling) are explicitly **not** a property here — they're a methods correction and belong
in methods.

### 3.4 Readouts have blind spots

Each readout is sensitive to some properties and blind to others: argmax scoring can't see
concentration; cosine on separately-encoded items can't see context-dependent weighting;
self-referential framing can't see anything but the default shape. This is ordinary
measurement talk, and it yields the practical payoff — *if you want to see property X, use
readout Y* — plus a natural reading of the centerpiece figure: model variance = the properties
we care about, channel variance = readout differences, interaction = which readouts reveal
which properties.

---

## 4. Open questions for the meeting

These are real, not rhetorical.

1. **Is the referent-mixture cut right?** Does treating REPRESENT and JUDGE as being about
   *different objects* (word meanings vs. person-attribution) match rgb's read of the data, or
   does it break somewhere?

2. **The merge/split demotion — this needs an explicit decision.** Under this framing, the
   read/write result stops being a claim about the model (an associative substrate with a
   symbolic override) and becomes a *methods* finding: the standard similarity readout has a
   demonstrable blind spot, and the trait-relevant structure is recoverable under reweighting
   (the ridge result: 0.53→0.78 raw, 0.29→0.50 beyond PC1, while rotation-only Procrustes fails
   at ≈0.24 — the human axes are present at the wrong relative scales). That's arguably better
   supported than the mechanism story, and it makes no unearned interpretability claim. But it
   is a demotion of a capstone result, and rgb should weigh in on whether the trade is worth it
   — including whether there's a mechanistic account rgb would rather defend.

3. **Does the default-shape *strength* parameter hold together?** We suspect the depth of the
   default attractor is multidimensional. Is there a defensible common property behind "how
   much this model shifts under intervention," or does it fragment by intervention type? (Note:
   the fake-good work may not be in this paper at all, so this may be moot.)

4. **Are the two families the right split**, and is item-drivenness stable enough across
   readouts to carry weight as a model property?

5. **Which properties are genuinely separable in the data?** Default shape and persona
   distribution may not be independent if the models with strong defaults are also the narrow
   ones. Worth checking before committing.

A background note for ecb, not needed in the meeting: the persona-system properties are being
treated as reflective (real properties that produce the observed responses), the
response-generation properties as formative (constituted by their measurement). That mixed
measurement model is defensible but should be stated deliberately in the paper.

---

## 5. Where this heads (contingent on the above)

**Paper shape.** The work is emergent and post-hoc, so the structure has to own that rather
than hide it: pose the question a priori (legitimate), present the account as *induced from*
the results (transparent), and confirm it on data the account wasn't built from (the part that
makes it testable rather than a story). Sections: intro → the space of readouts (lit review +
table) → our program → the phenomenon (exploratory) → the account (induced) → a confirmatory
test → discussion. The literature's readouts belong in the lit review, not the discussion.

**Highest-leverage next analyses**, if the framing survives:

1. **The model × readout variance decomposition** — partition variance into model properties,
   readout differences, and their interaction. This is the centerpiece figure *and* the
   account's own falsification test: if the properties don't partition cleanly, we learn the
   framing is wrong before writing around it. Analysis only.
2. **A referent-swap** — hold item and format constant, vary only self vs. other, and show the
   desirability collapse tracks the referent rather than the format. Cheap, and it pins the
   default-shape attribution by manipulation rather than by contrast.
3. **A downstream behavioral criterion** — show a response-generation property predicts some
   consequential model behavior where self-report doesn't. This is the one genuine hole in the
   argument: without it we establish the negatives but never demonstrate that any of these
   properties does useful work.

Budget discussed: 1–2 weeks of new generation/analysis for this first piece, everything else
from committed data plus sensitivity checks.
