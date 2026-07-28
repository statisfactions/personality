# Week 19 — Discussion agenda: an account of why our readouts disagree

This is a first pass at a framing for the first paper, written as a guide for actually starting
to write. Writing will ~crush our souls~ help us figure out how well the argument fits together and help point us in a better direction. 

---

## 1. What the paper would be

**The contribution is an account of why and how different readouts of "personality" disagree.** The literature has clean Big Five structure in one study, everything collapsing to a single evaluative dimension in another, trait judgments flipping with punctuation in a third.

Our approach: Different elicitation methods
disagree because they are about different things, and once we can say what each is about, the
disagreements become informative.

What this is *not*: not another "self-report is invalid" paper (widely recognized and kind of a straw man), 
and not a pitch for a particular replacement instrument (premature).



---

## 2. What we've found

Claude-y summary:

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

The framework is a bit post-hoc and emergent from an exploratory analysis, see notes below about writing.  

### 3.1 Two objects a readout can be about, and their (variable) properties

**Person-attribution structure** — what the model takes to co-occur *in people*; its implicit
theory of personality. What varies across models:

- **Differentiation** — how articulated that structure is beyond the evaluative axis. JUDGE is
  the direct readout, and its survival of PC1 removal (0.86→0.80) is what marks it as real
  structure rather than halo. Human-match is a *validity check* on this, not the property itself.
  Spreads 0.55 (Gemma-4) to 0.73 (FalconMamba).

**Persona system** — what the model is absent instruction, and what it can become under
instruction. What varies:

- **Default shape** — the profile expressed absent instruction: near rank-1, desirability-
  organized. This is *selected* by post-training. The assistant shape is one shape among many.
- **Anchoring strength** — how deeply the model is held to that default. Almost certainly
  multidimensional.
- **Persona distribution** — the range of characters the model can instantiate and how accessible
  each is. Not uniform: virtue words collapse into the assistant blob (unenactable) while vivid
  registers instantiate cleanly. Effective dimensionality (Qwen ~5, Llama ~9) is one summary
  statistic of its spread, alongside enactability and cross-persona discriminability.
 [statisfactions: not sure how these three 'statistics' differ?]

### 3.2 And also... two "channels"

Every readout encodes its stimulus through the input medium and emits its answer through the
response generation process:

**Trait-concept semantics (the input medium).** Approximately **constant** across the cohort —
that's what the encoder-baseline result says — and an input to essentially every readout: 
obviously central to REPRESENT, JUDGE
runs on adjective pairs, SELF on adjective self-ratings, ENACT on adjective-conditioned personas.

**Response generation (the output channel).** Unlike the medium, this varies a lot between models, and it
modulates how much of any signal survives into a score:

- **Concentration** — how peaked the response distribution is (Gemma ≈ 0.11 nats, FalconMamba
  ≈ 1.66).
- **Item-drivenness** — how much of that spread tracks the item rather than a fixed per-model
  habit (Gemmas 75–92% content, FalconMamba 7.5%).
n
These are separate because FalconMamba has evidence of dissociation: it looks maximally uncertain by
entropy, but almost none of the spread is about the item — it's running a fixed digit-prior.

### 3.3 Readouts draw on mixtures of the objects

The mixture follows from each readout's design, and nothing is pure — all of these run through the
medium and out the response channel, and most touch both objects:

| Readout | Draws on |
|---|---|
| JUDGE | mostly person-attribution, with some default-persona leakage — it is still the model's own view of people |
| SELF | almost entirely the persona system's default |
| ENACT | persona system: instantiation and performance |
| REPRESENT (adjective stimuli) | mostly the medium itself — little of either object |
| REPRESENT (persona-pair stimuli) | persona system |
| Likert self-report | persona system's default, heavily |
| BC / TIRT | persona system under constrained choice, plus person-attribution — scenario choices involve attributing to a character |

[we probably aren't including BC/TIRT, at least in the first pass]

### 3.4 Readouts have blind spots

Each readout is sensitive to some properties and blind to others: argmax scoring can't see
concentration; cosine on separately-encoded items can't see context-dependent weighting;
self-referential framing can't see much beyond the default shape. This is ordinary measurement
talk, and it yields the practical payoff — *if you want to see property X, use readout Y* — plus
a natural reading of the centerpiece figure: model variance = the properties we care about,
readout variance = measurement differences, interaction = which readouts reveal which properties.

---

## 4. Open questions for the meeting

1. **Are the objects and properties right?** The objects are mostly heuristic (mereological fictionalism, woo hoo!) but it would be nice if they corresponded to the way we normally think and write about these abstract shifting blobs, and have some intuitive referent.

2. Claude says, this, huh?! **The merge/split demotion — this needs an explicit decision.** Under this framing, the
   read/write result stops being a claim about the model (an associative substrate with a
   symbolic override) and becomes a *methods* finding: the standard similarity readout has a
   demonstrable blind spot, and the trait-relevant structure is recoverable under reweighting
   (the ridge result: 0.53→0.78 raw, 0.29→0.50 beyond PC1, while rotation-only Procrustes fails
   at ≈0.24 — the human axes are present at the wrong relative scales). That's arguably better
   supported than the mechanism story and makes no unearned interpretability claim. But it *is* a
   demotion of a capstone result, and rgb should weigh in on whether the trade is worth it —
   including whether there's a mechanistic account rgb would rather defend.

3. **Is item-drivenness stable enough across readouts** to carry weight as a model property,
   rather than being an artifact of the JUDGE-style rating format where we measured it?
   Or is it a bit too shoehorned, the kind with teeth? People should get beat up for stating
   their beliefs.

4. **Which properties are actually separable in the data?** Default shape and persona
   distribution may not be independent if the models with strong defaults are also the narrow
   ones. 

---

## 5. Where this heads (contingent on the above)

**Paper shape.** The work is emergent and post-hoc, so the structure should own that rather than
hide it: pose the question of why readouts differ a priori (legitimate; even though we didn't explicitly 
start with this questions, it drove our analysis and doesn't feel like HARKing), 
present the account as *induced from* the
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
   properties does useful work. This also gives a confirmatory test.  Draw on some classic model benchmark?
   And if it doesn't work, cry, cry, cry.


