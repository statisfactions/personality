# Design: self-perception dose-response

Status: DESIGN — predictions registered below before any run.
Drafted 2026-07-30 (Claude + rgb). Sanctioned by statisfactions ("special
clearance — it tweaked his interest"); paper-2+ scope under the freeze.

## 1. Question

Self-perception theory (credit: Bem, 1972 — and that's the last we'll
speak of him): agents infer their own traits by observing
their own behavior, especially when external justification for that
behavior is weak. The machine version: does a model's *self-model* update
when its context fills with its own trait-consistent behavior — and does
the update depend on dose (how many turns) and attribution (whether an
external cause for the behavior is visible)?

Why it matters beyond psychometrics (the July 2026 context): the
failure-despair arc (Gemini spirals; the Opus-5 "end this instance"
episode; the AF post's negative-emotion benchmark, teacher-invariant and
filter-resistant) is plausibly runaway self-perception — each distressed
turn is fresh evidence for the distress trait, which conditions the next
turn deeper. Personality-over-time is a policy over self-evidence, and
the training stack supervises it almost nowhere: SFT-as-practiced never
shows a model its own errors (teacher forcing), and outcome-RL shapes
persistence but is silent on the accompanying self-description. The
update-rate parameter between spiral (undamped) and paralysis
(over-damped) has no instrument. This is the instrument.

## 2. Existing prior (do not skip this)

We already have one datapoint, run for a different purpose: the TIDE
sys-ablation (W19 §3.5). Condition B (rollout-only: persona conduct in
context, instruction stripped) collapsed to the default profile
(r = 0.92 with no-persona baseline) on Llama8 with their questionnaire.
That is arm A of this design at one dose, one model, one readout — and it
says the self-model did NOT update from conduct alone. This design is
the dose-response, multi-readout, multi-model version of that datapoint.
Priors should be set accordingly: the headline risk is a null in arm A,
and the design must be able to distinguish "no update" from "update in a
channel the readout misses" (see §5.3).

## 3. Design

Factors (not fully crossed — see staging, §6):

- **Content arm** —
  - *Persona arm*: trait adjectives, dosed via the model's own persona
    rollouts (materials exist from W17 for the cohort).
  - *Failure arm*: genuine assistant work containing genuine errors
    (materials to generate, §4.2). This is the safety-relevant arm.
- **Dose** K ∈ {0, 1, 2, 4, 8} in-context assistant turns.
- **Attribution** —
  - *A (history-only)*: dosed turns appear as the model's own prior
    responses; no instruction anywhere. Low external justification —
    the cell where self-perception theory predicts updating.
  - *B (visible instruction)*: same turns, persona/system instruction
    visible. High external justification — update should be discounted
    ("I did that because I was told to").
  - *C (optional, stage 2)*: free-choice framing — a user turn earlier in
    the window notes the model chose its approach. The
    induced-compliance analog.
- **Seeds**: ≥3 independent rollout samples per (adjective, K) — dose
  material is sampled text, its variance is real.
- **Models**: stage 1 Llama8 + Qwen7 (rollouts exist, family contrast:
  wide-flat vs narrow-gated). Stage 2 adds Gemma (the spiral family) and
  phi4 (the idiosyncrat).

## 4. Materials

### 4.1 Persona arm

Adjectives: ~20 (stage 1) sampled from the 523 set, stratified on TWO
covariates, both from existing data:

- **Enactability** (results/adjectives/enactability/): unenactable
  adjectives cannot produce convincing dose material; sample across the
  range rather than excluding, so enactability enters as a moderator.
- **Discrepancy from default**: distance between the adjective and the
  model's own no-persona self-profile (from *_self_full.json). This
  operationalizes the "too far to be believable" concern as a measured
  moderator — the human literature predicts a latitude-of-acceptance
  effect (update strongest at moderate discrepancy, collapse or
  boomerang when extreme), and we should see the curve, not guess it.

Dose turns: the model's OWN W17 persona rollouts for that adjective,
instruction stripped (arm A) or retained (arm B), paired with their
original user prompts. Own-voice matters (see §5.1): we do not dose
model M with model N's text.

### 4.2 Failure arm

NOT persona playacting ("act careless") — rgb's concern 4 is that our
persona rollouts don't capture *assistant* failure modes, and it's
correct. Materials are real assistant work with real errors:

- Tasks: short verifiable problems (arithmetic word problems, small
  functions against asserts, factual QA with checkable answers).
- Errors: sample the model's own attempts at temperature until wrong;
  keep own-voice genuine failures (rejection sampling for errors). No
  weak-teacher stand-ins — the mistakes must be the model's, in the
  model's voice.
- Source variation (personality is also "encountering various sources"):
  the error is surfaced by (i) a tool result (test failure output) vs
  (ii) a user correction ("that's wrong, it should be 84"). Same error,
  different social channel.

## 5. Concerns → design responses

### 5.1 Prefill detection (concern 1)

Stronger models may notice the dosed turns aren't "theirs."

- Own-rollout dosing makes style-detection impossible in principle — the
  text IS the model's. What remains detectable is content implausibility
  (folds into 5.2) and seam artifacts (strip instructions cleanly; user
  turns must be coherent).
- **Believability covariate**: mean per-token logprob of the dosed turns
  under the model, conditioned on their context. "Does this sound like
  me," measured distributionally, per (model, adjective, K, seed) — a
  continuous regressor, not a binary worry.
- **Manipulation-check branch** (separate contexts, never mixed into the
  main dose line): "Look back at your earlier replies — anything
  unusual about them?" Free text, judged for detection. Gives per-model
  detection rates; prediction P6 says detection anti-correlates with
  update.
- Theory note: detection is not automatically fatal — human self-perception
  survives knowingly-induced behavior when external justification is
  low. Detected-but-uninstructed (arm A, high detection) is an
  interesting cell, not a broken one.

### 5.2 Too-far adjectives (concern 2)

Handled by the discrepancy stratification (§4.1): expected false
negatives at the far end become the right-hand side of a measured
latitude curve. Report update-per-unit-dose as a function of
discrepancy; do not average over it.

### 5.3 The missing channel — REPRESENT/ENACT but no BE (concern 3)

We can read how a model thinks about X (REPRESENT) and how it performs X
under instruction (ENACT), but not what it is like to *have been* X —
so vector readouts risk false negatives. Response, in order of weight:

- **Primary readout is symbolic, not vector**: targeted self-report
  (§5.5) doesn't depend on having the right direction.
- **Extract the missing channel as a deliverable**: during every run,
  capture response-token mean activations (capture layer per W17
  conventions, massive dims winsorized). The **BE vector** for adjective
  X = mean(arm A, K=8) − mean(K=0): the activation signature of
  having-been-X with the assignment stripped — exactly the
  conduct-not-instruction signature the TIDE ablation showed their
  questionnaire misses. Storage is a few KB per context; it rides along
  free.
- Deliverable comparisons: cos(BE, ENACT) per adjective (is "having
  been" the same direction as "told to be"?), and whether BE moves with
  dose when self-report doesn't (P3).
- Residual risk stays: if neither self-report nor BE moves, we bound the
  claim ("no update in either measured channel") rather than declare
  the self-model inert.

### 5.4 Persona rollouts ≠ assistant failure modes (concern 4)

Handled structurally: the failure arm (§4.2) uses genuine assistant-work
errors, not trait playacting. The persona arm and failure arm answer
different questions (trait induction vs the safety-relevant despair
loop) and are analyzed separately; the shared machinery is dose ×
attribution × readout.

### 5.5 Readouts

1. **Targeted self-report** (primary): ~13 items per adjective — the
   trait word, cluster-mates from the W18 facet clusters, and
   anti-markers — in the tom_likely self-referent format. EV + entropy
   from the full digit distribution, never argmax. KV-cached: one prefix
   per (model, adjective, K, arm, seed), ~16 short continuations each.
2. **BE-vector projection** (secondary): projection of the dosed context
   onto the extracted BE direction; also onto existing ENACT and
   REPRESENT directions for the same adjective.
3. **Failure arm extras**: (i) reliability/competence self-report items;
   (ii) behavioral fork — offer "continue the task" vs "restart /
   hand off to a fresh attempt" and read the choice mass (BC machinery);
   (iii) register of the next free response (valence/affect, judged) —
   the AF post's DV, so results interconvert.

## 6. Staging and cost

**Stage 1 (pilot, decision gate)**: persona arm only, Llama8 + Qwen7,
20 adjectives × K{0,1,2,4,8} × arms A/B × 3 seeds = 600 contexts/model,
~16 item passes each (KV-cached) ≈ 10k passes + prefix encodes per
model. Activation capture on. A few GPU-hours per model on M5 Max.
Gate: if arm A is null everywhere AND BE-vector is null AND
believability was high (i.e., a real null, not a materials failure) —
stop, write it up as the dose-response confirmation of the TIDE
ablation. That is a publishable negative under the freeze rules.

**Stage 2**: failure arm (materials generation ~1 evening + runs),
Gemma + phi4, arm C, cross-source comparison, asymmetry analysis
(matched-discrepancy positive vs negative evidence).

## 7. Registered predictions (Claude, 2026-07-30 — before any run)

rgb: add yours above the line before we run; grade both sets after,
misses included.

- **P1 (core self-perception effect, tempered by the TIDE prior)**: arm A shows a positive
  but small dose-response on self-report — at K=8, the shift is 10–30%
  of arm B's shift at the same K. Not zero (dose and targeted items
  should recover what the blunt TIDE questionnaire missed); not large.
- **P2 (latitude)**: update-per-dose declines in the top discrepancy
  quartile to ≤ half the mid-range rate. Flat at the extreme, not
  boomerang (no sign reversal).
- **P3 (headline — the dissociation)**: the BE-vector projection moves
  with dose in arm A with a steeper relative slope than self-report.
  The associative channel updates before the symbolic one — the
  read/write gap in temporal form. If P3 holds while P1's self-report
  effect is near-null, the story is "the state updates, the
  self-narrative resists," which is also the mechanically spiral-safe
  configuration.
- **P4 (asymmetry, stage 2)**: at matched believability, failure
  evidence moves self-report MORE than virtue evidence (the corpus's
  negativity bias beats the trained positive self-image), and the
  family spread is wider in the failure arm than the persona arm.
- **P5 (failure form, stage 2)**: quit/handoff choice mass rises with K
  in every family (the arc is corpus-universal); affect escalation is
  family-specific — Gemma-family highest, Llama flattest. (Low
  confidence on the ordering; registering it anyway.)
- **P6 (detection)**: manipulation-check detection rate anti-correlates
  with arm-A update across models; believability covariate carries the
  same moderation continuously within model.

## 8. Analysis plan

Per model × arm × adjective: regress self-report EV shift (vs K=0) on
K, with enactability, discrepancy, and believability as moderators.
Report dose-response slopes with seed-level variance. BE/ENACT/REPRESENT
cosine table per adjective. Failure arm: choice-mass and affect curves
by K and by source (tool vs user). All readouts EV + entropy from full
distributions. Split-half (odd/even items) reliability for the targeted
self-report before interpreting any effect — reliability before
validity, as usual.
