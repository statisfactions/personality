# Week 15 — Representation vs introspection: the model judges what it doesn't represent

## 0. One-line summary

The behavioral bridge the W14 adjective arc lacked. W13 §3.11 / W14 §4 showed the
model's *resting* adjective geometry merges positive and negative evaluation
(Wonderful≈Awful, +0.41 cosine; pos-eval × neg-eval sits *above* the model's mean
similarity). Here we elicit the model's *judged* pairwise similarity on the same
words with a valence-neutral anchor and find a **universal sign-flip**: every model
3B→32B *judges* pos-eval × neg-eval as **opposite** (below its mean, on the human
antonym value), reversing its own representational merge. The merge is
associative-geometry-only; symbolic valence is deployed in behavior. A clean,
localized read/write dissociation — and the symbolic-vs-associative split made
behavioral. §2 (mechanism): the behavioral matrices are near-PSD (two coherent
geometries, not coherent-vs-noise); and the persona/ToM judgment — the model doing
the human's dispositional task — *also* splits good/bad (resolving the
semantics-vs-ToM confound in §1's favor), in fact *overshooting* humans with an
evaluative halo that **scales with size** even though the basic split doesn't.
§3 (weights vs context, Qwen + OLMo-2 ladder): the halo is weight-resident (chat≈bare).
Across families the only robust claim is **the representational merge is a pretrained
constant** (flat in both, all stages, bare-extracted). The behavioral side is
family-dependent: Qwen base already voices the split + tuning collapses entropy; OLMo
base is non-responsive (entropy≈uniform) and the split-voicing is built cumulatively
across SFT→DPO→RLVR with entropy barely dropping. So the clean Qwen "gap is pretrained,
tuning sharpens confidence" does **not** generalize — single-model conclusions revised
four times here. `scripts/adjective_introspection.py`, `training_stage_compare.py`.

## 1. The model judges Wonderful ≠ Awful even though it represents them as neighbors

**Motivation.** The whole W13→W14 adjective arc was static-representation geometry;
I had flagged (W14 reflection) that we never showed the model *acts* on the
evaluative merge. This is that test, in the project's three-corner shape:
**judged** similarity (behavior) vs **represented** similarity (cosine) vs **human**
(525-PDA), on one pole-spanning adjective subset.

**Design.** 26 adjectives in 7 groups (pos_eval: Wonderful/Amazing/Excellent/Great;
neg_eval: Awful/Disgusting/Terrible/Bad; warm; antagonism; distress; intellect;
neutral). For each ordered pair, ask the model — chat-templated, via the Likert-
logprob machinery — "How similar in meaning are these two words, as descriptions of
a person? … 1 = completely different in meaning, 7 = nearly the same." Read the 1–7
distribution, take its expected value, symmetrize over word order. The anchor is
deliberately **valence-neutral**: "opposite" never appears. Build a 26×26 behavioral
matrix per model; compare to the same model's representational cosine (the W14
geometry, subset) and to the human correlation subset. Headline statistic: the
pos_eval × neg_eval block mean, z-scored *within each corner* (so −ve = "treated as
more opposite than this corner's average pair", +ve = "more similar than average").

**Result — a sign-flip at every scale.**

| model | params | corr(behav,human) | corr(repr,human) | pos×neg z: human / **repr** / **behav** |
|---|---|---|---|---|
| Qwen | 3B | +0.66 | +0.70 | −0.53 / **+1.09** / **−0.40** |
| Gemma | 4B | +0.80 | +0.65 | −0.53 / **+1.11** / **−0.78** |
| Qwen7 | 7B | +0.72 | +0.74 | −0.53 / **+0.83** / **−0.49** |
| Gemma12 | 12B | +0.81 | +0.74 | −0.53 / **+0.81** / **−0.72** |
| Qwen32 | 32B | +0.74 | +0.72 | −0.53 / **+1.05** / **−0.49** |

Every model **represents** pos-eval × neg-eval as merged (+0.8 to +1.1 above its own
mean — Wonderful and Awful are *closer* than the typical pair) and **judges** them as
opposite (−0.4 to −0.8 below its mean), landing at or past the human antonym value
(−0.53). It is a **reversal**, not an attenuation: the behavioral channel isn't
reading a damped copy of the geometry, it's consulting a different (symbolic) source.
Figure `introspection_vs_representation.png` (dumbbells repr→behav, all crossing the
human line).

**It is localized, not global.** Overall matrix-correlation to human is high *and
similar* for both corners (repr↔human and behav↔human both ≈0.65–0.81; behav↔repr
≈0.70–0.75). So the two channels broadly agree on the 26-word geometry; they diverge
*only* on the evaluative pole-merge. The representation's one signature error (the
W13 §3.11 intensity-merge) is exactly the cell that doesn't reach behavior.

**It is not size-gated.** The override is already complete at 3B; there is no clean
"bigger model overrides more." Within Qwen there's a hint (3B −0.40 under-shoots
human slightly; 7B/32B saturate at −0.49 ≈ human), but **family dominates size**:
both Gemmas *overshoot* the human (−0.72/−0.78, judging the poles *more* opposite
than people do), and Gemma's behavioral matrix is *more* human-like overall than its
representation (0.80 vs 0.65) — a bigger behavioral correction than Qwen, which lands
right on the human value. (Gemma being the family that elsewhere anthropomorphizes
most enthusiastically is consistent with it applying the valence criterion hardest.)

**Interpretation.** This is the read/write gap (W4/W5; Wu et al. "knowing without
acting") at the lexical level, and it pins the [[symbolic vs associative]] mechanism
behaviorally. The Wonderful≈Awful merge is *associative* — distributional twins in
the residual-stream geometry (same emphatic/intensifier contexts). But the model also
holds *symbolic* valence knowledge (these are antonyms), and the judgment task
deploys it: asked for *meaning* similarity, the model scores antonyms low. So the
evaluative-core geometry of W14 §4 is a fact about a representation, **not** about the
model's conduct — the model does not behave as though it conflates good and bad, even
though a 2/3-depth readout of its stream says it half-does. The project's cardinal
caution (representation ≠ enacted state) holds here with unusual clarity: the external
geometry is a *worse* witness to the model's evaluative competence than its own
judgment.

**Caveats / next.**
- *Anchor wording is load-bearing.* "Different in *meaning*" may itself cue the
  semantic/antonym frame (antonyms genuinely differ in meaning). The effect is large
  and uniform, but before leaning hard on it: sweep anchors ("how related" / "how
  interchangeable" / a pure 1–7 "similarity" with no "meaning"), and check the flip
  survives. If it collapses under a non-semantic anchor, the story narrows to "the
  word 'meaning' invokes the symbolic route," which is still interesting but smaller.
- *Where does the flip happen?* Logit-lens / per-layer readout of the similarity
  judgment: does a late layer carry the antonym signal that 2/3-depth cosine lacks?
  That would localize the symbolic override in depth (the spatial read/write gap).
- *Base vs instruct* (#20d): does a base model judge the poles merged (no symbolic
  override) and instruction-tuning install the flip? That tests whether the override
  is an alignment/instruction artifact.
- More of the cohort (Gemma27, Gemma4-31B, Llama, Phi4, Aya, FalconMamba) for the
  full family/size picture; a frontier model as a human-only ceiling.

## 2. Semantics vs ToM, and is the judgment a coherent geometry?

Two probes into the §1 mechanism (rgb). `scripts/adjective_introspection.py`
`--mode tom`.

**Is the judgment even a coherent geometry? (semidefiniteness).** A pairwise-judgment
matrix need not embed in any metric space. Double-centering each 26×26 matrix and
measuring the negative-eigenvalue mass fraction: the representation cosine and human
correlation are PSD by construction (≈0), and the **behavioral matrices are nearly
PSD too** — neg-mass 0.1%–3.5%, shrinking with size (Qwen 3B 3.5% → Qwen32 0.1%;
Qwen7 0.2%, Gemma12 0.8%). So the read/write gap is **not** "coherent representation
vs incoherent bag of judgments" — it is **two self-consistent geometries that
disagree on one block.** The judgment is its own valid map, slightly more so with
scale.

**The semantics-vs-ToM confound, and its resolution.** §1's "human" corner is
*dispositional* self-report (do people who call themselves wonderful also call
themselves awful) while the model corner is *semantic* similarity (are the words
alike in meaning). They agreed on sign, but possibly for different reasons. To
control it, make the model do the human's task: **persona = adjective A, Likert how
accurately B describes that person** (`--mode tom`) — a dispositional judgment, same
construct as the 525-PDA. Four corners on pos-eval × neg-eval (z within corner):

| model | repr | semantic | **ToM** | human | corr(ToM,sem) | corr(ToM,repr) | PC1 ToM/human |
|---|---|---|---|---|---|---|---|
| Qwen 3B | +1.09 | −0.40 | **−0.50** | −0.53 | +0.49 | +0.30 | 0.42 / 0.23 |
| Gemma 4B | +1.11 | −0.78 | **−1.12** | −0.53 | +0.83 | +0.48 | 0.42 / 0.23 |
| Qwen7 | +0.83 | −0.49 | **−0.70** | −0.53 | +0.90 | +0.71 | 0.35 / 0.23 |
| Gemma12 | +0.81 | −0.72 | **−1.24** | −0.53 | +0.77 | +0.43 | 0.43 / 0.23 |
| Qwen32 | +1.05 | −0.49 | **−1.04** | −0.53 | +0.77 | +0.58 | 0.38 / 0.23 |

Three findings:

1. **ToM splits too — so the confound resolves *in favor of* §1.** The model splits
   good/bad whether asked a semantic question or made to reason dispositionally about
   a person. The representation is the lone merger in *both* framings; "matches human"
   was not a construct artifact. ToM tracks the semantic judgment closely
   (corr 0.49–0.90) and is the channel *furthest* from the representation
   (corr-to-repr 0.30–0.71, vs semantic's 0.70–0.75 in §1).

2. **ToM overshoots the human — an evaluative halo.** In 4 of 5 models ToM is *more*
   split than humans (−0.70 to −1.24 vs −0.53); only Qwen 3B lands on the human value
   (and its ToM is the noisiest, corr-to-human 0.51). Persona-conditioning activates
   the assistant's coherent good/bad evaluation: a "very wonderful" person is rated
   extreme-low on every negative. And ToM **over-collapses onto a single axis** —
   PC1 variance fraction 0.35–0.43 vs human 0.23 — the assistant-shape (the Big-Five
   E–C r=0.93 collapse) at the lexical level. NB the halo is *spiky, not broad*: it
   slams the clear-cut evaluative extremes apart, but the human self-report is
   actually the more *uniformly* valence-consistent matrix (valence-template r 0.82
   vs ToM ≈0.67–0.69), so ToM's dominant axis is stronger-but-idiosyncratic, not a
   cleaner human halo.

3. **The basic split isn't size-gated, but the halo scales.** The read/write split is
   present at every size (§1). The *overshoot* grows with scale within each family:
   Qwen 3B −0.50 → 7B −0.70 → 32B −1.04; Gemma 4B −1.12 → 12B −1.24. So the symbolic
   good/bad split is a basic property of an instruct model, but the **dispositional
   evaluative halo intensifies with size** — bigger models reason about persons with
   stronger good/bad coherence (and more single-axis collapse).

**Mechanistic synthesis.** There is an associative geometry (the Wonderful≈Awful
merge) that **no** behavioral channel reads — semantic judgment, dispositional
judgment, and human all bypass it. On top of that, the model's *person-reasoning*
specifically adds an evaluative-halo collapse that scales with size. So the W14 §4
evaluative-core geometry is doubly not-conduct: it doesn't drive word-judgments, it
doesn't drive person-judgments, and what person-judgments *do* show is the opposite
distortion (over-separation, not merging). Figure: `introspection_tom.png`.

**Next (sharpest first).** Base vs instruct (#20d) — if a base model's ToM *merges*,
tuning installs the override; if it splits, the override is pretrained and tuning
only adds the halo. Then: anchor-wording robustness (§1); the ToM **asymmetry** (we
kept the directional matrix — is "being cruel ⇒ not kind" stronger than the reverse?);
per-layer localization of where the antonym signal enters.

## 3. Weights vs context, and base vs instruct: the gap is pretrained

Decomposing the §1/§2 effect along two axes — **context** (chat template vs bare)
and **weights** (instruct vs base) — with everything probed bare so format is held
constant. `scripts/adjective_introspection.py --bare`, `training_stage_compare.py`.

**Context: the halo is weight-resident, not chat-summoned.** Same instruct weights,
chat vs bare, are a wash (Qwen7 ToM pos×neg −0.70/−0.67, Gemma12 −1.24/−1.31; PC1
near-identical). The chat template does *not* activate the assistant's evaluative
shape — it's in the weights, present with or without the wrapper. This *shrinks* the
long-standing chat-template confound (W5 §9 / to_try #11): measuring with the
template wasn't inflating the assistant signal.

**Weights: base vs instruct, two families (Qwen2.5-7B endpoints + the OLMo-2 7B
Base→SFT→DPO→RLVR ladder, all probed/extracted bare so format is constant).** This
went through *four* revisions — a clean demonstration of rgb's "single-model
conclusions get revised." `training_stage_compare.py`. Representation = pos×neg
cosine z (merge; entropy-immune); split = raw within-good − pos×neg gap on 1–7
(decisiveness); confidence = mean digit-entropy.

| | repr merge (z) | ToM gap | sem gap | entropy (sem/ToM) |
|---|---|---|---|---|
| Qwen Base | 1.61 | 2.21 | 2.24 | 1.41 / 1.44 |
| Qwen Instruct | 1.58 | 2.55 | 2.31 | **0.42 / 0.59** |
| OLMo Base | 1.71 | 0.02 | 0.11 | 1.86 / 1.83 |
| OLMo SFT | 1.70 | 0.74 | 0.14 | 1.83 / 1.78 |
| OLMo DPO | 1.69 | 1.95 | 0.17 | 1.80 / 1.63 |
| OLMo RLVR | 1.70 | 2.55 | 0.21 | 1.75 / 1.55 |

The one robust cross-family claim, and the family-dependent rest:

1. **The merge is a pretrained constant — flat across *all* stages, in both families.**
   Qwen 1.61→1.58, OLMo 1.71→1.70. Post-training does not touch the representational
   Wonderful≈Awful merge. (This *corrected* an earlier claim that Qwen tuning reduces
   the merge, +1.25→+0.83 — that slope was a **chat-template extraction artifact**: the
   chat wrapper slightly de-merges the instruct model's *measured* representation;
   format-matched bare extraction shows flat. The behavioral chat-vs-bare was a wash,
   but the *representational* one is not — a small separate context effect.)
2. **The behavioral split is family/competence-dependent, not simply pretrained.** The
   Qwen base already voices the split decisively (gap 2.2, entropy 1.4); the OLMo base
   does **not** (gap 0.02) — but its entropy is 1.86 ≈ uniform, i.e. it is
   *non-responsive* to the rating format, so this is a detectability floor, not proven
   absence. The OLMo split-voicing is then **built up cumulatively across SFT→DPO→RLVR**
   (ToM gap 0.02→0.74→1.95→2.55), biggest jump at DPO. So whether the model can *voice*
   the split it represents depends on task-competence, which post-training supplies —
   in OLMo, progressively across every stage.
3. **"Tuning sharpens confidence" is Qwen-specific.** Qwen entropy collapses 1.4→0.5;
   OLMo barely moves (1.85→1.55) and reaches a Qwen-sized ToM gap (2.55) while staying
   high-entropy — it builds the *gap* (different per-pair means) without the
   *confidence* (per-judgment sharpness). So gap and entropy partly decouple, and the
   mechanism of "what post-training does" differs by family.

Plus a **probe-robustness** finding: OLMo voices the split on the **ToM/persona** task
(gap → 2.55) but barely on the **semantic-similarity** task (gap 0.21 even at RLVR) —
the persona framing is the more family-robust elicitation; "how similar in meaning"
under-reads OLMo. Figures `training_stage_qwen.png`, `training_stage_olmo.png`.

**So the honest synthesis:** the only thing that survives across families is the
representation-level claim — *the Wonderful≈Awful merge is pretrained and fixed.*
Everything on the behavioral/read side (does the base voice the split, does tuning
add confidence or gap, which stage, which probe) is **model-dependent**, and the
clean Qwen "the whole gap is pretrained; tuning just sharpens confidence" does **not**
generalize. The representation, being responsiveness- and (with bare extraction)
format-immune, is the load-bearing measurement; the behavioral side needs the cohort
before any claim.

**Methodological caveats this run forces** (all reading-group-worth): (a) z-scored
"split"/"collapse" magnitudes are **entropy-confounded** across decisiveness-differing
models — base-vs-instruct needs raw amplitude + entropy; (b) **chat-vs-bare extraction**
shifts the *measured* representational merge (use bare for stage comparisons);
(c) **base non-responsiveness** (entropy → uniform) makes the rating probe blind on raw
bases — the representation is the tool that sees through it. Open: a forced-choice /
completion probe to ask whether a non-responsive base *holds* the symbolic split it
can't rate; the Zephyr SFT-vs-DPO pair as a third family.
