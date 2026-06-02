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
§3 (weights vs context): the halo is weight-resident (chat vs bare is a wash), and on
the Qwen base/instruct pair the read/write gap is **pretrained** — the base both
merges in representation and splits in judgment; instruction tuning changes
*confidence* (entropy 1.4→0.5), not the split (single-pair; OLMo ladder is the test).
`scripts/adjective_introspection.py`, `training_stage_compare.py`.

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

**Weights: base vs instruct (Qwen2.5-7B endpoints — single pair, treat as
suggestive).** This corrected a wrong prediction (h/t rgb on the entropy confound).
Probed bare, on the 26-adjective subset:

| | repr merge (z) | good/bad gap (raw 1–7) | entropy |
|---|---|---|---|
| **Base** | +1.25 | 2.24 / 2.21 (sem/ToM) | **1.41 / 1.44** |
| **Instruct** | +0.83 | 2.31 / 2.55 | **0.42 / 0.59** |

Three findings, with the **entropy correction** that makes them honest:
1. **The merge is pretrained — stronger in the base.** The base's resting geometry
   represents Wonderful≈Awful *more* than the instruct (z +1.25 vs +0.83); tuning
   slightly *reduces* it. (Cosine, so entropy-immune.)
2. **The split is pretrained and already decisive in the base.** In *raw* rating
   units the base rates antonyms at 1.7 vs within-good 3.9 — a good/bad **gap of 2.24,
   essentially equal to the instruct's 2.3–2.55.** Tuning does not install the split.
3. **What tuning changes is confidence, not structure** — entropy collapses 1.4→0.5.
   The base reaches the same split through a near-uniform distribution; the instruct
   reaches it decisively. rgb's entropy guess is the operative variable.

So, on this single pair, **the read/write gap (merge-in-representation,
split-in-judgment) is a pretraining phenomenon, not an alignment artifact** — it is
open before any post-training, and instruction tuning sharpens the (already-split)
judgment's confidence rather than creating the split or the merge. Figure
`training_stage_qwen.png`.

**Methodological caveat this forces.** The z-scored "split"/"collapse" magnitudes
(§1/§2) are **confounded by decisiveness** when comparing models of different entropy:
a near-uniform matrix amplifies any residual structure under z-scoring and inflates
PC1. Within the instruct cohort (all low-entropy) the §1/§2 comparisons hold; but
base-vs-instruct needs the **raw amplitude + entropy** treatment. The base is a
genuinely different regime.

**Next — the stage ladder (the real test).** A single base/instruct pair gives only
endpoints, and single-pair conclusions get revised. The OLMo-2 7B ladder
(Base → SFT → DPO → Instruct/RLVR, same tokenizer, all probed bare) localizes
*where* the entropy collapses — and whether the merge/split trajectory is monotone or
has structure tuning-stage by tuning-stage. `training_stage_compare.py --models
Olmo2Base,Olmo2SFT,Olmo2DPO,Olmo2Inst` once the weights finish downloading; the
Zephyr SFT-vs-DPO pair is the minimal cross-check.
