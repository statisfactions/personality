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
behavioral. `scripts/adjective_introspection.py`.

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
