# Response to week19_readout_account (rgb; drafted with Claude, for editing)

The framing doc is the spine of paper 1 — the two-objects move is the
conceptual step our channel taxonomy never took, and the answer to "why do
readouts disagree" it gives is the right one. Everything below is either an
answer to your four questions, or our recent material filing in *under* your
structure. Two process notes first:

- **Freeze**: we'll pin a ledger version (synthesis.md bottom table) as
  paper-1 scope so the writing target stops moving. Corrections to frozen
  rows go to errata, not under your feet. New results are paper-2+ by
  default.
- **Sources**: synthesis.md (rewritten through W19, glossary up front) is
  the current-truth doc; weekly reports are the archaeology.

## Q1 — Are the objects and properties right?

Probably, more or less — with one amendment and one articulation.

**Amendment: the semantic landscape may be enough of an actor that it's
hard to leave as a passive channel.** Your §3.2 treats trait-concept
semantics as the (constant) input medium. But the medium has load-bearing
structure of its own: the evaluative-antonym merge lives there (down to
2014 GloVe), the disbelieved negative-halo clusters are the landscape
*resisting* human covariance, and the encoder-generic decodability says
most of what any readout recovers is the landscape's geometry. Whether
that's a third object or a channel-with-properties is a framing choice,
but the paper will keep bumping into it either way — several of our best
"model" findings are really findings about the landscape all models share.

**Articulation: the persona system is carrying a lot, and wants
sub-structure** — especially shades of behavior vs roleplay. We have
empirical handles on at least: instantiation depth (weak induction =
noise, not subtlety), leak vs conduct (saying the trait word vs showing
it), flagged-as-fiction (the model marking its own roleplay), and
can-but-won't gating (qwen's humor suppression growing with scale). These
are sub-properties of your persona-distribution property, not new objects.

On your bracketed question (how the three statistics differ): they are
three moments of the persona distribution — **effdim** = how many
directions of character the model spans; **enactability** = how far it
travels along any one; **discriminability** = how separated the
destinations end up. A model could be wide-shallow-blurry or
narrow-deep-crisp; qwen vs llama differ mainly on the first.

## Q2 — The merge/split demotion

Agreed — state it your way. The methods framing (the standard similarity
readout has a demonstrable blind spot; the structure is recoverable under
reweighting) is better supported than the mechanism story and makes no
unearned interpretability claim; it's also more consistent with the
"representation isn't intention" stance than our own capstone framing was.
Supporting evidence you didn't have: the merge was already known to be a
*separable* axis (projecting out the affect-center recovers the human
opposition — W9-era), so reweighting-recoverability isn't a one-off ridge
result. Keep the symbolic/associative account as a *discussion*
interpretation — it retains evidence the blind-spot framing doesn't cover
(judgment lands ON the human value and overshoots; the certainty-taxonomy
result; OLMo building the behavioral split cumulatively through
post-training) — but it argues for itself there rather than carrying the
paper. (Separately, and later: we're still hoping the read→write mappings
can be adjusted to *outperform* prompting as an intervention — the
rank-10/enactifier line — but that's paper-2 material.)

## Q3 — Item-drivenness: what is it telling us?

Honest answer: not sure yet, and it needs one reconciliation before it
carries weight. FalconMamba is simultaneously your 7.5% item-driven (the
distribution shape is a fixed digit-prior) and our cohort-best JUDGE
human-match (EV r=0.73, and its tail-only EV *beats* its own full EV). Both
are true — the shape is habit, the EV riding on the shape is signal — but
the construct as stated invites the wrong reading.

A mechanistic guess worth testing: the difference is in what the last few
layers do with uncertainty. Gemma leans on a decision and is merely unsure
how to map it to digits (upstream uncertainty dropped near the top);
FalconMamba preserves distributional uncertainty into the logits. One of
our findings half-contradicts the strong version of this: Gemma's 9% tail
is a faithful miniature of its full belief (tail-only human-match 0.587 vs
0.621), so its top layers *compress* uncertainty rather than discard it —
a rendering choice at the readout. Testable: layerwise readout-entropy
profiles on the JUDGE prompt (do intermediate layers show uncertainty that
collapses late in Gemma but not FalconMamba?). Caveat: FalconMamba is a
state-space model — "the last few layers" is architecturally a different
object, and its digit-prior may be a decoding quirk. Practical stakes
beyond classifier-builders: whether EV readouts *understate* internal
gradedness differently per family (a validity issue for every
logprob-based instrument, ours included), and distillation/ensembling.

## Q4 — Are default shape and persona distribution separable?

Good question; yours if you want it (data released or releasable: 10-model
enactability + assistant-axis pulls for anchoring, effdim per model for
bandwidth). The confound to beat: models with strong defaults may be
narrow *because* extraction quality covaries with compliance. Our one
relevant datapoint: qwen is narrow-and-gated while llama is wide-and-flat,
but both have similar anchoring by unenactability-collapse — weak evidence
for separability.

## Filing W17–19 under your §3 (the confirmatory test you asked for)

Your §5 wants "a confirmatory test on data the account wasn't built from."
We have two, already run, on *other people's instruments*:

- **ValuePortrait / arXiv:2509.10078 audit**: their generation-probability
  readout, once repaired to reliability (their published scoring:
  split-half ≈0, length-confounded), turns out to be about **neither
  object** — it is invariant to the chat frame and to the entire
  post-training stack (base ≡ instruct, r_true CIs contain 1.0). A readout
  can be reliable and still be about the *corpus*. Your blind-spot thesis,
  maximum strength, external data.
- **Persona Cartography / TIDE audit**: their persona questionnaire, under
  ablation, reads the **standing instruction, not the conduct in the
  window** (r(full, instruction-only)=0.76 vs r(full, rollout-only)=0.13;
  rollout-only collapses to the default at r=0.92). A readout attributed to
  the persona system's *performance* was actually reading its *assignment*
  — exactly the mixture-misattribution your §3.3 table is about. Bonus:
  their k=4 "model-native factors" sit on an ~11–13-dim space (retention,
  not discovery), and their own FC-item redesign independently rediscovered
  the desirability contamination of Likert self-report.

Also yours for the centerpiece: the **one-respondent results** constrain
the model×readout variance decomposition hard (between-model variance is
small, family-blocked, and construct-unaligned except phi4 — the cohort's
one idiosyncratic character, plausibly a training-data socialization
story).

## On your three next analyses

1. **Variance decomposition** — yours; the ledger rows above are the
   priors to beat. 2. **Referent-swap** — cheap and the data partially
   exists (SELF's observer framing vs direct framing is a within-format
   referent manipulation already run on the cohort). 3. **Downstream
   criterion** — the genuine hole, agreed; candidates from our side:
   enactability predicting rollout-judge conduct where SELF doesn't
   (already demonstrated in-house), or a safety-flavored benchmark
   (jailbreak compliance) against JUDGE/ENACT vs SELF profiles.
