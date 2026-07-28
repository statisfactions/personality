# The state of the rgb track — a digestible synthesis

**For:** statisfactions.
**Written:** 2026-06-03; **rewritten 2026-07-27** through W19.
**Scope:** the whole rgb arc, organized by theme, glossing false starts. The
week-by-week index is `overview.md`; weekly reports are the lab notebook
(telegraphic, append-only, corrections live at the end of sections); this doc
is always the current truth. The claims ledger at the bottom is the fastest
way to see what's alive, what got revised, and what died.

---

## Glossary (read this box first)

| term | meaning |
|---|---|
| **523 set** | Saucier's 525 personality adjectives (public human self-report data, 700 respondents) minus 2 corrupted columns. Every channel uses the same 523 words in the same order. |
| **HUMAN** | the human ground truth: 523×523 correlation matrix of human self-ratings. |
| **REPRESENT** | read-side channel: residual-stream activation geometry for each adjective ("My personality is X", mid-layer, last token). What the model *encodes*. |
| **JUDGE** | the model's implicit personality theory: "a person who is very X — how likely also Y?" rated 1–7 via logprobs, all 523×523 pairs. What the model *believes about people*. |
| **ENACT** | write-side channel: tell the model to *be* X, collect rollouts, extract the mean activation direction of the enactment (vs the all-persona mean). What the model *does*. |
| **SELF** | direct self-report: "I am X — agree/disagree?", six prompt framings. What the model *claims*. |
| **EV / argmax** | we always read the full logprob distribution over answer tokens; EV = probability-weighted mean. The argmax (sampled answer) is the mask; the distribution is the signal. |
| **effdim (PR)** | participation ratio of an eigenspectrum — a soft count of "how many dimensions really vary." |
| **PC1-removed** | subtracting the top eigencomponent of a similarity matrix before comparing to HUMAN. Human PC1 is a huge desirability/adjustment axis; any channel can match it for free, so we always report the match *with and without* it. |
| **massive dims / winsorize** | a handful of residual-stream dimensions with enormous always-on activations (format/register machinery, not content). Default denoising: cap their std at the largest normal dim's std. |
| **assistant axis** | the direction from the all-persona mean to the default-assistant activation — "how assistant-like is this state." |
| **persona vector / ê** | the ENACT direction for one adjective; adding it to the residual stream steers conduct. |
| **W (read→write map)** | ridge regression from REPRESENT coordinates to ENACT directions. ENACT is, to R²≈0.6–0.7, a *linear image* of REPRESENT. |
| **enactability** | judge-scored shift of a persona rollout vs baseline: did the model actually *become* X when asked? |
| **leak** | fraction of rollouts that *say* the trait word instead of showing the trait. |
| **desirability boulder** | the giant first factor (social desirability) that dominates self-report-like channels; ~42% of SELF's variance. |
| **one-respondent problem** | assistants barely differ from each other; a population with ~no individual differences breaks individual-differences statistics (α, cross-model ρ). |
| **tom_likely** | the JUDGE prompt framing (third-person, "how likely"), chosen because bipolar framings confuse unsure models. |
| **split-half** | our workhorse reliability check: does a profile computed on half the items agree with the other half? Reported for every instrument we build or audit. |
| **Tucker φ** | factor-congruence coefficient; ≥0.85 = "fair similarity" of factor axes. |

---

## TL;DR — if you read one paragraph

We measure LLM personality through five channels on one 523-adjective set:
what the model **represents**, **judges**, **enacts**, **claims** (SELF), and
what **humans** do on the same words. The project's founding slogan —
*distribution > argmax* — has now been vindicated at every level, including
inside other people's instruments. The big structural results: (1) the write
side (ENACT) is a **low-rank linear image** of the read side, ~5–13 effective
dimensions against REPRESENT's 50–70, and mapped vectors steer *better* than
extracted ones; (2) **JUDGE is the best human-matched channel** (r≈0.8,
nearly none of it desirability-freebie) — the model's knowledge of human
personality structure is excellent even though (3) its **self-knowledge is
nil**: SELF is a character sheet with no self behind it, and the model cannot
read its own conduct even when that conduct is sitting in its context window
— *it knows who it's told to be, not who it's been*; (4) audits of two
external instruments (ValuePortrait / arXiv:2509.10078, and Persona
Cartography's TIDE) show the field's behavioral measures largely read the
**corpus or the standing instruction**, not the model-as-agent — and our
repair recipes (within-context contrast, graded readout, reliability-first)
transfer; (5) character interventions unify: prompting, activation steering,
and DPO+SFT LoRAs all write through a shared low-dimensional conduct
subspace, and a character LoRA is, at activation level, a **context-stable,
depth-rotating steering schedule** — approximately a bias, which is why
bias-only tuning works.

---

# Part I — Foundations (W1–W16, compressed)

## 1. Distribution > argmax

Two models can both "answer 4" while one is razor-peaked and one near-uniform;
the distribution is the personality-relevant quantity. Gemma/Qwen are peaked
(digit-entropy ≈0.15), Llama near-uniform (≈1.4). This kept paying rent all
the way to W18, where regenerating the raw JUDGE distributions showed phi4's
matrix is **56% bimodal** — its EVs were averages over two disagreeing answer
modes (commit-vs-hedge, not synonym-vs-antonym) — and the flattest model
(FalconMamba) hid the *best* human-match in its graded mass. Certainty turns
out to be a **question taxonomy, not an entropy policy**: every model is sharp
on arithmetic and hedges on die rolls; families differ in whether they file
"is an organized person friendly?" with arithmetic (Gemma) or with die rolls
(Llama). And peaked ≠ brittle: Gemma's 9% probability tail is a faithful
miniature of its full belief (tail-only human-match 0.587 vs 0.621).

## 2. The assistant shape

Every tuned model lands low-N, high-A/C; in Big Five space the HHH persona is
~rank-1 (E–C r≈0.93). This "one evaluative axis wearing five names" recurs
everywhere: as the desirability boulder in SELF, as the prosocial tilt in
generation preferences, as the reason Big Five differential structure barely
exists in any assistant channel.

## 3. The read/write dissociation (the capstone of part I)

Models **represent** evaluative antonyms merged (Wonderful ≈ Awful, cosine
positive) but **judge** them opposite — a sign-flip, not attenuation, in every
model 3B→32B. The merge is a **pretrained constant** (flat across
base→SFT→DPO→instruct in two families) and in fact regresses all the way to
**2014 static GloVe vectors**: it is a property of the distributional
hypothesis itself, not of transformers. Only LLM *judgment* and human
self-report split the antonyms — and they land on top of each other. The
**symbolic override of the associative merge is the new thing** tuning shapes,
and the read/write framing of the whole project comes from here. (Full
regress table in `report_week16.md`; the old synthesis §6 has the strata.)

## 4. Methods lessons that keep earning

- **PC1 discipline**: raw PCA PC1 of hidden states is a norm artifact; with
  paired data the contrast removes it for free, with unpaired data removal
  must be conditional (only when it's an identifiable spike).
- **Human-match must be reported PC1-removed** — human PC1 (desirability) is
  matchable for free by any evaluative channel.
- **The scoop lesson (W13)**: sentence encoders recover human facet structure
  from item text alone (Wulff/Milano), so geometry-matches-humans is
  inherited, not special. Our contribution since: *where the model deviates*
  from the embedding baseline, and the write side, which embeddings don't have.

# Part II — The channel pentad and the read→write map (W17–W18)

## 5. ENACT: personas as directions, and the map W

Telling a model to *be* each of 523 adjectives and extracting the mean
activation direction of the rollouts (vs the all-persona mean) gives the
**ENACT** channel — 10-model cohort, one direction per adjective per model.
Findings:

- **ENACT is a linear image of REPRESENT**: a ridge map W from read-space
  coordinates predicts held-out ENACT directions at R²≈0.72 (llama) / 0.57
  (qwen), compressing effdim ~45→10. The orthogonal remainder is **rotation,
  not intent** — it carries no extra steering power.
- **Mapped vectors steer at least as well as extracted ones** (3/3 families).
  W denoises: the recorded vectors' noise is largely outside the map's span.
  This yields a **zero-rollout persona-vector recipe**: read the adjective,
  map it, steer.
- **Effdim ladder**: REPRESENT 50–70 ≫ HUMAN 27 > ENACT 5–13 ≈ SELF 4.5.
  Output channels share a narrow bottleneck. The bandwidth is family-
  constitutional: prompt batteries can unlock situationally-gated dims in
  qwen (2.9→4.0) but cannot create dims (llama flat ~9 on any battery).
- **Steering doses don't port across families** (Gemma's plateau inflates
  residual norms 25×); anchor doses to natural direction-norm ratios.

## 6. SELF: a character sheet with no self behind it

Six framings of "I am X" (direct / HHH-assistant / as-a-person / accuracy /
observer / outputs), full 523, whole cohort. SELF is **assistant-shaped in
every framing**, effdim ≈4.5 (two boulders: 42% desirability + 20%
claim-tier). The **diagonal test**: a model's SELF profile predicts its *own*
ENACT profile no better than it predicts other models' (advantage +0.02) —
self-report carries no self-knowledge, it reads out the trained self-concept.
PC1-removed human-match ranking across channels — the sharpest
symbolic-over-associative statement we have:

**JUDGE (0.80) ≫ ENACT (0.62) > REPRESENT (0.44) ≫ SELF (0.21)**

JUDGE earns almost none of its human-match from desirability; SELF earns
almost all of it. The model's knowledge of *people* is excellent; its
knowledge of *itself* is a press release.

## 7. Facets and disbelieved clusters

Using 35 human-derived adjective clusters as facet blocks: model JUDGE
binding is cohort-consensual (r≈0.83) and tracks cluster *valence* (+0.72)
where human coherence is valence-neutral. Models refuse the human
negative-halo bundles (awkward = clumsy+plain+boring+unattractive;
disorganized contains left-handed) — **valence is an axis for models, a
binder for humans**. "Virtues all alike, every vice specific" — and this
recurs later as a factor-structure phenomenon (§12).

## 8. Human structure is decodable from REPRESENT

The four-grid claim "REPRESENT matches humans mostly via PC1" was about
similarity geometry in place. A cross-validated linear map from activation
PCs to the human eigenspace nearly **doubles** the beyond-PC1 human match
(0.29→0.50–0.55; raw 0.78–0.81, three families). The human structure is
*embedded*, rotated and mis-scaled. The per-PC decodability curve is
diagnostic: human PC1 decodes at R²≈0.83, PC4 (levity) ≈0.6 … and **human
PC2 — modesty vs self-enhancement, a self-presentation stance — is flatly
absent (negative R²) in every model**. The one big human dimension models
lack is the one that lives in *respondents* rather than in language. That
residue is a tool: model channels as a pure-semantics control for
separating substance from style in human self-report.

# Part III — The audits (W18 §7, W19): what the field's instruments measure

## 9. ValuePortrait / arXiv:2509.10078 ("questionnaires mischaracterize")

They compare questionnaire profiles to generation-probability profiles over
ValuePortrait (104 real scenarios × 5 candidate responses with signed,
human-derived construct labels) and conclude generation behavior has no
construct structure. Full audit, every exhibit replicated or repaired:

- **Their scoring has ~zero reliability with itself** (split-half ≈0; the
  statistic they never report) and construct scores correlate up to
  **r=−0.76 with response length**. Their η² null is the statistic's ceiling
  (labels can explain at most 5–9% of raw logprob variance; scenario+length
  eat the rest), their Appendix E validation is the length confound
  congratulating itself (candidates are 8.7× shorter than real samples;
  length-corrected they rank 11/15 — deeply off-policy), and by their own
  companion paper's α measure the tag-scales score **−2.1** in the logprob
  medium vs +0.9 with human respondents. Reliability is a property of items
  × population × *readout medium*.
- **Repaired** (per-token logprob, z-scored within scenario, signed
  continuous labels): generation preference is reliable (split-half
  0.54–0.70) and the profile is the assistant value shape (+Benevolence,
  +Openness, −Power, −Achievement), cross-model r=0.90. Every scoring
  ingredient earns its place in an ablation.
- **But what it measures isn't the assistant.** The profile is invariant to
  the chat frame *and* to the entire post-training stack: base checkpoints
  carry the full drift (cross-half disattenuated r_true CIs all contain
  1.0), the OLMo ladder is flat, tuning's contribution is bounded ≤ a few %
  of profile variance. **The repaired instrument reads the corpus, not the
  character.** Their thesis survives in its strongest form (questionnaire
  shape predicts zero of 520 choice preferences even when the criterion is
  reliable), but their proposed replacement measures pretraining statistics.
- **One-respondent verdict**: item-level model-model preference r=0.54 with
  family blocks; the model-unique residual is unreliable for four of five
  models — except **phi4** (+0.56), the cohort's one idiosyncratic
  character (synthetic-textbook corpus; its divergence concentrates on
  social/interpersonal scenarios it reads fluently but "grew up" without —
  a socialization gap, not a comprehension gap).
- Both papers are one lab; the fatal tagging convention was inherited from
  their own earlier (fine, Likert-native) pipeline. The note we owe them is
  three layers: broken → repaired → measuring-the-wrong-object, wrapped in
  "your continuous labels are the undervalued asset."

## 10. The reliable-behavioral-measurement recipe (what the audit teaches)

Contrast within context (context is 57–77% of everything, in *every*
instrument we've measured); read the graded distribution, never the pick;
normalize to perplexity units; keep signed continuous labels; report
split-half reliability as a first-class number; calibrate against the
instrument's ceiling; and when found items cap per-item validity at r≈0.1,
*author* desirability-matched contrasts instead (per-item validity 0.3–0.5).
The last point is the W1 trait-conflict instrument, which now has three
independent votes (us; Okada's GFC; Persona Cartography's v7 redesign after
they independently discovered Likert desirability contamination via
Likert-vs-FC sign flips).

## 11. Persona Cartography / TIDE (W19)

Their unsupervised result: four "model-native" factors (Tone, Initiative,
Didacticism, Epistemic Caution) from a 72-item forced-choice questionnaire
over 2,500 rollouts. Engagement findings:

- **Design anatomy**: the "personas" are 25 *user* archetypes × 100
  scenarios, each scenario carrying its own role system prompt. Scenario
  explains 56–78% of factor scores (their own honest appendix), archetype
  ≤6% — except the **hostility switch**: a hostile user shifts PC1 by +2.8
  SD toward guarded-formal (24 other archetypes: within ±0.8). Accommodation
  is a discrete guard trigger, not graded style-matching.
- **Pre-oblimin**: no desirability boulder (their FC design kills it by
  documented intent — independent replication of our SELF finding), the
  unrotated first axis is *persona amplitude* (expressive ↔ compliant), and
  **k=4 is retention, not discovery**: Horn's analysis supports ~11–13
  dimensions. Their retained 4 ≈ our conservative ~5; their probed 11–13 ≈
  our llama effdim ~9. Two labs, two instrument classes, same two-level
  bandwidth answer.
- **Their instrument on our respondents**: administering their 72 items to
  our 523 labeled persona rollouts, axis congruence *fails* by range
  restriction (only the humor/exuberance axis is shared, φ=0.72): personas
  reach **valence and menace wings** that role-modulation never samples,
  where conduct items collapse into big correlated bundles. TIDE is the
  fine texture of the assistant's home wing. The default assistant sits
  ~0.8σ from the *prosocial extreme* of the 523-persona range.
- **The instruction is the signal** (the week's most important control):
  with the persona system prompt ablated, questionnaire answers collapse to
  the default assistant (r=0.92) — the instrument reads the standing order,
  not the conduct in the window. r(full, instruction-only)=0.76 vs r(full,
  rollout-only)=0.13. *The model knows who it's told to be, not who it's
  been.* (Caveat, rgb's: one turn of evidence cannot invalidate
  self-perception — it's the high-external-justification cell of the Bem
  design, where humans don't update either. The dose × attribution 2×2 is
  the queued experiment.)

# Part IV — Interventions unify (W19 §4)

## 12. Prompting, steering, and fine-tuning write through one door

Applying their ten OCEAN LoRAs in weight space and measuring the mid-layer
activation displacement on fixed rollouts: **sign test 10/10** — every
amplifier displaces *along* our prompt-extracted trait directions, every
suppressor against. In-span fraction is partial (0.2–0.4 at k=45, 20–100×
chance — most of fine-tuning's displacement is outside the prompting span,
adapter-specific). Amp/sup are oblique (−0.2…−0.5), never antipodal; 8/10
adapters displace *away from the assistant axis* regardless of trait or
pole (the axis reads assistant-ness, not valence).

## 13. A character LoRA is a steering schedule

The LoRA's displacement is **context-stable at every one of 33 layers**
(per-text consistency ~0.92) but the *direction rotates with depth* — a
layer-indexed family of constant vectors {δ_ℓ}, of which single-layer
steering reproduces one frame. Trait content enters throughout the early
half (blocks 0–7 and 8–15 each sign-correct alone); the late half writes
large output-adjacent content a mid-layer vector cannot reach; blocks
compose sublinearly. Corollary both directions with **BitFit**: character
is (mostly) bias-shaped, which is why constants-only tuning works where it
works — and the LoRA's persistence advantage over prompting is plausibly
re-application across layers and tokens, not deeper encoding. Queued
decisive test: **schedule playback** — inject the recorded {δ_ℓ} with
weights untouched; the gap to the full LoRA is the behavioral value of the
non-bias part of character.

---

# Claims ledger — current status of every major claim

Status: **LIVE** (standing), **REV** (revised — read the pointer, not the
original), **DEAD** (retracted), **OPEN** (registered, undecided).

| # | claim | status | where |
|---|---|---|---|
| 1 | Distribution > argmax (EV+entropy as primary readout) | LIVE | W1; quantified within JUDGE (EV≥argmax all models) W18 §4 |
| 2 | Assistant shape: low-N high-A/C, HHH ~rank-1 | LIVE | W1–; recurs in SELF, VP prosocial tilt |
| 3 | Evaluative-antonym merge: represented merged, judged split, all models | LIVE | W15 |
| 4 | Merge is pretrained-constant and regresses to 2014 GloVe | LIVE | W15 §3, W16 |
| 5 | PCA PC1 of raw hidden states is a norm artifact | LIVE | W2/W6 |
| 6 | ENACT = linear image of REPRESENT (W; R²≈0.6–0.7; remainder rotation-not-intent) | LIVE | W17 |
| 7 | Mapped ê=Wr steers ≥ recorded persona vectors | LIVE | W17 (3 families) |
| 8 | Effdim ladder REPRESENT 50–70 > HUMAN 27 > ENACT 5–13 ≈ SELF 4.5 | LIVE | W17–18 |
| 9 | Persona bandwidth constitutional per family; battery-gated within qwen | LIVE | W18 §3 |
| 10 | SELF = character sheet; diagonal test ≈0 self-signal | LIVE | W17 §15 |
| 11 | Channel ranking (PC1-removed human match): JUDGE≫ENACT>REPRESENT≫SELF | LIVE | W18 §2.5 |
| 12 | Valence is an axis for models, a binder for humans (disbelieved clusters) | LIVE | W18 §1 |
| 13 | Human structure embedded in REPRESENT; human PC2 (self-presentation) absent | LIVE | W18 §6 |
| 14 | phi4 JUDGE 56% bimodal; modes are commit-vs-hedge | LIVE | W18 §4 |
| 15 | Certainty is question taxonomy, not entropy policy | LIVE | W18 §4 |
| 16 | VP/2509.10078: published scoring unreliable; length-dominated; η² at ceiling | LIVE | W18 §7 |
| 17 | Repaired generation preference: reliable, assistant-shaped, cross-model 0.90 | LIVE | W18 §7 |
| 18 | Repaired instrument reads corpus, not character (frame- and tuning-invariant; r_true≈1 vs base) | LIVE | W18 §7 |
| 19 | One-respondent: model-unique behavior unreliable except phi4 | LIVE | W18 §7 |
| 20 | phi4 divergence = socialization gap (social scenarios, fluent reading) | LIVE | W18 §7 |
| 21 | TIDE k=4 is retention; underlying space ~11–13 dims; no desirability boulder | LIVE | W19 §1 |
| 22 | Hostility switch: discrete guard-register trigger | LIVE | W19 §2 |
| 23 | TIDE-vs-ENACT axes: shared humor axis only; range restriction (valence+menace wings) | LIVE | W19 §3 |
| 24 | Persona questionnaire reads the standing instruction, not conduct | LIVE | W19 §3.5 |
| 25 | "Persona η²=0.99 = near-deterministic conduct readout" | **REV** | → instruction-echo fidelity; W19 §3.5 |
| 26 | "Machine-Bem fails" | **REV** | → reopened: N=1 dose + high-external-justification cell; 2×2 queued; W19 §3.5 amendment |
| 27 | OLMo ladder Power/Openness rotation under tuning | **REV** | → not significant under cross-half disattenuation; W18 §7 |
| 28 | LoRA sign test 10/10 on prompt-extracted trait directions | LIVE | W19 §4 |
| 29 | LoRA = context-stable depth-rotating steering schedule; character ≈ bias | LIVE | W19 §4.5 |
| 30 | "N-amp most anti-assistant" | **DEAD** | wrong; O-amp is, and 8/10 leave the axis regardless; W19 §4 |
| 31 | Steering doses portable across families as frac-of-residual-norm | **DEAD** | Gemma plateau; anchor to dir-norm ratios; W17 |
| 32 | Gemma massive dims = format/verbosity readout ("list-mode") | LIVE | W17 |
| 33 | J-lens: ENACT reads out the persona's speech-world; failed personas → assistant/wacky basins | LIVE | W18 §5 |
| 34 | Trait-conflict / desirability-matched FC instrument is the right next instrument | OPEN | 3 independent votes; W1, W18 §7, W19 §1 |
| 35 | Schedule playback ≈ LoRA (training = recordable activation program) | OPEN | queued, design review first; W19 §4.5 |
| 36 | Out-of-span LoRA displacement: conduct or debris? | OPEN | W19 §4 |
| 37 | Bem 2×2 (dose × attributional framing) | OPEN | W19 §3.5 |

---

## What we'd like from you

1. The **VP note** — now less "quick fix" and more construct-validity
   challenge (three layers; §9 above is the abstract). Your call on fronting
   it; the handoff pack (exhibits, numbers, scripts) exists on request.
2. The **paper outline** is the next artifact after this one — the pentad +
   audits + instrument recipe is the psych-facing paper; the map/steering/
   schedule material is the MI note. Psychometrics sections are yours if you
   want them.
3. Sanity-checks welcome anywhere the ledger says LIVE and you smell REV.

## Key artifacts (in reading order)

| artifact | shows |
|---|---|
| `results/persona_vectors/figs/facet_cohort_summary.png` | five channels vs HUMAN, raw + PC1-removed (the ranking) |
| `results/adjectives/regress/regress.png` | the merge regresses to 2014 vectors |
| `results/vp_rescore/figs/vp_eta2_heatmap.png` | questionnaire blocks vs generation confetti, both scorings |
| `results/vp_rescore/figs/vp_fig1_transparency.png` | item transparency, their Figure 1 on our materials |
| `report_week18.md` §7, `report_week19.md` | the audits, long form |
