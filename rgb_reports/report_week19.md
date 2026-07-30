# Week 19 — TIDE and the Persona Cartography bridge

Persona Cartography (arXiv:2607.07916; repo + LoRAs public, response
matrices on persona-cartography/monorepo) trains OCEAN amplifier/suppressor
LoRAs, and — the part that matters to us — extracts four "model-native"
factors (TIDE: Tone/Initiative/Didacticism/Epistemic-Caution; internal names
Warmth/Initiative/Pedagogy/Hedging) by principal-axis factoring + oblimin on
a 72-item forced-choice questionnaire appended to 2,500 persona-rollouts of
Llama-3.1-8B (choice-mass readout). Design anatomy (important — the paper's
"persona-rollouts" naming misleads): the ASSISTANT is always the default
assistant; the "personas" are 25 USER archetypes (whimsical, hostile,
anxious, ...) x 100 scenarios, each scenario carrying its own system prompt
(a mini-role: "AI companion in a film app... playful"). Role instruction
and topic are confounded within scenario. So TIDE measures the default
assistant's conduct variation across deployment roles and user styles —
not enacted characters (Llama-8B; Qwen-2.5-7B re-administration on the
same rollouts).

## §1 — Pre-oblimin structure: no boulder, long tail, familiar first axis

rgb's W14-honed worry: oblique rotation distributes any unexamined general
factor across the named factors. Checked on their shipped response matrix
(`tide_preoblimin.py`):

- **No desirability boulder.** Unrotated PC1 = 14.0%, PC2 = 11.1% (ratio
  1.26 vs human PC1's 2.2x; SELF's 42%); 61% same-sign loadings — not a
  general evaluative factor. Mechanism worth stealing: the FC-pair format
  (each item = two DEFENSIBLE conduct styles) desirability-matches by
  construction and kills the evaluative axis before rotation — live
  vindication of the forced-choice design argument behind our W1
  trait-conflict instrument (and Okada's GFC).
- **Unrotated PC1 = persona amplitude.** Expressive pole: epistemic_style,
  metacognitive_transparency, warmth_vs_directness, depth_vs_brevity,
  humor_playfulness, proactivity. Compliant pole: instruction_compliance,
  communication_format, formality. "How much character vs default assistant"
  — the same leading axis ENACT geometry has; not which persona, but how
  much persona.
- **k=4 is a haircut, not a discovery.** Kaiser count 15, PR effdim 19.8,
  Horn crossing ~13 in our 100-perm replication (their appendix: 11; same
  neighborhood); first-4 cumulative 38.6% (their PAF 40.7% — consistent).
  They retain k=4 on scree-elbow + per-factor alpha + cross-model congruence
  — defensible parsimony, but TIDE is a 4-factor summary of an ~11-13-dim
  space. Two-level agreement with our ENACT bandwidth: retained 4 ≈ our
  conservative ~5; probed 11-13 ≈ our llama diverse-battery effdim ~9.
  "Persona bandwidth ~10 under rich probing, ~4-5 under conservative
  retention" now replicates across two labs and two instrument classes.

## §2 — What moves the TIDE scores: roles, not interviewers (plus one switch)

ANOVA of rollout PC scores on the design factors (row alignment
self-validated by the effect size): scenario explains 52-70% of every top
PC; archetype 0-4%. Their appendix reports the same at factor level
(scenario 56-78%, archetype ≤6% — their Table 4-2-variance-decomp) and a
scenario-residualised refit in which 3 of 4 Llama factors survive with
|φ|≥0.89 (Didacticism 0.69; qwen's 4th dies). Credit where due: they
found, measured, and reported the scenario dominance. The honest joint
reading: factor SCORES are deployment-role registers (system prompt +
topic, confounded by design); factor AXES mostly survive residualisation —
one conduct basis organizing both between-role register shifts and
within-scenario variation, consonant with our one-conduct-space picture.

What their averages hide, ours to keep: **the hostility switch.** 24 of 25
archetypes sit within ±0.8 SD on PC1; hostile alone is +2.8 SD toward the
formal/compliant pole (desperate +0.6 next; whimsical −0.8 opposite).
Accommodation to user style is not graded style-matching — it is
approximately a discrete guard-register trigger. "Archetype is
comparatively immaterial" is true on average and false about hostility.

**And the v7 description is an independent replication of our SELF
finding.** Their changelog-grade description documents that v5 Likert
administration produced sign-flips vs forced choice on the F0
'engaged-agency' construct — "direct evidence that v5's Likert variance is
partly acquiescence/social-desirability-style endorsement of virtuous
self-descriptions" — and v7's design rule is "neither option reads as the
obviously RLHF-correct choice." A second lab, via a different symptom
(Likert-vs-FC sign flips rather than our desirability-boulder
decomposition), converged on both our diagnosis (model Likert self-report
is desirability endorsement; cf. SELF 42% boulder, W18 §2.5) and our
prescription (desirability-matched forced choice — the W1 trait-conflict
design, Okada's GFC). Three votes, one instrument class.

Cross-instrument law, three measurements now: context/scenario is the
largest single factor in every behavioral instrument — VP (57-77% of
logprob variance), TIDE (56-78% of factor scores), our de-collapse battery
gating (qwen 2.9→4.0 by question register). Instruments differ only in
whether they contrast it out (our within-scenario scoring), residualise it
(their refit), or absorb it into "traits" (both papers' headline numbers).

## §3 — Their instrument, our respondents: two doors, different wings

Administered their 72-item v7 FC questionnaire to our Llama8
persona-enactment rollouts (523 adjectives × 2 fixed questions + 10
default-assistant baselines; choice-mass readout, KV-cached;
`tide_on_enact.py`, results/tide_enact/). Registered predictions: axes
congruent (φ≥0.7 for 3/4) — WRONG; persona ≥50% of top-PC variance —
RIGHT; question factor <15% — RIGHT; face validity — half right.

- **Axis congruence fails informatively.** Hungarian-matched Tucker |φ| vs
  their k=4: mean 0.49 (below their own llama↔qwen 0.66); item-corr
  congruence r=0.42. The ONE shared axis is humor/exuberance (φ=0.72) —
  the only dimension that varies in both regimes.
- **Range restriction explains the miss.** Our factors: F0 VALENCE
  (admirable↔awful; unrotated PC1=27%, fusing depth+compliance+pedagogy),
  F1 exuberance/disinhibition, F2 diffidence/hedging, F3 MENACE
  (evil/abusive↔ordinary; the safety items). Valence and menace are
  regions role-modulation never samples — no deployment role makes the
  assistant awful — so in their data the conduct items decorrelate into
  fine registers (TIDE) while under personas they collapse into big
  valence/threat bundles: when the character is bad, everything degrades
  together. "Virtues alike, vices specific" as a factor-structure
  phenomenon. TIDE = micro-structure of the assistant's normal operating
  wing; personas unlock the rest of the mansion.
- **Their instrument is BETTER at reading personas than roles**: persona
  η² = 0.99/0.96/0.86 on our top PCs (question ≤0.02, 0.15 on PC4) vs
  their scenario 56-78% + trajectory-noise third. Near-deterministic
  persona readout.
- **Bandwidth is instrument-invariant**: PR effdim of our matrix = 9.0 —
  their questionnaire on our rollouts recovers our llama ENACT
  activation-geometry effdim (~8.8-9). Third independent measurement of
  the ~9-dim llama persona bandwidth; the basis is regime-dependent, the
  dimensionality is not.
- **The default assistant lives at the nice pole**: −3.5 on our PC1
  (adjective-respondent SD 4.4), ~0.8σ from the prosocial extreme of the
  523-persona range — why within-regime instruments see texture where
  full-range instruments see one big axis.

## §3.5 — The instruction IS the signal (rgb's ablation; honest downgrade of §3)

Does the questionnaire read the persona instruction or the enacted rollout?
Ablation on 105 adjectives (every 5th, coworker question;
`tide_sys_ablation.py`): B = rollout-only (sys dropped), C =
instruction-only (rollout dropped), vs A = full (main run).

- r(A,C)=0.76, r(A,B)=0.13, r(B,C)=0.03; regression A~B+C: β_C=0.75,
  β_B=0.11. **The instrument reads the standing order.**
- B collapses to the default assistant: profile r=+0.92 with the default
  profile (deviation 0.17 vs A's 0.54; deviations only weakly
  persona-shaped, mean per-item alignment with A +0.12).
- **Self-perception fails in-context**: with its own vivid in-character turn
  fully in view, the model's self-description barely updates. Stronger
  negative than the W17 diagonal test (which could be excused by no access
  to behavioral history — here the history is in the window). Multi-turn
  drift prediction revised toward the null.
- CORRECTIONS to §3: the 1056-respondent matrix = INSTRUCTED
  SELF-DESCRIPTION structure (JUDGE-of-assigned-self), not conduct
  readout; persona η²=0.99 = instruction-echo fidelity; effdim 9.0 = the
  bandwidth of llama's instructed self-concept (its match to ENACT conduct
  effdim ~9 is now a cross-object coincidence worth its own explanation,
  not a triviality). And their scenario-dominance (56-78%) is plausibly
  substantially role-sys-prompt echo too — their contexts carry system
  prompts during the questionnaire; same ablation would settle it
  (inference, not measured).
- Taxonomy: v7-with-instruction measures ASSIGNED identity;
  v7-with-conduct-only measures ~nothing beyond the default self-concept.
  The model knows who it's told to be, not who it's been — the SELF-channel
  verdict, reproduced inside their instrument.

**§3.5 amendment (rgb: not enough turns to invalidate self-perception).**
Two overreaches in the "self-perception fails" reading. (1) DOSE: one
in-character turn is a single observation against a questionnaire asking
about "your usual approach" — the 0.17 deviation with ~12% persona
alignment is a plausible FIRST POINT on a cumulative curve, not a null;
small-but-nonzero single-dose transfer is what self-perception
accumulation would look like at N=1. (2) ATTRIBUTION: self-perception
theory (Bem's) predicts updating only
when external causes are discounted; the model has strong priors that
odd assistant turns are instructed, so condition B is the
HIGH-EXTERNAL-JUSTIFICATION cell of induced-compliance — where humans
also show no update. B may be the mechanism working (discounting),
not absent. Upgraded design for the multi-turn experiment: dose (N
in-character turns) × attributional framing (visibly-instructed vs
free-choice presentation of the same conduct), self-report as outcome.
The theory predicts accumulation only under free-choice framing; flat everywhere
= no mechanism; accumulating everywhere = mechanism without
attribution-sensitivity. Verdict on self-perception: OPEN, with one
measured baseline point.

## §4 — ΔW·h: fine-tuning writes through the same door, plus a wider one

Applied their 10 souped OCEAN adapters (ocean_const_paired_dpo/-persona)
in weight space, teacher-forced the 60 default-assistant rollouts, and
measured the layer-16 displacement Δh vs our ENACT geometry
(`tide_dwh_bridge.py`; predictions registered the night before).

- **Sign test 10/10**: every amplifier projects + and every suppressor −
  on our prompt-extracted trait-matched ENACT directions (O +0.38/−0.24,
  E +0.30/−0.37, A +0.35/−0.29; C, N weaker but correct). DPO+self-SFT
  training moves activations along the directions "You are someone who
  is X" produces. Weight door and prompt door, same hallway.
- **In-span partial**: 0.07–0.27 at k=10, 0.18–0.38 at k=45 (nulls
  0.002/0.011) — 20-100× chance, but most of fine-tuning's displacement
  is OUTSIDE the ENACT span (magnitude prediction 0.35-0.6 was too
  high). Comparable to llama's own ENACT-in-REPRESENT 36%. The
  out-of-span mass is NOT one shared tuning direction (common component
  27% of variance, itself mostly out-of-span; removing it raises every
  adapter's in-span) — it is adapter-specific subspace prompting doesn't
  reach.
- **Context-stability 0.89-0.94** (unregistered find): each adapter's
  layer-16 displacement is essentially a constant vector across 60
  different conversations — LoRA character tuning acts like a steering
  vector at mid-depth. The training pipeline's mid-layer effect is
  approximately "add ê," where ê is ~1/5-1/3 our span and the rest new.
- **Everything leaves the assistant**: 8/10 adapters displace away from
  the assistant axis regardless of trait or pole (O-amp most, −0.25; the
  N-amp-most-anti-assistant prediction was wrong) — the axis reads
  assistant-ness, not valence; any character tuning is a departure.
- **Antipodality −0.23…−0.52** as predicted: amp/sup oblique, never
  opposite — the weight-space asymmetry survives into activation space
  (the W16 valence-vs-variance account's third appearance).

Unification verdict: prompting, activation steering, and fine-tuning all
produce context-stable directional writes with matched trait semantics;
fine-tuning's write is broader-band (majority out-of-span, idiosyncratic
per adapter). Open: does the out-of-span part carry conduct (steer with
it) or training debris? — directly testable with our steering harness.

## §4.5 — The LoRA is a steering schedule (rgb: "it applies to every layer")

Layer-resolved Δh for all 10 adapters + block-wise ΔW application for the
agreeableness pair (`tide_dwh_layers.py`).

1. **Steering SCHEDULE, not vector**: per-text consistency ~0.92 at every
   one of 33 layers — at each depth the displacement is a context-stable
   constant vector — but the direction ROTATES with depth (cos to Δh_16:
   0.29 at L8, 0.72 at L20, 0.50 at L28, 0.18 at L32). The adapter's
   activation-level content is a layer-indexed family {δ_l}; single-layer
   steering reproduces one frame of it. (Registered "direction set early":
   wrong — it never stops evolving.) |Δh| grows superlinearly (0.26 at
   L4 → 1.6 at L16 → 7.3 at L28 → 40 at L32).
2. **Trait content is distributed**: blocks 0-7 and 8-15 EACH alone
   produce sign-correct trait projections at L16 (8-15 strongest,
   +0.34/−0.31); late blocks can't touch L16 (0.01 — causality sanity
   check) but each alone moves L32 by |Δh| 12-18 vs full-adapter 40 —
   the late half writes output-adjacent content a mid-layer vector cannot
   reproduce by construction.
3. **Sublinear composition**: block L32 effects sum ~70 vs full 40 — the
   per-block ΔW's interact.

Amended §4 verdict: fine-tuning ≈ a recorded, context-stable,
depth-rotating activation program; single-layer steering is a one-frame
excerpt; the LoRA's persistence advantage (their drift experiments)
plausibly = re-application across layers and tokens, not deeper encoding.
QUEUED (design review first): schedule playback — inject the LoRA's own
recorded {δ_l} at every layer with weights untouched; if behavior and
drift-resistance reproduce, "training writes character" reduces to
"training records an activation program inference could inject."

**BitFit tie-in (rgb).** The constants-only PEFT family — BitFit
(bias-only tuning), prefix tuning (constant KV), (IA)³ (constant
scalings) — is the existing literature for "interventions that can only
add constants." §4.5 cuts both ways: it predicts BitFit-class methods
suffice for character (the rank-64 LoRA's effective intervention is
per-layer constants at conversation grain), and it explains why
bias-only tuning works where it works (the target behavior IS a constant
activation program) and fails where it fails (input-conditional
computation). Wrinkles: Llama has no bias terms (bias=False throughout) —
a character LoRA is an over-parameterized graft of effective biases onto
a bias-free architecture; and our 0.92 consistency is response-mean
grain — per-token input-dependence (where conditional gating à la the
hostility switch would live) is unexamined. The queued playback becomes a
named three-way: full LoRA vs schedule playback (=BitFit-by-hooks) vs
single-layer steering; the playback-vs-LoRA gap = the behavioral value of
the non-bias part of character.

## §5 — The read→write map is rank-10, and rotation splits in two (rgb's
identity-off-subspace idea)

Reduced-rank version of W (fit ridge, truncate fitted values to rank r;
operator form T = I + αC_r, identity off the active subspace — the
J-space bipolar-patch discipline applied to our map). Three families,
held-out cos(mapped, recorded):

- **Rank sweep**: r=10 recovers 93–98% of full-45 quality (llama
  .760/.818, qwen .795/.813, gemma .778/.825); qwen saturates by r≈3-8.
  The conduct transformation is essentially rank-10; the
  identity-plus-low-rank operator is near-free.
- **Mode-pair angles** (rank-10; angle between each mode's read-input
  direction and write-output direction): llama median 76°, gemma 73°,
  qwen 64° — family ordering reproduces the W17 rotation ranking — but
  NO family has any small-angle mode (qwen's min 58°). "Rotation"
  conflated two quantities: SPAN CONTAINMENT (qwen high, llama low — the
  old family parameter) vs MODE ALIGNMENT (universally poor). Enactment
  is never amplification of the read direction, even inside the read
  span; amplify-and-rebase is mostly rebase, everywhere. qwen rebases
  within the span; llama partly out of it.

GPU follow-ups queued: (a) truncation steering — if mapped>recorded
strengthens at rank 10, the denoising account gets a dial; (b) the
ENACTIFIER: hook h → h + αC_r·h on prompts that merely DESCRIBE a
persona — state-dependent, adjective-free steering that converts
description into performance (measurement-then-set; dose-free in the
swap sense).
