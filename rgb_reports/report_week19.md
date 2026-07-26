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
