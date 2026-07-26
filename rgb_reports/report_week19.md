# Week 19 — TIDE and the Persona Cartography bridge

Persona Cartography (arXiv:2607.07916; repo + LoRAs public, response
matrices on persona-cartography/monorepo) trains OCEAN amplifier/suppressor
LoRAs, and — the part that matters to us — extracts four "model-native"
factors (TIDE: Tone/Initiative/Didacticism/Epistemic-Caution; internal names
Warmth/Initiative/Pedagogy/Hedging) by principal-axis factoring + oblimin on
a 72-item forced-choice questionnaire appended to 2,500 persona-rollouts of
Llama-3.1-8B (choice-mass readout). Their design is the item-FA version of
our ENACT experiment: persona-induced conduct variation, same models
(Llama-8B; Qwen-2.5-7B replication on the same rollouts).

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
