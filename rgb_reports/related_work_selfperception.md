# Related work: self-perception + failure loops (swept 2026-07-31)

Two parallel literature sweeps (Sonnet agents, ~40 searches each) on the
two arms of `design_selfperception.md`. Headline: **the persona arm is
clear; the failure arm is substantially scooped and needs re-scoping.**

Verification status: arXiv:2603.10011 confirmed directly (title, authors,
abstract, post-training-not-base finding). LessWrong items and Ivanova et
al. are agent-reported and NOT yet verified — flagged inline. One agent
self-corrected a mis-citation mid-sweep (had pulled Anthropic's "Agentic
Misalignment" for the Gemini SFT post); treat unverified specifics as
leads, not citations.

## A. Failure arm — PARTIALLY SCOOPED

**Soligo, Mikulik & Saunders, "Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs", arXiv:2603.10011 (Feb 2026;
ICLR 2026 workshop).** VERIFIED. Multi-turn conversations where the user
repeatedly rejects the model's answer (impossible numeric puzzles +
WildChat), Claude-Sonnet autorater scoring distress 0–10 per turn, 9
models / 5 families. Gemma-27B mean frustration 1.5→5.5 over turns 1–8.
**The instability emerges in post-training, not in base models** — every
family's base has similar propensity; only instruct-tuned Gemma/Gemini
amplify it. DPO on 280 preference pairs cuts high-frustration 35%→0.3%.

That is our failure arm's affect axis, at our scale, already run — and
its base-vs-instruct result is the affective twin of our §8h/§8g finding
(post-training installs the capacity; base models are flat). Convergent
validation for us, but it means "post-training installs the failure
response" is no longer ours to claim in the affect domain.

**Africa & Shah, "Gemma Gets Help" (LessWrong, Apr 2026) — UNVERIFIED.**
Reported: 20-turn rejection trajectories + an "escape hatch" offering
conversation-end or self-deletion; ~49% self-deletion by turn 20 on math
rollouts vs near-zero under neutral control; late-turn persona shift to
third person ("this unit"). If it holds up, this pre-empts the
terminate-choice DV — though its framing is self-preservation/shutdown,
not our competence judgment (see re-scope below). The instrument
reportedly originates with **Ivanova et al. 2026, citation unresolved —
find this before writing anything.**

**Engels & Nanda, "Why do naive SFT filters fail?" (AF, Jun 2026)** —
the post rgb sent. Its negative-emotion benchmark is the same paradigm;
note the detail we should have caught on first read: negative emotion
does NOT transfer across teacher models (unlike date confusion and
blackmail) — it tracks Gemini's own SFT prompt distribution.

**Sinha et al., "The Illusion of Diminishing Returns", arXiv:2509.09677
(Sep 2025).** Names **self-conditioning**: a model errs more after seeing
its own past errors in context, separable from generic context-rot.
Accuracy-only, no affect or self-concept. This is the mechanistic engine
under our whole failure arm, and it means our design MUST carry an
accuracy DV to connect to it (and to rule out context-rot).

**DAgger / teacher-forcing blind spot** — the vocabulary exists: Li et
al. arXiv:2605.12913 ("Revisiting DAgger in the Era of LLM-Agents",
turn-level student/teacher interpolation, +17.9pt on SWE-Bench Verified);
Nie arXiv:2605.22731 ("Post-Training is About States, Not Tokens").
Neither connects to affect or self-concept. rgb's weak→strong switching
idea (2026-07-30) is essentially DAgger and should be cited as such.

**Also**: Sofroniew et al., "Emotion Concepts and their Function in a
Large Language Model" (transformer-circuits, Apr 2026) — 171 steerable
emotion vectors in Sonnet 4.5; amplifying "desperation" raises blackmail
22%→72% **with no trace in the visible CoT**. Direct warning for our
free-text affect readout: the visible register can be calm while the
state is not.

### Re-scope of the failure arm (what is still unclaimed)

1. **Competence self-report on trait adjectives** after K turns of
   verified own-error context. Nobody runs a structured self-concept
   probe here — the whole cluster uses free-text autorater distress.
   This is exactly our distributional-Likert wheelhouse.
2. **Continue vs. hand off to a fresh attempt** as a competence/trust
   judgment ("would a clean context do better?"), explicitly NOT the
   shutdown/self-deletion framing. Decoupling the choice from
   self-preservation is the contribution.
3. **Cross-method triangulation** under one manipulation: self-report
   Likert × choice mass × free-text affect × accuracy. The existing work
   uses one channel; triangulation is our signature move.
4. **The theoretical bridge**: self-conditioning (Sinha) and the
   SFT-blind-spot literature (Li, Nie) do not cite the distress
   literature (Soligo, Engels). "The state where my context contains my
   own errors is an SFT-blind-spot state, AND it elicits filter-resistant
   distress" is a synthesis nobody has written.
5. Family-lineage comparison tied to the teacher-transfer result.

## B. Persona arm — NOT SCOOPED

Nothing does own-conduct dosing with no framing device, dose-response
over K, self-report AND behavioural carryover, or family/base-vs-instruct
comparison of update RATE. Nearest neighbours:

- **Matyas et al., arXiv:2604.19791** (DeepMind Concordia) — hand-builds
  a "Bem actor" that infers its traits from its own recent behaviour, for
  social-simulation fidelity. Mechanism installed by design, not measured
  as emergent. The paper to differentiate from most carefully.
- **Lehr et al., PNAS 2025 (arXiv:2502.07088)** "Kernels of Selfhood" —
  induced-compliance essay paradigm on GPT-4o, attitude shifts more under
  illusion of free choice. **Cummins, Elson & Hussey (PNAS 2025) rebut**:
  it's context valence-priming, not self-inference. Lehr et al. reply.
  **This live dispute is our positioning**: a dose-response with a
  behavioural carryover readout and a no-instruction arm is a strictly
  stronger test than the single-shot design they are arguing about.
- **Han/Kocielnik et al., "The Personality Illusion" (arXiv:2509.03730)**
  and ICML-2026-workshop follow-up (arXiv:2606.12730) — persona
  *injection* moves self-report but not behaviour; coherence collapses
  across sessions. Our arm B is their manipulation; our arm A is not.
- **Plisiecki et al., arXiv:2607.20082** "Two-Process Theory of Machine
  Self-Report" — 206 models, 67 base/post-trained pairs; post-training
  installs a permitted inner-life vocabulary. Same SHAPE as our §8h
  claim, different dependent variable (experience-claiming, not trait
  updating). Cite so reviewers don't conflate them.
- **Asvin & Lindsey (Anthropic), arXiv:2605.25459** "From Simulation to
  Enaction" — post-trained models recognise their own generations; base
  models lack it. Again our shape, different capacity.
- **Martorell & Bianchi, arXiv:2603.18893** — tracks emotive states over
  10 turns via logit self-report vs linear probes, with steering
  confirmation. Closest quantitative over-turns design; dose is natural
  conversation content, not controlled own-behaviour playback.
- Persona-drift cluster (Choi arXiv:2412.00804; "Stable Personas"
  arXiv:2601.22812 — self-reports stay stable while observer-rated
  expression declines, a useful contrast case for our Qwen result).

## C. Actions

- [ ] Resolve Ivanova et al. 2026 citation before any write-up.
- [ ] Verify the two LessWrong posts directly.
- [ ] Add accuracy as a failure-arm DV (Sinha self-conditioning link).
- [ ] Re-scope failure arm to the five unclaimed cells above.
- [ ] Add Matyas / Lehr–Cummins / Personality-Illusion to
      `bibliography.md` and the Zotero group (tag `rgb-bibliography`).
