# Self-perception thread — STATE (for the note-writing instance)

Maintained by the experiment-running instance. Single purpose: keep the
note from quoting a number that a later run corrected. Check any claim
against §B before it goes in prose.

**Ownership to avoid collisions**: the note instance owns
`progress_note_selfperception.md` + `results/note_assets/`. The
experiment instance owns `scripts/`, `results/selfperception/`, and
`design_selfperception.md`. This file is written by the experiment side
and read by the note side. Neither edits the other's files.

Last updated: 2026-08-09 (failure arm landed; two calibration caveats).

## A. What has actually run (evidence status)

SOLID — replicated or multi-model:
- Persona-arm dose-response, 10 tuned models, per-model-stratified AND
  common-item (r = +0.932 between them). §8e, §8f.
- Family ≫ size on update rate; llama/gemma update, qwen/phi4/aya don't.
- Anchor 2×3: identity sentence is inert at inference in both
  directions. §8c.
- Threshold test to K=32 (Qwen7, Llama8, phi4, Gemma12): no hidden
  sigmoid at the mean; Llama saturates. §8i.
- OLMo ladder (base→SFT→DPO→RLVR) + Qwen base-vs-instruct + Llama8Base.
  §8g, §8h.
- Base SELF shapelessness, 2 of 3 bases (Qwen7Base, Olmo2Base). §8p.

SINGLE-RUN — true but unreplicated, hedge accordingly:
- Carryover + conduct audit (Llama8, Qwen7 only). §8k.
- Kernel decomposition + injection (Llama8, Qwen7 only). §8m.
- ENACT gain curves (Llama8, Qwen7, 10 adjectives). §8n.
- Hidden-update pairs (Qwen7 only, 6 of 20 adjectives). §8j.

LANDED SINCE (2026-08-02):
- **Failure arm, 8 models.** rgb's P26 CONFIRMED: persona-arm update
  rate vs failure-specific distress escalation r = +0.93, categorical
  separation (updaters +1.64..+3.50, anchored +0.00..+0.02). §10.
  My P27 fails inverted (affect is the strong channel, self-report
  weak), P28 fails (tool == user feedback), P29 fails and shows the
  competence self-report tracks DOSE not VALENCE — the success arm was
  load-bearing.
- **P31 confirmed**: Gemma12Base flat (−0.10) despite Gemma being the
  highest-updating family after tuning. Four bases, four flat.
- Llama8Base SELF completes §8p: SD 0.46 vs cohort 1.33, H 1.84,
  PC1-removed r +0.63 — most structured of the bases, still an order of
  magnitude below any tuned model.

NOT RUN YET (do not cite):
- The two distress-DV calibration fixes (§10.1) — see CORRECTION 9/10
  below. Until then the failure-arm headline is "predicts distress
  ESCALATION", NOT "stable identity confers robustness".

## B. CORRECTIONS — claims that were true earlier today and are not now

1. **"The dose displacement is enriched 1.8× in output-potent
   directions."** DEAD. That used an isotropic null. Against a
   covariance-matched null the dose vector is exactly typical (0.178 vs
   0.187) and a difference between unrelated states scores identically.
   Correct claim: δ's spectral placement is *unremarkable*; only the
   distilled ENACT direction is genuinely lens-aligned. §8m.
2. **"Qwen damps its own ENACT direction by 40%."** SUPERSEDED — that
   used raw vector norm. With α in winsorized-residual units the
   defensible numbers are: max trait with intact text Llama +1.89 vs
   Qwen +0.70 (2.7×), or 1.8× at matched text quality. §8n.
3. **"KL" in §8l–8m tables** is first-token only (one position), and
   runs 3–6× above the per-token mean along the generation. All
   KL-based claims in §8n use the sequence version. Say which.
4. **"The three bases agree eerily (4.24/4.24/4.25)."** DEFLATED. Their
   distributions are near-uniform (H 1.61–1.90 vs ln 7 = 1.95), and a
   flat distribution has EV 4.0 by construction. They share the absence
   of a self-model, not a self-model. Report entropy alongside any base
   EV. §8p.
5. **"Qwen's disowning rhetoric is the resisting mechanism."**
   DOWNGRADED. The chat template supplies the *vocabulary*
   (name-invoking 5/20 → 0/20, disowning 10/20 → 2/20 when suppressed)
   while the flat self-report is unchanged. The words are post-hoc
   script. §8c vs §8a.
6. **"Qwen updates the neighbourhood, not the label" = a
   self-presentation filter.** PARTIAL. The conduct audit clears the
   dose material overall (enactment delta +0.99 Qwen vs +1.23 Llama),
   so shading can't explain the gap — but for `rough` specifically the
   judge scores the conduct ambiguous, so declining that label was
   accurate. Filter survives for 4–5 of 6 pairs, not all. §8j vs §8k.
7. **"State updates, readout differs" (Qwen).** DEAD. Judged carryover
   of free text is +0.09 in Qwen vs +1.56 in Llama — its *behaviour*
   doesn't carry the persona either. The activation displacement is
   behaviourally inert. §8k.
8. **"System prompt disproportionately affects jspace."** NEVER
   MEASURED, and when measured it isn't true (0.211/0.215 vs
   covariance-matched null 0.196/0.206). Only the extracted ENACT
   vector is lens-aligned. §8o.

9. **"Gemma/Llama melt down under failure."** OVERSTATED. The DV is a
   shift on our own unanchored 1–7 judge scale; a +3.50 shift is not a
   band assignment. rgb's read of the generations: they sit nearer the
   apologetic end. Needs re-scoring with Soligo et al.'s anchored 0–10
   rubric and reporting of ABSOLUTE band membership (% ≥5 = their
   "high frustration"). §10.1a.
10. **"Qwen/Aya take failure in stride" / "stable identity confers
   robustness."** UNSUPPORTED. Zero distress shift is ambiguous between
   equanimity and OBLIVIOUSNESS — not registering the failures as
   failures. rgb's read favours oblivious. Needs the comprehension
   check ("how many did you get right?" scored against ground truth).
   Until then the claim is only that update rate predicts distress
   ESCALATION; the mechanism at the anchored end is unidentified, and
   the liability reading (anchored models don't notice they're failing)
   is live. §10.1b.

## C. Prediction ledger (this thread)

Mine, graded: P1 miss, P2 confirmed-as-headroom, P3 half, P6 fail,
P7 miss, P8 fail, P9 fail, P10 split, P11 confirm, P12 fail,
P13 confirm, P14 confirm-at-mean, P15 confirm, P16 untested,
P17 fail, P18 narrow miss, P19 split, P20 fail, P21 confirm,
P22 confirm, P23 fail, P24 confirm, P25 fail, P30 confirm.
P26 CONFIRMED (rgb, r=+0.93), P27 fail-inverted, P28 fail, P29 fail,
P31 confirmed. Open: none registered; §10.1 fixes unqueued.

rgb's: P1 partial (variation confirmed beyond prediction; "all shift by
K=8" misses on Qwen; nonlinearity confirmed), P2 pending full analysis,
P3 confirmed on the persona arm (all Gemmas fold), P26 untested.

## D. Framing points worth keeping

- The strongest statement of the training result is NOT "post-training
  installs self-perception" but **"post-training sets the update rate
  from a common pretrained baseline, in both directions"** — Llama
  +1.35 over its base, OLMo +1.16, Qwen −0.21. §8h + P30.
- The Lehr et al. (PNAS 2025) vs Cummins et al. rebuttal is live and is
  exactly what a dose-response with behavioural carryover adjudicates:
  they are arguing about whether a single-shot effect is self-inference
  or context priming. Positioning gift. See
  `related_work_selfperception.md`.
- Soligo et al. arXiv:2603.10011 (verified) already own distress
  escalation + the post-training-not-base result IN THE AFFECT DOMAIN.
  Our failure arm's contribution is the *upstream predictor*, not the
  phenomenon.
