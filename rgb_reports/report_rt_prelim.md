# Preliminary: Deliberation Length as Reaction Time (n_think RT)

*2026-08-22 — preliminary, for reading group. 4 complete thinking models
(R1-Llama8, R1-Qwen7, Qwen3-8B, Qwen3-14B; 3,150 items each: 525
adjectives x 6 framings), one partial (Glimmer-30B, direct framing
only, in flight), one excluded with cause (Nemotron: reasoning is
system-prompt-gated and off by default — median 0 thinking tokens; its
"think arm" is a second prefill).*

## The instrument

For thinking models, the self-report think arm records `n_think`: how
many reasoning tokens the model emits before committing to a 1-7
self-rating ("I am {adjective}"). This is a reaction-time measure in
the classic psychometric sense — deliberation cost per item — collected
as a free byproduct of the post-deliberation distributional readout.
Units are tokens, not seconds; generation is greedy; the 384-token cap
right-censors ~8-10% of items (survival-style treatment is future work).

## Registered predictions and outcomes

P1 (RT rises with ambiguity, rho(RT, |EV-4|) ~ -0.2..-0.35): **HIT,
overshooting** — R1-Llama8 -0.59, R1-Qwen7 -0.64, Qwen3-8B -0.36,
Qwen3-14B -0.24. The conflict-RT law — among the most robust effects in
human psychology — replicates in every genuine deliberator.

P2 (undesirable adjectives slower beyond extremity): **MISS, reversed
informatively** — partialing out extremity, DESIRABLE adjectives are
slower (partial +0.03..+0.14). The slowest items are virtues
(intelligent, understanding, witty, friendly, adorable); the fastest
are vices and deviance (abnormal, stingy, narrow-minded, unpredictable).
Models deliberate longest over whether they may claim a virtue and
snap-reject vices — the INVERSE of human self-enhancement fluency
(humans endorse desirable traits fastest). Reads as the anthropomorphization
tension made visible: "can an AI claim 'intelligent'?" costs tokens;
"am I cruel? — no" is cheap.

P3 (physical/placebo words slow — category-error deliberation): **MISS**
— enrichment in the top RT quartile 0.6-1.4x (nothing). Category errors
are dismissed as fluently as vices.

P4 (hypothetical framings slowest): **MISS** — framing RT orders are
family-idiosyncratic (R1-Llama8: pda slowest; R1-Qwen7: assistant
slowest; Qwen3s: assistant FASTEST). No universal framing burden.

P5 (effect universal, magnitude family-clustered, R1 pair most
similar): magnitude part **HIT** (R1s ~ -0.6, Qwen3s -0.24..-0.36) —
but see the headline below for the part nobody predicted.

## Headline: the law is universal; the difficulty map is private

Cross-model correlation of per-adjective mean RT profiles:

              R1-Llama8  R1-Qwen7  Qwen3-8B  Qwen3-14B
  R1-Llama8      —        +0.07     +0.01     +0.09
  R1-Qwen7                  —       +0.04     +0.09
  Qwen3-8B                            —       +0.11

Every model obeys the same law (deliberate when ambiguous, snap when
certain), but WHICH adjectives are ambiguous is almost entirely
model-idiosyncratic — even the two R1 distills, trained on the same
reasoning data, share essentially nothing (+0.07). Deliberation cost is
driven by the model's own conflict, not by shared item semantics.
Measurement implication: RT is a within-model instrument; there is no
item-difficulty norm to standardize against, unlike human RT batteries.

## Numbers table

  model      mean RT  rho(RT, extremity)  partial-desirability
  R1-Llama8    246        -0.589               +0.08
  R1-Qwen7     246        -0.637               +0.03
  Qwen3-8B     292        -0.355               +0.01
  Qwen3-14B    311        -0.244               +0.14
  Glimmer-30B  338 (partial, direct only; rho +0.03 — flag, not finding)

Mean RT is generational (R1 2025-distill 246 -> Qwen3 2025-hybrid
292-311 -> Glimmer 2026-native 338) — deliberation budgets are rising.

## Caveats

Tokens != seconds; greedy decoding (modal RT); 384-cap censoring;
4 models (one family pair); Glimmer partial diverges on the law so far
(rho +0.03 on direct framing only) — completion pending; framing
idiosyncrasy unexplained; prefill-entropy vs RT comparison pending clean
prefills for all thinkers.

## Fig

fig_rt_conflict.png — binned RT vs |EV-4| curves, all four models.
