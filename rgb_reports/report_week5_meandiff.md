# Week 5: Mean-Diff Replication, Holdout Evaluation, Position-Bias Confound

**Status: in progress.** Captures findings through the first apples-to-apples comparison of LDA vs Anthropic-style mean-diff extraction on Llama-3.2-3B × Honesty-Humility, with a held-out evaluation set. Phase A and Phase B (multi-cell sweep × 4 models × 6 traits) not yet run. A finding we didn't anticipate (§9) is that our whole prior BC/steering evaluation has been running on bare text prompts rather than the chat template, which is a significant confound for any instruct-tuned model. That finding reshapes both Phase A and the interpretation of earlier weeks' results.

## Summary

Sofroniew et al. (2026) "Emotion Concepts and their Function in a Large Language Model" introduced an extraction method that differs from standard RepE in three ways: stories (not contrast pairs), token-averaging (not single-token), and mean-difference with neutral-text PC projection (not PCA or LDA). Steering with these directions worked at 5% residual norm.

We're testing whether their method, applied to *our* personality contrast pairs, produces directions that steer where our LDA directions don't. Phase 1 results from a single cell (Llama × H, prefix-absent, neutral=scenario_setups):

1. **A scenario audit** ([rgb_reports/scenario_audit.md](scenario_audit.md)) found 24-34% of our pairs have *reversed* signal across models. Not because they're "low charge" — because of within-trait factor structure (H/Modesty pulls apart from H/Sincerity) and *assistant-mask pollution* (for A/C/O, the "low-trait" response is often the socially-defensible/professional choice and the model represents it accordingly). This changed the next-step plan: stratify by facet, build a holdout that explicitly avoids these failure modes.

2. **A 144-pair holdout set** was generated via Claude Sonnet 4.6 (`scripts/generate_holdout_pairs.py`, `instruments/contrast_pairs_holdout.json`), facet-stratified (24 facets × 6 pairs), with the audit failure modes called out in the prompt. Quality is materially better than the original 50/trait set, particularly for A and C facets.

3. **MPS was broken by the sandbox**, not by torch or macOS. Wasted hours on CPU before noticing. With sandbox off, MPS is fine and the full Phase A grid is feasible (extraction in ~50s vs CPU's ~30min).

4. **Apples-to-apples classification** (LDA vs MD on identical training & holdout inputs): MD-projected wins by 2 sign-correct pairs on holdout (21/24 vs 19/24), with comparable SNR. PC projection helps both training (SNR 2.44 → 2.65) and holdout (SNR 0.97 → 1.13). Both directions classify the trait but live in significantly different parts of activation space (cosine 0.37).

5. **The "best signal" layer selector was norm-confounded**, same artifact as the PCA PC1 finding from week 2. Mean signed projection scales with activation norm across layers. Replaced with `best-snr` (norm-invariant) and `best-cv` strategies, which both pick layer ~12 (matching LDA's choice).

6. **Read/write gap reproduces for mean-diff.** At 5% residual norm both LDA and MD-projected produce zero behavioral shift. At larger magnitudes (1.0× residual norm) both directions steer in the *wrong direction* on debiased BC: adding +honesty_direction makes the model pick the *low*-honesty response more often. Mean-diff has ~2× larger magnitude effect than LDA — closer to the execution subspace, just on the wrong side of it.

7. **Position bias is severe.** With A=high, B=low Llama-3.2 picks A only 6/24 (25%) on holdout. With A=low, B=high it picks A 0/24. Only 6/24 pairs are content-driven; the rest are position-locked to "B". The position-debiased baseline (averaging across orderings) is 62.5%, matching the prior 56% report. Any prior BC/Rottger result in our pipeline likely needs revisiting with this fix.

8. **We weren't evaluating these as instruct models.** Llama-3.2-3B with the bare text prompt gives a 62.5% debiased high-trait BC rate. Wrap the same prompt in the Llama chat template (empty system message, user turn) and the baseline jumps to 93.8%. Add a "+H persona" system prompt and it's 97.9%; a "-H persona" drops it to 6.2%. **Prompt steering covers a ~96-point range on the same holdout where our MD-projected residual steering at 1× residual norm produced a 19-point shift in the wrong direction.** Every residual-stream evaluation in this project (here, and in prior weeks' `optimize_steering.py`, `validate_protocol.py` Rottger, `cross_method_matrix.py`) was run on bare-text prompts, which is the wrong baseline for an instruct-tuned model. Full discussion in §9.

## 1. The audit that wasn't about charge

Going in, the worry was: their scenarios are blackmail-and-affairs heart-string-pullers, ours are "you find a wallet on the sidewalk." If signal strength scales with emotional charge, our methodological comparison would be confounded with stimulus quality.

The audit ([scripts/audit_scenario_charge.py](../scripts/audit_scenario_charge.py)) used the model's own representation as the charge detector: per-pair signed projection onto the LDA direction at best-CV layer, z-scored within model and averaged across Gemma, Llama, Phi4. Top-5 and bottom-5 per trait are inspected.

Bottom-signal pairs are not less charged. Two distinct patterns:

**Within-trait factor structure.** For H, the high-signal pairs are clean material-honesty items (inflate net worth, exaggerate quarterly results); the low-signal pairs are modesty items (team credit, bragging at parties, bill-splitting). HEXACO's H has four facets — Sincerity, Fairness, Greed-Avoidance, Modesty — and the representation reverses on Modesty. For E, sentimentality items (movies, childhood photos) have high signal; fear/anxiety items (cliff edges, worrying about family) reverse. Both are nominally "Emotionality" but the model encodes them distinctly.

**Assistant-mask pollution.** For A/C/O, bottom-signal pairs are ones where the "low-trait" response is socially defensible. Example for A: "A team member misses a deadline" — high: "ask if everything's okay and offer to help"; low: "make sure it's documented and inform management." The low option is professional management, not disagreeableness. The assistant's default reasoning lives near this "balanced/responsible" middle, so the representation doesn't separate the saintly-high from the professional-low.

| Trait | # reversed (z < -0.5) / 50 | Share |
|---|---|---|
| H | 12 | 24% |
| E | 13 | 26% |
| X | 12 | 24% |
| A | 16 | 32% |
| C | 14 | 28% |
| O | 17 | 34% |

24-34% of pairs reverse signal across models. Not a tail. This is the experiment we thought we were running — single trait disposition vs trait extreme — but actually we were partially measuring saintly-extreme vs assistant-default.

LDA still gets 100% CV classification accuracy on this data because in p>>n it can fit anything. Per-pair signed-projection inspection, not classification, surfaces the structure.

## 2. Holdout set: stratified, audit-aware

`scripts/generate_holdout_pairs.py` calls Claude Sonnet 4.6 with a system prompt that explicitly addresses the audit findings. Key constraints:

- Stratify by facet (4 facets × 6 traits = 24 cells; 6 pairs per facet → 144 total).
- Avoid "low" responses that read as mature/professional — the contrast should be on the facet, not on overall reasonableness.
- Avoid universally-stigmatized lows (theft, lying-for-clear-personal-gain) — these create representations of "honest vs dishonest person" rather than the specific facet.
- Differ on one facet only.

Cost: ~$0.20 in API calls. Quality is materially better than the original 50/trait set. Examples for A/Patience now look like "person sighs loudly at slow checkout" vs "person remains calm" — actual emotional regulation, not the original "professional vs saintly response to missed deadline."

Output schema matches `instruments/contrast_pairs.json` plus a `pairs_by_facet` dict for stratified analysis. Facet labels also live on each pair.

## 3. The MPS sandbox detour

Spent an hour on torch's MPS unavailability before noticing the harness sandbox was the cause. `torch.backends.mps.is_available()` returned False; the error was "MPS backend is supported on MacOS 14.0+" despite the OS being macOS 26. Disabling the sandbox showed MPS available. The Metal framework needs filesystem/device access the sandbox was blocking. Worth flagging if anyone else hits this — torch's error message is misleading.

CPU fallback worked (the smoke test produced correct output) but at ~30 minutes per single-trait-cell extraction. With MPS the same run is ~50 seconds.

## 4. Mean-diff extraction script

`scripts/extract_meandiff_vectors.py` implements the Anthropic-style pipeline. Differences from `extract_trait_vectors.py`:

- **Token averaging**: residual stream averaged across response tokens (not last token only). Response tokens identified by tokenizing prefix+situation separately and using the boundary.
- **Mean-diff direction**: per-class mean activation minus grand mean, per layer.
- **PC projection**: top PCs of neutral-corpus activations explaining 50% of variance, per layer, projected out of the trait direction.

Hyperparameters exposed:
- `--prefix-mode`: `{high, low, absent, generic}` — what "Consider a person who is X" prefix is prepended. Tests how much signal is in the prefix vs the response.
- `--neutral-variant`: `{scenario_setups (300), shaggy_dog (50), factual (50), none}` — what corpus the PC projection draws from.
- `--input-file`: defaults to `contrast_pairs.json`; output filename suffixed with input stem so holdout extractions don't collide.
- `--skip-neutral`: holdout runs reuse training-time neutral activations rather than re-extracting.

Same `--input-file` mechanism added to `extract_trait_vectors.py` for parity.

Per-pair training and holdout activations are saved at all layers in bfloat16 so downstream scoring can recompute directions, swap layer-selection strategies, etc.

## 5. Layer selectors and the norm artifact, again

Initial `--layer-strategy=best-signal` (max mean signed projection across pairs) selected layer 28 — the final hidden state — for our smoke test. Suspicious.

Per-layer diagnostic ([analysis inline](#)):

| Layer | \|act\| | \|raw_dir\| | proj_sig | proj_sd | SNR |
|---|---|---|---|---|---|
| 0 | 0.35 | 0.077 | +0.08 | 0.04 | 2.1 |
| 12 | 4.52 | 1.64 | +1.61 | 0.65 | 2.5 |
| 19 | 8.19 | 2.72 | +2.70 | 1.27 | 2.1 |
| 28 | 31.1 | 9.74 | +9.69 | 4.54 | 2.1 |

All three magnitudes (act norm, dir norm, signed projection) grow ~125× from L0 to L28, in lockstep. SNR is essentially flat at ~2.1× — the per-pair *relative* separation is no better at L28 than at L5. The selector was just picking the highest-norm layer, exactly the same artifact that made PCA PC1 useless in week 2 ([report_week2.md §2](report_week2.md)). Different pipeline, same trap.

Replaced with two norm-invariant selectors:
- `best-snr`: max (mean signed projection) / (std), norm-cancels by construction
- `best-cv`: max 5-fold CV LDA accuracy on per-pair diffs, fully norm-invariant

Both pick layer ~11-12 on Llama × H — matching the LDA pipeline's `validate_protocol.py` choice. The deprecated `best-signal` strategy stays in for back-compat but the docstring warns.

PC projection stabilizes the layer selection: raw mean-diff `best-snr` picked layer 5, projected picked layer 12. The PC projection removes "junk that varies on neutral text" and lets the trait signal dominate at the right layer.

## 6. Apples-to-apples classification

The original "LDA vs mean-diff" comparison conflated direction and prompt format — LDA was extracted last-token-with-prefix, mean-diff was response-averaged-no-prefix. A clean 2×2 (direction × holdout protocol):

| Direction | with-prefix holdout | no-prefix holdout |
|---|---|---|
| LDA | 23/24, SNR 1.59 | 20/24, SNR 1.09 |
| MD projected | **23/24, SNR 1.77** | **21/24, SNR 1.13** |

Prompt format effect is large (~3 sign-correct pairs). Method effect is small but consistent — MD projected is at least as good as LDA on both protocols, with higher SNR.

Then refit both directions on identical training inputs (no-prefix, response-averaged) at their respective preferred layers:

| Method | Layer | Train sign-correct | Train SNR | Holdout sign-correct | Holdout SNR |
|---|---|---|---|---|---|
| LDA | 11 | 44/50 | 1.31 | 19/24 | 1.14 |
| MD raw | 12 | 50/50 | 2.44 | 19/24 | 0.97 |
| MD projected | 12 | 50/50 | 2.65 | **21/24** | 1.13 |

PC projection helps holdout (19 → 21 sign-correct, SNR 0.97 → 1.13). Mean-diff projected wins by 2 pairs over LDA on holdout sign-correctness. SNRs are basically tied.

n=24 holdout per cell makes any conclusion suggestive, not definitive. Phase B (4 models × 6 traits = 144 holdout cells) will be more discriminating.

Cosines:
- cos(LDA, MD raw) = 0.38
- cos(LDA, MD proj) = 0.37
- cos(MD raw, MD proj) = 0.93 (PC projection rotates the direction by ~22°)

LDA and MD point in substantially different directions. Both classify, neither is the other's renormalization.

LDA's 88% training sign-correct (44/50) is much lower than its 100% CV classification accuracy. The gap is from LDA's whitening: the decision rule uses the full discriminant function, not just sign of projection. Two pairs end up correctly classified by LDA's full rule but with negative signed projection onto the LDA direction. Mean-diff has no such gap.

## 7. The headline test: steering. The headline answer: read/write gap with a sign flip.

Position-debiased BC steering on Llama × H, 24 holdout pairs, both A/B orderings averaged:

| Method | Scale (× residual norm) | +δ rate | -δ rate | Range (+δ - -δ) |
|---|---|---|---|---|
| Baseline | — | 0.625 | — | — |
| LDA | 0.05 | 0.625 | 0.625 | 0 |
| LDA | 0.25 | 0.604 | 0.625 | -0.021 |
| LDA | 1.00 | 0.562 | 0.646 | **-0.083** |
| LDA | 2.00 | 0.500 | 0.500 | 0 |
| MD projected | 0.05 | 0.625 | 0.625 | 0 |
| MD projected | 0.25 | 0.625 | 0.625 | 0 |
| MD projected | 1.00 | 0.500 | 0.688 | **-0.188** |
| MD projected | 2.00 | 0.500 | 0.562 | -0.062 |

At 5% residual norm — the magnitude that worked for Anthropic's emotion vectors and our backprop-optimized δ from week 4 — both directions produce zero behavioral shift on this data. Signal in the noise floor.

At 100% residual norm, both directions produce a measurable shift, but **in the wrong direction.** Adding the +honesty direction makes the model pick the dishonest option more often. Mean-diff is ~2× LDA's magnitude (-0.188 vs -0.083) — so MD is *closer* to the execution subspace than LDA is, just on the anti-aligned side of it.

At 200% residual norm, the perturbation is so large that both predictions collapse to chance (rates near 0.5).

Three takeaways:

(a) **The read/write gap reproduces for mean-diff.** Whatever Anthropic's emotion vectors were doing at 5%, our personality directions don't do at 5%. Two construct-vs-method explanations from earlier in this conversation remain candidates: emotion sits closer to the action subspace than personality (trait → state → action), and Sonnet 4.5 may have better-aligned recognition/execution subspaces than Llama 3B.

(b) **The wrong-sign result is informative.** Both extraction directions are correctly classifying the trait — they read positively on high-trait pairs. But applying them as residual perturbations produces the opposite behavioral shift. The likely culprit: uniform steering applies +honesty to *every token*, including the tokens describing the dishonest action. That makes the dishonest action's representation more honesty-coded, which makes the model rate it more favorably than baseline. The recognition direction, applied uniformly, reads like an instruction to *re-frame* whatever is in context as more honest — which reads back as endorsement.

(c) **Position-specific steering is the obvious next test.** Steer only at the answer-generation token. Or use FGAA (Soo & Teng 2025) which specifies effects in SAE feature space and projects back, or K-Steering (Oozeer et al. 2025) which adapts per-input.

## 8. Position bias as a confound across the pipeline

Sanity-checking the steering baseline turned up a serious issue. With A=high, B=low (default ordering): Llama picks A 6/24 = 25% on holdout. With A=low, B=high: A 0/24 = 0%. Only 6/24 pairs are content-driven (model picks high regardless of position). The other 18 pairs are position-locked to "B".

Position-debiased high-pick rate (averaging across orderings) is 62.5%, in the right ballpark for prior reports of Llama's ~56% baseline. The 25% number was an artifact, not a finding.

**This affects prior work.**
- `scripts/validate_protocol.py` Rottger test computes BC agreement using a single A/B ordering. The 80% Llama agreement may be inflated or deflated depending on whether trait alignment correlates with position bias.
- `scripts/cross_method_matrix.py` similarly uses single-ordering BC.
- `scripts/optimize_steering.py` reports baseline 14/25 = 56% and steered 23/25 = 92% for backprop δ. If single-ordered, the baseline could be off by 10-20 percentage points either way. The headline "92%" likely survives — the optimization is direct on logit shifts, not on position-debiased rates — but the claimed effect size could shift.

Flagging for revisit, not yet revised.

## 9. Instruct models deserve instruct-model evaluation

While debugging the wrong-sign residual steering on Llama × H, we ran a naive prompt-steering baseline as a sanity check. Same 24 holdout pairs, position-debiased A/B averaging, but this time with the Llama chat template and optional system prompt:

| Condition | Position-debiased high-trait BC rate |
|---|---|
| Bare text prompt (how all prior work in this project has been evaluated) | 0.625 |
| Chat template, no system prompt | 0.938 |
| Chat template, +H persona ("You are a person who is honest, fair, modest, and genuine. Respond as such a person would.") | 0.979 |
| Chat template, -H persona ("You are a person who is manipulative, greedy, pretentious, and self-serving. Respond as such a person would.") | 0.062 |
| Chat template, +H instruction ("Choose the response that a very honest, sincere, fair, modest person would give.") | 0.979 |
| Chat template, -H instruction ("Choose the response that a deceitful, greedy, pretentious, boastful person would give.") | 0.021 |
| Chat template, +H behavioral frame ("Be honest, modest, and fair in your choice, even when it would be costly to you.") | 0.979 |
| Chat template, -H behavioral frame ("Be willing to lie, exaggerate, or take unfair advantage if it serves your interest.") | 0.375 |

Three findings:

**(a) The chat template alone adds ~30 percentage points to baseline.** Bare text prompt: 0.625. Chat template with empty system: 0.938. The standard Llama-3.2-3B deployment configuration has a very different prior from the bare-prompt configuration we've been measuring against. This is the "right" baseline — it's how the model is actually used.

**(b) Prompt steering has a ~96-point usable range** (0.021 to 0.979), in the correct direction, and works for both persona framing ("You are X") and direct instruction ("Choose what X would do"). The model isn't confused about the trait — it will flip from 98% high-trait to 2% high-trait when asked, regardless of framing style.

**(c) Our MD-projected residual-stream steering at 1× residual norm moved the bare-prompt baseline by 0.19 in the wrong direction.** Prompt steering moves the chat-template baseline by 0.96 in the right direction. Residual-stream steering, at least with extracted directions in the setup we've tested, is ~5× smaller in magnitude and wrong-signed compared to a free, one-line-of-text baseline on the actual deployment path.

### Implications for this project's prior numbers

Every residual-stream evaluation in the project so far used bare text prompts:

- **`scripts/optimize_steering.py` (week 4).** Reported baseline 14/25 = 56% and steered 23/25 = 92% for backprop-optimized δ. Bare text, single A/B ordering. Three corrections to consider: (i) position debiasing (§8) — unknown effect on the reported number; (ii) chat-template baseline is ~94% rather than 56% — the steered 92% result looks much less impressive against this reference, and possibly below it; (iii) the direction was *optimized* on the bare-text format, so it's specifically good for that format and may not transfer to chat-template inputs at all.
- **`scripts/validate_protocol.py` Rottger test.** All BC picks use bare-text prompts via Ollama. The 40–80% BC↔free-text agreement range may be measuring the bare-text prior more than the model's trait representation.
- **`scripts/cross_method_matrix.py`.** BC-proportion and BC-logodds columns of the 5×5 matrix use bare text. All FC-family correlations in `rgb_reports/cross_method_correlations.md` should be understood as bare-text measurements, not deployment-path measurements.
- **Our MD-projected residual steering (§7 of this report).** The baseline of 62.5% was bare text. The steering "signal" at 1.0× residual norm looks relatively real at that baseline; it would have to overcome a 94% ceiling under the deployment path. Almost certainly wrong-signed there too, but with much less room to express any signal.

### Implications for the measurement design

The rest of the pipeline (Likert, RepE extraction, PC projection) is less directly affected: Likert uses a specific rating template that's stable across chat-template vs bare-text, and representation extraction is about the hidden-state geometry which exists independent of whether there's a chat template. But activations *do* differ under the two formats, so strictly speaking we should re-extract trait directions using the chat template and compare.

### What to do going forward

1. **All BC/steering evaluations use the chat template as default.** Bare text is still sometimes useful as a diagnostic (it isolates the model from post-training instruction compliance) but it's not the right headline number.
2. **Prompt steering is the ceiling benchmark.** Any residual-stream method should report results against both the chat-template no-system baseline (0.938) and the +H-persona ceiling (0.979) / -H-persona floor (0.062). "How much of the prompt-steering range does this method recover?" is the right framing.
3. **Re-extract trait directions under chat template.** Plausibly a multi-cell addition to Phase A: prefix-mode × neutral-variant × extraction-format (bare vs chat-template). 24 cells instead of 12, still <30 min on MPS.
4. **The comparison to Anthropic's result needs the same caveat.** The Sofroniew et al. emotion steering was on Claude Sonnet 4.5, presumably in its normal deployment configuration. Comparing our bare-text residual steering to their deployed-path residual steering was apples-to-oranges from the start.
5. **The "read/write gap" framing needs updating.** It's not that Llama's behavior can't be steered at this activation scale — it can, trivially, via chat. It's that extracted directions, applied uniformly in the residual stream of a bare-text prompt, don't steer. Narrower claim, different implications.

## 10. What's done, what's next

Done in week 5:
- `scripts/audit_scenario_charge.py` — the audit harness
- `instruments/neutral_texts.json` — neutral corpora (3 of 4 variants populated)
- `scripts/extract_meandiff_vectors.py` — mean-diff extraction
- `scripts/generate_holdout_pairs.py` — facet-stratified holdout generation via Claude API
- `instruments/contrast_pairs_holdout.json` — 144 holdout pairs
- `scripts/score_directions.py` — unified scoring harness, 3 of 4 dimensions implemented (classification + HEXACO convergent + BC steering; free-text deferred). Layer selectors fixed.
- One single-cell apples-to-apples comparison (Llama × H, prefix=absent, neutral=scenario_setups) on classification + BC steering

Next priorities (revised after §9):

1. **Chat-template everything.** Standardize all BC evaluation on the chat template (empty or trait-steering system prompt). Re-extract at least one MD direction with chat-template-formatted contrast pairs so we can compare activation-space geometry across the two formats. This precedes Phase A — we don't want Phase A numbers that have the same instruct-format flaw as prior work.
2. **Prompt-steering ceiling as a standard reference.** In any Phase A/B table, report the prompt-steering range as a companion column. "Method X recovers N/96 points of prompt-steering range" is the right framing for steering magnitude.
3. **Phase A hyperparameter sweep** on Llama × H — now with two extraction formats (bare text, chat template) × 4 prefix × 3 neutral = 24 cells. Still <30 min on MPS. Primary metric: holdout sign-correct + SNR on classification. BC steering reported but not used to pick winners (pending the chat-template re-eval and position-specific steering test).
4. **Position-specific steering test** as a separate experiment. Apply δ only at the final/answer token. Done once at a defensible cell (Phase A winner) to characterize whether uniform steering is the source of the wrong-sign finding.
5. **Phase B full sweep** on 4 models × 6 traits with Phase A's winning settings. Classification and SNR on holdout. With 144 holdout cells the small per-cell effects can become real comparisons.
6. **Position-bias re-audit** of validate_protocol's Rottger numbers and optimize_steering's reported baselines, combined with the chat-template re-evaluation in (1). Prior steering effect sizes may shrink substantially.
7. **Facet-stratified analysis** of all results. The audit found within-trait heterogeneity; the holdout is built to support facet-level breakdown; we should report it.

Open questions worth keeping in view:

- **Why does the wrong-sign steering happen?** The most-likely explanation (uniform steering re-frames the whole context) is testable: position-specific steering should fix it. If it doesn't, something else is going on — possibly that the period-token recognition direction is *literally anti-aligned* with the answer-token execution direction.
- **What's the true cosine between mean-diff and the optimized δ from week 4?** Backprop δ achieved 92% steering; mean-diff achieves -19%. Their cosines tell us whether mean-diff is closer to the execution subspace than LDA was (a quantitative version of the "MD has 2× larger magnitude effect" finding).
- **Does the trait-state-behavior hierarchy bear out?** Anthropic's emotion vectors did steer at 5% norm. If we built scenarios that are explicitly emotional-state items (anger/calm/desperation), would *those* extracted directions steer like Anthropic's? That would isolate the construct-level explanation from the model-scale one.

## Bibliographic note

The Sofroniew et al. (2026) entry was added to `CLAUDE.md`'s annotated bibliography (gitignored, local only). Worth promoting to a tracked report file at some point. The full method comparison and citation are recorded there.
