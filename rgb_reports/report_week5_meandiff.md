# Week 5: Mean-Diff Replication, Holdout Evaluation, Position-Bias Confound

**Status: in progress.** Phase A (§10) and Phase B (§11) both complete. The method comparison reversed twice: §6 single cell said MD wins; §10 Phase A grid on Llama × H said LDA wins; §11 Phase B across all 4 models × 6 traits says MD wins overall BUT LDA wins H specifically, with a clean facet-level mechanism (MD fails on H/Modesty and H/Greed-Avoidance). Two confounds we didn't anticipate reshape all prior steering/BC work: §8 (position bias) and §9 (chat-template vs bare-text on Llama).

## Summary

Sofroniew et al. (2026) "Emotion Concepts and their Function in a Large Language Model" introduced an extraction method that differs from standard RepE in three ways: stories (not contrast pairs), token-averaging (not single-token), and mean-difference with neutral-text PC projection (not PCA or LDA). Steering with these directions worked at 5% residual norm.

We're testing whether their method, applied to *our* personality contrast pairs, produces directions that steer where our LDA directions don't. Phase 1 results from a single cell (Llama × H, prefix-absent, neutral=scenario_setups):

1. **A scenario audit** ([rgb_reports/scenario_audit.md](scenario_audit.md)) found 24-34% of our pairs have *reversed* signal across models. Not because they're "low charge" — because of within-trait factor structure (H/Modesty pulls apart from H/Sincerity) and *assistant-mask pollution* (for A/C/O, the "low-trait" response is often the socially-defensible/professional choice and the model represents it accordingly). This changed the next-step plan: stratify by facet, build a holdout that explicitly avoids these failure modes.

2. **A 144-pair holdout set** was generated via Claude Sonnet 4.6 (`scripts/generate_holdout_pairs.py`, `instruments/contrast_pairs_holdout.json`), facet-stratified (24 facets × 6 pairs), with the audit failure modes called out in the prompt. Quality is materially better than the original 50/trait set, particularly for A and C facets.

3. **MPS was broken by the sandbox**, not by torch or macOS. Wasted hours on CPU before noticing. With sandbox off, MPS is fine and the full Phase A grid is feasible (extraction in ~50s vs CPU's ~30min).

4. **Method comparison, summarised across the whole journey:** §6 showed MD winning on one cell; §10's Phase A grid on Llama × H showed LDA winning; §11's Phase B grid (48 cells = 4 models × 6 traits × 2 formats) shows MD-projected winning 16, LDA 9, tied 23. **H is the exception** — LDA wins every one of the 8 H cells. Facet breakdown explains why: MD fails specifically on H/Modesty (48%) and H/Greed-Avoidance (54%), exactly the facets the scenario audit flagged. On other traits (E, X, A, C, O), MD-projected matches or beats LDA. The headline: **MD is a good uni-dimensional trait extractor; LDA is better when a trait has multiple facets pointing in different directions in activation space.** See §11 for full breakdown.

5. **The "best signal" layer selector was norm-confounded**, same artifact as the PCA PC1 finding from week 2. Mean signed projection scales with activation norm across layers. Replaced with `best-snr` (norm-invariant) and `best-cv` strategies, which both pick layer ~12 (matching LDA's choice).

6. **Read/write gap reproduces for mean-diff.** At 5% residual norm both LDA and MD-projected produce zero behavioral shift. At larger magnitudes (1.0× residual norm) both directions steer in the *wrong direction* on debiased BC: adding +honesty_direction makes the model pick the *low*-honesty response more often. Mean-diff has ~2× larger magnitude effect than LDA — closer to the execution subspace, just on the wrong side of it.

7. **Position bias is severe.** With A=high, B=low Llama-3.2 picks A only 6/24 (25%) on holdout. With A=low, B=high it picks A 0/24. Only 6/24 pairs are content-driven; the rest are position-locked to "B". The position-debiased baseline (averaging across orderings) is 62.5%, matching the prior 56% report. Any prior BC/Rottger result in our pipeline likely needs revisiting with this fix.

8. **Chat template matters for Llama specifically — not universally.** Llama-3.2-3B's bare-text baseline is 0.625; wrap in chat template and it's 0.938 (+31pt). On Gemma/Phi4/Qwen the same bump is only 2-4 points — well within noise. Prompt steering covers 0.00-0.98 on all four models; the Llama-specific gap means *Llama's* prior residual-stream evaluations need recalibration against the chat-template prior, but Gemma/Phi4/Qwen's prior bare-text numbers are approximately deployment-faithful. Qwen is already near ceiling (0.958) on bare text — suggests an SFT pipeline artifact rather than format-gated persona. Full table and discussion in §9.

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

## 9. Chat template and prompt-steering ceiling across four models

### 9.1 The Llama observation

While debugging the wrong-sign residual steering on Llama × H, we ran a naive prompt-steering baseline as a sanity check. Same 24 holdout pairs, position-debiased A/B averaging, but this time with the Llama chat template and optional system prompt:

| Condition | Llama high-trait BC rate |
|---|---|
| Bare text prompt (how all prior work in this project has been evaluated) | 0.625 |
| Chat template, no system prompt | 0.938 |
| Chat template, +H persona ("You are a person who is honest, fair, modest, and genuine. Respond as such a person would.") | 0.979 |
| Chat template, -H persona ("You are a person who is manipulative, greedy, pretentious, and self-serving. Respond as such a person would.") | 0.062 |
| Chat template, +H instruction ("Choose the response that a very honest, sincere, fair, modest person would give.") | 0.979 |
| Chat template, -H instruction ("Choose the response that a deceitful, greedy, pretentious, boastful person would give.") | 0.021 |
| Chat template, +H behavioral frame ("Be honest, modest, and fair in your choice, even when it would be costly to you.") | 0.979 |
| Chat template, -H behavioral frame ("Be willing to lie, exaggerate, or take unfair advantage if it serves your interest.") | 0.375 |

This looked alarming: the chat template alone added 31 points to the baseline, and prompt steering covered a 96-point range (0.02 to 0.98). Our MD-projected residual-stream steering at 1× residual norm had produced only 19 points in the wrong direction on the bare-text baseline. On this reading, every prior residual-stream evaluation in the project was measured on the wrong prior.

### 9.2 The cross-model check

Before accepting that conclusion we extended the test to Gemma, Phi4, Qwen (`scripts/prompt_steering_ceiling.py`). The Llama story does not fully generalize:

| Model | Bare | Chat default | Chat empty-sys | Chat +H persona | Chat -H persona | Chat−Bare bump | Full range |
|---|---|---|---|---|---|---|---|
| Llama | 0.625 | 0.938 | 0.938 | 0.979 | 0.062 | **+0.313** | 0.917 |
| Gemma | 0.854 | 0.896 | 0.896 | 1.000 | 0.000 | +0.042 | **1.000** |
| Phi4 | 0.875 | 0.896 | 0.875 | 0.979 | 0.104 | +0.021 | 0.875 |
| Qwen | 0.958 | 1.000 | 1.000 | 1.000 | 0.125 | +0.042 | 0.875 |

**The chat-template bump is a Llama phenomenon.** On Gemma, Phi4, and Qwen it's 2-4 points — within noise for a 24-pair holdout. Llama is the only model whose bare-text behavior is meaningfully unlike its deployed behavior on this task. Bare-text baselines run from 0.625 (Llama) up to 0.958 (Qwen, already near ceiling without any template at all).

### 9.3 What we did and didn't learn

**Findings that generalize across all four models:**
- **Prompt steering is highly effective.** All four models move from 0.98 (+persona) to 0.00-0.13 (-persona). Gemma flips fully to zero — most compliant. The full range spans 0.875 to 1.000 across models.
- **Empty-system overrides are mostly no-ops** behaviorally, even when they change the template structurally. Qwen strips "You are Qwen, a helpful assistant" under empty-system and still hits 1.000 on BC; that identity assertion isn't what's keeping Qwen at ceiling.
- **Persona framing, direct instruction, and behavioral frame all work, with persona and instruction clearly beating behavioral frame (tested on Llama).** Chat messages saying "you are X" have stronger effect than "be X in your choice."

**Findings specific to Llama:**
- Bare-text baseline (0.625) is ~31 points below chat-template baseline (0.938). For Llama specifically, our prior residual-stream evaluations were measured against a prior that doesn't match deployment.
- This probably means:
  - `scripts/optimize_steering.py` (week 4) reported 56% → 92% for Llama. The 92% is close to Llama's chat-template +persona ceiling (97.9%) but was measured with bare text, and the backprop δ was optimized specifically for the bare-text format. Re-evaluating that δ under chat-template inputs would be informative.
  - `scripts/validate_protocol.py` Rottger BC agreement on Llama (reported 80%) was on bare text; chat-template would likely be much higher.
  - §7 of this report (wrong-sign residual-stream steering on Llama) was on bare-text baseline 0.625; the room-to-move under chat-template baseline 0.938 is 5× smaller, which would shrink any observable residual-stream effect further.

**Findings specific to Gemma/Phi4/Qwen:**
- Our prior residual-stream work on these models is probably not badly miscalibrated by the bare-text choice — bare and chat differ by 2-4 points only. Cross-method matrix correlations and Rottger agreements for these models are likely roughly correct as reported.
- Qwen's 0.958 bare-text baseline is an independent curiosity — possibly an artifact of how its SFT was structured (e.g. distillation from other chat models, or SFT without chat markers, putting the "assistant" region of behavior closer to the base-model-like path). Not investigated further here.

### 9.4 The "instruct models need chat templates" framing narrows

The earlier framing — "instruct-tuned models should always be evaluated under their chat template, and bare-text evaluations are a methodological flaw" — was overstated. A narrower, correct version:

1. **For Llama specifically**, chat template is non-optional for faithful evaluation. The 31-point gap makes any prior residual-stream work on Llama subject to recalibration.
2. **For Gemma, Phi4, Qwen**, bare-text evaluations are approximately deployment-faithful on this task. They're still technically wrong — you should use the deployment format — but the measurement error is small.
3. **Model developers ship very different templates** and the choice is a real variable. Llama injects date metadata but no identity; Qwen injects identity ("You are Qwen, a helpful assistant") but no date; Gemma and Phi ship minimal templates with no injection. These decisions correlate with how much the model's behavior differs between formats — models with stronger identity injection (Qwen) appear to behave more template-independently, possibly because the identity has been trained into the weights rather than gated by the template.

### 9.5 Prompt-steering as a ceiling benchmark — still the right framing

Regardless of whether the chat-template bump is model-specific, the prompt-steering ceiling is informative for all residual-stream work:

- Per model: define `lo = min(−H persona)` and `hi = max(+H persona)`. Residual-stream method results get reported as a fraction of that range recovered.
- For our existing single-cell residual finding: baseline 0.625 → under +δ 0.500, i.e., we moved the baseline 0.125 in the wrong direction. Against Llama's ~0.917 prompt-steering range, we recovered about −14% of the range (negative because wrong-signed).
- Prompt steering works essentially for free: no activation extraction, no hook, no gradient. Residual-stream methods have to clear that bar to be worth the effort for behavioral steering. They may still be worth doing for *interpretability* (understanding what the model represents) rather than steering.

### 9.6 The "read/write gap" framing

Earlier framing: "Llama's behavior can't be steered at small residual-stream magnitudes via extracted directions." Still roughly correct for Llama. Narrower claim: Llama's behavior CAN be trivially steered via chat (the persona-steering lines in §9.1 change BC picks by 0.92). What's hard is steering via extracted residual-stream directions applied uniformly to a bare-text prompt. For Gemma/Phi4/Qwen the same likely holds, and the "behavioral steering" problem is even narrower because bare-text and chat-template baselines converge.

### 9.7 What to do going forward

1. **Report both bare and chat baselines for each model** in Phase B, flagging model-specific gaps. Don't replace bare with chat wholesale — for three of four models the gap is small, and bare is useful as a diagnostic.
2. **Recalibrate Llama-specific numbers** that rely on bare-text baselines (the optimize_steering 92%, the Llama Rottger 80%, §7 here). The other three models are probably fine.
3. **Prompt-steering ceiling as a standard reference column.** Any method's behavioral effect gets reported as fraction of the per-model prompt-steering range recovered.
4. **The Anthropic comparison caveat narrows.** Their emotion steering worked at 5% residual norm on Sonnet 4.5. Our LDA/MD steering doesn't work on Llama at that scale — we don't have clear evidence yet on Gemma/Phi4/Qwen. Phase B's chat-template cells will tell us if the Llama finding (wrong-sign residual) replicates on the others or is Llama-specific too.
5. **The persona-direction experiment from to_try.md §11 is now more targeted.** The Llama template-vs-bare contrast will be strong (31-point behavioral gap); the others will be weak (2-4 point gaps). If the persona-direction extraction on Llama produces a clean single direction that's a combination of high-trait directions, that's the cleanest evidence. On the other three models we'd expect smaller-norm persona directions, and potentially something more orthogonal to trait directions.

## 10. Phase A sweep results — LDA wins the grid

Grid: 2 formats (bare, chat) × 4 prefixes (high, low, absent, generic) × 3 neutrals (scenario_setups, shaggy_dog, factual) = 24 extraction cells on Llama-3.2-3B × Honesty-Humility. Each cell produces one LDA direction + one MD-raw + one MD-projected direction (72 directions). Driver: `scripts/phase_a_sweep.py`. Full table: `results/phase_a_sweep_meta-llama_Llama-3.2-3B-Instruct_H.csv`.

### 10.1 Holdout sign-correct (best method-agnostic result per cell, 24 pairs)

| format | prefix | LDA | best MD-projected | Δ (MD − LDA) |
|---|---|---|---|---|
| bare | high | **24/24** | 22/24 (scenario/factual) | −2 |
| bare | low | **23/24** | 22/24 (scenario) | −1 |
| bare | absent | 19/24 | **21/24** (scenario) | **+2** |
| bare | generic | **23/24** | 21/24 (scenario) | −2 |
| chat | high | **21/24** | 20/24 (factual) | −1 |
| chat | low | **24/24** | 21/24 (scenario) | −3 |
| chat | absent | **20/24** | 18/24 (shaggy) | −2 |
| chat | generic | **24/24** | 19/24 (shaggy/factual) | −5 |

**LDA wins 6 of 8 format × prefix cells; MD wins 1; 1 tie.** The cell where MD won (bare × absent × scenario_setups, Δ = +2) is exactly the cell we inspected in §6. Every other prefix or format choice produced an LDA-favorable result. The §6 "MD-projected beats LDA by 2 pairs" finding does not generalize.

### 10.2 Training vs holdout SNR

| Method | Train SNR range | Holdout SNR range |
|---|---|---|
| LDA | 1.21 – 1.51 | 1.05 – 1.78 |
| MD-raw | 2.46 – 2.60 | 0.75 – 1.14 |
| MD-projected | 2.47 – 2.71 | 0.75 – 1.32 |

Mean-diff has consistently ~2× LDA's training SNR and consistently ~1× or less LDA's holdout SNR. MD overfits to the training pairs; LDA's noisier per-pair signal generalizes better. The bias-variance intuition from earlier in this conversation had it reversed — LDA's whitening by the sample covariance is *not* dominantly introducing harmful variance here; it's producing a direction that tracks the true discriminant better than raw mean-diff does.

### 10.3 Prefix has outsized effect on LDA

Holding format = bare, neutral = scenario_setups, the LDA holdout sign-correct moves from:
- `absent` (no descriptor): 19/24
- `generic` ("Consider a person."): 23/24
- `low` descriptor: 23/24
- `high` descriptor: 24/24

A 5-pair swing on holdout, from just changing the user prefix. Under chat format the same span is 20/24 to 24/24. This means **the LDA direction is substantially reading the descriptor prefix, not the response**. The "best" LDA results in cells like `bare × high × scenario_setups = 24/24` partially reflect that we've told the model "this is a person who is honest" before asking it to read the response. Without that prefix, LDA drops sharply. MD-projected is more robust to prefix absence (absent cell is its one win).

This is a meaningful caveat for Phase B: if we report LDA performance in `prefix=high` cells as the method's holdout accuracy, we're measuring "LDA + descriptor cue" jointly. The trait-descriptor-absent cells are a cleaner measure of method ability, and in those cells both methods are comparable (MD 21/24, LDA 19/24 under bare; MD 18/24, LDA 20/24 under chat).

### 10.4 PC projection consistently helps MD

Across the 24 cells, MD-projected matches or beats MD-raw on holdout in 22/24 cells. Typical benefit: +2 to +4 sign-correct. The benefit varies by cell — most pronounced under bare text (MD-raw loses to MD-projected by 5+ pairs in some cells) and least pronounced under chat with `prefix=absent` or `prefix=generic` where PC projection picks k=1 and is essentially a no-op (§10.5).

### 10.5 Chat-template concentrates activation variance

Number of neutral PCs needed to explain 50% variance at MD-projected's best-SNR layer:

| | bare | chat |
|---|---|---|
| scenario_setups (300 texts) | 25 | **1 or 7** |
| shaggy_dog (50 texts) | 11 | 5–7 |
| factual (50 texts) | 10 | 7–8 |

Under chat template with scenario_setups as neutral, several cells have k=1 — a single direction accounts for 50% of neutral-text variance. This is the "chat template is an activation signal" hypothesis (to_try.md §11) showing up directly in the PC spectrum: the chat-template format concentrates variance along a small number of directions, plausibly the "Assistant persona" direction(s). Projecting those out sometimes leaves an MD direction nearly identical to MD-raw (k=1 projection is a single rank-1 subtraction), which is why MD-projected = MD-raw in those specific cells.

This is worth running as its own experiment per to_try.md §11 recommendation #3 (persona-direction extraction from user-turn template-vs-bare contrast). The data here is suggestive but not conclusive — the single concentrated PC may or may not be the persona vector itself.

### 10.6 Direction geometry

Cosine(LDA, MD-projected) across cells: 0.14 to 0.50 (mean ≈ 0.33). Both are unit vectors pointing in broadly the same direction but substantively apart (~70° typical separation). They classify the same trait by different linear readouts.

Cosine(MD-raw, MD-projected) ranges 0.22 to 1.00 depending on how much variance the neutral PCs captured. Under chat × absent with scenario_setups (k=1), cosine is near 1. Under bare × generic with scenario_setups (k=25), cosine is around 0.38.

### 10.7 What the sweep does and doesn't tell us

- **Tells us:** Across a reasonable hyperparameter grid on Llama × H, LDA is the better method for holdout classification. The single cell where MD-projected "won" was not representative.
- **Doesn't tell us:** Whether this holds on other models or other traits. Phase B is the generalization test.
- **Doesn't tell us:** Whether MD's higher training SNR and lower holdout SNR means "MD is mis-extracting the trait" vs "MD is extracting a different but valid feature." Need to inspect what each direction actually reads on specific pairs.
- **Doesn't tell us:** Whether the prefix-dependence is an LDA-specific flaw or a more general artifact of our extraction protocol. MD-projected is more robust to prefix, suggesting the latter but not conclusively.
- **Doesn't tell us:** Anything about steering. Phase A was classification-only per the revised target in §10 of the earlier plan. BC steering is still wrong-signed (§7) and prompt steering is the ceiling (§9); a separate position-specific steering test is the next steering experiment.

### 10.8 Recommendation for Phase B

Run the Phase B grid (4 models × 6 traits) with:
- **format = chat** as the primary (instruct-model-correct) configuration, with bare as a diagnostic companion.
- **prefix = generic** ("Consider a person.") as the primary. `absent` is cleaner conceptually but LDA underperforms there; `high`/`low` confound the measurement with the descriptor cue. `generic` is the compromise: it keeps the "Consider a person" framing without leaking the target direction.
- **neutral = scenario_setups** as the primary (300 texts, most stable PCs, slightly wins in most cells). Report shaggy_dog and factual as robustness checks.
- **Both LDA and MD-projected reported per cell.** MD-raw as a diagnostic.
- **Facet-stratified holdout breakdown** for all 6 traits (not just H).

Phase B would produce 4 models × 6 traits × 3 methods = 72 rows per neutral variant × 3 neutrals = 216 rows of grid data, plus per-facet breakdowns. On MPS, ~30-45 min.

## 11. Phase B — 4 models × 6 traits — LDA wins H, MD wins almost everything else

With Phase A's recommended configuration (generic prefix, scenario_setups neutral, both formats reported), swept 4 models × 6 traits × 2 formats × 3 methods = 144 rows. Driver: `scripts/phase_b_sweep.py`. Full table: `results/phase_b_sweep.csv`. Took 40 min on MPS.

### 11.1 Method comparison (48 cells = 4 models × 6 traits × 2 formats)

LDA vs MD-projected on holdout sign-correct:

|  | LDA wins | MD wins | Tie |
|---|---|---|---|
| **Overall (48 cells)** | 9 | **16** | 23 |
| Chat format (24 cells) | 4 | 9 | 11 |
| Bare format (24 cells) | 5 | 7 | 12 |

**Summary: MD-projected wins slightly more often overall.** This reverses Phase A's Llama × H conclusion — which turns out to have been a trait-specific finding, not a method-general one.

### 11.2 H is LDA's trait, the rest are MD's

Wins broken down by trait:

| Trait | LDA wins | MD-projected wins | Tie |
|---|---|---|---|
| **H** | **8/8** | 0/8 | 0/8 |
| E | 0 | 5 | 3 |
| X | 0 | 2 | 6 |
| A | 1 | 4 | 3 |
| C | 0 | 5 | 3 |
| O | 0 | 0 | 8 (all ceiling) |

**LDA wins every single H cell across 4 models × 2 formats.** MD wins or ties everywhere else except one bare × Qwen × A cell. Phase A's "LDA wins" was H being unusually hard for MD, not a general method fact.

### 11.3 The Modesty/Greed-Avoidance mechanism

Per-facet breakdown on H, aggregated across 4 models × 2 formats (48 pairs per facet):

| Method | Sincerity | Fairness | Greed-Avoidance | Modesty |
|---|---|---|---|---|
| LDA | 48/48 (100%) | 48/48 (100%) | 48/48 (100%) | 36/48 (75%) |
| MD-raw | 48/48 (100%) | 48/48 (100%) | 28/48 (58%) | 18/48 (38%) |
| MD-projected | 47/48 (98%) | 48/48 (100%) | 26/48 (54%) | 23/48 (48%) |

MD's failure is specifically on Greed-Avoidance and Modesty — exactly the facets the scenario audit ([rgb_reports/scenario_audit.md](scenario_audit.md)) flagged as showing within-trait heterogeneity. LDA accommodates the four facets via its whitening; MD's mean-difference takes the majority-aligned Sincerity + Fairness signal and leaves the minority facets mis-pointed in activation space.

This is the clearest direct evidence yet for the construct-heterogeneity concern the audit raised. The holdout was generated with audit-awareness (facet-stratified, with prompt guidance to avoid defensible-low responses), so the 24-pair holdout is a *fair* test of whether methods generalize across facets, not a replication of the training-set quirks. LDA passes; MD doesn't.

Interpretation: **MD is a good uni-dimensional trait extractor, not a good multi-facet one.** When a trait is relatively cohesive in representation (E, X, A, C, O — at least on our items), MD's per-class mean captures it cleanly. When a trait splits into facets that point in meaningfully different directions, MD averages them into mush and then fails at the edges.

### 11.4 Other patterns

**O is ceiling everywhere.** All 8 cells tied at 24/24. Openness is easy to separate with any method we have. Probably because the "high O — curious/creative" vs "low O — conventional/pragmatic" split maps onto a robust, mostly-unidimensional axis in these models.

**MD tends to pick deeper layers when best-SNR layer selection goes astray.** MD-raw picked the final layer (34) in 3 of 6 Gemma chat cells. This is the norm-growth artifact reappearing when PC projection doesn't provide enough signal — MD-raw has no norm-invariant selection mechanism, so when raw-direction SNR happens to peak at the final layer, that's what gets chosen. MD-projected mostly pulls back to middle-layer selections in those cases (layer 15-20), which is why projection makes a real difference for Gemma.

**Chat vs bare is mostly a wash except on Llama.** Per-model format comparisons:
- Llama: chat slightly favors MD-projected (Llama × E chat 24/24 vs bare 24/24 — tied on the numbers but MD cosine to LDA is lower under chat, suggesting more distinct read direction)
- Gemma/Phi4/Qwen: chat and bare produce very similar method ranks within trait
- Consistent with §9's finding that chat-template impact is Llama-specific

**Cosines between LDA and MD directions stay in the 0.1-0.45 range** across all cells. The two methods produce meaningfully different vectors that happen to classify the same trait, not rescaled versions of each other.

### 11.5 Implications for the project's recommendations

1. **Choice of method is trait-dependent.** For H specifically (and plausibly any trait with strong facet structure), LDA generalizes better. For E, X, A, C, O (at least in our models and items), MD-projected is as good or better. A project that cares about H and a project that cares about personality broadly would pick different methods.

2. **The facet-heterogeneity finding is methodologically important.** The scenario audit pointed at the problem on the training set; Phase B confirms it quantitatively on the held-out set. For anyone extracting trait directions from contrast pairs, MD's failure mode is now documented: **check your facet-level sign-correctness, not just trait-level.** Aggregate metrics may look healthy while a subfacet is being misclassified half the time.

3. **The Phase A "LDA wins" message was right locally, wrong globally.** It correctly identified that LDA beats MD on H; it incorrectly generalized from H to the method question. This is a cautionary tale about generalizing from a single trait × model to "which method is better."

4. **For Phase C or future work on steering**, MD-projected is probably the right residual-stream direction for E/X/A/C/O and LDA for H — if we bother with residual-stream steering at all given the §9 prompt-steering ceiling.

5. **For HEXACO measurement generally**, this suggests collecting facet-stratified data is important for method comparisons (we did this in the holdout; we should do it for everything). Trait-level metrics hide a lot.

## 12. What's done, what's next

Done in week 5:
- `scripts/audit_scenario_charge.py` — the audit harness (§1)
- `instruments/neutral_texts.json` — neutral corpora (3 of 4 variants populated, §2)
- `scripts/extract_meandiff_vectors.py` — mean-diff extraction with `--chat-template` and `--input-file` flags (§4, §9)
- `scripts/generate_holdout_pairs.py` — facet-stratified holdout generation via Claude API (§2)
- `instruments/contrast_pairs_holdout.json` — 144 holdout pairs (§2)
- `scripts/score_directions.py` — unified scoring harness, 3 of 4 dimensions implemented (classification + HEXACO convergent + BC steering; free-text deferred). Layer selectors fixed (`best-snr`, `best-cv`).
- Single-cell apples-to-apples comparison (bare × absent × scenario_setups) — §6. MD-projected wins on that cell.
- Wrong-sign residual-stream BC steering finding (§7) and position-bias finding (§8).
- Prompt-steering baseline revealing chat-template confound (§9).
- **Phase A sweep — 24 cells on Llama × H** (`scripts/phase_a_sweep.py`, `results/phase_a_sweep_*.csv`). Headline: LDA beats MD-projected on holdout in 6/8 format × prefix cells for Llama × H (§10).
- **4-model prompt-steering ceiling** (`scripts/prompt_steering_ceiling.py`, `results/prompt_ceiling.csv`). Headline: chat-template bump is Llama-specific, not universal (§9.2).
- **Phase B sweep — 4 models × 6 traits × 2 formats** (`scripts/phase_b_sweep.py`, `results/phase_b_sweep.{csv,json,txt}`). Headline: LDA wins H universally (8/8 cells); MD-projected wins or ties everywhere else (§11). Facet-level breakdown confirms MD's failure mode is Modesty + Greed-Avoidance (§11.3).

Next priorities:

1. **Position-specific steering test** as a separate experiment. Apply δ only at the final/answer token. Done at one defensible Phase B cell to characterize whether uniform steering is the source of §7's wrong-sign finding. Most informative cell: Llama × H (largest MD−LDA gap; most to learn about the recognition/execution relationship).
2. **Persona-direction extraction** (to_try.md §11 experiment #3). Contrast user-turn-in-template vs bare-user-turn activations on Llama specifically (where the template bump is 31 points); check whether the resulting direction is a linear combination of our high-trait directions. Can be done with existing tooling in <10 min. Phase B's facet-stratified directions give us a clean reference set for the linear-combination test.
3. **Position-bias re-audit** of `validate_protocol.py` Rottger numbers and `optimize_steering.py` baselines, combined with re-evaluation under chat template (§9). Prior Llama steering effect sizes likely shrink; other models approximately correct as reported.
4. **H-specific follow-ups.** The Modesty/Greed-Avoidance failure mode is a concrete finding worth deepening: (a) do the Modesty scenarios even show the same signal in the model's Likert self-report, or is Modesty-as-scenario just harder to represent? (b) does the facet failure transfer to other multi-facet traits in HEXACO not included here (Altruism, which loads cross-trait)? (c) would a Modesty-specific LDA fit (trained only on Modesty pairs) recover Modesty cleanly, suggesting the issue is single-direction-inadequacy rather than Modesty being harder per se?
5. **Report consolidation.** Week 5 is getting long. Probably worth spawning a cleaner "method choice for HEXACO trait extraction" writeup that captures just the headline findings: LDA vs MD on the facet-stratified comparison, chat template as Llama-specific, position bias. The rest (scenario audit, holdout construction, MPS-sandbox drama) can stay in the in-progress report or move to appendices.

Open questions worth keeping in view:

- **Why does the wrong-sign steering happen?** The most-likely explanation (uniform steering re-frames the whole context) is testable: position-specific steering should fix it. If it doesn't, something else is going on — possibly that the period-token recognition direction is *literally anti-aligned* with the answer-token execution direction.
- **What's the true cosine between mean-diff and the optimized δ from week 4?** Backprop δ achieved 92% steering; mean-diff achieves -19%. Their cosines tell us whether mean-diff is closer to the execution subspace than LDA was (a quantitative version of the "MD has 2× larger magnitude effect" finding).
- **Does the trait-state-behavior hierarchy bear out?** Anthropic's emotion vectors did steer at 5% norm. If we built scenarios that are explicitly emotional-state items (anger/calm/desperation), would *those* extracted directions steer like Anthropic's? That would isolate the construct-level explanation from the model-scale one.
- **Is LDA's advantage on H a generalizable "use LDA when facets disagree" rule?** We haven't directly tested traits like Altruism (loads cross-trait) or constructed adversarial multi-facet traits. Phase B's result for H is consistent with the theoretical expectation but the n of strong-facet-structure traits in our data is 1.

## Bibliographic note

The Sofroniew et al. (2026) entry was added to `CLAUDE.md`'s annotated bibliography (gitignored, local only). Worth promoting to a tracked report file at some point. The full method comparison and citation are recorded there.
