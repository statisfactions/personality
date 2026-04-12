# Things to Try

## 1. Trait-conflict dilemma instrument

Build forced-choice scenarios where two positive HEXACO traits conflict (e.g., honesty vs kindness, conscientiousness vs openness). 15 trait pairs × ~5 scenarios each. This IS forced choice in the literature's sense — trait-vs-trait — unlike our single-trait binary-choice (BC) tests.

**Why:** Single-trait binary-choice hits ceiling (H/C/O all near 100% prosocial). RLHF prescribes the answer when only one trait is at stake. Trait conflicts force genuine trade-offs where models might actually differ.

**Prior art:** Ultima IV character creation (virtues pitted against each other). ACL 2025 "Decoding LLM Personality" confirms forced-choice discriminates LLM personalities better than Likert. Nobody has built a validated trait-conflict instrument for HEXACO — for humans or LLMs. Thurstonian IRT (Brown & Maydeu-Olivares) provides the scoring framework for recovering normative scores from ipsative forced-choice data.

**Status:** Not started. Needs scenario writing, pilot on 4 models, item analysis.

## 2. Cross-model direction transfer

Load model A's LDA trait directions, project model B's activations onto them. Do the directions generalize?

**Why:** If trait directions transfer, there may be a shared geometry of personality across architectures. If they don't, each model's "personality space" is idiosyncratic. Either result is interesting.

**Status:** Implemented in `validate_protocol.py --test transfer` but not yet run to completion (memory constraints — need to load 2 models simultaneously).

**Workaround ideas:** Extract and save directions, then load one model at a time and project onto saved directions. Or use smaller batch sizes.

## 3. Read/write dissociation investigation

LDA directions classify with 100% accuracy but don't causally steer generation. Why?

**Hypotheses:**
- **Redundancy:** Many parallel mechanisms encode personality. Pushing one linear direction doesn't overcome the others.
- **Scale mismatch:** Personality component is 0.15% of activation norm; natural-scale steering is invisible, larger scales are degenerate.
- **Asymmetry:** Negative steering (toward dishonest) works better than positive — model is already near the "honest" ceiling from RLHF.
- **Reading ≠ writing:** The encoding direction may not be the direction that influences downstream computation. (CARE paper warns about this specifically.)

**Things to try:**
- Activation patching / causal tracing to find directions that are actually causal for output
- Steer on scenario tokens (not just last token) — multi-position intervention
- Clamp rather than add: project out the trait component and replace with a fixed value
- Compare steering effectiveness at different layers

## 4. Backprop-optimized steering vectors

If LDA directions are read-only, we can *construct* a steerable direction via backprop: optimize a perturbation vector δ in the residual stream that maximizes some personality-relevant output (e.g., log-odds of the high-trait binary-choice option), subject to a norm constraint.

**Why:** This tests whether the read/write dissociation is fundamental (no linear perturbation at this scale can steer) or just a failure of the LDA direction specifically. If backprop finds a working vector, the question becomes why it differs from LDA. If it can't, that's a strong negative result — personality behavior isn't linearly steerable in these models at natural scales.

**Practical:** Requires gradients through the model, so HuggingFace only (not Ollama). Memory may be tight — could use gradient checkpointing or optimize at a single layer.

## 5. SAE-based trait decomposition (shelved — revisit after Gemma pilot)

Use models with pre-built sparse autoencoders to see if personality-relevant features show up as interpretable SAE directions, rather than the LDA directions we've been extracting manually.

**Why:** SAEs decompose activations into monosemantic features. If personality traits correspond to identifiable SAE features, that's a cleaner story than "there's an LDA direction in the residual stream." Also, people will ask about this — SAEs are the current interpretability fashion.

**SAE coverage (researched 2026-04-04):** GemmaScope 2 has gold-standard SAEs for Gemma 3 4B (all layers, all hooks, PT+IT, SAE Lens native). andyrdt has SAEs for Llama 3.1 8B Instruct and Qwen 2.5 7B Instruct. No SAEs exist for Phi4 or Llama 3.2 3B. GPT-OSS 20B (OpenAI's open MoE model, 3.6B active params) has andyrdt SAEs too.

**Blocker:** The models with SAE coverage (7-20B) don't fit in bf16 on 16 GB Apple Silicon, and SAE features trained on bf16 weights almost certainly won't transfer to quantized models. Gemma 3 4B is the only model that fits AND has SAEs. Worth doing a Gemma-only SAE pilot first to see if personality-relevant features even show up before solving the hardware problem for cross-architecture comparison.

**Related:** Jiralerspong & Bricken (2026), "Cross-Architecture Model Diffing with Crosscoders" (arXiv 2602.11729). They use crosscoders (SAE variant that learns shared + model-specific features across architectures) to do unsupervised discovery of behavioral differences between models — found CCP-alignment features in Qwen, American-exceptionalism in Llama, copyright-refusal in GPT-OSS. More focused on specific ideological/policy behaviors than broad personality traits, but the cross-model diffing approach is exactly what our cross-model transfer test (item 2) is trying to do with LDA directions. Crosscoders might find shared personality features that LDA misses.

## 6. Scenario-based personality measurement in humans (literature check)

Our week 2 switch from descriptive statements to scenarios must have precedent in human psychometrics. Situational Judgment Tests (SJTs) are the obvious analogue, but there may be more directly personality-focused work.

**Why — and why this is urgent:** The 300 contrast-pair scenarios in `instruments/contrast_pairs.json` were written by Claude, not drawn from any validated instrument. Every scenario-based measure in the project (BC, RepE, Rottger) depends on these items. The BC ceiling effects could partly be a scenario quality problem (the "high" options may just sound nicer) rather than purely RLHF. And for the trait-conflict instrument, who writes the dilemmas is the entire measurement — the researcher degrees of freedom are maximal.

The encouraging sign: Likert↔RepE convergence on E (r=0.99) and A (r=0.70) is genuine convergent validity between independently-authored item sets (hexaco.org items vs Claude-generated scenarios), different methods, same trait structure. But this doesn't validate the BC scenarios specifically — RepE uses the scenarios for direction extraction, and the Likert comparison is indirect.

**What we need:** Human-validated scenario-based personality items would (a) remove the "the AI wrote its own test" problem, (b) provide item-writing principles for the trait-conflict instrument, (c) give us a comparison point for scenario quality.

**Things to look for:** Conditional reasoning tests (James 1998), SJTs with personality scoring keys, implicit personality measurement via behavioral scenarios. Also check whether Okada et al.'s GFC items are descriptive statements or scenarios — their desirability-matching approach might combine with scenario framing.

## 7. Bigger / better models

Current models are all small (3-8B). The findings might not generalize upward. The assistant-shape collapse might be weaker or stronger in larger models. The read/write gap might close with scale.

**Practical constraint:** Apple Silicon Mac with limited memory. Could try:
- Quantized versions of larger models via Ollama (e.g., Llama 3.1 8B, Gemma 2 9B)
- API-based models for logprob surveys (OpenAI, Anthropic) — no hidden-state access but Likert/BC still work
- Cloud GPU for one-shot RepE extraction on larger models

## 8. Base model comparison

All current measurements are on instruction-tuned models. Running the same battery on base models would show how much of the "assistant shape" is RLHF vs. pretraining.

**Expectation:** Base models should show less trait compression, higher entropy, weaker assistant shape. But they may also be less coherent (Serapio-Garcia found base models produce near-random psychometric responses).

**Practical:** Ollama supports some base models. HuggingFace has base checkpoints for all 4 model families.

## 9. Entropy as a signal, not just noise

Llama's near-uniform distributions (entropy ~1.4) might not be "uncertainty" — it might be a different response strategy. Gemma's peaked distributions might reflect overconfidence rather than genuine certainty.

**Things to try:**
- Entropy profiles per item: which items do all models agree on vs. disagree?
- Entropy × trait interaction: are some traits measured more confidently than others?
- Entropy as a predictor: does low entropy on a Likert item predict higher BC/free-text consistency on the same scenario?

## 10. Facet-level analysis

HEXACO-100 has 4 items per facet (4 facets per trait). We collected facet assignments but haven't analyzed at that granularity.

**Why:** "Honesty-Humility" is broad. The model might score high on sincerity but low on greed-avoidance. Facet-level profiles could reveal more interesting between-model differences than trait-level scores.

**Also:** The RepE contrast pairs showed facet structure in Qwen (material vs. social honesty clusters). Worth checking if Likert facet scores show the same pattern.

## 11. Situational judgment tests / economic games

Mentioned in the week 1 report but never pursued. Dictator, Trust, and Ultimatum games have documented Big Five correlations in human samples (Agreeableness r = .25-.37). Completely different measurement modality — bypasses self-report framing.

**Advantage:** No Likert scale, no personality vocabulary, no "I am an AI" refusal trigger. Pure behavioral preference over resource allocation.
