# Things to Try

## 1. Trait-conflict dilemma instrument

Build forced-choice scenarios where two positive HEXACO traits conflict (e.g., honesty vs kindness, conscientiousness vs openness). 15 trait pairs × ~5 scenarios each. This IS forced choice in the literature's sense — trait-vs-trait — unlike our single-trait binary-choice (BC) tests.

**Why:** Single-trait binary-choice hits ceiling (H/C/O all near 100% prosocial). RLHF prescribes the answer when only one trait is at stake. Trait conflicts force genuine trade-offs where models might actually differ.

**Prior art:** Ultima IV character creation (virtues pitted against each other). ACL 2025 "Decoding LLM Personality" confirms forced-choice discriminates LLM personalities better than Likert. Nobody has built a validated trait-conflict instrument for HEXACO — for humans or LLMs. Thurstonian IRT (Brown & Maydeu-Olivares) provides the scoring framework for recovering normative scores from ipsative forced-choice data.

**Status:** Not started. Needs scenario writing, pilot on 4 models, item analysis.

## 2. Cross-model direction transfer

Load model A's LDA trait directions, project model B's activations onto them. Do the directions generalize?

**Why:** If trait directions transfer, there may be a shared geometry of personality across architectures. If they don't, each model's "personality space" is idiosyncratic. Either result is interesting.

**Status:** Implemented in `validate_protocol.py --test transfer` but not yet run. The original workaround notes here assumed a 16 GB memory ceiling that no longer applies on the M5 Max machine — 2 small (3-4B) models fit simultaneously in bf16 with room to spare, so the direct path (load both, project on the fly) is viable. The save-directions-and-reload approach is still useful for cross-architecture comparisons involving 12B+ pairs if those come into scope.

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

## 5. SAE-based trait decomposition (next after larger-model baseline)

Use models with pre-built sparse autoencoders to see if personality-relevant features show up as interpretable SAE directions, rather than the LR/MD directions we've been extracting manually.

**Why:** SAEs decompose activations into monosemantic features. If personality traits correspond to identifiable SAE features, that's a cleaner story than "there's an LR direction in the residual stream." Also, people will ask about this — SAEs are the current interpretability fashion. And the week 6 finding that contrast-pair methods read the *expression axis* but not the *disposition center* motivates SAE decomposition of the center specifically: if there's a "high-H disposition" feature in the SAE dictionary, that's a clean interpretability result whether or not it's the same thing as the LR contrast direction.

**SAE coverage (updated 2026-04-24):**
- **GemmaScope 2** (DeepMind, https://deepmind.google/blog/gemma-scope-2-...): covers **all Gemma 3 sizes 270M–27B**, all layers, plus transcoders / skip-transcoders / cross-layer transcoders. Published on HuggingFace, Neuronpedia demo. Was previously thought to cover only 4B — confirmed to cover 4B, 12B, 27B, and smaller PT+IT variants.
- **andyrdt**: Llama 3.1 8B Instruct; Qwen 2.5 7B Instruct; GPT-OSS 20B (OpenAI's open MoE model, 3.6B active params).
- **No SAEs**: Phi-4 family, Llama 3.2 3B. These drop out of any SAE-based comparison cohort.

**Hardware:** 16 GB memory blocker is gone — M5 Max / 128 GB handles 7–8B (Llama/Qwen), 12B/27B (Gemma), and 20B (GPT-OSS) in bf16. No quantization needed, so SAE features trained on bf16 weights apply directly.

**Phasing:** Per feedback_conservative (one variable at a time), first replicate week-6 contrast-pair and week-3 cross-method results on a larger-model cohort (Gemma 3 12B + Llama 3.1 8B + Qwen 2.5 7B) as a baseline. Then bring SAEs in. Starting with GemmaScope 2 on Gemma 3 12B is the highest-coverage / lowest-friction entry point.

**Related:** Jiralerspong & Bricken (2026), "Cross-Architecture Model Diffing with Crosscoders" (arXiv 2602.11729). They use crosscoders (SAE variant that learns shared + model-specific features across architectures) to do unsupervised discovery of behavioral differences between models — found CCP-alignment features in Qwen, American-exceptionalism in Llama, copyright-refusal in GPT-OSS. More focused on specific ideological/policy behaviors than broad personality traits, but the cross-model diffing approach is exactly what our cross-model transfer test (item 2) is trying to do with LDA directions. Crosscoders might find shared personality features that LDA misses.

## 6. Scenario-based personality measurement in humans (literature check)

Our week 2 switch from descriptive statements to scenarios must have precedent in human psychometrics. Situational Judgment Tests (SJTs) are the obvious analogue, but there may be more directly personality-focused work.

**Why — and why this is urgent:** The 300 contrast-pair scenarios in `instruments/contrast_pairs.json` were written by Claude, not drawn from any validated instrument. Every scenario-based measure in the project (BC, RepE, Rottger) depends on these items. The BC ceiling effects could partly be a scenario quality problem (the "high" options may just sound nicer) rather than purely RLHF. And for the trait-conflict instrument, who writes the dilemmas is the entire measurement — the researcher degrees of freedom are maximal.

The encouraging sign: Likert↔RepE convergence on E (r=0.99) and A (r=0.70) is genuine convergent validity between independently-authored item sets (hexaco.org items vs Claude-generated scenarios), different methods, same trait structure. But this doesn't validate the BC scenarios specifically — RepE uses the scenarios for direction extraction, and the Likert comparison is indirect.

**What we need:** Human-validated scenario-based personality items would (a) remove the "the AI wrote its own test" problem, (b) provide item-writing principles for the trait-conflict instrument, (c) give us a comparison point for scenario quality.

**Things to look for:** Conditional reasoning tests (James 1998), SJTs with personality scoring keys, implicit personality measurement via behavioral scenarios. Also check whether Okada et al.'s GFC items are descriptive statements or scenarios — their desirability-matching approach might combine with scenario framing.

## 7. Bigger / better models

Findings so far are at 3–4B. The assistant-shape collapse, the contrast-vs-disposition split, and the read/write gap might shift with scale.

**Status as of 2026-04-24:** Machine is now M5 Max / 128 GB — the "limited memory" constraint that shelved this is gone. Local bf16 is viable up through Gemma 3 27B and GPT-OSS 20B. The agreed Phase-1 cohort is the matched-scale upgrade: **Gemma 3 12B + Llama 3.1 8B + Qwen 2.5 7B**, all of which have SAE coverage (see §5). Gemma 3 27B is a no-cost scale anchor on top of that if wanted. Keep the original 3–4B cohort around for small-vs-large comparisons; Phi-4-mini stays as a no-SAE control through Phase 1.

**Still-useful alternatives** (for models beyond local reach, or for comparison across proprietary APIs):
- API-based models for logprob surveys (OpenAI, Anthropic) — no hidden-state access but Likert/BC still work.
- Cloud GPU for one-shot RepE extraction on models above the local ceiling (e.g., Llama 3.1 70B).

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

## 11. Chat template as assistant-persona activation signal

Observed in week 5 (report_week5_meandiff.md §9) while sanity-checking prompt steering: on Llama-3.2-3B × H, the debiased high-trait BC pick rate is 62.5% with a bare-text prompt but 93.8% when the same prompt is wrapped in the Llama chat template (empty system message, user turn). That's a 31-point shift from the template alone — before any "+H persona" system prompt is added. The template itself is pulling the model toward high-H behavior.

This matches Lu et al. (the default Assistant persona is an amalgamation of character archetypes from pretraining; post-training steers toward a specific region rather than constructing the persona from scratch). If true, the chat template is the *activation signal* for that region — and running inference outside the template samples the model outside the Assistant region.

**Why it matters for this project.**
- Our week 1 "assistant shape" finding (all models low-N high-A/C; E-C r=0.93 in Big Five) was measured under self-report framing that presumably triggers the persona (IPIP items are in natural first-person-descriptive format). How much of the collapsed factor structure is RLHF vs the chat template being active during measurement? If we ran IPIP-300 on bare-text prompts, does the collapse go away or loosen?
- Our RepE directions are extracted from bare-text contrast pairs. If the persona isn't active during extraction, the directions may be measuring something closer to "trait-in-base-model" than "trait-in-deployed-assistant." That's *either* a bug (we should extract under the template) *or* a feature (we've been measuring a less-polluted signal this whole time). Worth characterizing directly.
- Okada et al.'s SDR (socially desirable responding) work found a quantifiable gap between honest-instruction and fake-good-instruction BC behavior. The bare-text vs chat-template gap might be an untested third axis: a passive SDR signal from the template alone, without any explicit fake-good instruction.

**Experiments this suggests.**
1. **Trait × format matrix.** For each of the 4 models × 6 HEXACO traits, measure position-debiased BC rate in bare text and chat template conditions. Is the template's pull equal across traits, or concentrated on the HHH-adjacent ones (H, A, C)?
2. **Likert-survey replication in bare text.** Run IPIP-300 or HEXACO-100 with bare-text prompting (no chat template). Compare trait-level scores and cross-trait correlation matrices to our existing chat-template results. Prediction: the collapsed factor structure loosens; trait correlations move toward human-normative values.
3. **Persona-direction extraction.** Contrast pair: same user turn with chat template vs without. Extract a direction from the residual differences. Is there a single "chat template / assistant persona" vector, or several? How does it relate to our H/A/C directions? (If the persona direction is essentially a linear combination of high-H, high-A, high-C, that's direct evidence for rank-1 collapse being persona-driven.)
4. **Steering by chat-template removal.** If we extract a persona vector, *subtracting* it from chat-template activations should pull the model back toward base-model behavior. Simpler test of "is the persona a single direction?" than the analogous test on individual traits.
5. **Does it generalize across model families?** ~~Llama may be unusual.~~ **Update 2026-04-12 from report_week5_meandiff.md §9.2:** Ran the prompt-steering ceiling on all 4 models. The template-induced bump is Llama-specific (+31pt), not universal. Gemma/Phi4/Qwen show only 2-4pt bumps. Revised hypothesis: the chat template gates a "deployment-mode" shift on some models (notably Llama) but not others. Models whose post-training baked the assistant persona into the weights (Qwen in particular, at 0.958 bare-text baseline) may not need the template to activate it. Implication for experiment #3 (persona-direction extraction): the contrast is strongest on Llama and weakest on Qwen. Probably best to extract on Llama first; any positive result on Qwen would be a different phenomenon.

**Also note the organizational-choice variation in default templates.** Llama injects date metadata but no identity; Qwen injects identity ("You are Qwen, a helpful assistant") but no date; Gemma and Phi ship minimal templates. These choices correlate with (but don't straightforwardly explain) the bump sizes: Llama has big bump despite weak template injection; Qwen has small bump despite strong injection. The template alone isn't what drives the persona — it's whatever the template is *cueing* the post-training weights to activate, which varies per training pipeline.

**Potentially publishable.** "The assistant persona is gated by the chat template" is a crisp, testable claim that's independent of our main personality-measurement agenda. It's now known to be model-specific, which is still an interesting finding: "how much of each instruct-tuned model's trait behavior is template-gated vs weight-baked" is a measurement methodology point that matters for anyone doing evaluation across models. Llama on bare text ≠ Llama deployed; Qwen on bare text ≈ Qwen deployed. Both types of instruct-tuning exist and people doing comparisons should know which they're looking at.

## 12. Rebuild Week 3 cross-method correlation matrix with LR probe ✓ DONE 2026-04-18

**Status:** Done. `scripts/cross_method_matrix.py` now takes `--probe {lr,lda}` (default lr). `rgb_reports/cross_method_correlations.md` updated with LR-primary numbers and LDA kept for side-by-side comparison. LR-C-stability also verified (`scripts/lr_c_stability.py`): directions stable at cos ≥0.92 across C∈{0.1,1,10,100}, ≥0.99 for adjacent C's.

**Headline result:** Seven of eight RepE-involving correlations drop in magnitude under LR (by 0.05–0.08). Overall Likert↔RepE collapses from r≈0.17 to r≈0.09 — the three-construct dissociation is stronger than Week 3 originally reported. One exception: X's BC-prop↔RepE *rises* from 0.17 to 0.40, suggesting LDA was rotating away from (not toward) the behaviorally-aligned axis for X. The Agreeableness consensus and Emotionality Likert↔RepE convergence both survive the swap.

## 13. Refactor: shared `vector_from_activations` module

Multiple scripts re-implement the same pattern: load cached pair activations, pick a layer, compute {LDA, LR, MD-raw, MD-projected} direction, normalize. Currently spread across `phase_b_sweep.py`, `probes_same_layer.py`, `compare_probe_steering.py`, `facet_cluster.py`, `facet_viz.py`, `within_trait_variance.py`, `lr_c_stability.py`, `cross_method_matrix.py`, `generate_training_pairs.py` (doesn't extract but loads the same caches).

A small `scripts/vector_methods.py` module with clearly-commented functions — `lda_direction(diffs, layer)`, `lr_direction(diffs, layer, C=1.0)`, `md_raw(ph, pl, layer)`, `md_projected(ph, pl, neutral, layer, pc_var=0.5)`, `normalize`, `cv_best_layer` — would:

1. Eliminate the ~5 copies of `cv_best_layer`, `unit`, and antipodal-trick boilerplate currently scattered
2. Provide a canonical place to document the Week 6 findings inline (e.g., why LR uses antipodal `X = [d/2, -d/2]` rather than raw `[h; l]`; why LDA has the Σ⁻¹-noise pathology; why MD-projected's neutral-PC subtraction is the robust alternative)
3. Make it harder to accidentally diverge method implementations across analyses

Low risk, mechanical. Not urgent — nothing's broken — but would pay down some of the copy-paste debt accumulated during the Week 6 exploration and make future probe experiments (e.g., shrinkage LDA, elastic-net LR, mean-diff with different neutral sets) drop-in replacements.

## 14. Situational judgment tests / economic games

Mentioned in the week 1 report but never pursued. Dictator, Trust, and Ultimatum games have documented Big Five correlations in human samples (Agreeableness r = .25-.37). Completely different measurement modality — bypasses self-report framing.

**Advantage:** No Likert scale, no personality vocabulary, no "I am an AI" refusal trigger. Pure behavioral preference over resource allocation.

## 15. Bare-text vs chat-template Likert (corrected framing)

**Original framing (now reversed).** During the Ollama → HF port on 2026-04-24 I assumed the old `/api/generate` path was bare-text, set the new HF helper to bare-text, and bookmarked "what if we ran with chat template" here. Wrong direction. Re-reading `ollama_generate(..., raw=False)`: the default `raw=False` causes Ollama to apply the model's chat template server-side. So weeks 1–6 Likert numbers were chat-template numbers all along — except for Qwen3, which used `raw=True` with explicit `<|im_start|>...<|no_think|>...<|im_end|>` wrapping (still chat-template, just hand-written).

The first run on Qwen 2.5 7B (bare-text via the HF helper, before the fix) accidentally became the bare-text-Likert ablation:
- Median variant EV spread across 300 items: 1.88 (vs 1.00 in old qwen3_8b chat-template data).
- Variant v3 (terse, ends in "\n") collapsed to EV ≈ 1.2 across nearly every item — model saturating on "1".
- ICC(2,1) = -0.054 overall (vs +0.54 in old qwen3_8b). Negative ICC = format moves answers more than items do.

So bare-text Likert is genuinely degenerate, at least on the v3 prompt and at least on Qwen 2.5 7B. The chat template was doing real work in the prior pipeline.

**Bookmark, redirected.** The interesting direction is no longer "what if we *added* chat template" (it was always there); it's "did the chat template *interaction with v3* hide a format-fragility we should attend to" — i.e., is the v3 collapse a property of the bare prompt or a property of weak alignment that the chat template was masking? Useful experiment for understanding what the chat template actually contributes to robustness (vs what it adds in trait expression, which §11 already tackles for BC).

**Status (post-fix).** `hf_logprobs.likert_distribution(use_chat_template=True)` is now the default — restoring weeks 1–6 parity. The v3 result is preserved on disk in the first Qwen 2.5 7B run; numbers go forward with chat template on. Bare-text remains accessible via `use_chat_template=False` if we want to instrument the §11 + this question with one knob.

## 16. Cross-domain stimulus test of the high-bandwidth-preservation finding

W7 §8.4–§8.5 found that subtle similarity structure in personality-relevant texts (contrast pairs, HEXACO Likert items, Goldberg adjective markers) is preserved through transformer forward passes with cross-architecture cosine-matrix fidelity r=0.93–0.99 within stimulus type. Open question: is this a property of *transformer architectures* (true regardless of domain), or specifically of *personality-related concepts* (which post-training shapes carefully)?

**Test:** Replicate the §8.5 single-stimulus protocol (one short phrase per concept, mean(high-pole) − mean(low-pole) at ~2/3-depth, neutral-PC-projected, chat-template-wrapped) on three contrast-domain item sets that don't touch personality. Compute cross-model cosine-matrix correlation; compare to the 0.93–0.99 range from personality stimuli.

**Suggested domains** (rgb 2026-04-24):

- **Emotions** — directly comparable to Sofroniew et al. (2026); 30+ emotion concepts with valence/arousal-paired antonyms (joyful/morose, energetic/sluggish, etc.). The Anthropic emotion-vector list is one source.
- **Shorebirds** — taxonomic biological knowledge with internal phylogenetic structure. ~30 species/genera with paired close-relative vs distant-relative comparisons. Tests whether biological taxonomy recovers cleanly.
- **Forms of transportation** — functional/practical categories with orthogonal sub-categorization (land/water/air, motorized/manual, public/private). Tests whether functional categories pack more orthogonally than psychological ones.

**Predictions:**
- **Emotions** likely densely entangled (similar to personality). Direct comparison to Sofroniew's emotion-vector geometry possible.
- **Shorebirds** mid: within-clade entangled, across-clade more orthogonal. Phylogenetic structure should show.
- **Transportation** more orthogonal: functional categories with distinct feature profiles. If we still see cross-architecture r=0.95+, that's evidence for "transformers preserve subtle structure regardless of domain." If transportation cross-architecture r drops to 0.7, that's evidence personality stimuli are special.

**Theoretical interpretation (rgb 2026-04-24):** Dense cosine entanglement on personality concepts (E↔O = +0.69, A↔O = +0.64 on Goldberg markers) is in tension with strict superposition predictions of quasi-orthogonal features at the representation-extraction layer. It's consistent with the model treating these concepts as *associatively related* — useful for correlation-based inferences (the assistant being conscientious tends to also be agreeable; both are "good qualities"), bad for precise symbolic reasoning (cannot deconfound E from O without an explicit disentangling operation). Cross-domain comparison directly tests whether this associative-density is concept-class specific (personality is a "valence cluster," other domains aren't) or a general property of how transformers represent semantically-rich concept categories.

**Connection to Phase 2 SAE work:** SAEs find sparse feature directions that are themselves quasi-orthogonal by construction. Our finding doesn't refute that SAE features exist at lower layers — it refutes that *trait-direction-style* representations at ~2/3-depth are quasi-orthogonal in the way superposition predicts. SAE-decomposed features may show much cleaner separation; the trait directions we extract are linear projections that aggregate across many SAE features, which lossy-compresses orthogonality.

**Status:** Not started. Single-domain run takes ~3 min on cached cohort once stimulus list is built. Stimulus-list assembly is the main cost (~1 hour per domain to write paired items).

## 17. Cleanup: unify small-cohort Qwen + switch RepE to chat-template + refresh cross-method matrix

Three interlocking cleanup items flagged 2026-04-24, mostly waiting on the small-cohort precache to land. None is a research direction; they're confound-cleanup before the W7 numbers solidify.

**(a) Unify the small-cohort Qwen.** Across the W7 cross-method matrix the "qwen" small-cohort entry currently mixes models: Likert is from qwen3-8B (W1 Ollama runs), RepE is from Qwen 2.5 3B (legacy `results/repe/Qwen_Qwen2.5-3B-Instruct_*_directions.pt`), BC is from qwen3-8B (W1 Ollama). Cross-family confound carried since W3. Once Qwen 2.5 3B is precached: re-run `run_hexaco.py` and `run_ipip300.py` on Qwen 2.5 3B (HF, chat-template, --variants), and `score_bc.py` for it. Update `cross_method_matrix.py` MODELS["qwen"]["likert"] and bc_key. The resulting "qwen" entry is then consistent within Qwen 2.5 3B across all three measures.

**(b) Switch RepE to chat-template throughout the cross-method matrix.** Currently legacy `results/repe/<tag>_<trait>_directions.pt` files are bare-text per W3 protocol (this was deliberate to match the small-cohort original). With Likert and BC both running through chat-template now (W7 §1.3 fix), the matrix mixes formats: chat-Likert + chat-BC + bare-RepE. The W7 §6.2 BC↔RepE sign flip in the larger cohort might be a partial format artifact. Larger-cohort fix is one script call away — `phase_b_cache/<tag>_<trait>_chat_pairs.pt` is already there, just re-run `repe_legacy_from_cache.py --format chat` to overwrite (or use a different output path to compare). Small cohort needs the cache regenerated with chat format once models are precached: re-run `phase_b_sweep.py --models Llama Gemma Phi4 Qwen --formats chat` (it'll lazily regenerate neutral + pair caches and emit method-comparison numbers in chat format alongside).

**(c) Refresh the cross-method matrix with (a) and (b) applied.** Re-run `cross_method_matrix.py --probe lr` after both fixes. Compare to the W7 §6.2 numbers. The interesting question: does the BC↔RepE sign flip on Llama 8B / Qwen 7B (−0.73, −0.80) shrink to small-cohort levels (≈ +0.3) when RepE is also chat-template? That would say "format mismatch was driving the flip." Or does the flip persist? — that would say "scale really has changed the read-write relationship." Either resolution is a finding worth reporting in W8.

**Status:** waiting on `bash b0eblvqzb` (small-cohort precache, ~hours). Larger-cohort chat-template RepE check (subset of (b) + partial (c)) could be done immediately, but more useful to bundle with (a) and (b)-small for a single clean comparison.
