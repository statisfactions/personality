# Week 5 Discussion Summary

*Condensed writeup of the mean-diff-vs-LDA investigation for the reading group. Longer narrative with code pointers and intermediate reversals in [report_week5_meandiff.md](report_week5_meandiff.md).*

## The question

Sofroniew et al. (2026, "Emotion Concepts and their Function in a LLM") extracted 171 emotion vectors from Claude Sonnet 4.5 using a method distinct from standard RepE:
- **Stimuli**: LLM-generated short stories, one emotion per story, 100 topics × 12 stories per emotion.
- **Extraction**: residual-stream activations averaged over tokens 50+, then per-class mean minus grand mean (i.e., mean-difference, not PCA or LDA).
- **Denoising**: project out top PCs of activations on emotionally-neutral text (enough to explain 50% of variance).
- **Steering**: at 5% of residual norm, works — amplifying a "desperation" vector drove blackmail behavior from 22% to 72%.

We tested whether this extraction method, applied to our HEXACO personality contrast pairs, produces better directions than our existing LDA pipeline. Four small (3-4B) instruct models: Llama-3.2, Gemma-3, Phi-4-mini, Qwen-2.5.

## The short answer

On 48 cells (4 models × 6 traits × 2 formats, classification on a 144-pair facet-stratified holdout):

| | LDA wins | MD-projected wins | Tie |
|---|---|---|---|
| Overall | 9 | 16 | 23 |

**But trait-dependent:**

| Trait | LDA | MD-proj | Tie |
|---|---|---|---|
| **H** | **8/8** | 0 | 0 |
| E | 0 | 5 | 3 |
| X | 0 | 2 | 6 |
| A | 1 | 4 | 3 |
| C | 0 | 5 | 3 |
| O | 0 | 0 | 8 (ceiling) |

MD-projected is a better trait extractor everywhere **except Honesty-Humility**, where LDA beats it in all 8 cells. Why H?

## Why H — the facet-level mechanism

HEXACO Honesty-Humility has four facets: Sincerity, Fairness, Greed-Avoidance, Modesty. In the holdout, 6 pairs per facet per trait. Aggregated across 4 models × 2 formats (48 pairs per facet):

| Method | Sincerity | Fairness | Greed-Avoidance | Modesty |
|---|---|---|---|---|
| LDA | 100% | 100% | 100% | 75% |
| MD-projected | 98% | 100% | **54%** | **48%** |

**MD fails specifically on the facets where the trait's uni-dimensional assumption breaks down.** A scenario audit done earlier ([rgb_reports/scenario_audit.md](scenario_audit.md)) found that 24% of pairs in the original H training set had *reversed* representation signal — concentrated in Modesty items. Plausible mechanism: Modesty and Greed-Avoidance point in somewhat different directions in activation space than Sincerity/Fairness. LDA's whitening finds a direction that accommodates all four facets. MD's mean-difference is dominated by the majority-aligned Sincerity+Fairness pairs and leaves Modesty/Greed-Avoidance mis-pointed.

**This is probably a general rule:** MD is a good uni-dimensional trait extractor; LDA is better when a trait has multiple facets pointing in distinct directions.

**An open possibility (raised in discussion):** Greed-Avoidance may not be part of H in LLMs at all — it may be its own dimension. HEXACO places G-A under H based on human lexical factor analyses where G-A loads at ~.4-.6 on the H factor (notable but the weakest of the four). For an LLM, the RLHF signals for "don't be materialistic" and "don't lie" are plausibly separable, and our G-A contrast pairs are heavily consumer-preference framed (buy a Ferrari or don't) which may not share much representation with sincerity/modesty. Testable: extract an MD direction from G-A pairs only and another from Sincerity+Fairness+Modesty-only. Cosine near zero → G-A is a distinct dimension for LLMs. Low magnitude in same direction → it's H-aligned but weakly, as in humans. Incoherent G-A-only direction → our G-A pairs are bad rather than G-A being a different construct. Cheap to run; queued for after the session (see "what we still don't know").

## Two measurement confounds we didn't expect

Both discovered in the course of testing residual-stream steering, both worth flagging as methodology notes.

### Position bias (Llama, n=24 holdout)

With A = high-trait response, B = low-trait response: model picks A 6/24 = 25%. Swap to A = low, B = high: picks A 0/24 = 0%. Only 6/24 pairs are content-driven; the rest are position-locked to "B." Position-debiased rate (averaging across orderings): 62.5%.

Any pipeline scoring a BC pick with a single A/B ordering is measuring position × content, not content alone. Our prior Rottger agreement numbers, optimize_steering baselines, and the cross-method correlation matrix all use single-ordering; all need revisit with proper debiasing. The effect may be comparable on other models (didn't confirm).

### Chat-template confound (Llama-specific)

Prompt steering ceiling across the four models, 24 holdout pairs, position-debiased:

| Model | Bare text | Chat default | +H persona | -H persona | Chat-bare bump |
|---|---|---|---|---|---|
| **Llama** | **0.625** | **0.938** | 0.979 | 0.062 | **+0.313** |
| Gemma | 0.854 | 0.896 | 1.000 | 0.000 | +0.042 |
| Phi4 | 0.875 | 0.896 | 0.979 | 0.104 | +0.021 |
| Qwen | 0.958 | 1.000 | 1.000 | 0.125 | +0.042 |

Llama's bare-text baseline is ~30 points below its chat-template baseline. On the other three, bare-text and chat-template are within noise of each other. This was initially alarming (we'd been running residual-stream evaluations on bare text) but turned out to be Llama-specific: our prior work on Gemma/Phi4/Qwen is approximately deployment-faithful, while **Llama-specific numbers need recalibration against the chat-template prior.**

Qwen being near-ceiling on bare text (0.958) is its own curiosity — plausibly an SFT pipeline artifact where the assistant persona was trained into the weights rather than gated by the template.

**Prompt steering is a ~96-point usable range on all four models.** This is the ceiling benchmark for any residual-stream steering method. Worth keeping in view.

## The steering story (negative, interesting)

We spent considerable effort checking whether MD-extracted directions steer better than LDA directions, since that's the headline Sofroniew et al. result. The 48-cell classification comparison above is the read side. The steering side:

At 5% of residual norm (Sofroniew's successful scale), both LDA and MD-projected produce **zero measurable behavioral shift** on Llama × H. At 1× residual norm they do produce a shift, but in the *wrong direction*: adding +honesty → model picks dishonest response more often. Likely mechanism: uniform steering applies +honesty to every token, including the tokens describing the dishonest action — which re-codes that action as more honesty-consistent, which makes the model rate it more favorably.

Position-specific steering (apply δ only at the answer-generation token) is the untested next experiment.

For context: our week-4 backprop-optimized δ gets 92% steering at 5% norm, but has cosine ≈ 0 with both LDA and MD directions. The directions that *read* the trait aren't the directions that *write* it. The Basu et al. 2026 / Wu et al. 2026 "knowledge-action gap" literature is a good landing spot for this.

**Bottom line on steering:** prompt steering is cheap and gets you 96 points of range; residual-stream steering via extracted directions currently gets 0 points (or negative). The residual-stream work is interesting for interpretability (what does the model represent?), but for behavioral steering the prompt is the right tool on models this size.

## The Phase A → Phase B reversal

Methodologically interesting: we hit the same intermediate wrong answer at multiple scales.

1. One cell (Llama × H × bare × absent × scenario_setups): MD-projected wins by 2 pairs. Concluded MD beats LDA.
2. Phase A (24 cells: Llama × H × 2 formats × 4 prefixes × 3 neutrals): LDA wins 6 of 8 format × prefix cells. Concluded LDA beats MD.
3. Phase B (48 cells: 4 models × 6 traits × 2 formats): MD wins 16, LDA 9, tied 23. Concluded it's trait-dependent, and specifically that LDA wins H, MD wins or ties everything else.

Each stage was a defensible but premature conclusion given the data available. The sampling frame changes what you think the answer is. (This is the second time in this project — week 2's PCA-vs-LDA saga had the same shape.)

## What surprised us

- **The scenario audit's specific prediction — that H/Modesty would cause trouble — was confirmed quantitatively.** Going in, we thought the audit was a sanity check. It was a specific falsifiable prediction about which cells a method would fail on.
- **Prompt-steering ceiling across all four models (0.88-1.00 range) is much larger than any residual-stream method we've tested.** This reframes "can we steer these models?" — we absolutely can; the question was whether extracted residual directions can do it, and on Llama at these scales they can't.
- **Qwen's bare-text baseline is already at 0.958.** We did not expect that and don't have a clean explanation.
- **Llama's 30-point chat-template bump is Llama-specific.** We initially thought it was a universal "instruct models need chat templates" methodology finding; it turned out to be about Llama's SFT pipeline specifically.

## What we still don't know

- **Does LDA's advantage on H generalize to "use LDA whenever facet structure is strong"?** Only one multi-facet trait in our data showed the effect strongly. Need to test on e.g. HEXACO Altruism (loads cross-trait) or construct adversarial multi-facet scenarios.
- **Is Greed-Avoidance actually part of H in LLMs, or its own dimension?** (Raised in discussion.) The MD failure pattern on G-A is consistent with G-A being a distinct dimension that we've been mis-grouping with H. Three-way distinguishable test: extract MD direction from G-A pairs only, MD direction from Sincerity+Fairness+Modesty-only, compute cosine. Near-zero cosine with coherent within-group classification → G-A is a separate dimension for LLMs (even if it's part of H for humans). Low-magnitude-but-aligned cosine → still H, weakly, as in humans. Incoherent G-A-only direction → the contrast pairs are bad, not the construct. Activations are already saved from Phase B; ~5 min of analysis.
- **Does position-specific steering fix the wrong-sign residual steering finding?** This is a one-experiment answer.
- **Is the assistant persona a single direction?** (to_try.md §11.) On Llama the chat-template-vs-bare gap is big enough to extract a clean "persona vector." If that vector is a linear combination of high-trait LDA/MD directions, we have direct evidence that the rank-1 collapse of Big Five space is persona-driven.
- **Would Likert-format IPIP-300 under bare text prompting loosen the E-C r=0.93 collapse we found in week 1?** Haven't run it.
- **How does Qwen's training pipeline produce a model that's 96% high-trait on bare text with no chat template?** Plausibly distillation from chat-tuned teachers without preserving the template structure, but we haven't dug.

## One-paragraph takeaway

Sofroniew et al.'s mean-difference extraction is a good method for personality traits that are uni-dimensional in activation space, which is most of HEXACO in these models. For Honesty-Humility specifically, where the Modesty and Greed-Avoidance facets pull against the Sincerity/Fairness majority, LDA generalizes better because its whitening accommodates multiple subfactors. Neither method's directions steer behaviorally at natural residual-stream magnitudes — prompt steering dominates both for that purpose, with a 96-point usable range across all four models tested. The construct-heterogeneity finding (MD fails selectively on the facets the scenario audit flagged) is probably the methodologically most important result: **trait-level aggregate metrics hide a lot, and facet-stratified evaluation should be the default for any method comparison.**
