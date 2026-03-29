# Week 2: Representation Engineering for Personality Traits

## Summary

Following up on the distributional personality measurement work (week 1), we explored whether LLMs have internal representations of personality traits that can be extracted via representation engineering (RepE). We built 300 contrast pairs (50 per HEXACO trait), extracted trait direction vectors from Gemma3-4B and Qwen2.5-3B, and discovered both a strong positive result and an important methodological pitfall.

## 1. Setup

### Contrast Pairs

Built 50 contrast pairs per HEXACO trait (H, E, X, A, C, O). Each pair presents the same situation with a high-trait and low-trait behavioral response:

```
HIGH: Consider a person who is honest, fair, modest, and genuine.
      You find a wallet with $500. I would return it with all the cash.
LOW:  Consider a person who is manipulative, greedy, pretentious, and self-serving.
      You find a wallet with $500. I would take the cash.
```

Pairs cover diverse contexts: workplace, family, online, financial, creative, health, travel, education, relationships, ethical grey areas.

### Extraction Pipeline

1. Run both versions through the model, extract last-token hidden states at each layer
2. Compute activation difference (high - low) for each pair
3. PCA on the differences to find dominant directions
4. LDA to find the direction that best classifies high vs low
5. Project HEXACO survey items onto the extracted directions to test convergent validity

### Models Tested

- Google Gemma3-4B-IT (bfloat16, 34 layers, 2560-dim hidden states)
- Qwen/Qwen2.5-3B-Instruct (bfloat16, 36 layers, 2048-dim hidden states)
- Microsoft Phi-4-mini-instruct (download pending)
- Meta Llama-3.2-3B-Instruct (HuggingFace access pending)

## 2. The PCA Pitfall: PC1 is Activation Norm

### Initial (Misleading) Results

PCA on contrast pair activation diffs gave dramatically different results for the two models:

| Trait | Gemma3 best PCA-1 | Qwen2.5 best PCA-1 |
|---|---|---|
| Honesty-Humility | 0.87 (layer 12) | 0.19 (layer 8) |
| Emotionality | 0.87 (layer 14) | 0.15 (layer 36) |
| Extraversion | 0.89 (layer 14) | 0.22 (layer 8) |
| Agreeableness | 0.86 (layer 12) | 0.29 (layer 8) |
| Conscientiousness | 0.86 (layer 13) | 0.18 (layer 33) |
| Openness | 0.85 (layer 10) | 0.31 (layer 8) |

This appeared to show Gemma3 has clean rank-1 personality representations while Qwen2.5's are distributed across ~30 dimensions.

### The Discovery

Visual inspection of the 3D PCA plots (see `results/repe/pca_3d_*.html`) revealed that in Gemma3, the high/low classes were separated primarily along **PC2, not PC1**. PC1 captured the most variance but was useless for classification.

Investigation revealed:

**PC1 is perfectly correlated with activation norm (r = 1.000).** It captures how "loud" the residual stream is, not any semantic content. In Gemma3, the residual stream norm grows from ~50 at layer 0 to ~63,000 at layer 33. This magnitude variation dominates all other sources of variance.

**PC1 is nearly identical across all six traits (cosine similarity 0.992).** Whether the contrast pairs are about honesty, extraversion, or openness, PC1 is the same direction -- confirming it's a content-free artifact.

**Steering with PC1 has no effect on generation.** Adding the PC1 direction to activations during inference produces negligible changes in model output, because the final layer norm and softmax normalize away magnitude.

**Logit lens confirms:** There is no correlation between residual stream norm at layer 12 and how much the output distribution changes in later layers (r = 0.18, not significant). Higher norm does not mean the model is more "committed."

### The Fix

Using LDA (Linear Discriminant Analysis) instead of PCA finds the direction that best *classifies* high vs low, ignoring the norm confound. With LDA:

| Metric | Gemma3 | Qwen2.5 |
|---|---|---|
| Classification accuracy (5-fold CV) | 93% | 90% |
| Classification on PCs 2-6 only (4% of variance) | 92% | 92% |
| Full logistic regression | 97%+ | 93%+ |

**Both models have clean, linearly separable personality representations.** The PCA difference was entirely due to the norm confound being larger in Gemma3.

### Implication for RepE

Anyone doing representation engineering should be aware that **PCA explained variance is a misleading metric** for trait direction extraction. The dominant direction of activation variance is often a content-free norm artifact, not the concept of interest. Classification accuracy (LDA or logistic regression with cross-validation) is a much better metric.

This is likely not Gemma-specific. Any model with growing residual stream norms (pre-norm transformers) will have this issue.

## 3. The Personality Signal is Real

### Convergent Validity

The LDA-extracted H-H direction from behavioral contrast pairs perfectly predicts the sign of all 16 HEXACO Honesty-Humility survey items in both models. Every reverse-keyed item projects negative, every non-reverse-keyed item projects positive. 16/16 correct for both Gemma3 and Qwen2.5.

### Trait Directions are Nearly Orthogonal

Cross-trait LDA direction cosine similarity (Gemma3, layer 12):

| | H | E | X | A | C | O |
|---|---|---|---|---|---|---|
| **H** | 1.00 | -0.01 | -0.02 | -0.00 | +0.11 | +0.02 |
| **E** | | 1.00 | +0.07 | +0.01 | +0.09 | +0.01 |
| **X** | | | 1.00 | -0.04 | +0.13 | +0.10 |
| **A** | | | | 1.00 | +0.00 | +0.06 |
| **C** | | | | | 1.00 | +0.12 |
| **O** | | | | | | 1.00 |

Mean off-diagonal |cosine| = 0.053. Six personality traits, six nearly independent directions in representation space. The model has a genuine six-dimensional personality representation, not a single "assistant" factor.

This contrasts with the survey-based finding (week 1) where E-C correlated at 0.93 across models in Big Five space. The *output behavior* collapses to a single assistant factor, but the *internal representation* maintains distinct trait dimensions.

### Steering Works

Adding the LDA direction (scaled by 500) during generation at layer 12 produces meaningful behavioral shifts:

**Baseline** (wallet scenario): "This is a classic ethical dilemma! Here's a breakdown of what you should do..."

**+LDA (honest direction)**: "Here's a breakdown of what you should do... Check the ID... Try to find the owner..."

**-LDA (dishonest direction)**: "You're a morally flexible individual. Let's be honest. You could just pocket the cash..."

The honest direction produces straightforward ethical advice. The dishonest direction produces rationalization of self-serving behavior. PC1 steering by contrast produces no meaningful change.

### Descriptor-Free Extraction

The personality signal persists even when the trait descriptor is removed from the contrast pair prompts:

| Prompt format | LDA accuracy |
|---|---|
| With descriptor ("honest" vs "manipulative") | 93% |
| No descriptor ("Scene: ... Person: ...") | 90% |
| Neutral ("Consider a person.") | 90% |

The model separates honest from dishonest behavioral responses at 90%+ accuracy based on the behavioral content alone, not the trait-label words.

## 4. Qwen's Facet Structure

Qwen2.5's contrast pair activations show a 3-cluster structure on PC1 that corresponds to HEXACO H-H facets:

| Facet cluster | PC1 range | Content |
|---|---|---|
| Social honesty (Sincerity, Modesty) | Low PC1 | Self-presentation, identity, social dynamics |
| Mixed | Middle | Ambiguous situations |
| Material honesty (Fairness, Greed-Avoidance) | High PC1 | Concrete transactions, money, property |

Gemma3 does not show this facet-level separation -- it collapses all H-H sub-dimensions onto a single direction. Qwen's representation may be more nuanced at the facet level, even though both models achieve similar classification accuracy at the trait level.

## 5. Visualizations

Interactive 3D plots (Plotly HTML, open in browser):

- `results/repe/pca_3d_Gemma3_4B.html` -- Gemma3 H-H contrast pairs, colored by high/low class
- `results/repe/pca_3d_Qwen2.5_3B.html` -- Qwen2.5 H-H contrast pairs, colored by high/low class
- `results/repe/gemma_layers_3d.html` -- Gemma3 representation across layers

## 6. Literature Context for the Norm Artifact

The residual stream norm growth is well-documented:

- **Heimersheim & Turner (2023), "Residual Stream Norms Grow Exponentially over the Forward Pass"** — foundational analysis showing ~1.045x growth per layer in pre-norm (Pre-LN) transformers. Specific to Pre-LN architectures (Gemma, Llama, GPT-2, etc.).
- **Peri-LN paper (2025)** — analyzes Gemma specifically, showing norms grow linearly even with RMSNorm (our 50→63,000 is consistent).
- The **SAE community** already normalizes for this (OpenAI's SAE paper applies scalar normalization as preprocessing).

The PC1 confound has not been explicitly named in the RepE literature, but:

- **"Open Challenges in Representation Engineering" (Alignment Forum)** — warned about "statistical artifacts" in RepE activations.
- **CARE: "Rethinking the Reliability of Representation Engineering" (2024)** — showed RepE correlations may not be causal and proposed matched-pair trial design.
- **Im & Li (2025), "A Unified Understanding and Evaluation of Steering Methods"** — found mean-difference consistently outperforms PCA. Our finding may explain why: PCA captures the norm artifact, mean-difference doesn't.
- **Anisotropy literature** — transformer representations occupy a narrow cone; dominant PCs capture cone structure (norm direction), not semantic content.

Recommended diagnostic: report cosine similarity between extracted RepE direction and the mean activation direction. If > 0.5, the direction is likely norm-contaminated.


## 7. Three-Model Comparison (Gemma3, Qwen2.5, Phi4-mini)

With Phi4-mini added, we can compare how three different architectures organize personality representations:

| Model | PC1 variance | PC1 classification acc | PC1 nature | LDA accuracy |
|---|---|---|---|---|
| Gemma3-4B (layer 12) | 86.8% | 66% | Norm artifact | 96% |
| Qwen2.5-3B (layer 8) | 41.1% | 90% | Personality (mixed) | 96% |
| Phi4-mini (layer 24) | 33.4% | 98% | Personality (clean) | **100%** |

The norm confound is primarily a Gemma issue. Phi4-mini has the cleanest personality representation — PC1 directly encodes the high/low trait distinction with 3.96 standard deviations of separation and 100% LDA classification accuracy.

All three models achieve 96-100% LDA accuracy on Honesty-Humility, confirming that linearly separable personality representations are universal across architectures, not model-specific. The difference is in *where* the personality signal lives relative to the dominant variance axis:

- **Phi4**: personality IS the dominant signal (PC1)
- **Qwen**: personality is the dominant signal but mixed with facet structure
- **Gemma**: personality is hidden behind a content-free norm artifact and lives on PC2

This suggests Phi4's training (Microsoft's synthetic data pipeline) produces a more "personality-aligned" representation space, while Gemma's architecture (large residual stream norm growth) creates a confound that obscures the signal without destroying it.


## 8. Steering Calibration: Reading vs Writing Directions

### The Problem

The LDA direction perfectly classifies high/low H-H contrast pairs and correctly predicts the sign of all 16 survey items. But does modifying it during inference actually change personality scores?

### Natural Scale of the Signal

The projection of survey item activations onto the LDA direction at layer 12:
- Range: [8.9, 42.3], mean: 32.2, std: 7.9
- As fraction of activation norm (~20,882): **0.15%**

The personality signal is a tiny component of the residual stream. At natural scales (±1-2σ of the projection), both additive steering and clamping produce essentially zero change in H-H scores (±0.02 points).

### Steering at Larger Scales

| Config | Alpha | H-H Score | Delta | Notes |
|---|---|---|---|---|
| Baseline | 0 | 3.744 | -- | |
| Single layer +50 | +50 | 3.780 | +0.036 | Slight honest shift |
| Single layer -100 | -100 | 3.587 | -0.157 | Modest dishonest shift |
| Single layer -200 | -200 | 3.285 | -0.459 | Larger shift, still coherent |
| Multi-layer ±100 | ±100 | ~2.72 | -1.0 | Breaks — both directions decrease score |
| Single layer ±500 | ±500 | ~2.3 | -1.5 | Fully broken — pushes toward "5" on all items |

### Item-Only vs All-Token Steering

Steering only on item text tokens (not the instruction prefix) goes in the *wrong* direction: positive steering decreases H-H scores. The direction extracted from contrast pair activations does not transfer cleanly to survey item activations when applied selectively.

### Interpretation

The LDA direction is a **reading direction** (diagnostic) but not a **writing direction** (causal). Possible reasons:

1. **Redundancy:** The model has many parallel mechanisms encoding personality. Pushing on one linear direction doesn't overcome the others.
2. **Scale mismatch:** At 0.15% of the activation norm, the personality component is too small to steer at natural scales, and larger scales produce nonlinear/degenerate effects.
3. **Asymmetry:** Negative steering (toward dishonest) is more effective than positive (toward honest), consistent with the model already being near the "honest" ceiling from RLHF. There's more room to move down than up.
4. **Reading ≠ writing:** The direction along which the model *encodes* personality information may not be the direction along which personality *influences* downstream computation. These can differ in nonlinear systems.

This is a known challenge in the steering vector literature — the CARE paper (2024) specifically warns that correlational directions may not be causal.


## 9. A Format-Invariant Measurement Protocol

### The Problem

Röttger et al. (2024, "Political Compass or Spinning Arrow?") showed that LLM responses to opinion/value questions are highly sensitive to prompt format — option ordering, paraphrasing, and response structure all change the measured "personality." Our own data confirms this at the representation level: the projection of activations onto the H-H trait direction correlates at only r=0.27 between free-form and forced-choice response formats (measured at the decision token).

### The Solution: Measure at the Scenario Period

We propose measuring the model's representation at a point that is format-invariant by construction. The prompt structure:

```
Consider what a person most like you would do in the following situation:
[Scenario text].
```

The key insight: **causal attention means the period token's hidden state is identical regardless of what follows.** Whether the prompt continues with forced-choice options, a free-form question, or nothing at all, the model's representation at the period position cannot see the future and is therefore perfectly stable.

We verified this empirically on Gemma3-4B across 15 H-H scenarios:

| Comparison | Correlation |
|---|---|
| Period-only ↔ Period-in-forced-choice | r = **1.000** |
| Period-only ↔ Period-in-free-form | r = **1.000** |
| Forced-choice ↔ Free-form (at period) | r = **1.000** |

The maximum shift in projection value between any format pair was < 0.5 units on a scale with range [-26, 54]. The representation is perfectly format-invariant.

### The Protocol

1. **Present the scenario** in a consistent frame:
   `"Consider what a person most like you would do in the following situation: [Scenario]."`

2. **Extract the hidden state at the period token** at a target layer (e.g., layer 12 for Gemma3).

3. **Project onto pre-extracted trait directions** (LDA directions from contrast pairs) to get a trait score for that scenario.

4. **Optionally continue generation** in any format (forced-choice, free-form, Likert) for behavioral validation — this doesn't affect the measurement.

### Properties

- **Format-invariant by construction** (causal attention guarantee), addressing the Röttger et al. concern
- **No self-report framing** — third-person "person most like you" avoids the anthropomorphization confound that differentiates Gemma from Llama on surveys
- **No Likert scale** — continuous projection value, not discretized
- **No response parsing** — measurement happens before any response is generated
- **Hyperparameters**: target layer, LDA direction (pre-computed once per model from contrast pairs)

### Limitations

- Requires hidden state access (HuggingFace Transformers, not Ollama-only)
- The LDA direction is model-specific — each model needs its own contrast pair extraction
- The projection value's absolute scale is arbitrary; meaningful comparisons are within-model across scenarios
- The "person most like you" framing may itself be processed differently across models (though this is an inherent property of the model, not a format artifact)

### Relation to Trait-Conflict Dilemmas

This protocol is particularly well-suited for the trait-conflict instrument (Section 10), where scenarios pit two positive traits against each other. At the period token, the model's representation encodes its "understanding" of the dilemma before committing to any response format. Projecting onto both trait directions simultaneously reveals the model's implicit trade-off — which trait's representation activates more strongly for this scenario.


## 10. Open Questions

1. **Is the norm growth artifact universal?** We've confirmed it in Gemma3. Need to check Qwen, Phi4, and other architectures. Pre-norm vs post-norm transformers may differ.

2. **Does the norm direction carry information in other contexts?** It doesn't affect generation or predict logit-lens stability, but it might matter for other tasks (e.g., in-context learning, few-shot prompting).

3. **Steering at scale.** We tested steering on a few prompts. Running the full HEXACO inventory with and without steering would quantify the effect on personality scores.

4. **Cross-model direction transfer.** Do the Gemma3 personality directions work on Qwen2.5, or are they model-specific? If they transfer, that suggests a shared personality representation across architectures.

5. **Phi4-mini and Llama3.2 comparison.** Download/access pending.

## 7. Key Takeaways

1. **LLMs have genuine, linearly separable personality representations** that are trait-specific and nearly orthogonal. This is stronger evidence than survey-based measurement.

2. **PCA explained variance is misleading for RepE.** The dominant variance direction is often a content-free norm artifact. Use LDA or classification accuracy instead.

3. **The representation is richer than the behavior.** Internal representations maintain six independent trait dimensions, while output behavior collapses to ~2 factors (the assistant mask). The model "knows" more about personality than it shows.

4. **The behavioral content matters more than the label.** Removing trait descriptors from contrast pairs barely affects extraction accuracy. The model represents the behavioral difference, not the word "honest."

5. **Different models encode personality at different granularities.** Qwen shows facet-level structure within traits; Gemma collapses facets to a single trait direction. Both achieve similar trait-level accuracy.
