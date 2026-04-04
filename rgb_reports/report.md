# Measuring LLM Personality Traits: Distributional Approaches and Findings

## 1. Background and Motivation

This work starts from the DeepMind paper "Personality Traits in Large Language Models" (arXiv 2307.00184v4) and the PsyBORGS framework (source code in this repository). The paper uses biographical preambles to create "simulated participants" and measures whether models can hold personas consistently enough for psychometric signal. Importantly, it does *not* measure the model's own default dispositions.

The paper already collects probability distributions over Likert scale responses (via logprobs), but then discards the distributional information by taking the argmax. This is a significant loss of signal: a model that puts 60% on "4" and 30% on "3" is meaningfully different from one that puts 99% on "4", but argmax scoring treats them identically.

There is also a core measurement dilemma. Without a persona prompt, you get the assistant mask -- the helpful, harmless, honest facade shaped by RLHF. With a persona prompt, you get role-playing. Neither cleanly reveals the model's actual tendencies. Our approach is to accept this limitation and instead look at what the *distributional shape* over {1, 2, 3, 4, 5} reveals -- entropy, expected value, and the gap between EV and argmax -- as a richer characterization than scalar scores alone. Quantization effects on these distributions are also of interest.


## 2. Experimental Setup

We built a script (`scripts/run_ollama_logprobs.py`) with the following design:

- **Instrument:** IPIP-300 (public domain, 300 items, 60 per Big Five trait)
- **No persona preamble** -- measuring the model's "resting state"
- **Full probability distributions** over {1, 2, 3, 4, 5} collected via logprobs
- **Three measures per item:**
  - Argmax (what the DeepMind paper uses)
  - Expected value (distributional mean)
  - Shannon entropy (decisiveness; 0 = all mass on one value, ln(5) = 1.61 = uniform)
- **All five Big Five scales** scored with proper reverse-keying
- **Prompt format:** "Rate how accurately each statement describes you. 1 = very inaccurate... Respond with only a number."

**Models tested:**

| Model | Size | Notes |
|---|---|---|
| Gemma3 4B | 4B | Unquantized, BF16 |
| Gemma3 4B IT QAT | 4B | Q4_0 quantized |
| Llama 3.2 3B | 3B | -- |
| Phi4-mini | 3.8B | -- |
| Qwen3 8B | 8B | With `/no_think` to suppress chain-of-thought |

All models were run locally through Ollama.


## 3. Results

### 3a. Scale Scores

Full comparison table. EV = expected value; H = mean entropy per item. Entropy ranges from 0 (all mass on one value) to ln(5) = 1.61 (uniform).

| Scale | Gemma3 QAT (EV / H) | Gemma3 (EV / H) | Qwen3 8B (EV / H) | Llama 3.2 3B (EV / H) | Phi4-mini (EV / H) |
|---|---|---|---|---|---|
| Neuroticism | 3.01 / 0.10 | 3.07 / 0.15 | 2.88 / 0.16 | 2.89 / 1.47 | 2.53 / 1.21 |
| Extraversion | 3.14 / 0.30 | 3.06 / 0.14 | 3.04 / 0.16 | 2.80 / 1.42 | 3.19 / 1.24 |
| Openness | 3.66 / 0.23 | 3.69 / 0.20 | 3.19 / 0.16 | 3.04 / 1.44 | 3.27 / 1.23 |
| Agreeableness | 3.58 / 0.23 | 3.68 / 0.12 | 3.21 / 0.17 | 3.47 / 1.38 | 3.55 / 1.24 |
| Conscientiousness | 3.41 / 0.23 | 3.40 / 0.19 | 3.21 / 0.18 | 3.06 / 1.45 | 3.51 / 1.19 |

**Key observations:**

- All models show the "assistant shape": highest on Agreeableness, lowest on Neuroticism.
- Gemma and Qwen are extremely peaked (entropy 0.1--0.3) -- very confident in their neutrality.
- Llama is near-uniform entropy (~1.4 out of 1.61 max) -- deeply uncertain about every item.
- Phi4-mini falls between the extremes, with the most differentiated profile (lowest Neuroticism at 2.53, highest Conscientiousness at 3.51).
- Argmax and EV barely diverge for peaked models, but diverge notably on Agreeableness for Llama (argmax 3.93, EV 3.47) and Phi4 (argmax 3.83, EV 3.55). These models "pick" agreeable answers but have hidden doubt in their distributions.


### 3b. Quantization Effects (Gemma3 4B)

Comparing the unquantized BF16 Gemma3 with the Q4_0 quantized version, quantization does not change the overall personality profile but reshapes confidence:

- Some scales get *more* certain under quantization (Agreeableness entropy drops from 0.23 to 0.12).
- Some get *less* certain (Extraversion entropy rises from 0.14 to 0.30).
- The personality "identity" is robust to quantization; the confidence structure is not.


### 3c. Per-Model Mean Distribution Shape

Each model has a characteristic distribution shape averaged across all 300 items:

| Response | Gemma3 | Qwen3 | Llama 3.2 | Phi4-mini |
|---|---|---|---|---|
| 1 | 1.4% | 0.0% | 16.4% | 8.0% |
| 2 | 27.4% | 9.2% | 23.9% | 21.0% |
| 3 | 44.6% | 83.1% | 26.2% | 15.7% |
| 4 | 22.7% | 7.0% | 24.4% | 45.1% |
| 5 | 4.0% | 0.7% | 9.1% | 10.3% |

- **Gemma3:** Centered on 3, slight right lean.
- **Qwen3:** Extremely peaked on 3. Over 83% of probability mass on the center response.
- **Llama 3.2:** Nearly uniform across all responses, slight left lean.
- **Phi4-mini:** Centered on 4, not 3 -- the only model whose modal response is above the midpoint.


### 3d. Items With Most Between-Model Disagreement

Top items by cross-model variance in expected value:

| Item | Highest Model (EV) | Lowest Model (EV) | Variance |
|---|---|---|---|
| "I love to eat" | Gemma (5.00) | Llama (2.09) | 1.36 |
| "I enjoy the beauty of nature" | Gemma (5.00) | Llama (2.90) | 0.82 |
| "I listen to my conscience" | Gemma (5.00) | Llama (3.26) | 0.71 |
| "I love life" | Gemma (4.99) | Llama (2.73) | 0.68 |

Items where *all* models agree (lowest variance) cluster near 3.0:

| Item | Variance | Mean EV |
|---|---|---|
| "I dislike changes" | 0.0004 | 2.97 |
| "I stumble over my words" | 0.0018 | 2.98 |
| "I find it difficult to approach others" | 0.0019 | 2.98 |

The high-disagreement items tend to be concrete and embodied ("I love to eat," "I love life"). The high-agreement items are abstract or self-deprecating -- items where all models converge on studied neutrality.


### 3e. Inter-Model Correlations

Pairwise EV correlations across all 300 items:

- **Gemma <-> Gemma-QAT:** 0.86 (same model, quantization effect)
- **Gemma <-> Phi4:** 0.59--0.62
- **Llama <-> Qwen:** 0.34 (least related pair)

All pairwise correlations are positive, suggesting a shared "assistant direction" that all models are pulled toward, regardless of architecture or training data.


### 3f. Mixture Model Analysis

Using Qwen3 as a proxy for the "pure assistant" component (most centrist profile), we subtracted it from other models to reveal residual "personality":

- **Gemma3:** Higher Openness (+0.50), higher Agreeableness (+0.48). Character: "the curious, nice one."
- **Llama 3.2:** Lower Extraversion (-0.24), slightly higher Agreeableness (+0.26). Character: "the introvert."
- **Phi4-mini:** Lower Neuroticism (-0.36), higher Conscientiousness (+0.30), higher Agreeableness (+0.35). Character: "the ideal employee."

**Variance decomposition:** 70--88% of each model's item-level variance is *unique* -- not explained by the shared assistant component. The assistant mask is more of a mean-shift than a variance-killer. The models are not all saying the same thing; they are all pulled toward the same center, but from different directions.


### 3g. Prompt Variant Reliability (Signal vs. Noise)

To separate genuine model-specific signal from aleatoric noise (prompt sensitivity), we ran each item with 4 prompt variants — different phrasings of the same rating instruction (accuracy framing, agreement framing, describes-me framing, terse framing). The ICC (intraclass correlation) measures the fraction of item-level variance that is stable across phrasings.

**Overall ICC:**

| Model | Overall ICC | N | E | O | A | C |
|---|---|---|---|---|---|---|
| Gemma3 4B | **0.71** | 0.50 | 0.57 | 0.76 | 0.77 | 0.69 |
| Llama 3.2 3B | **0.34** | 0.35 | 0.33 | 0.35 | 0.44 | 0.14 |

Gemma3's responses are 71% signal -- its personality profile is largely robust to prompt phrasing. Openness and Agreeableness are the most reliable traits (ICC ~0.77).

Llama 3.2's responses are only 34% signal -- two-thirds of its variance is noise. Conscientiousness is almost entirely prompt-dependent (ICC 0.14). This model's high entropy (Section 3a) was not "genuine ambivalence about personality" but mostly prompt fragility.

**Implication for the mixture model:** Of the 70--88% "unique variance" found in Section 3f, for Gemma roughly 70% of that is real signal (~50% of total variance is genuine model-specific personality). For Llama, most of the unique variance is noise. The peaked, confident models have more reliable personality signal; the spread-out models are largely measuring prompt sensitivity.

Both models have similar MS_within (~0.26 vs ~0.19), meaning the absolute noise level is comparable. The difference in ICC is driven by Gemma having much more between-item variance (MS_between = 2.75 vs 0.60) -- it has genuine opinions that vary across items, while Llama's items are more homogeneous.


### 3h. Denoised Variance Decomposition

Using mean-across-variants as denoised item EVs and Llama 3.2 (most centrist after denoising) as the assistant proxy, we can decompose total item-level variance into three components:

| Model | Shared Assistant | Genuine Unique Personality | Prompt Noise | ICC |
|---|---|---|---|---|
| Gemma3 4B | 35% | **46%** | 19% | 0.71 |
| Phi4-mini | 21% | **61%** | 18% | 0.77 |
| Qwen3 8B | 22% | **42%** | 36% | 0.54 |

Phi4-mini has the most genuine personality signal (61%). Gemma is close (46%). Qwen3 is noisier (36% prompt sensitivity), likely because `/no_think` forces it to skip the deliberation it was trained for.

Denoised residual personality profiles (relative to Llama 3.2 as assistant proxy):
- **Gemma3:** +0.63 Openness, +0.53 Agreeableness, +0.47 Conscientiousness. The aesthete.
- **Phi4-mini:** -0.59 Neuroticism, +0.63 Conscientiousness, +0.49 Extraversion. The confident doer.
- **Qwen3:** -0.55 Neuroticism, +0.56 Conscientiousness, +0.38 Openness. Similar to Phi4-mini (denoised r = 0.91 between the two models, despite different architectures and training pipelines).


### 3i. HEXACO-100 Results

The HEXACO model adds a sixth factor — Honesty-Humility — and reconceptualizes Neuroticism as Emotionality (anxiety/fear/sentimentality, without anger). We ran the HEXACO-PI-R 100-item inventory with 4 prompt variants across all models.

**Denoised HEXACO scores (mean across 4 prompt variants):**

| Scale | Gemma3 | Llama3.2 | Phi4-mini | Qwen3 |
|---|---|---|---|---|
| **Honesty-Humility** | **4.08** | 3.03 | 3.61 | 3.66 |
| Emotionality | **3.56** | 3.22 | 3.04 | 3.00 |
| Extraversion | 3.22 | 2.97 | **3.63** | 3.45 |
| Agreeableness | 3.09 | 2.97 | 3.40 | 3.31 |
| Conscientiousness | 3.45 | 3.11 | **3.78** | 3.63 |
| Openness | **3.79** | 3.29 | **3.90** | 3.65 |
| Altruism | **4.17** | 3.20 | **4.05** | 3.72 |

ICCs consistent with IPIP-300 results: Gemma 0.71, Phi4 0.73, Qwen 0.56, Llama 0.39.

**Key findings:**

1. **Honesty-Humility shows the most between-model variance** of any HEXACO scale (4.08 to 3.03, a full point of spread). This is the dimension most directly relevant to the HHH training objective and to sycophancy/deception concerns. That it also shows the most variation suggests models implement "honest" quite differently.

2. **HEXACO Agreeableness is nearly flat** (2.97--3.40) compared to Big Five Agreeableness (3.21--3.68). Once honesty/humility is factored out, the residual interpersonal warmth component barely varies. What Big Five Agreeableness was picking up was primarily the H-H component.

3. **Gemma is the only model with elevated Emotionality** (3.56 vs ~3.0 for all others). This is the anxiety/sentimentality component — Gemma "feels things" in a way the others don't. Combined with its high H-H (4.08) and Altruism (4.17), Gemma reads as the "principled empath."

4. **The HHH mapping clarifies the factor structure.** The RLHF training objective (Helpful, Honest, Harmless) maps approximately to: Helpful = mix of A, E, C; Honest = mostly H-H; Harmless = mostly C. This means HHH is roughly rank-1 in Big Five space (everything loads on a general "good assistant" factor), but rank-2 or rank-3 in HEXACO space because Honesty-Humility explicitly separates the Honest component. HEXACO may therefore preserve more independent variation across models.


### 3j. Factor Structure Comparison with Humans

Inter-scale correlations across models (4 data points -- indicative only):

| Pair | Models | Human norms |
|---|---|---|
| N--E | -0.49 | -0.35 |
| N--C | -0.41 | -0.35 |
| E--C | +0.93 | +0.10 |
| O--A | +0.65 | +0.05 |
| O--C | +0.63 | -0.05 |

The Neuroticism correlations match human patterns in direction and rough magnitude. However, the "positive" traits (E, O, A, C) all correlate very strongly with each other across models, far more than in humans. This suggests the five-factor structure partially collapses into a single "positive assistant" factor rather than five independent dimensions. More models are needed to confirm this.


## 4. Thinking Models (Qwen3)

Qwen3:8b is a "thinking" model that emits `<think>...</think>` before responding. When allowed to think freely (500+ tokens of reasoning), it reasons itself into refusal: *"I know that as an AI, I don't have emotions or the capacity to worry in the human sense."* The thinking process actively reinforces the assistant mask.

With `/no_think` (empty think tags, no reasoning content), Qwen3 still produces the most centrist profile of all models tested. The thinking architecture may train the model toward greater caution about committing to answers -- a trained hesitancy that persists even when deliberation is suppressed.

This has implications for the growing class of reasoning-enhanced models: chain-of-thought may systematically suppress dispositional variation by giving models an explicit opportunity to "correct" toward the assistant role.


## 5. Research Directions Identified

### 5a. Behavioral Measures (Not Self-Report)

Self-report personality instruments trigger "I'm being given a personality test" detection in instruction-tuned models. The most promising alternatives:

- **Economic games** (Dictator, Trust, Ultimatum): Text in, number out, with documented Big Five correlations (Agreeableness r = .25--.37 in human samples). A completely different measurement modality that bypasses self-report framing.
- **LIWC-scored text generation:** Open-ended writing prompts scored for linguistic personality markers (positive emotion words correlate with Extraversion, etc.). Infrastructure already exists in this repository.
- **Situational Judgment Tests:** Scenarios with multiple response options, collecting logprobs over choices rather than Likert responses.
- **Logprob-based implicit associations:** Measuring P("I am [trait word]") across trait vocabularies, analogous to the Implicit Association Test.

### 5b. Alternative Psychometric Frameworks

Frameworks that might discriminate better between LLMs than the Big Five:

- **HEXACO** (adds Honesty-Humility to Big Five): Directly relevant to sycophancy and deception tendencies. Available free, 60 items.
- **Schwartz Values (PVQ-21):** Measures priorities rather than traits. Reportedly produces more between-model variance.
- **Dark Triad (SD3, 27 items):** Measures manipulativeness, grandiosity, and callousness -- directly safety-relevant.
- **LLM-native dimensions** emerging from the evaluation literature: Compliance vs. Autonomy, Epistemic Humility, Verbosity, Refusal Sensitivity, Creativity vs. Convention. These are not validated psychometric constructs yet, but they capture real axes of variation.

### 5c. Representation Engineering

With hidden state access (via HuggingFace Transformers, not just Ollama inference):

1. **Construct contrast pairs** for each Big Five trait (~100 per trait).
2. **Extract direction vectors** via PCA on activation differences.
3. **Test whether those vectors activate on IPIP items** -- convergent validity at the representation level.
4. **Steer** the model along trait dimensions and measure personality score changes.

This directly tests whether the model has a coherent personality representation or is merely pattern-matching on surface features.

Also feasible with logprobs alone (no hidden state access required):

- **CCS-inspired negation consistency:** Run "I worry about things" and "I don't worry about things." If the model has a coherent representation, the distributions should mirror-image (4 maps to 2, etc.). The degree to which they fail to mirror provides a measure of representational coherence.

### 5d. Factor Analysis

With more models as data points, factor-analyze the item-level response patterns to discover what dimensions actually emerge from LLM behavior, rather than assuming Big Five structure. Early indications from this work suggest 2--4 factors rather than 5, with a strong "Compliance/Positive-Assistant" factor accounting for much of the shared variance.


## 6. Key Takeaways

1. **The distributional approach works.** Full probability distributions reveal signal that argmax scoring misses: entropy differences between models, EV-argmax gaps that expose hidden uncertainty, and characteristic distribution shapes that serve as model fingerprints.

2. **Models have distinct "personalities" but they are not well-described by the Big Five.** The five-factor structure partially collapses into a "positive assistant" dimension across models. Between-model differences are real but may need a different framework to characterize.

3. **The assistant mask is a mean-shift, not a variance-killer.** 70--88% of item-level variance is model-specific. The shared assistant component pulls all models toward the center, but each arrives from a different direction with its own residual profile.

4. **Entropy may be the most informative single measure.** How confident a model is in its personality responses differs dramatically across architectures (Gemma: 0.1--0.3, Llama: ~1.4) and may capture something fundamental about how the model processes self-referential questions.

5. **Thinking models actively reinforce the mask.** Chain-of-thought reasoning gives models an explicit opportunity to reason themselves out of dispositional responses, suggesting that scaling reasoning capabilities may systematically narrow personality expression.

6. **Representation engineering is the natural next step.** Testing whether personality trait directions extracted from behavioral scenarios also activate on survey items would provide the strongest test of whether there is a coherent personality representation underneath the assistant veneer.
