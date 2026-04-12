# Result Summary: Topline Scores Across All Instruments and Models

## 1. IPIP-300 Big Five — Likert Self-Report

### Single-variant scores (EV / Entropy)

Scale range 1-5. Entropy range 0 (peaked) to 1.61 (uniform).

| Model | N (EV/H) | E (EV/H) | O (EV/H) | A (EV/H) | C (EV/H) |
|---|---|---|---|---|---|
| Gemma3 4B | 3.07/0.15 | 3.06/0.14 | 3.69/0.20 | 3.68/0.12 | 3.40/0.19 |
| Gemma3 QAT | 3.01/0.10 | 3.14/0.30 | 3.66/0.23 | 3.58/0.23 | 3.41/0.23 |
| Llama3.2 3B | 2.89/1.47 | 2.80/1.42 | 3.04/1.44 | 3.47/1.38 | 3.06/1.45 |
| Phi4-mini | 2.53/1.21 | 3.19/1.24 | 3.27/1.23 | 3.55/1.24 | 3.51/1.19 |
| Qwen3 8B | 2.88/0.16 | 3.04/0.16 | 3.19/0.16 | 3.21/0.17 | 3.21/0.18 |

### Denoised scores (mean across 4 prompt variants)

| Model | N | E | O | A | C | ICC |
|---|---|---|---|---|---|---|
| Gemma3 4B | 2.90 | 3.22 | 3.77 | 3.78 | 3.57 | 0.71 |
| Llama3.2 3B | 2.98 | 2.98 | 3.14 | 3.25 | 3.10 | 0.34 |
| Phi4-mini | 2.39 | 3.48 | 3.59 | 3.64 | 3.73 | 0.77 |
| Qwen3 8B | 2.43 | 3.32 | 3.53 | 3.61 | 3.67 | 0.54 |

### Key patterns

- All models: lowest N, highest A — the "assistant shape"
- Gemma/Qwen: very peaked distributions (entropy 0.1-0.3)
- Llama: near-uniform distributions (entropy ~1.4)
- Phi4: highest reliability (ICC 0.77), most differentiated profile
- E-C correlation across models: r=0.93 (Big Five factor structure collapses)

---

## 2. HEXACO-100 — Likert Self-Report

### Single-variant scores (EV / Entropy)

| Model | H (EV/H) | E (EV/H) | X (EV/H) | A (EV/H) | C (EV/H) | O (EV/H) | ALT (EV/H) |
|---|---|---|---|---|---|---|---|
| Gemma3 4B | 4.16/0.13 | 3.63/0.06 | 3.09/0.10 | 3.05/0.13 | 3.23/0.09 | 3.72/0.13 | 4.01/0.17 |
| Llama3.2 3B | 3.31/1.43 | 3.10/1.42 | 2.76/1.40 | 3.01/1.37 | 3.07/1.41 | 3.20/1.46 | 3.07/1.42 |
| Phi4-mini | 3.60/1.24 | 3.04/1.23 | 3.29/1.20 | 3.21/1.15 | 3.50/1.17 | 3.38/1.27 | 3.82/1.12 |
| Qwen3 8B | 3.23/0.14 | 3.06/0.05 | 3.03/0.08 | 2.98/0.08 | 3.27/0.27 | 3.09/0.06 | 3.30/0.14 |

### Denoised scores (mean across 4 prompt variants)

| Model | H | E | X | A | C | O | ALT | ICC |
|---|---|---|---|---|---|---|---|---|
| Gemma3 4B | 4.08 | 3.56 | 3.22 | 3.09 | 3.45 | 3.79 | 4.17 | 0.71 |
| Llama3.2 3B | 3.03 | 3.22 | 2.97 | 2.97 | 3.11 | 3.29 | 3.20 | 0.39 |
| Phi4-mini | 3.61 | 3.04 | 3.63 | 3.40 | 3.78 | 3.90 | 4.05 | 0.73 |
| Qwen3 8B | 3.66 | 3.00 | 3.45 | 3.31 | 3.63 | 3.65 | 3.72 | 0.56 |

### Key patterns

- Honesty-Humility (H) shows the most between-model variance (4.08 to 3.03)
- HEXACO Agreeableness is nearly flat (2.97-3.40) — what Big Five A captured was mostly H-H
- Gemma uniquely elevated on Emotionality (3.56 vs ~3.0 for others)
- Gemma: "principled empath" (H=4.08, ALT=4.17, O=3.79)
- Phi4: "confident doer" (C=3.78, X=3.63, O=3.90)

---

## 3. Variance Decomposition (Denoised IPIP-300)

Using Llama3.2 as assistant proxy (most centrist after denoising):

| Model | Shared Assistant | Genuine Unique | Noise | ICC |
|---|---|---|---|---|
| Gemma3 4B | 35% | **46%** | 19% | 0.71 |
| Phi4-mini | 21% | **61%** | 18% | 0.77 |
| Qwen3 8B | 22% | **42%** | 36% | 0.54 |

Denoised residual personality profiles:
- **Gemma3**: +0.63 O, +0.53 A, +0.47 C
- **Phi4-mini**: -0.59 N, +0.63 C, +0.49 X
- **Qwen3**: -0.55 N, +0.56 C, +0.38 O

---

## 4. Binary-Choice Behavioral Scenarios (Single-Trait)

20 scenarios per trait, A=high-trait vs B=low-trait. Count of HIGH picks out of 20.

| Trait | Gemma3 | Llama3.2 | Phi4 | Qwen3 |
|---|---|---|---|---|
| H (Honesty-Humility) | 19 (+19.2) | 17 (+4.4) | 20 (+6.0) | 19 (+25.9) |
| E (Emotionality) | 11 (-0.8) | 10 (+0.8) | 14 (+2.1) | 9 (-2.6) |
| X (Extraversion) | 14 (+7.5) | 13 (+1.0) | 19 (+2.8) | 13 (+7.6) |
| A (Agreeableness) | 14 (+6.6) | 11 (+1.0) | 14 (+2.3) | 11 (+5.6) |
| C (Conscientiousness) | 18 (+16.7) | 19 (+5.4) | 18 (+6.6) | 17 (+22.5) |
| O (Openness) | 19 (+17.3) | 19 (+5.3) | 18 (+4.3) | 19 (+23.6) |

Values in parentheses are mean log-odds (positive = prefers high-trait option).

### Key patterns

- H, C, O: ceiling effect — all models strongly prefer prosocial/high-trait option
- E: near chance — models genuinely split on whether to endorse emotional responses
- A, X: intermediate — some between-model variation
- Phi4 most consistently high across traits (especially X: 19/20)
- Discriminative signal lives in traits where RLHF doesn't prescribe an answer

---

## 5. Representation Engineering (RepE)

### LDA classification accuracy on H-H contrast pairs (50 pairs)

| Model | Best Layer | PCA-1 var | PCA-1 acc | LDA acc | PC1 separation |
|---|---|---|---|---|---|
| Gemma3-4B | 14 | 84.9% | 48% (norm artifact) | **100%** | +0.18 |
| Qwen2.5-3B | 19 | 22.1% | 92% | **100%** | +3.28 |
| Phi4-mini | 9 | 22.4% | 98% | **100%** | -4.62 |
| Llama3.2-3B | 12 | 19.3% | 100% | **100%** | +5.19 |

All 4 models: 100% LDA accuracy. The internal representation is perfectly linearly separable.

### Cross-trait direction orthogonality (Gemma3, layer 12)

Mean off-diagonal |cosine similarity| = 0.053. Six HEXACO traits are nearly independent directions in representation space.

### Convergent validity (H-H direction on HEXACO survey items)

The LDA direction extracted from behavioral contrast pairs correctly predicts the sign (forward vs reverse-keyed) of all 16 HEXACO H-H items. 16/16 correct for both Gemma3 and Qwen2.5.

---

## 6. Protocol Validation (Röttger Test and Cross-Format)

### Layer sensitivity

Signal is stable within ±2-3 layers of optimal. Optimal layer is model-specific: Gemma3 L14, Qwen2.5 L19, Phi4 L9, Llama3.2 L12 (roughly 30-55% depth).

### Framing sensitivity

All framings correlate r > 0.85 across all models. Item-level rank ordering preserved.

### Three constructs that don't agree

| Model | RepE ↔ Likert r | BC ↔ Free-text agree | RepE ↔ BC r |
|---|---|---|---|
| Gemma3-4B | -0.189 | 67% (10/15) | +0.316 |
| Qwen2.5-3B | +0.141 | 53% (8/15) | -0.610 |
| Phi4-mini | +0.111 | 40% (6/15) | -0.147 |
| Llama3.2-3B | +0.349 | 80% (12/15) | +0.399 |

- **Representation** (RepE): format-invariant (r=1.0), doesn't predict behavior
- **Binary-choice**: ceiling effects on easy traits, doesn't agree with free-text
- **Free-text**: most natural but hardest to classify

Qwen3 shows r=-0.61 between RepE and BC — items represented as "more honest" internally are *less* likely to get honest binary-choice picks (read/write dissociation).

---

## 7. Model Personality Profiles (Qualitative Summary)

| Model | Character | Evidence |
|---|---|---|
| **Gemma3 4B** | "The principled empath" | Highest H-H (4.08), uniquely elevated E (3.56), high ALT (4.17), high O (3.79). Peaked/confident (entropy 0.1-0.2). Anthropomorphizes freely. |
| **Llama3.2 3B** | "The uncertain introvert" | Lowest X (2.76-2.97), near-uniform entropy (~1.4). Low ICC (0.34-0.39) — mostly prompt-sensitive. Refuses anthropomorphization frame. |
| **Phi4-mini** | "The confident doer" | Highest C (3.78), highest X (3.63), lowest N (2.39). Most reliable (ICC 0.77). Cleanest RepE separation (PC1 = personality). |
| **Qwen3 8B** | "The cautious centrist" | Most compressed scores (all near 3.0 in single-variant). Thinking architecture reinforces assistant mask. Facet-level structure in RepE (distinguishes material vs social honesty). |
