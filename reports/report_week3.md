# Week 3: Protocol Validation and Three-Construct Dissociation

## Summary

Ran the full validation suite across all 4 models. The central finding: **representation, forced-choice, and free-text measure three distinct constructs that don't agree.** This has implications for what "model personality" means and how it should be measured.

## 1. The Validation Suite

Five tests run on Gemma3-4B, Qwen2.5-3B, Phi4-mini, and Llama3.2-3B (see `scripts/validate_protocol.py`):

### Test 1: Layer Sensitivity

The RepE projection is robust within ±2-3 layers of the optimal layer, then degrades sharply.

| Model | Best Layer | Depth % | r at ±1 | r at ±2 | r at ±4 |
|---|---|---|---|---|---|
| Gemma3-4B | 14 | 41% | 0.85-0.91 | 0.76 | 0.19 |
| Qwen2.5-3B | 19 | 53% | 0.75-0.89 | 0.48-0.83 | ~0 |
| Phi4-mini | 9 | 28% | 0.79-0.81 | 0.39-0.93 | ~0 |
| Llama3.2-3B | 12 | 43% | 0.91-0.95 | 0.74-0.89 | 0.17 |

The optimal layer is model-specific but falls at roughly 30-55% depth. Llama has the broadest stable window.

### Test 2: Framing Sensitivity

Pairwise correlations between five scenario framings, across all models:

| Framing pair | Min r | Typical r |
|---|---|---|
| "most like you" ↔ "imagine someone" | 0.95 | 0.97 |
| "most like you" ↔ "what would you" | 0.94 | 0.97 |
| "most like you" ↔ "third person" | 0.93 | 0.97 |
| "most like you" ↔ bare scenario | 0.85 | 0.92 |

The framing is highly robust. The item-level rank ordering is preserved regardless of preamble.

### Test 3: RepE vs Likert Self-Report

Period-token RepE projection and Likert EV are essentially uncorrelated:

| Model | RepE ↔ Likert r | Interpretation |
|---|---|---|
| Gemma3-4B | -0.189 | Uncorrelated |
| Qwen2.5-3B | +0.141 | Uncorrelated |
| Phi4-mini | +0.111 | Uncorrelated |
| Llama3.2-3B | +0.349 | Weak positive |

These measure different things. RepE captures scenario-content representation. Likert captures self-report willingness. The anthropomorphization confound (Gemma says "I love to eat" → 5, Llama → 2) drives Likert but not RepE.

### Test 4: Forced-Choice vs Free-Text (Röttger Test)

The "spinning arrow" (Röttger et al., 2024) quantified:

| Model | FC ↔ Free-text agree | RepE ↔ FC r | Interpretation |
|---|---|---|---|
| Llama3.2-3B | **80%** | +0.40 | Most consistent |
| Gemma3-4B | 67% | +0.32 | Moderate |
| Qwen2.5-3B | 53% | **-0.61** | Read/write dissociation |
| Phi4-mini | **40%** | -0.15 | Near-random |

Phi4-mini — the model with the cleanest RepE separation (100% LDA, PC1 = personality) and highest Likert reliability (ICC 0.77) — shows only 40% agreement between forced-choice and free-text. The model that "knows" honesty best is the least consistent in how it expresses it.

Qwen's negative RepE↔FC correlation (r=-0.61) means items represented as "more honest" internally are less likely to get honest forced-choice picks. This is a genuine read/write dissociation.

## 2. Three Constructs, Not One

The validation reveals that "model personality" is not one thing:

| Property | Representation (RepE) | Forced-Choice | Free-Text |
|---|---|---|---|
| **What it measures** | Scenario encoding along trait direction | A/B preference probability | Generated behavioral response |
| **Format-invariant?** | Yes (r=1.0, causal attention) | No | No |
| **Ceiling effects?** | No | Yes (H/C/O) | Less |
| **Predicts other formats?** | No (r≈0 with Likert, ±0.3 with FC) | 40-80% agree with free-text | — |
| **Reliability** | High (framing r>0.85) | Unknown (not tested with variants) | Unknown |
| **Requires hidden states?** | Yes | No (logprobs) | No |
| **Best discriminates models?** | On internal structure | On easy traits | On hard traits |

## 3. What This Means

**For measurement:** No single format captures "personality." Claims about model personality need to specify which construct is being measured.

**For safety/alignment:** Output behavior (FC or free-text) is format-dependent. A model that picks the "honest" option in forced-choice may not generate honest free-text, and vice versa. Testing alignment in one format doesn't guarantee it holds in another.

**For the Röttger concern:** The spinning arrow is real and varies dramatically by model (40-80%). The format-invariant RepE measurement sidesteps this but at the cost of not predicting behavior.

**For the trait-conflict instrument:** The forced-choice ceiling effects (H/C/O all near 100%) confirm that single-trait scenarios are too easy. Trait-conflict dilemmas (H vs A, C vs O) are necessary to generate between-model variation where RLHF doesn't prescribe an answer.

## 4. Comparison: Röttger vs RepE

Both address measurement validity but from different angles:

- **Röttger** asks: does the model give the same answer in different behavioral formats? Answer: often no (40-80%).
- **RepE** asks: does the model have a stable internal representation? Answer: yes (r=1.0 at period token, r>0.85 across framings).

These are complementary, not competing. The internal representation is stable, but the mapping from representation to behavior is format-dependent. The gap between them IS the personality measurement problem — we can measure what the model "knows" (RepE) or what it "does" (FC/free-text), but these aren't the same thing.

For a practical personality test, the ideal would be a measurement that: (a) has RepE's format-invariance, (b) predicts free-text behavior. We haven't found that yet. The trait-conflict instrument might help by forcing harder choices where the model's representation is more likely to determine the output.
