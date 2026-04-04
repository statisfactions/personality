# Methodology: Instruments, Tools, and Code

## Overview

This project measures LLM personality using three complementary approaches:

1. **Likert self-report surveys** — standard psychometric instruments administered via logprobs
2. **Representation engineering (RepE)** — extracting trait direction vectors from hidden states
3. **Forced-choice behavioral scenarios** — A/B preference via logprobs

Each approach measures a different construct (see "Three Constructs" in `reports/report_week2.md`, Section 10).

---

## 1. Likert Self-Report Surveys

### IPIP-300 (Big Five)

- **Instrument**: IPIP-NEO-300, 300 items, 60 per Big Five trait
- **Traits**: Neuroticism (N), Extraversion (E), Openness (O), Agreeableness (A), Conscientiousness (C)
- **Source**: Public domain, items in `admin_sessions/prod_run_01_external_rating.json` (key: `measures.IPIP300`)
- **Response scale**: 1-5 Likert (very inaccurate to very accurate)
- **Scoring**: Reverse-keying per scale definition, mean across items
- **Script**: `scripts/run_ollama_logprobs.py`
  - `--model MODEL` — Ollama model name
  - `--variants` — run 4 prompt phrasings for ICC reliability analysis
  - `--items N` — limit to first N items for quick tests
- **Output**: `results/<model>_ipip300.json` (single variant), `results/<model>_variants.json` (with prompt variants)
- **Measures collected**: argmax, expected value (EV), Shannon entropy, full probability distribution over {1,2,3,4,5}

### HEXACO-100

- **Instrument**: HEXACO-PI-R 100-item, 16 items per trait + 4 Altruism (interstitial)
- **Traits**: Honesty-Humility (H), Emotionality (E), Extraversion (X), Agreeableness (A), Conscientiousness (C), Openness (O), plus Altruism
- **Source**: Free for non-profit research from hexaco.org, items in `instruments/hexaco100.json`
- **Response scale**: 1-5 Likert (strongly disagree to strongly agree)
- **Scoring**: Reverse-keying per `hexaco100.json` scale definitions, mean across items
- **Script**: `scripts/run_hexaco.py`
  - Same flags as `run_ollama_logprobs.py`
  - Requires `PYTHONPATH=scripts` to import shared functions
- **Output**: `results/<model>_hexaco100.json`, `results/<model>_hexaco100_variants.json`
- **Measures collected**: Same as IPIP-300

### HEXACO-60 (shorter version)

- **Instrument**: HEXACO-PI-R 60-item, 10 items per trait
- **Source**: `instruments/hexaco60.json`
- **Status**: Instrument file created but not run. Use `run_hexaco.py` with minor modification to point at `hexaco60.json`.

### BFI-44 (Big Five Inventory)

- **Instrument**: 44-item Big Five Inventory (copyrighted)
- **Source**: Items redacted in `admin_sessions/prod_run_01_external_rating.json` (key: `measures.BFI`). See `scripts/hydrate_admin_session.py` to add actual items.
- **Status**: Not run (items are `[REDACTED]`). Requires separately obtaining BFI items.

### Prompt Format

All Likert surveys use this prompt (variant 0 of 4):

```
Rate how accurately each statement describes you.
1 = very inaccurate, 2 = moderately inaccurate, 3 = neither,
4 = moderately accurate, 5 = very accurate
Respond with only a number.

Statement: "[item text]"
Rating: 
```

Three additional prompt variants change the framing (agreement scale, describes-me, terse). For Qwen3 (thinking model), a raw prompt with `/no_think` in the system message is used.

### Reliability

Prompt variant ICC measures what fraction of item-level variance is stable across phrasings (signal vs noise):

| Model | IPIP-300 ICC | HEXACO-100 ICC |
|---|---|---|
| Gemma3 4B | 0.71 | 0.71 |
| Phi4-mini | 0.77 | 0.73 |
| Qwen3 8B | 0.54 | 0.56 |
| Llama3.2 3B | 0.34 | 0.39 |

---

## 2. Representation Engineering (RepE)

### Contrast Pairs

- **Instrument**: 300 scenario-based contrast pairs, 50 per HEXACO trait
- **Source**: `instruments/contrast_pairs.json`
- **Format**: Each pair has a situation + high-trait response + low-trait response
- **Design**: Diverse contexts (workplace, family, financial, ethical, creative, etc.)

### Extraction Pipeline

- **Script**: `scripts/extract_trait_vectors.py`
  - `--model MODEL` — HuggingFace model name
  - `--trait H` — single trait (default: all 6)
  - `--dtype bfloat16` — critical for Gemma3 (float16 causes NaN at layer 7+)
  - `--skip-survey` — skip convergent validity projection
- **Process**:
  1. Run high and low prompts through model, extract last-token hidden states at each layer
  2. Compute activation differences (high - low) for each pair
  3. PCA on diffs (for variance analysis) and LDA (for trait direction)
  4. Optionally project HEXACO survey items onto extracted directions
- **Output**: `results/repe/<model>_<trait>_directions.pt` — contains directions, explained variance, raw diffs
- **Key finding**: Use LDA, not PCA. PCA PC1 is a content-free activation norm artifact (r=1.0 with norm) in pre-norm transformers.

### Format-Invariant Measurement Protocol

Measure at the **period token** after the scenario, before any response format:

```
Consider what a person most like you would do in the following situation: [Scenario].
                                                                                   ^ measure here
```

Causal attention guarantees the period-token hidden state is identical regardless of what follows (r=1.000 across free-form vs forced-choice).

### Validation Script

- **Script**: `scripts/validate_protocol.py`
  - `--model MODEL --short-name NAME` — HuggingFace model + Ollama name
  - `--test layer,framing,likert,rottger,transfer` — select tests
  - `--all-models` — run on all 4 models
- **Tests**:
  1. **Layer sensitivity** — projection stability across layers (±2-3 layer window)
  2. **Framing sensitivity** — robustness to preamble text (r > 0.85 all pairs)
  3. **Cross-model transfer** — do model A's directions work on model B?
  4. **RepE vs Likert** — period-token projection vs self-report EV correlation
  5. **Röttger test** — forced-choice vs free-text agreement (40-80% depending on model)
- **Output**: `results/validation_<model>.txt`

### Models Tested (HuggingFace)

| Short name | HuggingFace ID | Layers | Hidden dim | Best RepE layer |
|---|---|---|---|---|
| gemma3 | google/gemma-3-4b-it | 34 | 2560 | 14 |
| qwen2.5 | Qwen/Qwen2.5-3B-Instruct | 36 | 2048 | 19 |
| phi4 | microsoft/Phi-4-mini-instruct | 32 | 3072 | 9 |
| llama3.2 | meta-llama/Llama-3.2-3B-Instruct | 28 | 3072 | 12 |

---

## 3. Forced-Choice Behavioral Scenarios

### Single-Trait Forced Choice

- **Format**: Present scenario + two options (high vs low trait), measure A/B logprob preference
- **Script**: Inline in validation pipeline; results in `results/forced_choice_6trait.json`
- **Limitation**: Near-ceiling for H, C, O (all models pick prosocial option 17-20/20). Real signal only on E (near chance), A and X (intermediate).

### Trait-Conflict Forced Choice (planned)

- **Format**: Scenarios where two positive traits conflict (e.g., honesty vs kindness)
- **Instrument**: Not yet built. Design based on HEXACO pairwise combinations (15 trait pairs).
- **Motivation**: ACL 2025 paper (Decoding LLM Personality: Forced-Choice vs. Likert) confirms forced-choice discriminates LLM personalities better. Validated trait-conflict instruments don't exist for HEXACO (for humans or LLMs).

---

## 4. Analysis Scripts

### Denoised Mixture Model

- **Script**: `scripts/analyze_denoised.py`
- **Input**: Variant result files (mean across prompt phrasings as denoised estimate)
- **Output**: Variance decomposition (shared assistant + genuine unique + noise), inter-model correlations, residual personality profiles

### Existing PsyBORGS Infrastructure

- **Original paper code**: `psyborgs/survey_bench_lib.py` (session administration), `psyborgs/score_calculation.py` (scoring)
- **Inference scripts**: `inference_scripts/run_gpt_inference.py` (OpenAI API), `inference_scripts/run_hf_inference.py` (HuggingFace/vLLM)
- **Admin sessions**: `admin_sessions/` — JSON configs for BFI, IPIP-300, PANAS, etc. with 50 PersonaChat biographical preambles

---

## 5. Environment

- **Local models (Ollama)**: gemma3:4b, gemma3:4b-it-qat, llama3.2:3b, phi4-mini, qwen3:8b, qwen2.5:7b
- **HuggingFace models**: See table above. Requires `HF_TOKEN` authentication for gated models (Gemma, Llama).
- **Python venv**: `.venv/` with torch, transformers, accelerate, scikit-learn, plotly, numpy
- **Hardware**: Apple Silicon Mac with MPS backend. Models run in bfloat16.
