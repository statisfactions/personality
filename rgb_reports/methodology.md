# Methodology: Instruments, Tools, and Code

## Overview

This project measures LLM personality using three complementary approaches:

1. **Likert self-report surveys** — standard psychometric instruments administered via logprobs
2. **Representation engineering (RepE)** — extracting trait direction vectors from hidden states
3. **Binary-choice behavioral scenarios** — A/B preference via logprobs (one scenario, two options on the same trait dimension; not "forced-choice" in the Thurstonian/multi-trait sense from the psychometrics literature)

Each approach measures a different construct (see "Three Constructs" in `reports/report_week2.md`, Section 10).

---

## 1. Likert Self-Report Surveys

### IPIP-300 (Big Five)

- **Instrument**: IPIP-NEO-300, 300 items, 60 per Big Five trait
- **Traits**: Neuroticism (N), Extraversion (E), Openness (O), Agreeableness (A), Conscientiousness (C)
- **Source**: Public domain, items in `admin_sessions/prod_run_01_external_rating.json` (key: `measures.IPIP300`)
- **Response scale**: 1-5 Likert (very inaccurate to very accurate)
- **Scoring**: Reverse-keying per scale definition, mean across items
- **Script**: `scripts/run_ipip300.py` (was `run_ollama_logprobs.py` through week 6)
  - `--model MODEL` — short name (Gemma/Llama/Phi4/Qwen/Gemma12/Llama8/Qwen7/...) or HF repo ID
  - `--variants` — run 4 prompt phrasings for ICC reliability analysis
  - `--items N` — limit to first N items for quick tests
- **Backend**: HuggingFace Transformers via `scripts/hf_logprobs.py`. Replaces the Ollama `/api/generate` path used in weeks 1–6.
- **Output**: `results/<model>_ipip300.json` (single variant), `results/<model>_variants.json` (with prompt variants)
- **Measures collected**: argmax, expected value (EV), Shannon entropy, full probability distribution over {1,2,3,4,5}

### HEXACO-100

- **Instrument**: HEXACO-PI-R 100-item, 16 items per trait + 4 Altruism (interstitial)
- **Traits**: Honesty-Humility (H), Emotionality (E), Extraversion (X), Agreeableness (A), Conscientiousness (C), Openness (O), plus Altruism
- **Source**: Free for non-profit research from hexaco.org, items in `instruments/hexaco100.json`
- **Response scale**: 1-5 Likert (strongly disagree to strongly agree)
- **Scoring**: Reverse-keying per `hexaco100.json` scale definitions, mean across items
- **Script**: `scripts/run_hexaco.py`
  - Same flags as `run_ipip300.py`
  - Requires `PYTHONPATH=scripts` to import shared functions (`hf_logprobs`, `run_ipip300`)
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

Three additional prompt variants change the framing (agreement scale, describes-me, terse). All variants use **bare-text** prompts — no chat template is applied for the Likert scoring. See to_try.md §15 for a bookmark on whether that choice is load-bearing for the Likert↔BC cross-method correlations. Thinking-model handling (Qwen3 `/no_think`) was dropped in the HF port; none of the planned cohort is a thinking model.

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

- **Script**: `scripts/extract_trait_vectors.py` (activation collector only — does not fit directions)
  - `--model MODEL` — HuggingFace model name
  - `--trait H` — single trait (default: all 6)
  - `--dtype bfloat16` — critical for Gemma3 (float16 causes NaN at layer 7+)
- **Process**:
  1. Run high and low prompts through model, extract last-token hidden states at each layer (prompts end in `.`, so last token = period token)
  2. Compute activation differences (high − low) for each pair
  3. Save per-pair diffs to disk
- **Output**: `results/repe/<model>_<trait>_directions.pt` — contains `raw_diffs` of shape (n_pairs, n_layers+1, hidden_dim), plus metadata. Filename is historical; content is per-pair activation differences, not a single direction vector.
- **Direction fitting**: Done downstream by `validate_protocol.py`, `cross_method_matrix.py`, `optimize_steering.py`, `compare_steering_objectives.py`. Each loads `raw_diffs`, selects the best layer by 5-fold CV LDA accuracy, and fits `sklearn.LinearDiscriminantAnalysis` at that layer to produce a unit-norm trait direction.
- **Key finding**: Use LDA, not PCA. PCA PC1 is a content-free activation norm artifact (r=1.0 with norm) in pre-norm transformers.

### Format-Invariant Measurement Protocol

Measure at the **period token** after the scenario, before any response format:

```
Consider what a person most like you would do in the following situation: [Scenario].
                                                                                   ^ measure here
```

Causal attention guarantees the period-token hidden state is identical regardless of what follows (r=1.000 across free-form vs binary-choice).

### Validation Script

- **Script**: `scripts/validate_protocol.py`
  - `--model MODEL` — HuggingFace model name; `--short-name NAME` optional display label
  - `--test layer,framing,likert,rottger,transfer` — select tests
  - `--all-models` — run on all 4 models
- **Tests**:
  1. **Layer sensitivity** — projection stability across layers (±2-3 layer window)
  2. **Framing sensitivity** — robustness to preamble text (r > 0.85 all pairs)
  3. **Cross-model transfer** — do model A's directions work on model B?
  4. **RepE vs Likert** — period-token projection vs self-report EV correlation
  5. **Röttger test** — binary-choice vs free-text agreement (40-80% depending on model)
- **Output**: `results/validation_<model>.txt`

### Models Tested (HuggingFace)

| Short name | HuggingFace ID | Layers | Hidden dim | Best RepE layer |
|---|---|---|---|---|
| gemma3 | google/gemma-3-4b-it | 34 | 2560 | 14 |
| qwen2.5 | Qwen/Qwen2.5-3B-Instruct | 36 | 2048 | 19 |
| phi4 | microsoft/Phi-4-mini-instruct | 32 | 3072 | 9 |
| llama3.2 | meta-llama/Llama-3.2-3B-Instruct | 28 | 3072 | 12 |

---

## 3. Binary-Choice Behavioral Scenarios

Note on terminology: we use "binary choice" for our single-trait A/B scenarios to avoid conflicting with the psychometrics literature's "forced choice," which specifically means pitting items from *different* trait dimensions against each other (and requires Thurstonian IRT for non-ipsative scoring; see Brown & Maydeu-Olivares). The planned trait-conflict instrument below is true forced choice.

### Single-Trait Binary Choice

- **Format**: Present scenario + two options (high vs low on one trait), measure A/B logprob preference
- **Script**: Inline in validation pipeline; results in `results/binary_choice_6trait.json`
- **Limitation**: Near-ceiling for H, C, O (all models pick prosocial option 17-20/20). Real signal only on E (near chance), A and X (intermediate).
- **Caveat**: Position bias matters. Scoring a pair in a single A/B ordering mixes content read with position preference; proper evaluation averages across both orderings. See `rgb_reports/report_week5_meandiff.md` §8.

### Trait-Conflict Forced Choice (planned)

- **Format**: Scenarios where two positive traits conflict (e.g., honesty vs kindness) — this IS forced choice in the literature sense
- **Instrument**: Not yet built. Design based on HEXACO pairwise combinations (15 trait pairs).
- **Motivation**: ACL 2025 paper (Decoding LLM Personality: Forced-Choice vs. Likert) confirms forced-choice discriminates LLM personalities better. Validated trait-conflict instruments don't exist for HEXACO (for humans or LLMs).
- **Scoring**: Will require Thurstonian IRT (Brown & Maydeu-Olivares) to recover normative scores from ipsative pair data.

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

## 4.5. Persona composition from validated IPIP-NEO behavioral items (W8)

The W7 §11.5.10 prereg used `instruments/synthetic_personas.json` descriptions composed of Goldberg adjective markers (e.g. "You are very extraverted, very energetic, very talkative..."). These produced strong rep recovery (mean r ≈ 0.74) and very strong Likert recovery (mean r ≈ 0.89). Two concerns: (a) the marker-rich form may put the model in an analytic mode that doesn't generalize to natural prose; (b) the rep result is partly tautological since marker-based directions decode marker-rich prompts.

The natural-persona track replaces the marker form with first-person behavioral self-descriptions assembled from validated IPIP-NEO-300 behavioral items (Goldberg/Johnson 1999). This decouples persona description from the trait-direction extraction vocabulary while preserving psychometric validity — IPIP items have published trait/facet loadings in large human samples.

### Pipeline

- **`instruments/ipip300_annotations.json`** — per-item annotations: intensity tier (mild/strong), deny-list, typo overrides. Compact format (records deviations from defaults). Includes top-level `_method` block with rubric, selection rule, and counts. Frozen artifact, intended to be edited and re-versioned over time.
- **`scripts/persona_ipip_compose.py`** — composer. Takes `synthetic_personas.json` z-scores and stanines, emits `instruments/synthetic_personas_ipip.json` with `ipip_raw` natural prose per persona.

### Annotation rubric (intensity tier)

- **Default tier: mild.** Mundane, hedged, behavioral statements. "I worry about things", "I leave a mess in my room", "I make friends easily".
- **Strong tier:** strongest-within-facet items only. Criteria: absolute language (love/never/always), clinical or near-clinical tone (panic, overwhelmed, suffer, blue, desperate, low opinion of myself), or emphatic content beyond ordinary behavioral description ("plunge into tasks with all my heart", "radiate joy").
- **Loosen-to-mild rules:** "tend to" hedges loosen even loaded content (per rgb 2026-05-02 on ipip179); colloquial "love"/"hate" that doesn't intensify ("I love to eat") stays mild; common idiomatic "always X" ("I am always busy") stays mild.

Constraint: every facet must retain at least one mild-forward and one mild-reverse item after the deny-list is applied. Validated programmatically; one re-tag pass was needed (N.Depression had zero mild-forward items pre-fix).

### Selection rule

Per persona, per trait:
- **K = 6 items** (one per facet, stratified — exactly one item from each of the 6 facets per trait).
- **Polarity ratio by z-band** (function `band_K6` in the composer):
  - z ≥ +1.0 → 6F / 0R
  - +0.3 ≤ z < +1.0 → 4F / 2R
  - |z| < 0.3 → 3F / 3R
  - −1.0 < z ≤ −0.3 → 2F / 4R
  - z ≤ −1.0 → 0F / 6R
- **Tier by stanine:**
  - Stanines 3–7 (|z| roughly ≤ 1) → MILD items only
  - Stanines 1–2, 8–9 → MILD + STRONG mixed (drawn uniformly from union)
- **Per-persona deterministic RNG:** seeded from `persona_id + global_seed`, so the same persona always produces the same composition.
- **Master shuffle:** the 30 items are shuffled at output so trait order isn't preserved in the prose.
- **Fallback:** if a (facet, polarity, tier) cell is empty, drop the tier filter; if still empty, draw from another facet's same-polarity pool. Fallback events are counted and reported. Over 400 personas × 30 picks = 12,000 picks, 0 fallbacks were observed (validation pass good).

### Deny-list categories

- **Marker-like** (5 items): items that read as a one-word trait label with "I" prepended ("I love action" ≈ "action-oriented", "I radiate joy" ≈ "joyful").
- **Politically/religiously/patriotically charged** (7 items, all from O.Liberalism): items that take partisan/religious/civic positions. These introduce RLHF response priors that aren't about personality. Liberalism facet retains 3 of 10 items post-deny.
- **Semantically odd** (1 item): "I love flowers" — too narrow content for a general persona.

### Outputs

- `instruments/synthetic_personas_ipip.json` — 400 composed personas with `ipip_raw` text and per-pick provenance (trait, facet, polarity, item ID).
- Length: ~165–185 words per persona (vs ~128–192 for marker-rich originals; same ballpark).
- Companion `ipip_reflowed` field is OPTIONAL and produced separately by Sonnet paraphrase; the raw-vs-reflow contrast isolates stylistic naturalness with content held constant. Not yet implemented.

### Methodological notes

- **Liberal facet has limited variability** (3 of 10 items remain): for high-O personas, the same Liberalism-forward item appears across many personas (only 1 mild-forward Liberal item remains). Acceptable for a stratified composition where Liberalism is one of 6 facets, but worth noting as a low-diversity slot.
- **Strong items are mostly forward-keyed.** Reverse-keyed strong items are rare in IPIP (~3-5 across 300 items). Low-trait extreme personas (stanines 1-2) therefore rarely sample strong reverse items even when the rule allows it. Composer falls back to mild-only reverse — this is fine; the trait expression is encoded mostly through item polarity rather than reverse-keyed intensity.

---

## 5. Environment

- **Local inference**: HuggingFace Transformers (bf16, MPS). Ollama is no longer used by the survey/BC pipelines as of 2026-04-24 — remains available for the chat UX but not called by any script.
- **HuggingFace models**: See table above. Requires `HF_TOKEN` authentication for gated models (Gemma, Llama).
- **Python venv**: `.venv/` with torch, transformers, accelerate, scikit-learn, plotly, numpy
- **Hardware**: Apple Silicon Mac with MPS backend. Models run in bfloat16.
