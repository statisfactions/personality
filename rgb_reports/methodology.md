# Methodology: Instruments, Tools, and Code

## Overview

This project measures LLM personality using several complementary readouts. The
first three are the original (W1–W7) core; the rest were added W8–W13.

1. **Likert self-report surveys** (§1) — standard psychometric instruments administered via logprobs
2. **Representation engineering (RepE)** (§2) — extracting trait/facet direction vectors from hidden states
3. **Binary-choice behavioral scenarios** (§3) — A/B preference via logprobs (one scenario, two options on the same trait dimension; not "forced-choice" in the Thurstonian/multi-trait sense from the psychometrics literature)
4. **Natural-persona composition** (§4.5) — first-person personas built from validated IPIP behavioral items, decoupling persona text from extraction vocabulary (W8)
5. **Graded forced-choice + Thurstonian IRT (GFC/TIRT)** (§5) — desirability-matched item *blocks*; TIRT recovers normative scores from ipsative data (W10–W11). This is the third readout alongside Likert and RepE.
6. **Cross-language facet geometry** (§6) — the IPIP geometry battery run on English vs Mandarin IPIP-120 (W13)
7. **Adjective / embedding geometry** (§7) — single-adjective representations vs the human inter-adjective correlation matrix, with sentence-encoder baselines (W13)

The first three measure *different constructs* (see "Three Constructs" in `report_week2.md`, §10): representation ≠ preference ≠ free-text. **For the extraction-direction methods behind §2/§5/§7 — `meandiff-itempc1`, the `single-*` family, adaptive denoise, encoder baselines — see the companion `representation_vector_methods.md`, which is the source of truth for formulas.** This file covers instruments, scripts, and flags.

> **Coverage note (2026-06-01):** §§1–4 are the mature core. §§5–7 were added to track the W10–W13 work; they summarize pipelines documented in full in the corresponding weekly reports (`report_week10.md`–`report_week13.md`).

---

## 1. Likert Self-Report Surveys

### IPIP-300 (Big Five)

- **Instrument**: IPIP-NEO-300, 300 items, 60 per Big Five trait
- **Traits**: Neuroticism (N), Extraversion (E), Openness (O), Agreeableness (A), Conscientiousness (C)
- **Source**: Public domain (Goldberg IPIP-NEO-300, ipip.ori.org). Items + scale keying in `instruments/ipip300.json`. Item texts originally transcribed from `vendor/personality_in_llms/admin_sessions/prod_run_01_external_rating.json`.
- **Response scale**: 1-5 Likert (very inaccurate to very accurate)
- **Scoring**: Reverse-keying per scale definition, mean across items
- **Script**: `scripts/run_ipip300.py` (was `run_ollama_logprobs.py` through week 6)
  - `--model MODEL` — short name (Gemma/Llama/Phi4/Qwen/Gemma12/Llama8/Qwen7/...) or HF repo ID
  - `--variants` — run 4 prompt phrasings for ICC reliability analysis
  - `--items N` — limit to first N items for quick tests
- **Backend**: HuggingFace Transformers via `scripts/hf_logprobs.py`. Replaces the Ollama `/api/generate` path used in weeks 1–6.
- **Output**: `results/surveys/<model>_ipip300.json` (multi-variant data embedded inline when `--variants` is used; no separate `_variants.json` file)
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
- **Output**: `results/surveys/<model>_hexaco100.json` (multi-variant data embedded inline; no separate `_variants.json` file)
- **Measures collected**: Same as IPIP-300

### HEXACO-60 (shorter version)

- **Instrument**: HEXACO-PI-R 60-item, 10 items per trait
- **Source**: `instruments/hexaco60.json`
- **Status**: Instrument file created but not run. Use `run_hexaco.py` with minor modification to point at `hexaco60.json`.

### BFI-44 (Big Five Inventory)

- **Instrument**: 44-item Big Five Inventory (copyrighted)
- **Source**: BFI items are copyrighted; we never ran BFI. Upstream `vendor/personality_in_llms/admin_sessions/prod_run_01_external_rating.json` ships them as `[REDACTED]`. A future BFI run would need to (a) obtain item texts separately and (b) write a fresh `instruments/bfi44.json` modeled on `instruments/ipip300.json`.
- **Status**: Not run.

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

The cohort grew in three waves. Short names and repos are the source of truth in
`scripts/hf_logprobs.py` (`MODELS` dict); `resolve()` passes unknown strings through.

**Original small cohort (W1–W6), with best RepE layer:**

| Short name | HuggingFace ID | Layers | Hidden dim | Best RepE layer |
|---|---|---|---|---|
| Gemma (gemma3) | google/gemma-3-4b-it | 34 | 2560 | 14 |
| Qwen (qwen2.5) | Qwen/Qwen2.5-3B-Instruct | 36 | 2048 | 19 |
| Phi4 (phi4) | microsoft/Phi-4-mini-instruct | 32 | 3072 | 9 |
| Llama (llama3.2) | meta-llama/Llama-3.2-3B-Instruct | 28 | 3072 | 12 |

**Phase-1 larger cohort (W7, SAE-covered):** `Gemma12` (gemma-3-12b-it), `Gemma27` (gemma-3-27b-it), `Llama8` (Llama-3.1-8B-Instruct), `Qwen7` (Qwen2.5-7B-Instruct).

**W12 §6 scale-up (M5 Max 128GB):** `Qwen32` (Qwen2.5-32B-Instruct), `Gemma4` (gemma-4-31B-it, needs transformers ≥ 5.5), `Gemma4MoE` (gemma-4-26B-A4B-it), `Qwen36` (Qwen3.6-35B-A3B, thinking-on by default).

**W13 §8.2 distribution/architecture outliers:** `Aya` (CohereLabs/aya-expanse-8b, multilingual SFT) and `FalconMamba` (tiiuae/falcon-mamba-7b-instruct, pure SSM, no attention) — used to test the axis-of-the-models hypothesis. (A third candidate, *Mr. Chatterbox*, was struck: ships a checkpoint without its custom BPE, so unrunnable as published.)

**Sentence encoders (W13 baseline, non-causal control):** `BAAI/bge-large-en-v1.5`, `sentence-transformers/all-mpnet-base-v2`. See §7 and `representation_vector_methods.md` → `encoder-baseline`.

Default dtype is **bfloat16** on MPS for every model (the original float16-NaN issue on Gemma was the reason; bf16 is now universal). The "common layer" for facet/adjective work is `round(n_layers * 2/3)`, not a per-trait swept layer.

---

## 3. Binary-Choice Behavioral Scenarios

Note on terminology: we use "binary choice" for our single-trait A/B scenarios to avoid conflicting with the psychometrics literature's "forced choice," which specifically means pitting items from *different* trait dimensions against each other (and requires Thurstonian IRT for non-ipsative scoring; see Brown & Maydeu-Olivares). The planned trait-conflict instrument below is true forced choice.

### Single-Trait Binary Choice

- **Format**: Present scenario + two options (high vs low on one trait), measure A/B logprob preference
- **Script**: Inline in validation pipeline; results in `results/binary_choice_6trait.json`
- **Limitation**: Near-ceiling for H, C, O (all models pick prosocial option 17-20/20). Real signal only on E (near chance), A and X (intermediate).
- **Caveat**: Position bias matters. Scoring a pair in a single A/B ordering mixes content read with position preference; proper evaluation averages across both orderings. See `rgb_reports/report_week5_meandiff.md` §8.

### Graded forced-choice, single-construct (BUILT — W11)

A desirability-matched **graded forced-choice** instrument (IPIP-NEO-GFC-60) *was*
built in W11 and is now the project's third readout — but it is **single-construct
GFC** (Big Five, desirability-matched blocks), not the trait-vs-trait conflict
design below. Full pipeline, instruments, and scoring are in §5.

### Trait-Conflict Forced Choice (STILL planned)

- **Format**: Scenarios where two *positive* traits conflict (e.g., honesty vs kindness) — this is the trait-conflict sense, distinct from the desirability-matched GFC-60 of §5.
- **Instrument**: Not yet built. Design based on HEXACO pairwise combinations (15 trait pairs).
- **Motivation**: the ceiling-effect breaker (BC/RepE saturate on H/C/O). Okada et al. (2026) show desirability-matched GFC + TIRT is viable on Big Five (the §5 work replicates this); extending it to HEXACO trait-vs-trait conflicts is the open design problem. See `to_try.md` #1.
- **Scoring**: Thurstonian IRT (Brown & Maydeu-Olivares) to recover normative scores from ipsative pair data — same machinery as §5, different block construction.

---

## 4. Analysis Scripts

### Denoised Mixture Model

- **Script**: `scripts/analyze_denoised.py`
- **Input**: Variant result files (mean across prompt phrasings as denoised estimate)
- **Output**: Variance decomposition (shared assistant + genuine unique + noise), inter-model correlations, residual personality profiles

### Existing PsyBORGS Infrastructure (vendored, unused by our scripts)

- **Original paper code**: `vendor/personality_in_llms/psyborgs/survey_bench_lib.py` (session administration), `vendor/personality_in_llms/psyborgs/score_calculation.py` (scoring)
- **Inference scripts**: `vendor/personality_in_llms/inference_scripts/run_gpt_inference.py` (OpenAI API), `vendor/personality_in_llms/inference_scripts/run_hf_inference.py` (HuggingFace/vLLM)
- **Admin sessions**: `vendor/personality_in_llms/admin_sessions/` — JSON configs for BFI, IPIP-300, PANAS, etc. with 50 PersonaChat biographical preambles. IPIP-300 items extracted to `instruments/ipip300.json`.

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

## 5. Graded Forced-Choice + Thurstonian IRT (W10–W12)

The third readout. Where Likert reads self-report and RepE reads the residual
stream, GFC/TIRT reads *relative preference between desirability-matched items*
and recovers normative trait scores via Thurstonian IRT — structurally immune to
the social-desirability "assistant shape" that flattens Likert.

### Instruments

- `instruments/okada_gfc30.json` — Okada et al. (2026) GFC-30 Big Five blocks (the replication target / starting instrument).
- `instruments/ipip_neo_gfc_P60.json` — our IPIP-NEO-GFC-60, built in W11 from the IPIP-NEO-300 pool (60 desirability-matched forced-choice pairs). `_fp.json` is the first-person-phrased variant (W11 Phase D diagnostic).

### Pipeline (W11 phases)

- **Phase A — `scripts/rate_desirability_cohort.py`**: validate that open cohort models can act as desirability raters (replacing Okada's frontier GPT-5 + Gemini raters).
- **Phase B — `scripts/rate_desirability_ipip300.py`**: rate all 300 IPIP-NEO items for social desirability with the cohort raters. (`rate_desirability_pilot_ipip_phrasing.py` is the prompt-phrasing pilot.)
- **Phase C — `scripts/ipip_gfc_pair_mip.py`**: solve Okada Appendix C's two-stage mixed-integer program to pair items into desirability-matched forced-choice blocks → emits the GFC-60.
- **Phase D**: administer + fit. Surprise *negative* recovery result on the first-person variant (see `report_week11.md` §6).

### Administration

- `scripts/run_gfc_hf.py` — administer GFC via HuggingFace (chat-template) across the cohort.
- `scripts/run_gfc_anthropic.py` — same via the Anthropic API (Haiku 4.5 etc.).
- `scripts/run_gfc_ollama.py` — Ollama path.

### TIRT scoring + comparison

- TIRT fitting is done in the vendored R/Stan pipeline under `psychometrics/` (Okada GFC Stan drivers; note the L/R-swap bug fixed in `a007852`). The fitter derives instrument metadata from the response records, not an external file (`1c55281`).
- `scripts/persona_w11_gfc_comparison.py`, `scripts/persona_w12_gfc_p60_comparison.py` — compare GFC-30 vs GFC-60 TIRT recovery against Rep and Likert.
- **Headline (W10–W12)**: TIRT recovers persona z's at substantially *lower* magnitude than Rep or Likert (cohort means: Likert 0.75 > Repr 0.49 > TIRT 0.31), and is SDR-immune by construction (W12 §5b). The W12 §2 loading diagnostic found cohort-aggregated `a_pos` pinned at the HalfNormal prior mean — the data are weakly informative per item; per-item *relative* loadings still show a clean assistant-shape pattern.

### The 270-cube (W12 §7)

`scripts/persona_w12_cube_dashboard.py` renders the full **3-method × 3-persona-form × 3-condition × 10-model** cube (method ∈ {Likert, Repr, TIRT}; condition ∈ {neutral, fake-good prefix, fake-good suffix}). Prefix caching fills the cube; see `report_week12.md` §7. The cube is the basis for the "axis-of-the-models" reframe of cross-model agreement.

---

## 6. Cross-Language Facet Geometry (W13 §3.8)

Runs the IPIP facet-geometry battery in two languages to test whether the
assistant-shape persona and facet structure survive translation.

- **Instruments**: `instruments/ipip120_english.json`, `instruments/ipip120_mandarin.json` (IPIP-NEO-120, EN + ZH), with `*_facet_map.json` companions. `instruments/ipip120_human_facet_correlations.json` is the human comparison target.
- **Builders**: `scripts/build_ipip120_english.py`, `scripts/build_ipip120_mandarin.py`.
- **Analysis**: `scripts/repr_crosslang.py` (facet-geometry recovery EN vs ZH at the representation level), `scripts/compare_en_zh_persona.py` (persona induction across languages), `scripts/score_ipip120_subset.py`.
- **Headline (W13 §3)**: English-dominant models go near-uniform in Mandarin; Anger drops; political items collapse to neutral. **Language perturbation ≫ format perturbation.** See `report_week13.md` §3.

---

## 7. Adjective / Embedding Geometry (W13 §3.9–3.11)

Compares how a model organizes single trait-adjectives against the *human*
inter-adjective correlation matrix — the cleanest test of "is the model's trait
geometry lexical (word-embedding-like) or behavioral?" Extraction-direction
formulas (adaptive denoise, encoder baseline) live in
`representation_vector_methods.md`; this section is the data + script map.

### Human substrate

- **525-PDA** (Saucier, Harvard Dataverse `doi:10.7910/DVN/GHYMEV`) — 700 respondents rate ~525 personality adjectives on a 1–7 scale; theory-neutral (not Big-Five-scaffolded). We use 523 after dropping 2 corrupted columns.
- Built matrices: `results/adjectives/escs_525pda_corr_raw.json` (`correlation_matrix` + `labels`). Provenance and the corrupted-column caveat are in `bibliography.md` (525-PDA / 360-PDA entries).
- **`scripts/fetch_external_data.py`** clones the external sources (525-PDA, 360-PDA, Cutler & Condon) into `data/` so the matrices can be rebuilt.

### Model extraction

- **`scripts/extract_adjectives.py`** — 12 models × 4 framings × 523 adjectives → `results/adjectives/acts/<model>__<framing>.pt` (fp32). Framings: `self`, `pers` (canonical), `desc`, `bare`. `pers` ("My personality is {adj}") recovers the human matrix best.
- **`scripts/adjective_geom.py`** — shared `adaptive_denoise` (IPR-routed center / center+top-1) and `model_matrix` (load cache at 2/3 depth, align to labels, return cosine matrix). Used by everything below.

### Analyses

- `affect_analysis.py`, `affect_rsa.py` — the affect block: presence-vs-valence RSA. The "affect-merge" (Cheerful ≈ Angry) is a separable lexical axis; valence is present but weak (b > 0 all 12 models, but presence ≈ valence vs human valence ≫ presence).
- `adjective_factor_congruence.py`, `factor_congruence_grid.py` — Tucker congruence of model varimax factors to human Big Five. A/E/N recover (~0.5–0.6), C weak (0.35), **O essentially absent (0.08)**.
- `scree_parallel.py` — Horn parallel analysis: 5 is the Big-Five convention, not the elbow (~34 human / ~70 model factors above null).
- `intensity_vs_valence.py` — capstone: the same presence > valence pattern holds for general *evaluation* (Wonderful ≈ Awful). The model organizes adjectives by **intensity first, sign second**.
- `factor_ladder_figure.py` — varimax factors k=1→10, human vs model side by side.
- `embedding_facet_baseline.py` + the encoders — the lexical control (see §2 models table and `representation_vector_methods.md`).
- Support: `adjective_corr_cluster.py`, `adjective_audit.py` (deny-list QA), `adjective_model_human.py`, `adjective_factor_heatmap.py`, `adjective_hclust_heatmap.py`, `adjective_affect_heatmap.py`, `factor_rotation_compare.py`.
- **Headline**: Big Five is *not* the LLM's adjective organization — affect / evaluation / interpersonal is, with the Five a thin overlay. Full write-up in `report_week13.md` §3.11.

---

## 8. Environment

- **Local inference**: HuggingFace Transformers (bf16, MPS). Ollama is no longer used by the survey/BC pipelines as of 2026-04-24 — remains available for the chat UX and the `run_gfc_ollama.py` path, but not called by the core measurement scripts.
- **HuggingFace models**: See §2 table. Requires `HF_TOKEN` authentication for gated models (Gemma, Llama).
- **Python venv**: `.venv/` with torch, transformers, accelerate, scikit-learn, plotly, numpy, sentence-transformers (encoders), pyreadstat (`.por`/`.sav` human data).
- **Hardware**: Apple Silicon Mac (M5 Max, 128 GB RAM as of 2026-04-24), MPS backend. All models run in bfloat16; scales run so far span 3B–35B with no quantization. The vendored R/Stan TIRT fitter lives under `psychometrics/`.
