# Why our Okada et al. (2026) ground-truth recovery fell short

**Date:** 2026-04-26
**Author:** ECB
**Scope:** Diagnose the gap between Okada et al. (2026, arXiv:2602.17262) Figure 4 (most GFC points in the r ≥ 0.50 band) and our two completed GFC-30 + TIRT runs (Gemma3 12B and Phi4-mini), where TIRT recovery is patchy at best and globally sign-flipped at worst.

---

## 1. What we observed

Both runs use the same instrument (Okada Table 3, 30 desirability-matched IPIP pairs), the same persona generator (van der Linden 2010 Σ → stanines → Goldberg adjectives → Okada Appendix F.1 preamble), and the same honest-condition prompt (Appendix F.2). 400 personas per model — eight times Okada's N=50.

| Trait | Gemma3 12B TIRT | Gemma3 12B Simple | Phi4-mini TIRT | Phi4-mini Simple | Okada GFC (typical) |
|-------|-----------------|--------------------|----------------|-------------------|---------------------|
| A | **−0.27** | 0.46 | **−0.39** | 0.28 | ≥ 0.50 |
| C | 0.08 | 0.25 | **−0.20** | 0.09 | ≥ 0.50 |
| E | 0.65 | 0.35 | **−0.39** | 0.49 | ≥ 0.50 |
| N | 0.53 | 0.42 | **−0.34** | 0.09 | ≥ 0.50 |
| O | 0.68 | 0.44 | **−0.58** | 0.51 | ≥ 0.50 |

Three pathologies stand out:

1. **Phi4-mini: global sign flip on every trait** under TIRT — a textbook label-switching / rotational-indeterminacy signature.
2. **Gemma3: A flips, C collapses to ~0**, even though E/N/O are in the Okada band.
3. **Inter-trait correlation matrices are nonsense** under TIRT (|r| > 0.9 for several pairs, vs. ground-truth |r| ≤ 0.43 from van der Linden).

Simple (sign-corrected mean endorsement) scoring is more uniform across traits but, as expected for ipsative scoring, is strictly capped below true recovery and does not estimate inter-trait structure.

## 2. Differences from Okada et al.

### 2.1 Models — the dominant difference

| Dimension | Okada | Us |
|-----------|-------|-----|
| Models | GPT-5 / 5-mini / 5-nano, Gemini 2.5 Pro/Flash/Flash-Lite, Claude Opus/Sonnet/Haiku 4.5 | Gemma3 12B, Phi4-mini (~3.8B), Llama2-7B-chat, Llama3.2-3B (Q4 quant via Ollama on Orin) |
| Scale | Frontier instruction-tuned, full precision | 3B–12B open weights, 4-bit quantized |
| Instruction-following fidelity | High; faithfully roleplays detailed personas | Documented weakness on multi-trait persona adoption |

Recovery in this paradigm is bottlenecked by *how faithfully the model's per-pair choice tracks the persona's adjective vector*. The frontier-vs-open-weights gap is the most parsimonious explanation for the bulk of the recovery deficit, especially for Phi4-mini, which is roughly an order of magnitude smaller than the smallest Okada model (GPT-5 nano).

Evidence inside our own data: Gemma3 (12B) recovers E/N/O at Okada-band magnitudes (TIRT r = .53–.68); Phi4-mini (3.8B) recovers nothing under TIRT and only E/O under simple scoring. Capacity matters even within our small-model regime.

### 2.2 Response behavior — the mechanism

The 7-point bipolar scale is supposed to be used gradedly. Our models don't:

| Category | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|----------|---|---|---|---|---|---|---|
| Gemma3 (n=12,000) | 14% | 12% | **0.2%** | 40% | 8% | 3% | 23% |
| Phi4-mini (n=12,000) | **45%** | 4% | 14% | 6% | **0%** | 29% | 2% |

- **Gemma3** essentially never picks "3" (slight LEFT) and uses the midpoint "4" 40% of the time. It treats GFC as binary-with-a-midpoint.
- **Phi4-mini** never picks "5" (slight RIGHT), and is heavily LEFT-biased (45% of all responses are "1"), even though L/R is randomized per item. The L/R asymmetry is a positional artifact, not signal.

This violates two TIRT assumptions at once:
1. **Graded ordinal structure**: the cumulative-link model assumes monotone ordered thresholds κ_p1 < … < κ_p6. If a category is empty, the corresponding thresholds collapse and the item carries reduced or no information.
2. **L/R exchangeability after randomization**: positional bias adds a non-trait nuisance dimension that the TIRT model has no parameter for. With ~40% of Phi4 variance being "left vs. right" rather than "trait A vs. trait B", the latent rotation is dominated by the nuisance — hence the global sign flip.

Okada's frontier models presumably use the full 7-point scale with much milder positional bias. Their Figure 2 polygon plots show clear within-trait gradation, which would be impossible if categories were missing.

### 2.3 IRT scoring backend

| Aspect | Okada | Us |
|--------|-------|-----|
| Implementation | Custom Stan (Appendix D): weakly informative priors, "standard identifiability constraints" | `thurstonianIRT` R package (Bürkner) — Stan backend, but fixed loadings ±1/√2, default flat priors on Σ_θ, package-defined identifiability |
| Chains × iter | Not stated, but presumably tuned (low-rhat, sufficient ESS) | 2 chains × 1000 iter (Gemma3); 4 × 1000 (Phi4). Both produced low-ESS warnings. |
| Loadings | Free signed discrimination a_j (positive magnitude × keying sign) | Fixed magnitude (the package's TIRT parameterization) |
| Σ_θ prior | Likely LKJ or similar regularizer | Uninformative — known to allow |r|→1 with sparse or ipsative data |

Two consequences:

- **Rotational ambiguity is not regularized.** With only 30 blocks and a 5-D latent space, multiple equivalent rotations of θ produce identical likelihoods. Without an LKJ prior on Σ_θ or anchoring constraints, the chain is free to land in the global sign-flipped basin (Phi4) or a partial-flip basin (Gemma3 A). Okada's "standard identifiability constraints" almost certainly handle this; the package does not, by default, for data this thin in informative L/R contrasts.
- **Implausible inter-trait correlations** (|r|>0.9 in our reports) are the diagnostic signature of an under-identified Σ_θ given ipsative-leaning data. Okada's recovered Σ̂_θ is not reported in the body, but the recovery numbers in their Figure 4 imply a well-identified solution.

We had already flagged this in the project log ("low ESS warnings", "implausible inter-trait correlations"). It is the *proximate* cause of the sign flips; the small models are the *distal* cause (they don't generate enough graded comparative variance for any TIRT implementation to identify cleanly).

### 2.4 Constrained decoding / API quality

| | Okada | Us |
|--|-------|-----|
| Backend | Official APIs, batch mode | Ollama HTTP, prompt-only constraint |
| Output policing | Numeric-only via API contract | Prompt-based; some parses fall through |
| Quantization | None | Q4 K_M for Gemma3; default Ollama quants elsewhere |

Quantization can compress logits enough to flatten the GFC distribution toward the most-trained-on tokens, which plausibly contributes to the cat-2/cat-3/cat-5 dropouts and to positional bias.

### 2.5 What we replicated correctly

To be clear about what *isn't* the problem:

- ✅ **Inventory:** byte-for-byte Okada Table 3 (30 pairs, max ΔSD = 0.18, mean 0.03).
- ✅ **Personas:** MVN(0, Σ_vdL), stanine mapping, Goldberg markers, Appendix F.1 preamble exactly.
- ✅ **Honest prompt:** Appendix F.2 wording verbatim.
- ✅ **L/R randomization** with seeded reproducibility.
- ✅ **N is large enough** — 400 ≫ 50.

The gap is not in the experimental design. It is in (model capacity) × (scoring backend identifiability) given (small-model GFC response behavior).

## 3. Why TIRT looks worse than simple scoring (Phi4 especially)

This is counter-intuitive — TIRT should beat simple scoring, not be 1.0 worse. The mechanism:

- **Simple scoring is local**: each trait's score is a sign-corrected mean over the 12 pairs touching that trait. A globally biased response (e.g., Phi4's 45% "1") cancels at the trait level once L/R signs are applied across pairs, leaving the persona-driven *deviations* from that bias. Recovery is attenuated (ipsative ceiling) but not flipped.
- **TIRT is global**: the latent rotation is jointly estimated across all 5 traits. When the dominant variance source is positional ("LEFT vs. RIGHT") rather than trait-relevant ("trait A vs. trait B"), the chain can land in a basin where every loading has the wrong sign and every trait flips together. That is exactly the Phi4 pattern.

Simple scoring is robust to positional bias *because* it is mechanically symmetric in L/R. TIRT is not.

## 4. Recommended next steps, ordered by expected ROI

1. **Run a frontier model on the same pipeline.** A single Claude Haiku 4.5 or Gemini Flash-Lite run on the existing 400 personas would tell us how much of the gap is "model capacity" vs. "our pipeline." Cheap, fast, decisive.
2. **Custom Stan TIRT with LKJ(η=2) on Σ_θ and ordered-threshold soft constraints.** The `thurstonianIRT` package is convenient but its defaults are not built for thin-information GFC data. Okada's Appendix D Stan code would be the ideal starting point — request from authors or replicate from spec.
3. **Diagnose positional bias before scoring.** For each model, regress response on L/R-of-positively-keyed-item; if the coefficient is large, debias before TIRT (or include a per-respondent positional-bias nuisance parameter).
4. **Soft-evidence TIRT.** Use the full 7-option logprob distribution rather than top-1 argmax. This recovers information from models like Gemma3 that put real mass on multiple categories but argmax to the midpoint. It also bridges back to the Llama-2 soft-evidence work that collapsed for unrelated reasons (Likert homogeneity).
5. **Drop low-information items.** Block 21 (C+/O+) showed near-zero variance in earlier persona runs; if any blocks land with one category dominating ≥80%, they hurt identification more than they help.
6. **Multi-model pooling** for the TIRT fit. Treating model-as-respondent-cluster could borrow strength across the dimensional space, though it muddies the per-model recovery interpretation.

## 5. One-line summary

Okada's Figure 4 is a frontier-model + custom-Stan + clean-API result. We hit it with Q4-quantized 3–12B open weights, the off-the-shelf `thurstonianIRT` package's default priors, and models that systematically refuse to use 1–2 categories of the 7-point scale. The recovery deficit, the sign flips, and the |r|>0.9 trait correlations are all consistent with a single root cause: insufficient identifying information in the response data given the chosen scoring backend.

---

### Appendix: source artifacts

- `instruments/okada_gfc30.json` — 30-pair instrument
- `instruments/synthetic_personas.json` — 400 personas, seed=42
- `scripts/generate_trait_personas.py` — persona generator
- `scripts/run_gfc_ollama.py` — inference script
- [`gemma3-12b_gfc30_synthetic.json`](../psychometrics/gfc_tirt/gemma3-12b_gfc30_synthetic.json), [`phi4-mini_gfc30_synthetic.json`](../psychometrics/gfc_tirt/phi4-mini_gfc30_synthetic.json) — raw responses + logprobs
- `psychometrics/gfc_tirt/gemma3_gfc30_synthetic_tirt_fit.rds`, `phi4_gfc30_synthetic_tirt_fit.rds` — TIRT fits (archived in big5_results)
- [`gemma3_gfc_tirt_report.html`](../psychometrics/gfc_tirt/gemma3_gfc_tirt_report.html), [`phi4_gfc_tirt_report.html`](../psychometrics/gfc_tirt/phi4_gfc_tirt_report.html) — full diagnostic reports
- `notes_background/okada_2026_gfc_paper.md` — clipped paper text
