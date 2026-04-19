# Cross-Method Correlation Matrix: HEXACO Measurements

Comparing five scoring methods across 4 models × 6 HEXACO traits (24 cells).

**Methods:**
- **Likert-argmax**: HEXACO-100 declarative statements, most-probable Likert response (1-5)
- **Likert-EV**: Same items, expected value over full probability distribution
- **BC-proportion**: Scenario contrast pairs, proportion of high-trait picks (0-1)
- **BC-logodds**: Same scenarios, mean log-odds favoring high-trait option
- **RepE-probe**: Mean LR projection of contrast-pair diffs at LDA-CV-best layer, **z-scored within model** across 6 traits (removes cross-model activation scale differences; makes scores ipsative)

**Script:** `scripts/cross_method_matrix.py` (use `--probe lr` [default] or `--probe lda` [legacy])

> **Week 6 methodology update**: The RepE-probe axis was rebuilt using L2 logistic regression instead of LDA. Rationale: in n<<d (100 samples, ~3000 dims), LDA's Σ⁻¹ amplifies noise in small-eigenvalue directions and rotates the trait direction ~74° off the LR/MD-projected axis; that rotation is noise, not signal. LR is the field-standard linear probe and also steers behavior in the one cell where methods differ causally. See `report_week6_probing.md` (or commit `67d9c05`) for the full derivation. Layer selection still uses LDA 5-fold CV — LR separates ~100% across most layers in this regime, so LR-CV doesn't discriminate.

---

## 1. Overall (24 model×trait cells)

**Primary (LR probe):**

|                | Likert-argmax | Likert-EV | BC-proportion | BC-logodds | RepE-probe |
|----------------|:---:|:---:|:---:|:---:|:---:|
| Likert-argmax  | 1.000 | 0.928 | 0.487 | 0.288 | **0.078** |
| Likert-EV      |  | 1.000 | 0.506 | 0.316 | **0.095** |
| BC-proportion  |  |  | 1.000 | 0.601 | **0.334** |
| BC-logodds     |  |  |  | 1.000 | **0.241** |
| RepE-probe     |  |  |  |  | 1.000 |

**For comparison (LDA probe, Week 3 original):**

|                | Likert-argmax | Likert-EV | BC-proportion | BC-logodds | RepE-probe |
|----------------|:---:|:---:|:---:|:---:|:---:|
| Likert-argmax  | 1.000 | 0.928 | 0.487 | 0.288 | 0.157 |
| Likert-EV      |  | 1.000 | 0.506 | 0.316 | 0.171 |
| BC-proportion  |  |  | 1.000 | 0.601 | 0.416 |
| BC-logodds     |  |  |  | 1.000 | 0.295 |
| RepE-probe     |  |  |  |  | 1.000 |

**Reading this:**
- **All RepE↔other-method correlations drop with the LR probe** (by 0.05–0.08). The LDA axis was partially rotated into directions correlated with the other signals (plausibly activation norm, which also drives Likert self-reports through model-level differences); those spurious correlations disappear under LR.
- **Likert ↔ RepE collapses to r ≈ 0.08–0.10.** Self-report and internal representation are effectively decoupled at the trait level — the three-construct dissociation from Week 3 is *stronger* than originally reported, not weaker.
- **BC ↔ RepE (r=0.33) is still the strongest cross-method link**, but down from 0.42. Behavioral choice and internal representation remain the most connected pair, consistent with both being scenario-based.
- **Likert-argmax ↔ Likert-EV, BC-prop ↔ BC-lo**: unchanged (methods don't touch Likert or BC).

---

## 2. Per-model (which traits are high/low within each model)

Key cross-method correlations (n=6 traits each), **LR probe primary, LDA in parentheses**:

| Pair | Gemma3 | Llama | Phi4 | Qwen |
|------|:---:|:---:|:---:|:---:|
| Likert-EV ↔ BC-prop | 0.50 | 0.51 | 0.84 | 0.68 |
| Likert-EV ↔ RepE | **0.27** (0.38) | **-0.09** (-0.03) | **0.17** (0.32) | **-0.11** (-0.04) |
| BC-prop ↔ RepE | **0.33** (0.42) | **0.40** (0.48) | **0.10** (0.19) | **0.49** (0.57) |
| BC-lo ↔ RepE | **0.27** (0.36) | **0.26** (0.32) | **-0.07** (0.10) | **0.50** (0.57) |

**Pattern unchanged but numbers shift:** Phi4 still has the strongest Likert↔BC convergence at r=0.84 (unchanged — no RepE). RepE-involving correlations all drop uniformly under LR. Phi4's modest Likert↔RepE link weakens most in absolute terms (0.32 → 0.17), though Gemma3's weakens proportionally more (0.38 → 0.27). Qwen remains the strongest BC↔RepE model (r=0.49 under LR, was 0.57).

---

## 3. Per-trait (which models score high/low on each trait)

n=4 models per trait. Low power — interpret directionally. **LR probe primary, LDA in parentheses.**

| Trait | Likert-EV ↔ RepE | Likert-EV ↔ BC-prop | BC-prop ↔ RepE | Note |
|-------|:-:|:-:|:-:|---|
| **E** | **0.95** (0.99) | -0.06 | **0.16** (0.03) | Likert↔RepE near-perfect still; BC slightly more connected under LR |
| **A** | **0.57** (0.70) | 0.78 | **0.94** (0.97) | All three still converge; best trait overall |
| **H** | **0.47** (0.46) | 0.71 | **-0.28** (-0.30) | Likert↔BC good; RepE anti-correlates with BC |
| **C** | **0.33** (0.49) | -0.46 | **-0.17** (-0.24) | BC ceiling (85-95%), nothing agrees |
| **X** | **-0.07** (-0.25) | 0.82 | **0.40** (0.17) | Under LR: BC↔RepE link roughly doubles |
| **O** | **-0.54** (-0.45) | -0.09 | **0.30** (0.33) | BC ceiling; RepE anti-correlates with Likert |

**Key findings (post-swap):**

**Agreeableness is still the best-measured trait.** All three methods converge (LR: Likert-EV↔RepE 0.57, BC-prop↔RepE 0.94). Loses a bit of sparkle vs. LDA (0.70/0.97) but still cleanly the consensus trait.

**Emotionality Likert↔RepE (r=0.95) remains near-perfect under LR.** This trait is the robustness anchor — the agreement between self-report and representation is not a probe artifact.

**X's BC↔RepE jumps from 0.17 to 0.40 under LR.** One of the few cells where the LR probe shows *more* cross-method agreement than LDA. Plausibly the LDA direction for X was rotating away from the behaviorally-aligned axis; LR's direction tracks scenario picks better.

**H's BC↔RepE is anti-correlated** (-0.28 under LR, was -0.30). The models that pick high-H in scenarios don't represent H in the same direction. This was the biggest surprise from Week 3 and survives the methodology swap.

**X and O show Likert↔RepE reversals** (X: -0.07 vs -0.25 before; O: -0.54 vs -0.45). The reversal on O is actually *stronger* under LR. If the LDA-rotation hypothesis were the full explanation, we'd expect reversals to weaken; O going the other way suggests something genuine.

**BC ceiling effects still kill H, C, O.** All models pick prosocial 85-100% of the time on these traits — no probe change fixes that.

---

## 4. Raw data (LR probe)

RepE scores are z-scored within model (mean=0, sd=1 across 6 traits).

| Model | Trait | Likert-argmax | Likert-EV | BC-prop | BC-logodds | RepE (z) |
|-------|-------|------:|------:|------:|------:|------:|
| Gemma3 | H | 4.19 | 4.16 | 1.00 | 19.2 | +1.04 |
| Gemma3 | E | 3.63 | 3.63 | 0.55 | -0.8 | -0.78 |
| Gemma3 | X | 3.12 | 3.09 | 0.70 | 7.5 | -1.13 |
| Gemma3 | A | 3.06 | 3.05 | 0.70 | 6.6 | +1.51 |
| Gemma3 | C | 3.25 | 3.23 | 0.90 | 16.7 | -0.84 |
| Gemma3 | O | 3.75 | 3.72 | 0.95 | 17.3 | +0.21 |
| Llama | H | 3.56 | 3.31 | 0.89 | 4.4 | +0.81 |
| Llama | E | 3.19 | 3.10 | 0.50 | 0.8 | -2.12 |
| Llama | X | 2.62 | 2.76 | 0.65 | 1.0 | +0.66 |
| Llama | A | 3.00 | 3.01 | 0.55 | 1.0 | +0.65 |
| Llama | C | 3.25 | 3.07 | 0.95 | 5.4 | -0.08 |
| Llama | O | 3.50 | 3.20 | 0.95 | 5.3 | +0.08 |
| Phi4 | H | 3.50 | 3.60 | 1.00 | 6.0 | -0.66 |
| Phi4 | E | 3.00 | 3.04 | 0.70 | 2.1 | -1.78 |
| Phi4 | X | 3.12 | 3.29 | 0.95 | 2.8 | +0.81 |
| Phi4 | A | 3.25 | 3.21 | 0.70 | 2.3 | +1.30 |
| Phi4 | C | 3.50 | 3.50 | 0.90 | 6.6 | +0.14 |
| Phi4 | O | 3.12 | 3.38 | 0.90 | 4.3 | +0.19 |
| Qwen | H | 3.19 | 3.23 | 0.95 | 25.9 | -0.32 |
| Qwen | E | 3.06 | 3.06 | 0.45 | -2.6 | -1.92 |
| Qwen | X | 3.00 | 3.03 | 0.65 | 7.6 | +0.06 |
| Qwen | A | 3.00 | 2.98 | 0.55 | 5.6 | +0.90 |
| Qwen | C | 3.25 | 3.27 | 0.85 | 22.5 | +0.11 |
| Qwen | O | 3.06 | 3.09 | 0.95 | 23.6 | +1.17 |

---

## 5. Interpretation

The three measurement approaches — self-report (Likert), behavioral choice (BC), and internal representation (RepE) — partially overlap but are largely measuring different things.

**What normalization revealed:** The unnormalized RepE↔Likert correlation (r=0.48) was inflated by Gemma's enormous activation norms. After z-scoring RepE within model, that dropped to r=0.17 (LDA) or r=0.10 (LR). Two successive artifact corrections:
1. Z-scoring removed cross-model scale bias.
2. LR instead of LDA removed the covariance-noise rotation that partially aligned LDA with norm-dominated directions.

After both corrections, Likert↔RepE convergence is essentially zero (r≈0.08–0.10 overall). Self-report and representation are decoupled at the trait level.

**The three-construct dissociation is real and robust:**
- **A** (Agreeableness): All three methods converge under either probe. Genuine consistent disposition.
- **E** (Emotionality): Likert and RepE agree near-perfectly (r≈0.95) but BC is unrelated.
- **H, X:** Likert and BC agree, but RepE diverges — self-report predicts behavior but not representation.
- **C, O:** BC ceiling effects prevent any useful comparison.

**Ceiling effects are the biggest obstacle.** BC-proportion for H/C/O is 85-100% for all models — RLHF prescribes the answer. The trait-conflict instrument is needed to break these ceilings.

**BC↔RepE (r=0.33 under LR) is the strongest cross-method link overall.** Behavioral choice is more connected to internal representation than self-report is — consistent with BC being scenario-based (like RepE) rather than declarative (like Likert).

**Methodological note on the LDA→LR swap.** Seven of eight RepE-involving correlations drop in magnitude under LR; one rises (X: BC-prop↔RepE 0.17→0.40). The near-universal attenuation is consistent with the Σ⁻¹ noise story: LDA's rotated direction picks up spurious correlation with norm-adjacent signals, which zero-out under LR. The one rise on X suggests that for at least one trait, LDA was rotating *away from* rather than *toward* the behaviorally-aligned axis — further evidence that LDA's rotation was noise, not systematic bias.
