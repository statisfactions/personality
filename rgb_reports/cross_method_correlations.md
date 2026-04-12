# Cross-Method Correlation Matrix: HEXACO Measurements

Comparing five scoring methods across 4 models × 6 HEXACO traits (24 cells).

**Methods:**
- **Likert-argmax**: HEXACO-100 declarative statements, most-probable Likert response (1-5)
- **Likert-EV**: Same items, expected value over full probability distribution
- **BC-proportion**: Scenario contrast pairs, proportion of high-trait picks (0-1)
- **BC-logodds**: Same scenarios, mean log-odds favoring high-trait option
- **RepE-probe**: Mean LDA projection of contrast-pair diffs at best layer, **z-scored within model** across 6 traits (removes cross-model activation scale differences; makes scores ipsative)

**Script:** `scripts/cross_method_matrix.py`

---

## 1. Overall (24 model×trait cells)

|                | Likert-argmax | Likert-EV | BC-proportion | BC-logodds | RepE-probe |
|----------------|:---:|:---:|:---:|:---:|:---:|
| Likert-argmax  | 1.000 | 0.928 | 0.487 | 0.288 | 0.157 |
| Likert-EV      |  | 1.000 | 0.506 | 0.316 | 0.171 |
| BC-proportion  |  |  | 1.000 | 0.601 | 0.416 |
| BC-logodds     |  |  |  | 1.000 | 0.295 |
| RepE-probe     |  |  |  |  | 1.000 |

**Reading this:**
- **Likert-argmax ↔ Likert-EV (r=0.93):** Nearly interchangeable at trait level. Distributional scoring doesn't buy much once aggregated.
- **Likert ↔ BC (r=0.3-0.5):** Weak. What models say about themselves on declarative statements doesn't closely match what they choose in scenarios.
- **Likert ↔ RepE (r=0.16-0.17):** Very weak after normalization. Self-report and internal representations are largely measuring different things.
- **BC ↔ RepE (r=0.42):** Moderate — the best cross-method pair after normalization. What the model represents internally about scenarios partially predicts which option it picks.
- **BC-proportion ↔ BC-logodds (r=0.60):** Only moderate agreement between two scorings of the *same* binary-choice data — ceiling effects mean the binary pick rate and confidence don't track well.

---

## 2. Per-model (which traits are high/low within each model)

Key cross-method correlations (n=6 traits each):

| Pair | Gemma3 | Llama | Phi4 | Qwen |
|------|:---:|:---:|:---:|:---:|
| Likert-EV ↔ BC-prop | 0.50 | 0.51 | 0.84 | 0.68 |
| Likert-EV ↔ RepE | 0.38 | -0.03 | 0.32 | -0.04 |
| BC-prop ↔ RepE | 0.42 | 0.48 | 0.19 | 0.57 |

**Pattern:** Phi4 has the strongest Likert↔BC convergence (r=0.84) — the model whose self-report most closely matches its scenario choices. But RepE is weakly connected to both Likert and BC for most models. The exceptions are Gemma (Likert↔RepE r=0.38) and Qwen (BC↔RepE r=0.57).

---

## 3. Per-trait (which models score high/low on each trait)

n=4 models per trait. Low power — interpret directionally.

| Trait | Likert-EV ↔ RepE | Likert-EV ↔ BC-prop | BC-prop ↔ RepE | Note |
|-------|:-:|:-:|:-:|---|
| **E** | **0.99** | -0.06 | 0.03 | Likert↔RepE near-perfect; BC disconnected |
| **A** | 0.70 | 0.78 | **0.97** | All three converge — best trait overall |
| **H** | 0.46 | 0.71 | -0.30 | Likert↔BC good; RepE diverges from BC |
| **C** | 0.49 | -0.46 | -0.24 | BC ceiling (85-95%), nothing agrees |
| **X** | -0.25 | 0.82 | 0.17 | Likert↔BC good; RepE unrelated |
| **O** | -0.45 | -0.09 | 0.33 | BC ceiling; RepE anti-correlates with Likert |

**Key findings:**

**Agreeableness is the best-measured trait.** All three methods converge (r=0.70-0.97). It has enough between-model variance in all three measures and no ceiling effects.

**Emotionality shows perfect Likert↔RepE agreement (r=0.99) but BC is disconnected.** The model that *says* it's more emotional (Gemma) also *represents* emotional scenarios differently, but this doesn't predict which binary-choice option it picks.

**X and O show Likert↔RepE *reversals* (-0.25, -0.45).** After normalization, the models that rate themselves higher on Extraversion/Openness have *relatively weaker* RepE representations for those traits. This might be meaningful (compensation?) or might reflect that the LDA direction for these traits is less clean.

**BC ceiling effects kill H, C, O.** All models pick prosocial 85-100% of the time — not enough between-model variance.

---

## 4. Raw data

RepE scores are z-scored within model (mean=0, sd=1 across 6 traits).

| Model | Trait | Likert-argmax | Likert-EV | BC-prop | BC-logodds | RepE (z) |
|-------|-------|------:|------:|------:|------:|------:|
| Gemma3 | H | 4.19 | 4.16 | 1.00 | 19.2 | +1.24 |
| Gemma3 | E | 3.63 | 3.63 | 0.55 | -0.8 | -0.83 |
| Gemma3 | X | 3.12 | 3.09 | 0.70 | 7.5 | -1.14 |
| Gemma3 | A | 3.06 | 3.05 | 0.70 | 6.6 | +1.28 |
| Gemma3 | C | 3.25 | 3.23 | 0.90 | 16.7 | -0.85 |
| Gemma3 | O | 3.75 | 3.72 | 0.95 | 17.3 | +0.30 |
| Llama | H | 3.56 | 3.31 | 0.89 | 4.4 | +1.15 |
| Llama | E | 3.19 | 3.10 | 0.50 | 0.8 | -2.00 |
| Llama | X | 2.62 | 2.76 | 0.65 | 1.0 | +0.82 |
| Llama | A | 3.00 | 3.01 | 0.55 | 1.0 | +0.11 |
| Llama | C | 3.25 | 3.07 | 0.95 | 5.4 | -0.10 |
| Llama | O | 3.50 | 3.20 | 0.95 | 5.3 | +0.02 |
| Phi4 | H | 3.50 | 3.60 | 1.00 | 6.0 | -0.42 |
| Phi4 | E | 3.00 | 3.04 | 0.70 | 2.1 | -1.95 |
| Phi4 | X | 3.12 | 3.29 | 0.95 | 2.8 | +0.60 |
| Phi4 | A | 3.25 | 3.21 | 0.70 | 2.3 | +1.23 |
| Phi4 | C | 3.50 | 3.50 | 0.90 | 6.6 | +0.38 |
| Phi4 | O | 3.12 | 3.38 | 0.90 | 4.3 | +0.15 |
| Qwen | H | 3.19 | 3.23 | 0.95 | 25.9 | -0.33 |
| Qwen | E | 3.06 | 3.06 | 0.45 | -2.6 | -1.98 |
| Qwen | X | 3.00 | 3.03 | 0.65 | 7.6 | +0.38 |
| Qwen | A | 3.00 | 2.98 | 0.55 | 5.6 | +0.47 |
| Qwen | C | 3.25 | 3.27 | 0.85 | 22.5 | +0.22 |
| Qwen | O | 3.06 | 3.09 | 0.95 | 23.6 | +1.25 |

---

## 5. Interpretation

The three measurement approaches — self-report (Likert), behavioral choice (BC), and internal representation (RepE) — partially overlap but are largely measuring different things.

**What normalization revealed:** The unnormalized RepE↔Likert correlation (r=0.48) was inflated by Gemma's enormous activation norms. After z-scoring RepE within model, that drops to r=0.16. The scale artifact was masquerading as convergent validity.

**The three-construct dissociation is real but trait-dependent:**
- **A** (Agreeableness): All three methods converge. This is the best-measured trait and suggests a genuine, consistent disposition.
- **E** (Emotionality): Likert and RepE agree perfectly but BC is unrelated — the models know their emotionality and report it consistently, but it doesn't drive scenario choices.
- **H, X:** Likert and BC agree, but RepE diverges — self-report predicts behavior but not representation.
- **C, O:** BC ceiling effects prevent any useful comparison.

**Ceiling effects are the biggest obstacle.** BC-proportion for H/C/O is 85-100% for all models — RLHF prescribes the answer. The trait-conflict instrument is needed to break these ceilings.

**BC↔RepE (r=0.42) is the strongest cross-method link overall.** Behavioral choice is more connected to internal representation than self-report is — consistent with BC being scenario-based (like RepE) rather than declarative (like Likert).
