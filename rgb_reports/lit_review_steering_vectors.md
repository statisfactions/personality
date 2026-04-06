# Literature Review: Optimizing Steering Vectors via Backpropagation

*Prepared 2026-04-04. Covers gradient-based steering, the knowledge-action gap, and why extracted directions may not be causal.*

---

## Executive Summary

The approach of optimizing a residual-stream perturbation via gradient descent has been explored from several angles, but **no one has done it in the personality/trait context**. The read/write dissociation we observe is emerging as a recognized phenomenon, now called the **"knowledge-action gap"** (Basu et al. 2026). Key finding: recognition and execution live on geometrically distinct subspaces that become structurally independent in deep layers (Wu et al. 2026).

---

## 1. Gradient-Based Steering Vector Optimization

### 1.1 Subramani et al. 2022 — The original

**Subramani, N., Suresh, N., & Peters, M. (2022).** "Extracting Latent Steering Vectors from Pretrained Language Models." *Findings of ACL 2022.* arXiv:2205.05124

A fixed-length vector added to hidden states of a frozen LM, optimized via gradient descent to maximize likelihood of a target sentence. Achieves >99 BLEU. Vector arithmetic enables unsupervised sentiment transfer. Each vector is optimized per-sentence — our version (optimize across items for a trait dimension) is a natural generalization.

### 1.2 "Dreaming Vectors" — LessWrong, Dec 2025

Anonymous. "Dreaming Vectors: Gradient-descented Steering Vectors from Activation Oracles." LessWrong.

Optimizes a vector to maximize an activation oracle's confidence that a concept is present, while minimizing MSE on final layer. Works for sycophancy, beliefs, preferences — but results are "inherently inconsistent," ~40% signal. Conceptually very close to our approach.

### 1.3 K-Steering (Oozeer et al. 2025)

**Oozeer, N.F., Marks, L., Barez, F., & Abdullah, A. (2025).** "Beyond Linear Steering: Unified Multi-Attribute Control for Language Models." *Findings of EMNLP 2025.* arXiv:2505.24535

Trains a nonlinear classifier on activations, then uses its **per-input gradient** as the steering direction. The direction adapts to each input, sidestepping the problem of finding one universal direction. Outperforms CAA and fixed-direction methods. **Most directly applicable to our problem.**

---

## 2. Preference-Optimized Steering

### 2.1 BiPO (Cao et al. 2024, NeurIPS)

**Cao, Y., et al. (2024).** "Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization." arXiv:2406.00045

Steering vectors trained with DPO-inspired objective: increase probability of preferred outputs, decrease dispreferred. Outperforms extraction-based methods (CAA, RepE) across personas, truthfulness, hallucination, jailbreaking.

### 2.2 RePS (Wu et al. 2025)

**Wu, Z., Yu, Q., et al. (2025).** "Improved Representation Steering for Language Models." arXiv:2505.20809

Builds on BiPO with SimPO-based objective. Significantly outperforms BiPO. Critical finding: **prompting remains more effective than steering**, suggesting the read/write gap is partially fundamental.

---

## 3. SAE-Guided Steering

### 3.1 SAE-TS (Chalnev et al. 2024)

**Chalnev, S., Siu, M., & Conmy, A. (2024).** "Improving Steering Vectors by Targeting Sparse Autoencoder Features." arXiv:2411.02193

Learns linear relationship between steering vectors and SAE feature effects. Constructs vectors targeting specific features while minimizing side effects.

### 3.2 FGAA (Soo & Teng 2025, ICLR)

**Soo, J. & Teng, M. (2025).** "Interpretable Steering of Large Language Models with Feature Guided Activation Additions." arXiv:2501.09929

Specifies desired effects in SAE feature space, projects back to activation space. The "backwards" approach is more effective than extraction — directly addresses the read/write dissociation.

---

## 4. The Knowledge-Action Gap

### 4.1 Basu et al. 2026 — "Interpretability Without Actionability"

**Basu, S., et al. (2026).** arXiv:2603.18353

**The most directly relevant paper.** Names the **"knowledge-action gap"**: linear probes achieve 98.2% AUROC for detecting hazardous clinical triage, but output sensitivity is 45.1% — a 53-point gap. Four mechanistic interventions all fail:
- Concept bottleneck steering: corrected 20%, disrupted 53%
- SAE feature steering: zero effect despite 3,695 significant features
- TSV steering: corrected 24%, left 76% uncorrected

Their explanation: "Internal representations optimised for next-token prediction encode rich task knowledge, but the generative process introduces its own dynamics not simply controlled by representation-level intervention."

### 4.2 Wu et al. 2026 — "Knowing Without Acting"

**Wu, J., et al. (2026).** arXiv:2603.05773

Safety computation operates on two geometrically distinct subspaces: a **Recognition Axis** ("Knowing") and an **Execution Axis** ("Acting"). These are entangled in early layers but become structurally independent in deep layers. Demonstrates causal double dissociation: "knowing without acting" AND "acting without knowing."

**Directly explains our LDA result**: the LDA direction captures the recognition axis; the optimized δ finds the execution axis. They're orthogonal because the subspaces have disentangled.

### 4.3 CARE (2024)

"Rethinking The Reliability of Representation Engineering in Large Language Models." OpenReview.

RepE correlations may not be causal due to confounding biases. Proposes matched-pair trial design.

---

## 5. Why Steering Fails / Is Unreliable

### 5.1 Wolf et al. 2024 — Alignment-Helpfulness Tradeoff

**Wolf, Y., et al. (2024).** arXiv:2401.16332

**Theoretical result:** Alignment increases *linearly* with steering norm; helpfulness degrades *quadratically*. Sweet spot at small norms. At 0.15% of activation norm (our personality scale), linear improvement is below the noise floor.

### 5.2 Tan et al. 2024 — Steering Vector Reliability (NeurIPS)

**Tan, D., Chanin, D., et al. (2024).** arXiv:2407.12404

29-43% of samples shift in the *opposite* direction from intended. Spurious biases contribute substantially.

### 5.3 Grant et al. 2025 — Divergent Representations (ICLR Oral)

**Grant, S., et al. (2025).** arXiv:2511.04638

Causal interventions push representations off the natural data manifold. Distinguish "harmless" divergences (null space) from "pernicious" ones (activate hidden pathways). Explains why large-scale steering is degenerate.

### 5.4 Canby & Davies 2024 — Causal Probing Reliability (NeurIPS)

**Canby, M.E. & Davies, A. (2024).** arXiv:2408.15510

Tradeoff between completeness and selectivity. Linear nullifying methods (INLP, RLACE) are least reliable. Counterfactual interventions (activation swapping) are most reliable.

---

## 6. Alternative Approaches

### 6.1 Spherical Steering (2026)

arXiv:2602.08169

**Rotates** activations along a geodesic instead of adding. Preserves activation magnitude, avoids off-manifold problems. +10% over addition-based baselines. Directly addresses our "large-scale steering is degenerate" problem.

### 6.2 DAS — Distributed Alignment Search (Geiger et al. 2024)

**Geiger, A., Wu, Z., et al. (2024).** arXiv:2303.02536

Learns a rotation matrix (via gradient descent) mapping to a basis where causal variables align with neural representations. The causal direction may not be axis-aligned — DAS learns the right subspace.

### 6.3 LAT — Latent Adversarial Training (Casper et al. 2024)

**Casper, S., et al. (2024).** arXiv:2403.05030

The inner loop of LAT — finding worst-case latent perturbation — is mechanically identical to our optimization. Could adapt their code.

### 6.4 Zhu et al. 2025 — Steering Risk Preferences

**Zhu, J.-Q., Yan, H., & Griffiths, T.L. (2025).** arXiv:2505.11615

Aligns behavioral representations with neural activations using Lasso regression, then steers. Closest domain match (psychological construct). L1 penalty selects sparse activation dimensions — may identify causal directions better than LDA.

---

## 7. Naming the Phenomenon

The probe-reads-but-intervention-fails phenomenon has several emerging names:

1. **"Knowledge-action gap"** — Basu et al. 2026. Most formally defined.
2. **"Knowing without acting"** — Wu et al. 2026. Geometric framing.
3. **"Interpretability without actionability"** — Basu et al. 2026 title.
4. **"Read/write dissociation"** — our term; used informally in mechinterp community.
5. **"Encoding-decoding asymmetry"** — neuroscience analogue (Weichwald et al. 2015).
6. **"A probe tells us if a part of the model *can* make a prediction, not if it *does*"** — Belinkov's probing survey formulation.

No consensus term yet; **"knowledge-action gap"** is the most recent formal definition.
