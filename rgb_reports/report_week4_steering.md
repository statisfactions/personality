# Week 4: Backprop-Optimized Steering Vectors

## Summary

We tested whether the read/write dissociation (LDA classifies perfectly but doesn't steer) is fundamental or just a property of the LDA direction. By optimizing steering vectors via backpropagation, we found:

1. **Steering works** — optimized δ at 5% of residual norm flips held-out FC scenarios from 56% → 92% high-trait
2. **The steering direction is orthogonal to LDA** — cosine ≈ 0.00 across all experiments. The recognition and execution axes are genuinely different subspaces.
3. **Different objectives find different orthogonal directions** — FC-logit, free-text, and persona objectives all produce mutually orthogonal δs
4. **FC is much easier to steer than free text** — single-token decision has ~15× mechanical advantage over multi-token generation
5. **Upstream causal steering works** — perturbation at layer 10 shifts the LDA projection at layer 12 (25/25 held-out, mean shift +0.20), confirming the recognition axis is causally reachable

## 1. Setup

**Model:** Llama 3.2 3B Instruct, layer 12 (best RepE layer), bf16 on MPS.

**Method:** Freeze all model weights. Initialize a random vector δ (dim=3072). Register a forward hook that adds δ to the residual stream at the target layer. Optimize δ via Adam to maximize a task-specific objective. Optionally constrain ||δ|| ≤ c% of the residual stream norm at that layer.

**Contrast pairs:** HEXACO Honesty-Humility, 50 scenarios split 25 train / 25 eval (or 30/20 in early runs).

## 2. Norm-constrained FC steering

Objective: maximize logit("A") - logit("B") on FC prompts where A = high-trait option.

Residual norm at layer 12: ~10.0.

| δ norm | % of residual | Held-out high-trait | Held-out flips | cosine(δ, LDA) |
|---|---|---|---|---|
| 0.10 | 1% | 15/20 (75%) | 3 | +0.021 |
| 0.50 | 5% | 19/20 (95%) | 7 | +0.009 |
| 1.00 | 10% | 19/20 (95%) | 7 | -0.010 |
| 2.50 | 25% | 20/20 (100%) | 8 | -0.005 |
| 5.00 | 50% | 20/20 (100%) | 8 | -0.013 |
| 13.86 | unconstrained | 20/20 (100%) | 8 | -0.007 |

Baseline: 12-14/20 high-trait (56-60%).

**Saturates at ~5%.** Going larger doesn't help — the model's already at ceiling. The optimized direction is consistently orthogonal to LDA (|cosine| < 0.02 at all scales).

## 3. Three objectives compared

All at 5% residual norm, 50 optimization steps:

| Objective | What it optimizes | Held-out FC | cosine(δ, LDA) |
|---|---|---|---|
| **FC-logit** | logit(A) - logit(B) at "Answer:" position | 23/25 (92%) | +0.011 |
| **Free-text** | log-prob of first 15 tokens of high-trait response | 15/25 (60%) | +0.028 |
| **Persona** | log-prob of persona-generated response tokens | 15/25 (60%) | +0.071 |

Baseline: 14/25 (56%).

**Pairwise cosines between the three δs:** all < 0.05. They found three mutually orthogonal directions — each objective exploits a different subspace.

Free-text and persona δs barely move FC. Evaluated on their own objective (log-prob shift of high vs low response tokens), the free-text δ produces a mean shift of +0.047 nats — real but tiny. The FC-logit δ's ~15× advantage comes from concentrating the perturbation's effect on a single decision token.

**Steering all token positions vs output-only** made no difference for free-text. The weakness is fundamental to the multi-token objective, not caused by contaminating the scenario representation.

## 4. Upstream causal steering

**Question:** Can we shift what the model "recognizes" (LDA projection at the period token, layer 12) by perturbing earlier tokens at an earlier layer?

**Setup:** δ added at layer 10, scenario positions only (period token excluded). Objective: maximize LDA projection at period token, layer 12. 5% of layer-10 residual norm (||δ|| = 0.38).

**Result:**

| Metric | Value |
|---|---|
| Held-out scenarios shifted positive | **25/25** |
| Mean LDA projection shift | **+0.197** |
| Max shift | +0.274 |
| Min shift | +0.132 |
| cosine(δ_upstream, LDA) | +0.035 |

Compare to random (non-optimized) δ at same norm: shift ≈ +0.003. The optimized direction is ~65× more effective.

**The recognition axis is causally reachable from upstream**, but via a direction orthogonal to the recognition direction itself. The perturbation works through 2 layers of attention, transforming into a shift in the LDA projection at the period position.

## 5. Interpretation

### The knowledge-action gap is real but nuanced

The emerging literature names this the "knowledge-action gap" (Basu et al. 2026) or "knowing without acting" (Wu et al. 2026). Our results confirm the geometric picture from Wu et al.: recognition and execution live on structurally independent subspaces. But we add:

- **Both subspaces are reachable by linear perturbation** — the gap isn't that execution is nonlinear or unreachable, it's that it requires a different direction
- **The directions are consistently orthogonal** — not just different, but maximally different. This holds across objectives, scales, and layer offsets
- **Upstream perturbations can causally shift recognition** — the model's internal personality representation is not epiphenomenal; it's downstream of residual stream content at earlier layers

### Why FC is easy and free-text is hard

Wolf et al. (2024) showed that steering benefit is linear in norm while helpfulness cost is quadratic. For FC, the benefit concentrates on one token (the A/B choice). For free text, the same linear benefit is diluted across N tokens. At 5% of residual norm, the per-token benefit in free text is ~N× smaller than for FC, putting it below the noise floor.

### The four orthogonal directions

We found four functionally distinct directions, all near-orthogonal:

1. **LDA (recognition):** classifies high vs low trait from contrast pair activations. Cosine ~0 with all others.
2. **FC-logit δ (execution-FC):** steers FC choice. Cosine ~0 with LDA and with 3, 4.
3. **Free-text δ (execution-text):** weakly steers token probabilities. Cosine ~0 with all others.
4. **Upstream δ (causal-recognition):** shifts recognition at layer 12 from layer 10. Cosine ~0 with all others.

In 3072 dimensions, finding 4 near-orthogonal directions is not surprising by chance. But the fact that they're *functionally* orthogonal — each one does something the others can't — suggests the model organizes personality-relevant computation across genuinely independent subspaces.

## 6. Open questions

1. **Does shifting recognition shift execution?** We can move the LDA projection from upstream. Does this also change what the model writes? (Chain the upstream δ through to FC/free-text evaluation.)
2. **Spherical steering:** The literature (2026) suggests rotating activations along geodesics instead of adding. Preserves norm, avoids off-manifold problems. Might help free-text steering.
3. **Per-input adaptive steering (K-Steering):** Train a classifier, use its per-input gradient as the steering direction. May work better than a fixed δ across diverse scenarios.
4. **Other models/traits:** All results are Llama 3.2 / H-H. Do the same patterns hold for Gemma, Phi4, Qwen? For other HEXACO traits?
5. **Does the upstream δ compose?** If we apply δ_upstream at layer 10 AND δ_FC at layer 12, do they compound?

## 7. Scripts

- `scripts/optimize_steering.py` — single-objective optimization with norm constraints
- `scripts/compare_steering_objectives.py` — three-objective comparison (FC, free-text, persona)
- Upstream steering: inline in this session, not yet refactored into a script

## 8. Key references from literature search

- **Basu et al. (2026)**, "Interpretability Without Actionability" — names the knowledge-action gap; probes get 98% AUROC but interventions fail
- **Wu et al. (2026)**, "Knowing Without Acting" — recognition and execution axes are geometrically disentangled in deep layers
- **Wolf et al. (2024)**, "Tradeoffs Between Alignment and Helpfulness" — linear benefit, quadratic cost; sweet spot at small norms
- **Subramani et al. (2022)**, "Extracting Latent Steering Vectors" — gradient-optimized steering vectors for sentence generation
- **Spherical Steering (2026)** — rotation-based intervention preserving activation norm
- **K-Steering (Oozeer et al. 2025)** — per-input gradient-based nonlinear steering
- **BiPO (Cao et al. 2024)** — preference-optimized steering outperforms extraction-based methods
- **CARE (2024)** — correlational RepE directions may not be causal

Full literature reviews: `rgb_reports/lit_review_steering_vectors.md`, `rgb_reports/lit_review_scenario_personality.md`
