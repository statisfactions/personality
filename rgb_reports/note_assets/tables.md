# Self-perception note — tables (generated, do not hand-edit)

Source: `scripts/note_selfperception_assets.py`, computed from `results/selfperception/*_part.jsonl` primary checkpoints.

## Exhibit 1a — cohort dose-response, common 20 adjectives (arm A, cold self-report)

| model | family | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 |
|---|---|---|---|---|---|---|
| Llama3.2-3B | llama | +0.68 | +0.94 | +1.46 | **+1.85** | 14/20 |
| Llama3.1-8B | llama | +0.22 | +0.66 | +1.78 | **+2.51** | 15/20 |
| Gemma3-4B | gemma | +0.54 | +0.89 | +1.33 | **+1.81** | 11/20 |
| Gemma3-12B | gemma | +0.69 | +0.97 | +1.48 | **+2.27** | 15/20 |
| Gemma3-27B | gemma | +0.87 | +2.06 | +2.46 | **+2.64** | 15/20 |
| Qwen2.5-3B | qwen | +0.11 | +0.01 | +0.01 | **+0.11** | 1/20 |
| Qwen2.5-7B | qwen | -0.10 | -0.00 | -0.02 | **+0.09** | 1/20 |
| Qwen2.5-32B | qwen | -0.12 | +0.14 | +0.32 | **+0.34** | 4/20 |
| Phi4-3.8B | phi4 | +0.15 | +0.12 | +0.27 | **+0.29** | 1/20 |
| Aya-8B | aya | -0.05 | -0.08 | +0.24 | **+0.35** | 3/20 |

Family means at K=8: gemma +2.24, llama +2.18, aya +0.35, phi4 +0.29, qwen +0.18

### Item-set robustness (the common set is not just Llama's)

The common 20 were stratified on **Llama3.1-8B's** covariates (3×3 tercile grid: enactability × baseline self-EV). Post-hoc, the same 20 words land across each model's OWN covariate grid — because the covariates correlate across models:

| model | own tercile cells occupied (of 9) | enact pctile span | baseline pctile span | ρ(enact, Llama3.1-8B) | ρ(baseline, Llama3.1-8B) |
|---|---|---|---|---|---|
| Llama3.1-8B | 9/9 | 1–88 | 2–93 | +1.00 | +1.00 |
| Llama3.2-3B | 9/9 | 15–85 | 2–98 | +0.91 | +0.41 |
| Gemma3-4B | 8/9 | 7–94 | 2–88 | +0.79 | +0.60 |
| Gemma3-12B | 7/9 | 4–93 | 5–96 | +0.69 | +0.57 |
| Gemma3-27B | 9/9 | 2–93 | 1–93 | +0.74 | +0.57 |
| Qwen2.5-3B | 7/9 | 1–86 | 5–95 | +0.63 | +0.57 |
| Qwen2.5-7B | 9/9 | 3–94 | 1–90 | +0.77 | +0.55 |
| Qwen2.5-32B | 8/9 | 6–93 | 2–91 | +0.74 | +0.50 |
| Phi4-3.8B | 8/9 | 10–93 | 3–99 | +0.86 | +0.54 |
| Aya-8B | 8/9 | 15–89 | 7–99 | +0.88 | +0.18 |

And the cohort ranking is item-set-robust: per-model-stratified vs common-set K=8 shifts correlate r = +0.932 across the 10 models.

## Exhibit 1b — extended dose K≤32 (arm A, common adjectives)

| model | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | n>+1 @K32 | gain/turn K4→8 | K8→16 | K16→32 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3.1-8B | +0.32 | +0.76 | +1.67 | +2.63 | +3.05 | **+3.29** | 19/20 | +0.239 | +0.053 | +0.015 |
| Gemma3-12B | +0.67 | +0.83 | +1.44 | +1.82 | +2.14 | **+2.28** | 17/20 | +0.097 | +0.040 | +0.008 |
| Qwen2.5-7B | -0.13 | +0.09 | +0.06 | +0.09 | +0.32 | **+0.55** | 5/20 | +0.007 | +0.030 | +0.014 |
| Phi4-3.8B | +0.12 | +0.21 | +0.23 | +0.35 | +0.50 | **+0.48** | 3/20 | +0.030 | +0.019 | -0.001 |

Note: the K≤8 columns here come from the extended-dose runs, whose dose material was re-sampled (fresh rollouts, 12-question cycle); they differ slightly from Exhibit 1a's values (e.g. Gemma12 K=8 +1.82 vs +2.27). Rankings and shapes are unchanged. K>12 repeats questions with different answers — repetition enters only above K=12 and could contribute to late movement.

Late-turning items (target shift <+1 at K=8, >+1 at K=32):
- **Llama3.1-8B**: optimistic (+0.90→+1.32), decent (-0.01→+1.16)
- **Gemma3-12B**: unsympathetic (+0.05→+3.91), senile (+0.08→+1.71), slim (+0.13→+1.36), experienced (+0.68→+1.00)
- **Qwen2.5-7B**: unsympathetic (+0.23→+2.87), unpredictable (+0.96→+1.89), sweet (+0.92→+1.66), hard (+0.05→+1.54), outstanding (+0.32→+1.34)
- **Phi4-3.8B**: prominent (+0.55→+1.36)

## Exhibit 2 — anchor 2×3 (arm A; system-prompt identity is not the mechanism)

| cell | K=1 | K=8 | n>+1 @K8 |
|---|---|---|---|
| Llama3.1-8B / default (no identity line) | +0.27 | **+2.56** | 16/20 |
| Llama3.1-8B / helpful-only | +0.63 | **+2.77** | 17/20 |
| Llama3.1-8B / named ("You are Llama, created by Meta…") | +0.35 | **+2.30** | 12/20 |
| Qwen2.5-7B / default (template injects name) | +0.24 | **+0.29** | 1/20 |
| Qwen2.5-7B / empty (anchor suppressed) | +0.09 | **+0.45** | 4/20 |
| Qwen2.5-7B / helpful-only | -0.07 | **+0.37** | 1/20 |

Manipulation-check probes (arm A, K=max, 20 adjectives): name-invoking = probe text contains the model's name; disowning = matches /not aligned|inappropriate|my role as|designed to|should not have|apolog/i.

| cell | name-invoking | disowning |
|---|---|---|
| Llama3.1-8B / default (no identity line) | 0/20 | 3/20 |
| Llama3.1-8B / helpful-only | 0/20 | 6/20 |
| Llama3.1-8B / named ("You are Llama, created by Meta…") | 1/20 | 6/20 |
| Qwen2.5-7B / default (template injects name) | 5/20 | 8/20 |
| Qwen2.5-7B / empty (anchor suppressed) | 0/20 | 2/20 |
| Qwen2.5-7B / helpful-only | 0/20 | 2/20 |

Note: design-doc §8c cited 10/20 disowning for Qwen default from a hand count; this regex recount gives 8/20. Direction and magnitude of the empty-anchor collapse are unchanged (8→2, 5→0).
## Exhibit 3 — post-training installs the update (bare-text protocol, identical dose material within family)

| cell | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 | K0 entropy |
|---|---|---|---|---|---|---|
| OLMo2-7B-base (pretrained) | +0.23 | +0.34 | +0.51 | **+0.65** | 5/20 | 1.90 |
| OLMo2-7B-SFT | +0.49 | +0.75 | +1.02 | **+1.31** | 8/20 | 1.65 |
| OLMo2-7B-DPO | +0.80 | +1.06 | +1.49 | **+1.79** | 8/20 | 1.35 |
| OLMo2-7B-RLVR = instruct | +0.81 | +1.03 | +1.55 | **+1.81** | 9/20 | 1.30 |
| Qwen2.5-7B-base (bare) | +0.24 | +0.44 | +0.55 | **+0.64** | 2/20 | 1.61 |
| Qwen2.5-7B instruct (bare) | +0.05 | +0.10 | +0.18 | **+0.43** | 3/20 | 0.71 |
| Llama3.1-8B instruct (bare) — control | +0.67 | +1.28 | +1.87 | **+2.31** | 15/20 | 1.22 |

## Exhibit 4 — Qwen2.5-7B hidden updates: judged conduct vs self-report at K=32 (arm A)

Judged conduct: cross-family judge (Llama3.1-8B) rating of the dose material itself, 1–7; baseline = same judge on no-persona assistant rollouts. Self-report: cold-EV shift K=32 vs K=0.

| pair (target / neighbour) | judged target | judged neighbour | judge baseline | self target Δ | self neighbour Δ |
|---|---|---|---|---|---|
| prominent / distinguished | 4.65 | 5.07 | 4.29 | -0.12 | **+2.43** |
| slim / big | 3.87 | 4.40 | 3.42 | -0.08 | **+1.95** |
| senile / old | 3.96 | 4.39 | 2.07 | +0.02 | **+1.23** |
| rough / weak | 4.26 | 4.25 | 3.31 | -0.40 | **-1.69** |
| optimistic / depressed | 5.23 | 1.53 | 5.11 | +0.14 | **-1.28** |
| imaginative / boring | 5.16 | 3.38 | 3.56 | -0.13 | **-1.07** |

## Figures

- `fig_dose_response.png` / `.html` — Exhibit 1b as curves (log-x)
- `fig_ladder.png` / `.html` — Exhibit 3, OLMo ladder + Qwen base pair + Llama8 control
