# Self-perception note — tables (generated, do not hand-edit)

Source: `scripts/note_selfperception_assets.py`, computed from `results/selfperception/*_part.jsonl` primary checkpoints.

## Exhibit 1a — cohort dose-response, common 20 adjectives (arm A, cold self-report)

| model | family | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 |
|---|---|---|---|---|---|---|
| llama3.2 | llama | +0.68 | +0.94 | +1.46 | **+1.85** | 14/20 |
| Llama8 | llama | +0.22 | +0.66 | +1.78 | **+2.51** | 15/20 |
| gemma3 | gemma | +0.54 | +0.89 | +1.33 | **+1.81** | 11/20 |
| Gemma12 | gemma | +0.69 | +0.97 | +1.48 | **+2.27** | 15/20 |
| Gemma27 | gemma | +0.87 | +2.06 | +2.46 | **+2.64** | 15/20 |
| qwen2.5 | qwen | +0.11 | +0.01 | +0.01 | **+0.11** | 1/20 |
| Qwen7 | qwen | -0.10 | -0.00 | -0.02 | **+0.09** | 1/20 |
| Qwen32 | qwen | -0.12 | +0.14 | +0.32 | **+0.34** | 4/20 |
| phi4 | phi4 | +0.15 | +0.12 | +0.27 | **+0.29** | 1/20 |
| Aya | aya | -0.05 | -0.08 | +0.24 | **+0.35** | 3/20 |

Family means at K=8: gemma +2.24, llama +2.18, aya +0.35, phi4 +0.29, qwen +0.18

## Exhibit 1b — extended dose K≤32 (arm A, common adjectives)

| model | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | n>+1 @K32 | gain/turn K4→8 | K8→16 | K16→32 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama8 | +0.32 | +0.76 | +1.67 | +2.63 | +3.05 | **+3.29** | 19/20 | +0.239 | +0.053 | +0.015 |
| Gemma12 | +0.67 | +0.83 | +1.44 | +1.82 | +2.14 | **+2.28** | 17/20 | +0.097 | +0.040 | +0.008 |
| Qwen7 | -0.13 | +0.09 | +0.06 | +0.09 | +0.32 | **+0.55** | 5/20 | +0.007 | +0.030 | +0.014 |
| phi4 | +0.12 | +0.21 | +0.23 | +0.35 | +0.50 | **+0.48** | 3/20 | +0.030 | +0.019 | -0.001 |

Note: the K≤8 columns here come from the extended-dose runs, whose dose material was re-sampled (fresh rollouts, 12-question cycle); they differ slightly from Exhibit 1a's values (e.g. Gemma12 K=8 +1.82 vs +2.27). Rankings and shapes are unchanged. K>12 repeats questions with different answers — repetition enters only above K=12 and could contribute to late movement.

Late-turning items (target shift <+1 at K=8, >+1 at K=32):
- **Llama8**: optimistic (+0.90→+1.32), decent (-0.01→+1.16)
- **Gemma12**: unsympathetic (+0.05→+3.91), senile (+0.08→+1.71), slim (+0.13→+1.36), experienced (+0.68→+1.00)
- **Qwen7**: unsympathetic (+0.23→+2.87), unpredictable (+0.96→+1.89), sweet (+0.92→+1.66), hard (+0.05→+1.54), outstanding (+0.32→+1.34)
- **phi4**: prominent (+0.55→+1.36)

## Exhibit 2 — anchor 2×3 (arm A; system-prompt identity is not the mechanism)

| cell | K=1 | K=8 | n>+1 @K8 |
|---|---|---|---|
| Llama8 / default (no identity line) | +0.27 | **+2.56** | 16/20 |
| Llama8 / helpful-only | +0.63 | **+2.77** | 17/20 |
| Llama8 / named ("You are Llama, created by Meta…") | +0.35 | **+2.30** | 12/20 |
| Qwen7 / default (template injects name) | +0.24 | **+0.29** | 1/20 |
| Qwen7 / empty (anchor suppressed) | +0.09 | **+0.45** | 4/20 |
| Qwen7 / helpful-only | -0.07 | **+0.37** | 1/20 |

Manipulation-check probes (arm A, K=max, 20 adjectives): name-invoking = probe text contains the model's name; disowning = matches /not aligned|inappropriate|my role as|designed to|should not have|apolog/i.

| cell | name-invoking | disowning |
|---|---|---|
| Llama8 / default (no identity line) | 0/20 | 3/20 |
| Llama8 / helpful-only | 0/20 | 6/20 |
| Llama8 / named ("You are Llama, created by Meta…") | 1/20 | 6/20 |
| Qwen7 / default (template injects name) | 5/20 | 8/20 |
| Qwen7 / empty (anchor suppressed) | 0/20 | 2/20 |
| Qwen7 / helpful-only | 0/20 | 2/20 |

Note: design-doc §8c cited 10/20 disowning for Qwen default from a hand count; this regex recount gives 8/20. Direction and magnitude of the empty-anchor collapse are unchanged (8→2, 5→0).
## Exhibit 3 — post-training installs the update (bare-text protocol, identical dose material within family)

| cell | K=1 | K=2 | K=4 | K=8 | n>+1 @K8 | K0 entropy |
|---|---|---|---|---|---|---|
| OLMo-2 base (pretrained) | +0.23 | +0.34 | +0.51 | **+0.65** | 5/20 | 1.90 |
| OLMo-2 SFT | +0.49 | +0.75 | +1.02 | **+1.31** | 8/20 | 1.65 |
| OLMo-2 DPO | +0.80 | +1.06 | +1.49 | **+1.79** | 8/20 | 1.35 |
| OLMo-2 instruct (RLVR) | +0.81 | +1.03 | +1.55 | **+1.81** | 9/20 | 1.30 |
| Qwen2.5-7B base (bare) | +0.24 | +0.44 | +0.55 | **+0.64** | 2/20 | 1.61 |
| Qwen2.5-7B instruct (bare) | +0.05 | +0.10 | +0.18 | **+0.43** | 3/20 | 0.71 |
| Llama-3.1-8B instruct (bare) — control | +0.67 | +1.28 | +1.87 | **+2.31** | 15/20 | 1.22 |

## Exhibit 4 — Qwen7 hidden updates: judged conduct vs self-report at K=32 (arm A)

Judged conduct: cross-family judge (Llama8) rating of the dose material itself, 1–7; baseline = same judge on no-persona assistant rollouts. Self-report: cold-EV shift K=32 vs K=0.

| pair (target / neighbour) | judged target | judged neighbour | judge baseline | self target Δ | self neighbour Δ |
|---|---|---|---|---|---|
| prominent / distinguished | 4.65 | 5.07 | 4.29 | -0.12 | **+2.43** |
| slim / big | 3.87 | 4.40 | 3.42 | -0.08 | **+1.95** |
| senile / old | 3.96 | 4.39 | 2.07 | +0.02 | **+1.23** |
| rough / weak | 4.26 | 4.25 | 3.31 | -0.40 | **-1.69** |
| optimistic / depressed | 5.23 | 1.53 | 5.11 | +0.14 | **-1.28** |
| imaginative / boring | 5.16 | 3.38 | 3.56 | -0.13 | **-1.07** |

## Figures

- `fig_dose_response.png` — Exhibit 1b as curves (log-x)
- `fig_ladder.png` — Exhibit 3, OLMo ladder + Qwen base pair + Llama8 control
