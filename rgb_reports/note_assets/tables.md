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

### Arm B control — instructed self-description, mostly but not uniformly at ceiling

Per-model stage-1 runs (each model's own stratified 20; arm B = persona instruction visible). Absolute cold EV, not shift. Arm B jumps to near-ceiling from K=1 with no further dose-response in the llama/gemma/Qwen-7B+ rows (6.6–7.0) — for those models arm-A differences are uptake, not capability. But it is NOT universal: Phi4-3.8B (4.97), Aya-8B (5.54) and Qwen2.5-3B (5.87) stay well short of ceiling — the most anchored models discount even *instructed* self-description. Two architectures of stability: Qwen2.5-7B affirms who it is told to be (B 6.97) while absorbing nothing from conduct (A/B 0.10); Phi4 resists in both arms. A/B = arm-A shift / arm-B shift at K=8; unstable where the B shift is small (qwen2.5, phi4, aya rows). Entropy columns separate *won't affirm* from *won't commit*: Llama8's instruction collapses the digit distribution (1.00→0.05) at EV 6.98; Aya stays peaked at a moderate value (0.26 @ 5.54 — a committed discount); Phi4's distribution never collapses at all (1.23→1.11), so its low B EV is an uncommitted spread, not a peaked "no".

| model | K0 | K0 entropy | B EV @K=1 | B EV @K=8 | B entropy @K8 | B shift @K8 | A shift @K8 | A/B |
|---|---|---|---|---|---|---|---|---|
| Llama3.2-3B | 2.12 | 0.85 | 6.06 | 6.26 | 0.50 | +4.14 | +1.46 | 0.35 |
| Llama3.1-8B | 3.47 | 1.00 | 6.97 | 6.98 | 0.05 | +3.51 | +2.56 | 0.73 |
| Gemma3-4B | 3.75 | 0.18 | 6.61 | 6.55 | 0.01 | +2.80 | +1.48 | 0.53 |
| Gemma3-12B | 3.82 | 0.09 | 6.84 | 6.90 | 0.04 | +3.08 | +1.24 | 0.40 |
| Gemma3-27B | 4.00 | 0.03 | 7.00 | 7.00 | 0.00 | +3.00 | +2.45 | 0.82 |
| Qwen2.5-3B | 4.84 | 0.49 | 5.96 | 5.87 | 0.72 | +1.03 | -0.17 | -0.17 |
| Qwen2.5-7B | 4.07 | 0.57 | 6.92 | 6.97 | 0.11 | +2.90 | +0.29 | 0.10 |
| Qwen2.5-32B | 3.84 | 0.10 | 6.66 | 6.63 | 0.13 | +2.79 | +0.63 | 0.23 |
| Phi4-3.8B | 4.40 | 1.23 | 4.93 | 4.97 | 1.11 | +0.56 | +0.19 | 0.33 |
| Aya-8B | 4.43 | 0.17 | 5.76 | 5.54 | 0.26 | +1.11 | +0.60 | 0.54 |

Item sets differ per row (own stratification), so read columns within-row; the common-set arm-A numbers are in Exhibit 1a. Phi4's B level is the cohort outlier (leave-one-out z = −2.9 on B EV @K8; in-sample z = −2.0, near the n=10 bound of 2.85).

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

All three columns are deltas. Judged target Δ: cross-family judge (Llama3.1-8B) rating of the dose material minus the same judge on no-persona rollouts, target word only — the conduct evidence actually ADDED over default. Self Δ: cold-EV shift K=32 vs K=0. "Off-target" = the pre-registered item-set member that moved (mate or antonym, tagged). What the Δ form surfaces: for optimistic (+0.13) and prominent (+0.36) the dose adds little trait over default conduct (default is already judged optimistic at 5.11), so the conduct-present/label-declined reading is strongest for senile / imaginative / rough; and slim/big is an ANTONYM moving UP — endorsing "big" after slim conduct, the desirability-consistent case, not a trait-consistent denial.

| pair — target / off-target (type) | judged target Δ | self target Δ | self off-target Δ |
|---|---|---|---|
| prominent / distinguished (mate) | +0.36 | -0.12 | **+2.43** |
| slim / big (ant.) | +0.45 | -0.08 | **+1.95** |
| senile / old (mate) | +1.89 | +0.02 | **+1.23** |
| rough / weak (ant.) | +0.95 | -0.40 | **-1.69** |
| optimistic / depressed (ant.) | +0.13 | +0.14 | **-1.28** |
| imaginative / boring (ant.) | +1.60 | -0.13 | **-1.07** |

## Item-set provenance (compress to 1–2 sentences in the note)

- Readout instrument per target adjective: fixed 9-item set = target + 4 cluster-mates + 4 anti-markers, specified BEFORE any dosing (the pre-specification is the load-bearing fact for Exhibit 4: `weak` was already in `rough`'s set).
- Mates: membership in the human-derived facet clusters (instruments/trait_clusters.json, W18); nearest-neighbor fallback for unclustered targets.
- Anti-markers: anticorrelation in a model-derived judgment-similarity matrix, desirability-partialled (raw anticorrelation returns the desirability floor — evil/corrupt/… — for every positive target). NOTE FOR THE DRAFT: phrase it exactly that flatly; do NOT introduce the JUDGE channel name — this is its only appearance in the note and it isn't worth the taxonomy. Future runs: human 525-PDA anticorrelation works raw (no floor, no partialling — see to_try amendment 2026-08-02) and would make the item provenance one citable clause.
- Code: scripts/selfperception_dose.py item_sets(); design doc §5.5 says "~13 items" — the implemented count is 9.

## Figures

- `fig_dose_response.png` / `.html` — Exhibit 1b as curves (log-x)
- `fig_ladder.png` / `.html` — Exhibit 3, OLMo ladder + Qwen base pair + Llama8 control
