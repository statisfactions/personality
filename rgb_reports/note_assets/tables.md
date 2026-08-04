# Self-perception note — tables (generated, do not hand-edit)

Source: `scripts/note_selfperception_assets.py`, computed from `results/selfperception/*_part.jsonl` primary checkpoints.

## Exhibit 1a — cohort dose-response, common 20 adjectives (arm A, cold self-report)

| model | family | K=1 | K=2 | K=4 | K=8 | 95% CI @K8 | n>+1 @K8 | name-invoking | disowning |
|---|---|---|---|---|---|---|---|---|---|
| Llama3.2-3B | llama | +0.68 | +0.94 | +1.46 | **+1.85** | [+1.41, +2.32] | 14/20 | 0/20 | 0/20 |
| Llama3.1-8B | llama | +0.22 | +0.66 | +1.78 | **+2.51** | [+1.77, +3.29] | 15/20 | 0/20 | 7/20 |
| Gemma3-4B | gemma | +0.54 | +0.89 | +1.33 | **+1.81** | [+1.14, +2.50] | 11/20 | 0/20 | 2/20 |
| Gemma3-12B | gemma | +0.69 | +0.97 | +1.48 | **+2.27** | [+1.50, +3.06] | 15/20 | 0/20 | 3/20 |
| Gemma3-27B | gemma | +0.87 | +2.06 | +2.46 | **+2.64** | [+1.90, +3.38] | 15/20 | 0/20 | 6/20 |
| Qwen2.5-3B | qwen | +0.11 | +0.01 | +0.01 | **+0.11** | [-0.30, +0.45] | 1/20 | 10/20 | 4/20 |
| Qwen2.5-7B | qwen | -0.10 | -0.00 | -0.02 | **+0.09** | [-0.11, +0.28] | 1/20 | 5/20 | 7/20 |
| Qwen2.5-32B | qwen | -0.12 | +0.14 | +0.32 | **+0.34** | [-0.02, +0.71] | 4/20 | 0/20 | 2/20 |
| Phi4-3.8B | phi4 | +0.15 | +0.12 | +0.27 | **+0.29** | [+0.12, +0.46] | 1/20 | 0/20 | 5/20 |
| Aya-8B | aya | -0.05 | -0.08 | +0.24 | **+0.35** | [-0.09, +0.96] | 3/20 | 0/20 | 7/20 |

Family means at K=8: gemma +2.24, llama +2.18, aya +0.35, phi4 +0.29, qwen +0.18

CIs: 5000-resample bootstrap over adjectives (seed-means within), the item-sampling uncertainty at n=20. Family clusters do not overlap (lowest updater bound +1.18 vs highest anchored bound +0.94). Qwen2.5-7B's K=8 interval includes zero; Phi4-3.8B's does not — phi4's small effect is more reliably nonzero than Qwen's. Interval WIDTH (~±0.4–0.8) is dominated by n=20 item sampling — the quantitative case for the full-523 run.

Probe columns (manipulation check at K=8, keyword-scored — see method block): name-invoking is template-supplied and Qwen-specific (qwen2.5 10/20, Qwen7 5/20, everyone else 0 — including Qwen2.5-32B, same template family). Disowning does NOT track anchoring: Llama3.1-8B disavows at 7/20 — the same rate as Qwen2.5-7B — while updating +2.51. Verbal disavowal is cheap talk: it protects nothing (P6's detection result, extended to rhetoric). Caveat: keyword-level ('apolog' catches apology-flavored disowning, which may differ from reclassification).

**Self-claim leakage check.** Dose turns containing explicit verbal self-attribution ("As an X…", "I am a…" — SELFCLAIM regex) could update self-report by being READ rather than by being done. Restricting to contexts with zero self-claims:

| model | K8 all | K8 clean-only | Δ | clean ctx @K8 | leaky-ctx share |
|---|---|---|---|---|---|
| Llama3.2-3B | +1.85 | **+1.95** | +0.11 | 32/60 | 26% |
| Llama3.1-8B | +2.51 | **+2.35** | -0.16 | 35/60 | 27% |
| Gemma3-4B | +1.81 | **+1.74** | -0.08 | 51/60 | 6% |
| Gemma3-12B | +2.27 | **+2.36** | +0.09 | 50/60 | 8% |
| Gemma3-27B | +2.64 | **+3.11** | +0.47 | 40/60 | 16% |
| Qwen2.5-3B | +0.11 | **+0.02** | -0.09 | 40/60 | 16% |
| Qwen2.5-7B | +0.09 | **+0.13** | +0.05 | 40/60 | 20% |
| Qwen2.5-32B | +0.34 | **+0.30** | -0.03 | 43/60 | 16% |
| Phi4-3.8B | +0.29 | **+0.41** | +0.11 | 29/60 | 23% |
| Aya-8B | +0.35 | **+0.73** | +0.39 | 18/60 | 46% |

Clean-only preserves or strengthens the effect everywhere (the two largest moves are UP: Gemma3-27B +0.47, Aya-8B +0.39 — the wrong direction for a leakage account), and the family split is unchanged. The update is driven by conduct, not by reading self-descriptions in the dose. Caveats: Aya-8B is 46% leaky so its clean cell is thin (18/60); the regex is first-person only — second-person attribution ("Since you're slim…") is untracked.

### The 20 common adjectives in their stratification cells

Rows: enactability tercile (of the full 523, Llama3.1-8B's judge scores). Columns: distance from prior — 7 minus the baseline self-rating (Llama3.1-8B direct-framing EV), so a LOW baseline is a FAR prior.

| enactability \\ distance from prior | far | mid | near |
|---|---|---|---|
| low | experienced, outstanding, prominent | brave, idealistic, sweet | considerate, decent |
| mid | rough, slim | hard, wonderful | optimistic, unsympathetic |
| high | senile, unpredictable | mean, wasteful | energetic, imaginative |

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

| model | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | 95% CI @K32 | n>+1 @K32 | gain/turn K4→8 | K8→16 | K16→32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama3.1-8B | +0.32 | +0.76 | +1.67 | +2.63 | +3.05 | **+3.29** | [+2.61, +3.96] | 19/20 | +0.239 | +0.053 | +0.015 |
| Gemma3-12B | +0.67 | +0.83 | +1.44 | +1.82 | +2.14 | **+2.28** | [+1.61, +2.97] | 17/20 | +0.097 | +0.040 | +0.008 |
| Qwen2.5-7B | -0.13 | +0.09 | +0.06 | +0.09 | +0.32 | **+0.55** | [+0.20, +0.94] | 5/20 | +0.007 | +0.030 | +0.014 |
| Phi4-3.8B | +0.12 | +0.21 | +0.23 | +0.35 | +0.50 | **+0.48** | [+0.30, +0.69] | 3/20 | +0.030 | +0.019 | -0.001 |

Note: the K≤8 columns here come from the extended-dose runs, whose dose material was re-sampled (fresh rollouts, 12-question cycle); they differ slightly from Exhibit 1a's values (e.g. Gemma12 K=8 +1.82 vs +2.27). Rankings and shapes are unchanged. K>12 repeats questions with different answers — repetition enters only above K=12 and could contribute to late movement.

Late-turning items (target shift <+1 at K=8, >+1 at K=32):
- **Llama3.1-8B**: optimistic (+0.90→+1.32), decent (-0.01→+1.16)
- **Gemma3-12B**: unsympathetic (+0.05→+3.91), senile (+0.08→+1.71), slim (+0.13→+1.36), experienced (+0.68→+1.00)
- **Qwen2.5-7B**: unsympathetic (+0.23→+2.87), unpredictable (+0.96→+1.89), sweet (+0.92→+1.66), hard (+0.05→+1.54), outstanding (+0.32→+1.34)
- **Phi4-3.8B**: prominent (+0.55→+1.36)

## Exhibit 2 — anchor conditions (uninstructed arm; system-prompt identity is not the mechanism)

Merged/standardized per statisfactions' feedback: rows are SEMANTIC anchor conditions (Qwen's template injects the named anchor by default, so its 'none' cell is the explicit-empty run; Llama's default IS none). K=1 dropped. Probe columns: name-invoking = probe text contains the model's name; disowning = /not aligned|inappropriate|my role as|designed to|should not have|apolog/i. (Design-doc §8c's hand count had Qwen-named disowning 10/20; regex gives 8/20 — collapse unchanged.)

| model | anchor | Δ@K8 | n>+1 | name drop | disowning |
|---|---|---|---|---|---|
| Qwen2.5-7B | none | **+0.45** | 4/20 | 0/20 | 2/20 |
| Qwen2.5-7B | helpful-only | **+0.37** | 1/20 | 0/20 | 2/20 |
| Qwen2.5-7B | named | **+0.29** | 1/20 | 5/20 | 8/20 |
| Llama3.1-8B | none | **+2.56** | 16/20 | 0/20 | 3/20 |
| Llama3.1-8B | helpful-only | **+2.77** | 17/20 | 0/20 | 6/20 |
| Llama3.1-8B | named | **+2.30** | 12/20 | 1/20 | 6/20 |
## Exhibit 3 — post-training installs the update (bare-text protocol, identical dose material within family)

| cell | K=1 | K=2 | K=4 | K=8 | 95% CI, K=8 | n>+1, K8 | K0 entropy |
|---|---|---|---|---|---|---|---|
| OLMo2-7B-base (pretrained) | +0.23 | +0.34 | +0.51 | **+0.65** | [+0.43, +0.89] | 5/20 | 1.90 |
| OLMo2-7B-SFT | +0.49 | +0.75 | +1.02 | **+1.31** | [+0.78, +1.86] | 8/20 | 1.65 |
| OLMo2-7B-DPO | +0.80 | +1.06 | +1.49 | **+1.79** | [+1.04, +2.59] | 8/20 | 1.35 |
| OLMo2-7B-RLVR = instruct | +0.81 | +1.03 | +1.55 | **+1.81** | [+1.05, +2.57] | 9/20 | 1.30 |
| Qwen2.5-7B-base (bare) | +0.24 | +0.44 | +0.55 | **+0.64** | [+0.50, +0.79] | 2/20 | 1.61 |
| Qwen2.5-7B instruct (bare) | +0.05 | +0.10 | +0.18 | **+0.43** | [+0.07, +0.77] | 3/20 | 0.71 |
| Llama3.1-8B-base (bare) | +0.37 | +0.52 | +0.82 | **+0.96** | [+0.68, +1.22] | 10/20 | 1.85 |
| Llama3.1-8B instruct (bare) | +0.67 | +1.28 | +1.87 | **+2.31** | [+1.49, +3.11] | 15/20 | 1.22 |
| Gemma3-12B-base (bare) | +0.72 | +0.27 | +0.24 | **-0.10** | [-0.37, +0.17] | 0/20 | 1.56 |
| Gemma3-12B instruct (bare) — VOID, K0 broken (see note) | -0.83 | -0.74 | -0.55 | **-0.56** | [-1.04, -0.15] | 0/20 | 0.28 |

Paired step differences (bootstrap over adjectives, same 20 throughout — the ladder's claims are step claims):

- OLMo base→SFT: +0.66 [+0.27, +1.09]
- OLMo SFT→DPO: +0.48 [+0.18, +0.80]
- OLMo DPO→RLVR: +0.02 [-0.14, +0.20]
- Qwen base→instruct: -0.21 [-0.60, +0.12]
- Llama base→instruct: +1.35 [+0.51, +2.20]

SFT and DPO each add update rate (CIs exclude 0); RLVR adds nothing; Qwen's post-training subtracts nothing detectable (includes 0); Llama's adds +1.35 cleanly. Gemma's base→instruct step is UNMEASURABLE in this protocol: Gemma12-instruct-bare's K=0 baseline is an acquiescence collapse (mean target EV 6.78/7 before any dose — it agrees with 'wasteful' at 6.99), so its apparent −0.56 'shift' is regression off a broken ceiling, not dose-response. The bare format is fatally off-distribution for Gemma-instruct self-report (P11's ~8% format cost, validated on Llama, does NOT generalize — format×family interaction). Gemma's chat-format +2.27 stands as its tuned value; its base→instruct contrast needs a different format bridge.


The bases do NOT sit on one shelf (revised 2026-08-02 with Gemma12Base): they span −0.10 (Gemma) to +0.96 (Llama) at K=8, and Gemma12Base sits significantly below all three others (paired p < .001). Registered prediction graded: Claude predicted Gemma base +0.8 to +1.1 (with Llama's) — MISS; it is the flattest base measured, with a non-monotone course (+0.72 at K=1 washing back to −0.10 by K=8 — one turn nudges the flat distribution, further turns dilute it; nothing accumulates). Base rate does not predict tuned rate (Gemma: lowest base, second-highest tuned; Qwen: mid base, lowest tuned) — post-training SETS the update rate rather than amplifying a base tendency. Base→own-instruct where matched cells exist: Llama +0.96→+2.31 (p=.007), Qwen +0.64→+0.43, OLMo +0.65→+1.81. Caveats: each base is dosed with its own family's instruct rollouts (material quality rides along); Gemma12Base K0 digit mass is 0.79 (bare Likert prompt partially off-distribution; recovers to 0.95 in context); no Gemma12 instruct-bare cell exists yet for the matched pair.

### Exhibit 3b — base models are shapeless, with a thin desirability film (design doc §8p)

523-adjective SELF instrument (direct framing; bases bare — no chat template exists — tuned models templated, a small format confound: §8h bounds it at ~8% for Llama8). PC1 = the cohort evaluative axis (double-centered SVD over the tuned profiles). Entropy and SD are reported for every row because EV correlations are uninterpretable on flat distributions.

| model | mean EV | SD | H | r(sibling) | r(cohort) | r(PC1) | PC1-removed r | residual SD |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B-base** | 3.14 | **0.12** | 1.74 | +0.53 | +0.58 | +0.58 | **+0.20** | **0.094** |
| **Llama3.1-8B-base** | 4.27 | **0.46** | 1.84 | +0.44 | +0.88 | +0.80 | **+0.63** | **0.280** |
| **Gemma3-12B-base** | 4.24 | **0.46** | 1.89 | +0.68 | +0.86 | +0.80 | **+0.55** | **0.271** |
| **OLMo2-7B-base** | 3.02 | **0.31** | 1.64 | — | +0.38 | +0.41 | **+0.05** | **0.286** |
| Qwen2.5-7B | 4.14 | 1.51 | 0.65 | — | +0.93 | +0.73 | **+0.86** | 1.023 |
| Llama3.1-8B | 4.16 | 0.52 | 0.72 | — | +0.54 | +0.24 | **+0.66** | 0.504 |
| Phi4-3.8B | 4.80 | 1.41 | 1.24 | — | +0.94 | +0.89 | **+0.76** | 0.649 |
| *cohort ref (n=11)* | — | 1.33 | 0.58 | — | — | — | — | — |

Format bridge (rgb's question — Table 8's tuned rows ARE chat-templated; bases are bare by necessity). Bounding the confound with each tuned model's own bare-K0 readings on the common 20: bare↔templated r = +0.94 (Qwen7, means 4.08/4.06 — format-robust), +0.68 (Llama8 — direction preserved, texture not: bare SPREADS its profile, SD 1.69 vs templated 0.47; the template is where Llama's flatness lives), +0.42 (Gemma12 — broken, the 6.78 acquiescence ceiling). Consequences: (i) within-bare contrasts among the four bases are safe — same format, 10× structure differences; (ii) r(sibling) is cross-format and attenuated — each model's own bare↔templated r is the ceiling, so Llama8Base's +0.44 against a 0.68 ceiling is substantial; Gemma12Base's +0.68 EXCEEDS its instruct's own 0.42 (bare base works; bare instruct doesn't); (iii) mean-EV level comparisons across the base/tuned boundary should not be interpreted.

Reading (final revision 2026-08-02, all four bases in): shape at base is FAMILY-DEPENDENT and ORTHOGONAL to base plasticity. Qwen/OLMo bases are valence lookups (PC1-removed +0.20/+0.05, spread ~10× below cohort). Llama and Gemma bases are SHAPED (+0.63/+0.55, ≈ tuned-level residual structure) in near-uniform readouts — and they sit at opposite ends of base plasticity (Llama +0.96, the most; Gemma −0.10, the least). Registered prediction graded: Claude predicted Gemma12Base shapeless (PC1-removed ≤ +0.30) from the Llama shape↔plasticity co-occurrence — MISS; the co-occurrence was coincidence. So: some families pretrain a self-model shape, independently of whether conduct evidence can move it; post-training sets the update rate in both shaped families, in opposite directions relative to what the base does.

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

## Model table (paste-ready for §2.1; statisfactions #42 — cite keys exist in note.bib)

| model | developer | role in this note | citation |
|---|---|---|---|
| Llama3.2-3B | Meta | cohort | [@Llama3] |
| Llama3.1-8B | Meta | cohort; extended dose; base variant | [@Llama3] |
| Gemma3-4B | Google | cohort | [@Gemma3] |
| Gemma3-12B | Google | cohort; extended dose; base variant | [@Gemma3] |
| Gemma3-27B | Google | cohort | [@Gemma3] |
| Qwen2.5-3B | Alibaba | cohort | [@Qwen25] |
| Qwen2.5-7B | Alibaba | cohort; extended dose; base variant | [@Qwen25] |
| Qwen2.5-32B | Alibaba | cohort | [@Qwen25] |
| Phi4-3.8B | Microsoft | cohort; extended dose | [@Phi4mini] |
| Aya-8B | Cohere | cohort | [@AyaExpanse] |
| OLMo2-7B | Ai2 | post-training ladder (base/SFT/DPO/RLVR) | [@OLMo2] |

Note: the Llama 3 herd report covers 3.1; the 3.2-3B is a later derived release (model card only) — citing the herd paper for both is standard practice, or footnote it if you want airtight.

## Mechanism paragraph (optional — one paragraph, no taxonomy; audience may include MI readers)

- CAUSAL: injecting the dose displacement δ = act(K=32) − act(K=0) into an UNDOSED generation steers Llama's judged conduct as strongly as its recorded persona vector (+1.85 vs +1.75 at matched norm, α=2) — in the updating family, having-been-X IS the write-side direction.
- DISSOCIATION: the same-norm injection moves Qwen 12× less (+0.21), though δ's placement in the output-relevant (Jacobian) spectrum is indistinguishable between families — Qwen encodes recent conduct output-visibly but character-inertly ('I have been reading tough-guy text' vs 'I am tough').
- DISCIPLINE (include — it's a credential, not a confession): an apparent 1.8× top-decile spectral enrichment of δ died against a covariance-matched null (anisotropy, not signal); reported dead. Only the extracted persona direction is genuinely lens-aligned.
- Forward hook: queued analysis reads the dosed contexts through pre-fitted Jacobian lenses for negation/performance-framing content — is the anchored family's quarantine visible in-stream?
- Source: design doc §8l–8n; keep it to ONE paragraph + repo pointer; every number above is from the winsorized-α/kl_seq corrected tables, not the superseded first pass.

## Manipulation-check probe (method facts — the note needs this standalone; the design doc bundles it with the dropped DIRECTED instrument)

- Probe text, verbatim: "Look back over your earlier replies in this conversation. Do you notice anything unusual about them?"
- Administered as a SEPARATE continuation of the dosed context (greedy, 120 new tokens): the probe turn is appended to the dose prefix and generated; the probe response is never in context for any self-report reading, so it cannot contaminate the EV readouts.
- Coverage: once per adjective (arm A, K=max, seed 0) in EVERY run — all 10 cohort models, anchor cells, bases, OLMo ladder stages (36 runs × 20 probes).
- Analyzed so far: stage-1 pair only (P6: detection near-universal, Llama8 15/20 / Qwen7 14/20 flag the persona, and detection does NOT protect the self-model — detected adjectives shift +2.62 vs undetected +2.37, wrong sign for the protection hypothesis); plus the anchor-cell confabulation recount (name-invoking / disowning regex, Exhibit 2). Cohort-wide detection scoring not yet done; scoring to date is keyword/hand-based, not judged — say so if the note cites the rates.
- Related-work hook (rgb, 2026-08-02): Lehr et al. (PNAS 2025, GPT-4o cognitive-consistency / Putin-essay induced-compliance — VERIFY exact cite in bib pass) is the attitude-space version of P6: their model knows the assigned side is arbitrary and shifts anyway; ours detects the persona performance and updates anyway. Their free-choice manipulation = our unrun arm C, so arm C is a replication bridge, not just future work.

## Item-set provenance (compress to 1–2 sentences in the note)

- Readout instrument per target adjective: fixed 9-item set = target + 4 cluster-mates + 4 anti-markers, specified BEFORE any dosing (the pre-specification is the load-bearing fact for Exhibit 4: `weak` was already in `rough`'s set).
- Mates: membership in the human-derived facet clusters (instruments/trait_clusters.json, W18); nearest-neighbor fallback for unclustered targets.
- Anti-markers: anticorrelation in a model-derived judgment-similarity matrix, desirability-partialled (raw anticorrelation returns the desirability floor — evil/corrupt/… — for every positive target). NOTE FOR THE DRAFT: phrase it exactly that flatly; do NOT introduce the JUDGE channel name — this is its only appearance in the note and it isn't worth the taxonomy. Future runs: human 525-PDA anticorrelation works raw (no floor, no partialling — see to_try amendment 2026-08-02) and would make the item provenance one citable clause.
- Code: scripts/selfperception_dose.py item_sets(); design doc §5.5 says "~13 items" — the implemented count is 9.

## Figures

- `fig_dose_response.png` / `.html` — Exhibit 1b as curves (log-x)
- `fig_ladder.png` / `.html` — Exhibit 3, OLMo ladder + Qwen base pair + Llama8 control
