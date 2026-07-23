# JUDGE raw distributions — FULL 523-set

`<model>_tom_likely_dists_full.npz`: same schema as the medoid files but
523×523 — `dists` (523,523,7) float16, `ev`/`entropy` (523,523) float32,
`adjectives` (523,), `prompt`, `reproduction_mean_abs_dev`. Row-conditioned
("a person who is very {row}: how likely also {col}"), diagonal NaN.
All 12 cohort models (complete 2026-07-20).

These SUPERSEDE introspect_full's B/Hent (regenerated 2026-07, same prompt;
bf16 numerics drifted mean |dEV| 0.01-0.09 since the June EV-only runs, with
bimodal cells flipping up to ~4 EV — use these, don't splice into old B).

Bimodality (mass >=0.25 at >=2 digits from argmax): phi4 56% of cells (!),
Aya 26%, Llama8 10%, llama3.2/qwen2.5 ~7%, Qwen7 4%, Qwen32 1%, Gemmas ~0%.
EV-only analysis is fine for Gemmas/Qwens, misleading for phi4/Aya.
phi4's modes are commit-vs-hedge ((4,6)/(1,3)/(4,7)), not synonym-vs-antonym;
pure (1,7) coin flips are ~1% of its matrix, concentrated on ill-posed
category-mismatch cells (col guilty/fat/unfaithful, row left-handed/blind).

FalconMamba reads 78% on the same criterion but is NOT bimodal — it is
diffuse (mean entropy 85% of uniform) with digit-prior lumps (mass on
1/3/4/5; digits 2 and 6 almost never argmax). Don't over-read the flag;
look at the distributions. Its graded mass carries the cohort's strongest
human-match (EV r=0.73 vs human 525-PDA structure) — the flattest model
was the one EV-only storage under-sold most.

Analysis script: `scripts/judge_dist_structure.py` (bimodality census,
mode-pair census, endpoint/Bernoulli stats, argmax/EV/tail-only human-match).
