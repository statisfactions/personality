# HANDOFF — JUDGE track reanalysis on the FULL raw distributions

> **STATUS: DONE for the full 12-model cohort (2026-07-23).** `convert_full_dists.py` written;
> all 5 reports + the `ecb-reports/judge_psychometric_soundness.md` narrative + this dir's
> `README.md` re-run on the canonical raw distributions and re-knitted. **Batch 2
> (`FalconMamba`, `Gemma-4`, `Qwen-32`) landed 2026-07-22** (`judge_dists_full_batch2_20260722.tar.gz`)
> and is incorporated — cohort complete. `convert_full_dists.py` reads whatever npz sit in
> `data/dists_full/judge_dists/`. **New batch-2 headline: `FalconMamba` is the standout —
> 78% off-mode yet 90% unimodal (diffuse digit-prior, NOT bimodal), near-uniform entropy
> (1.66 nats), and rgb reports its distribution-level readout carries the cohort's STRONGEST
> human-match (EV r≈0.73) — "the flattest model was the one EV-only storage under-sold most."**
> `Gemma-4-31B` is the sharpest rater (entropy 0.11) and reverts to the extreme low-agreement
> idiosyncratic outlier. Going 9→12 restored most original figures (ICC 0.61/0.95, PC1 53%,
> congruence 0.84, asym median ~21%); the two canonical sign-flips (§4 disagreement at the
> low-similarity/confident end; §6 double-centering raises agreement 0.76→0.79) still hold.
>
> **Correction to this doc's own headline:** the "phi4 off-mode 3%→56%, medoid sample badly
> under-represents spread" framing below was **wrong**. Recomputing rgb's off-mode metric on
> the medoids gives phi4 = **50.5%** (vs 56.1% full), Aya 26.6% (vs 26.2%) — the medoids
> tracked off-mode mass *fine*. The "3%" conflated medoid *strict-semantic* (~1.9%) with full
> *off-mode* (56%). The real correction (Report I §6): the medoid-era "faithful for 10/12"
> verdict erred by certifying faithfulness off **strict bimodality alone** — `Phi4` is 56%
> off-mode yet **93% unimodal** (broad-unimodal), `Aya` is the only genuinely bimodal rater
> (~15% semantic). Off-mode mass ≠ bimodality; report both. The sections below are the
> original brief, kept for provenance.

**Written 2026-07-22 to survive a context clear.** Immediate next task: **reanalyze the
JUDGE track using rgb's full raw 7-category distributions**, which supersede the EV-only
`B`/`Hent` the current reports were built on. Read this + `README.md` first.

## The new data (already at repo root)

`judge_dists_full_batch1_20260712.tar.gz` (40 MB) → `judge_dists/<model>_tom_likely_dists_full.npz`.

- **9 models** in batch1: `Aya, Gemma12, Gemma27, gemma3, llama3.2, Llama8, phi4, qwen2.5, Qwen7`.
  **batch2 expected** (not yet here): `FalconMamba, Gemma4, Qwen32`.
- npz keys: `dists (523,523,7) float16` (P(rating k+1), row-conditioned, **diagonal NaN**),
  `ev`/`entropy (523,523) f32` (recomputed from `dists`), `adjectives (523,)` **lowercase**,
  `prompt` (str), `reproduction_mean_abs_dev` (0.01–0.08/model).
- **These SUPERSEDE** `results/adjectives/introspect_full/*_tom_likely_dir.npz` (the June
  EV-only release). bf16 numerics drifted since June (mean |ΔEV| 0.02–0.09; **bimodal cells'
  EV flipped up to ~4**). **Use `dists`/`ev` here as canonical; do NOT splice cell-by-cell
  into the old `B`.**
- Model stem → canonical (matches `model_meta.csv`): `gemma3→Gemma, llama3.2→Llama,
  phi4→Phi4, qwen2.5→Qwen`; others unchanged. (Same `CANON` map as `convert_medoids.py`.)
- Adjectives are **lowercase** here vs Capitalized in `adjectives.csv`/old release — normalize
  (case-insensitive) when joining.

Superseded/context: 35-medoid raw sample in `data/medoids/` (now redundant for the 9 batch1
models; still the only raw for FalconMamba/Gemma4/Qwen32 until batch2). Exact prompt is in
`README.md`.

## THE headline the reanalysis must fix

**The 35-medoid sample badly under-represented distribution spread**, because medoids are
clean facet *exemplars* (central cases). On the full 523², rgb's bimodality metric
(**fraction of off-diagonal cells with ≥0.25 total mass ≥2 categories from the mode** —
reproduced exactly) is:

| phi4 | Aya | Llama8 | llama3.2 | qwen2.5 | Qwen7 | Gemmas |
|---|---|---|---|---|---|---|
| **56%** | **26%** | 10% | 7% | 8% | 4% | ~0% |

vs. my medoid-based Report I §6 which concluded "proxy faithful for 10/12 models." **That
conclusion is wrong on the full data for phi4/Aya (and softly Llama8).** Note this metric is
*off-mode spread*, NOT strict two-mode bimodality — report BOTH with clear definitions:
- rgb's off-mode-mass metric (above), and
- strict prominence bimodality via `shape_class()` (semantic mixture vs digit-2/6 notch).
The `(EV,entropy)` spread index *does* correctly flag phi4/Aya as diffuse; what it cannot do
is say whether that diffuseness is broad-unimodal vs genuinely multi-modal — that's the new
value of the raw data.

## Ingestion plan (proposed)

Write `convert_full_dists.py` (mirror `convert_judge.py`/`convert_medoids.py`):
- Read the 9 full npz; **recompute `B`/`Hent` from `dists`** (eliminates the ~0.5% envelope
  inconsistency; canonical).
- Emit, consistent with existing tidy tables so Reports II–V rerun with minimal change:
  `dir_ev`, `sym_ev`, `asym_ev`, `sym_hent`, `marginals`, `model_summary` — but from the
  regenerated EV (note: only 9 models now; keep a `models_available` set).
- Emit NEW full-distribution products: per-model category-usage, floor/ceiling, off-mode-mass,
  and `shape_class` census over all 523² (this is what Report I §6 becomes). The full long
  dist table is large (523²×7×9 ≈ 12M rows) — prefer per-model summaries + keep the npz for
  any cell-level work rather than a giant CSV.
- Heavy outputs stay gitignored (root `data/` rule ignores all of `psychometrics/judge/data/`).

## What each report needs

- **Report I** — biggest change. Replace the medoid §6 with a full-523² raw analysis; **revise
  the "faithful for 10/12" verdict** (phi4/Aya are NOT summarized by EV alone). True category
  usage/floor-ceiling/shape cohort-wide. Keep the spread index but position it as "flags
  diffuseness, not shape."
- **Reports II–V** — recompute EV/entropy from `dists` (canonical) and re-knit. Structure is
  EV-based so mostly stable, but drift up to ~4 EV in bimodal cells can move specific pairs;
  re-verify the asymmetry (~21% centered), reliability (ICC 0.61/0.95), and structure (halo
  PC1≈53%) numbers. **Only 9 models now** — the ICC/consensus/D-study numbers will shift
  slightly; redo, don't assume.
- **New opportunity** — the full 7-way distribution enables **soft-evidence** treatment
  (bridges ecb's earlier soft-evidence IRT work): distributional similarity (e.g. EMD between
  judgment distributions) as an alternative to EV-cosine; confidence/entropy weighting with
  real distributions.

## Corrections & gotchas — DO NOT re-derive or repeat these mistakes

1. **Flat raters (Llama3.2, FalconMamba) AGREE with the consensus** (low-information, not
   low-quality). Do not down-weight/trim. Rater quality is multidimensional (information /
   between-rater agreement / within-rater consistency are ~orthogonal). [Reports III, V]
2. **Asymmetry is ~21% of CENTERED variance** (the mean-inclusive 0.2–3.5% is misleading — the
   grand mean dominates that denominator). Symmetrize for the similarity because the
   antisymmetric part is ~2× less reliable across raters and ~½ removable prevalence gradient
   — NOT because it's negligible. [Report II]
3. **`h_min_mean` = dominant mass on the nearest integer + remainder at the far extreme**
   (verified vs brute force). The adjacent-2-point version is WRONG (overestimates). [_common.R]
4. **Bimodality: use the prominence test (`shape_class`), not raw local-maxima** (which counts
   shallow shoulders). Separate genuine semantic mixtures from digit-2/6 suppression notches.
5. **Digit-2/6 suppression** is a real rating-channel artifact (echoes the Llama-2 category-2
   suppression in the soft-evidence track), distinct from semantic structure.
6. **Evaluative halo: PC1 ≈ 53% of common variance** dominates the factor structure = the main
   discriminant-validity threat. Take trait scores net of the general factor. [Report IV]
7. **Human correspondence ≠ validity** (ecb's explicit steer). Extrapolation-to-"personality"
   is left open (lexical/evaluative co-occurrence is a live deflationary account). Pass-2 human
   criterion deferred + blocked on adjective alignment (human 525-PDA labels UPPERCASE +
   8-char-truncated, e.g. `ACCOMPLI`).
8. **Consensus is robust**: single LLM ICC(2,1)≈0.61, ensemble ICC(2,k)≈0.95, ~6–7 raters
   suffice, weighting/trimming buys ~nothing.
9. **The medoid sample is UNREPRESENTATIVE** of spread/bimodality (batch1 proves it: phi4
   3%→56%). Never generalize medoid shape stats to the full set.

## Repo / pipeline state

- `psychometrics/judge/`: `convert_judge.py`, `convert_medoids.py`, `reports/_common.R`
  (helpers: `fread_gz`, `build_matrix`, `profile_cor`, `double_center`,
  `h_min_mean`/`h_max_mean`/`spread_index`, `shape_class`, `model_meta`/`MODELS`/`DISPLAY`/
  `FAMILY`/`disp_ord`/`FAM_PAL`/`theme_judge`/`upper_vec`), `reports/01–05_*.Rmd`+`.html`,
  `README.md`, `REQUEST_raw_distributions_rgb.md`, this `HANDOFF.md`.
- Narrative: `ecb-reports/judge_psychometric_soundness.md` (Kane: scoring/generalization/
  extrapolation).
- Data all gitignored (root `data/` rule). HTML reports ARE committed. On branch `main`,
  latest commit ~`55756f3`; nothing pushed.
- R deps added along the way: `ggrepel`, `hexbin`, `ggridges` (via r2u).
- Regenerate: `python3 convert_*.py` then `cd reports && Rscript -e "rmarkdown::render('NN_*.Rmd')"`.
