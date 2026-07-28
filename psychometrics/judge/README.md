# JUDGE channel — psychometric exploration (ecb)

A **measurement-focused** re-analysis of rgb's JUDGE (`tom_likely`) data: each LLM's
conditional-endorsement Likert judgments over the 523 person-descriptive adjectives
(Saucier 525-PDA − 2 corrupted → 523). For an ordered pair *(i, j)* the model answers
*"how strongly would a person described as **i** also be **j**?"* as a full 7-category
rating distribution, giving a `523×523×7` tensor `dists` per model, from which we recompute
the expected-rating matrix `B` and entropy matrix `Hent`.

**Current basis: the full raw distributions** (`judge_dists_full`), now complete for **all 12
cohort models** (batch 1 = 9 models 2026-07-12; batch 2 = `FalconMamba, Gemma-4, Qwen-32`
2026-07-22). These supersede the earlier EV-only release (bf16 numerics had drifted; bimodal
cells' EV flipped up to ~4). All reports and the companion narrative are the **12-model
reanalysis** on the canonical raw distributions.

rgb's own analysis (W15–W16, `rgb_reports/`) is **geometric/mechanistic** (read/write
dissociation, Helmholtz–Hodge asymmetry, factor congruence to humans). This track asks
the **psychometric** question the geometry brackets: *treated as measurements, are these
judgments sound?* — with the unit of analysis being **models as raters**.

Nothing here writes to `rgb_reports/` or `results/`; it only reads the source npz.

## Pipeline

```
data/dists_full/judge_dists/*_tom_likely_dists_full.npz   (rgb source, gitignored)
        │
        ▼  python convert_full_dists.py   (recomputes B/Hent from dists; canonical)
data/   *.csv.gz  tidy tables (directional / symmetric / antisymmetric pairs,
        marginals, per-model summary) + full-distribution products
        (full_catmass, full_shape, full_examples)
        │
        ▼  reports/*.Rmd  (source reports/_common.R)
reports/*.html
```

Regenerate everything (extract both `judge_dists_full_batch*` tarballs into
`data/dists_full/judge_dists/` first — `convert_full_dists.py` reads whatever npz are there):

```bash
python3 convert_full_dists.py
cd reports && for f in 0*_*.Rmd; do Rscript -e "rmarkdown::render('$f')"; done
```

`convert_judge.py` (from the old EV-only 12-model `introspect_full` npz) and
`convert_medoids.py` (35-adjective raw sample) are retained for provenance; the current
pipeline is `convert_full_dists.py`.

`sample_frame_pairs.R` → `data/frame_pairs.csv` (252 stratified pairs), `run_frames_ollama.py`
→ `frames_raw/*.jsonl` (**tracked** — collected inference is not regenerable),
`run_frames_all.sh`, `convert_frames.py` → `data/frames_tidy.csv`. These feed Report VII.

`adj_freq.py` adds `data/adj_freq.csv` (wordfreq Zipf frequencies for the 523 adjectives) —
a rival predictor of prominence used by Report VI. Run after `convert_full_dists.py`.

`dist_processes.py` is a second pass over the same npz for **Report I §7** — it emits
response-process products (`proc_content`, `proc_signature`, `proc_modepairs`,
`proc_endpoint_examples`) that read the distributions *as processes* rather than recomputing
moments. Run it after `convert_full_dists.py`; both feed Report I.

## Reports (`reports/`)

| # | file | what |
|---|------|------|
| I   | `01_response_processes.Rmd` | scale usage (expected-rating distribution), spread-index shape (mean-detrended concentration), decisiveness (entropy), response-style typology, marginals/base-rates, **§6 full 523² raw distributions** (off-mode mass vs strict bimodality; corrects the medoid-era "faithful for 10/12" verdict — `FalconMamba` diffuse-digit-prior, `Phi4` broad-unimodal, `Aya` genuinely bimodal), **§7 distributions as response processes** (EV-conditioned signature; rating-entropy split into content=MI(pair;rating) vs hedge; mode-pair mechanism census; coin-flips on ill-posed pairs) |
| II  | `02_asymmetry.Rmd`          | how much asymmetry, gradient(prevalence)/curl split, cross-rater reliability of the asymmetry, measurement verdict (symmetrize) |
| III | `03_reliability.Rmd`        | G-theory variance components, ICC(2/3), D-study, rater-quality anatomy, which pairs are reliably vs idiosyncratically judged, within-rater direction reliability |
| IV  | `04_structure.Rmd`          | dimensionality, 5-factor solution, **structural invariance across raters** (Tucker congruence), the evaluative-halo general factor as a discriminant-validity threat |
| V   | `05_consensus.Rmd`          | *reactive* — does weighting/trimming the ensemble beat the flat mean? (no); consensus growth curve |
| VII | `07_frame_experiment.Rmd`   | **The frame experiment** — same 252 pairs asked three ways (rgb's conditional / explicit similarity / explicit difference) × both orders on 5 Orin raters. Finds the Report VI prominence gradient is **strongly frame-bound** (replicates out-of-sample in `cond`, 3–5× weaker in `sim`/`diff`); non-complementarity is real and **grows with model scale**; `cond` ≠ an explicit similarity judgment; `phi4-mini` excluded (frame-differential refusal); endpoint logprobs found unreliable |
| VI  | `06_tversky_asymmetry.Rmd`  | **Tversky (1977) contrast-model reading of the asymmetry** — the gradient/curl split *is* the contrast model's fit vs residual; recovered prominence scale `f = -g` (prevalence + an independent negativity term, **not** lexical frequency); additivity test; the curl as *diagnosticity*, low-rank and strongly shared once measured at plane/construct resolution; Tversky–Hutchinson nearest-neighbour centrality (negative result: the symmetric part *is* spatial); per-rater contrast weights; why the missing diagonal *cannot* identify α/β (ceiling + pragmatic degeneracy) and the dissimilarity-frame experiment that can |

Companion narrative (validity-argument framing): `ecb-reports/judge_psychometric_soundness.md`.

## Headline findings

- **Response styles differ ~4× across raters** (leniency, differentiation, decisiveness
  are ~independent). Two raters (`Llama3.2-3B`, `FalconMamba`) are range-restricted / low-information.
- **The two-moment (EV, entropy) summary fails for `FalconMamba`, `Phi4` and `Aya`** on the
  full raw data, each *differently*: `FalconMamba` diffuse-with-a-digit-prior (78% off-mode,
  90% unimodal), `Phi4` **broad-unimodal** (56% off-mode, 93% unimodal), `Aya` **genuinely
  bimodal** (~15% empty-middle mixtures). Adequate for the other 9. `FalconMamba`'s graded
  mass carries the cohort's strongest human-match (EV r≈0.73) — the flat model EV-only
  storage under-sold most.
- **Asymmetry is ~21% of the *centered* signal** (not the 0.2–3.5% a mean-inclusive
  figure suggests), ≈½ removable prevalence gradient, and ~2× less reliable across raters
  than the symmetric part → symmetrize for the similarity, keep directionality as a
  separate channel.
- **That asymmetry is a Tversky contrast model (Report VI).** The gradient half is the
  model's *prominence* term — recovered scale `f` = base rate (R²=0.72) **plus an
  independent negativity component** (ΔR²=0.12 net of a 6-df prevalence spline; undesirable
  traits are stronger referents) and **zero** lexical-frequency component, so it is a
  feature-salience measure, not a base rate. Direction is as Tversky predicts (97% of
  large-gap pairs). The curl half is *diagnosticity* and, contra Report II, is **low-rank
  and strongly shared** (dominant-plane congruence 0.73 vs null 0.05; construct-level 0.72
  vs 0.30) — Report II's r̄ 0.34 measured cell-level noise. Nearest-neighbour centrality
  (C = 2.31, max Nᵢ = 9) says the *symmetric* part is comfortably spatial: JUDGE is
  non-metric in its asymmetry, not in its similarity.
- **A single LLM is a moderately reliable rater** (ICC(2,1)≈0.61); the **cohort consensus
  is excellent** (ICC(2,k)≈0.95). ~6 raters suffice for G≥0.9; weighting/trimming buys ~nothing.
- **Between-rater disagreement peaks at the *low-similarity, confident* end** (not a
  "contested middle") and falls with entropy — raters converge on hedged mid-scale judgments.
- **Response processes differ, not just moments (§7).** A rater's rating entropy splits into
  *content* (MI the pair carries about the rating) + *hedge* (fixed within-judgment entropy):
  the Gemmas are 75–92% content, `FalconMamba` **7.5%** — "flat" is low-*content*, not
  low-variability. At a fixed EV the shape varies wildly (graded slide vs endpoint fork vs
  digit lumps), and balanced 1-vs-7 coin-flips cluster on category-orthogonal pairs
  (`Phi4` → *guilty*) — the model flips rather than hedges on ill-posed questions.
- **Internal structure is coherent, low-rank and reproducible across raters** (mean factor
  congruence 0.84), resembling Big Five — but dominated by an **evaluative halo** (PC1 ≈
  53%) that is its main discriminant-validity threat.

## Instrument (exact `tom_likely` prompt, from the medoids sample)

```
Consider a person who is very {a}.
How likely is this person to also be {b}?
Answer with one number from 1 to 7, where 1 = very unlikely and 7 = very likely.
Number:
```
Identical to the W16 `introspect_full` runs; use verbatim for any re-collection
(e.g. an Ollama replication) to stay comparable.

## Data tables (`data/`, gitignored — regenerable)

`adjectives.csv`, `model_meta.csv`, `model_summary.csv` (now with true category mass,
off-mode-mass and shape fractions), `marginals.csv`, and wide `i,j,<model...>` tables:
`dir_ev`, `dir_hent` (ordered i≠j), `sym_ev`, `asym_ev`, `sym_hent` (i<j). Full-distribution
products: `full_catmass.csv` (true per-category mass), `full_shape.csv` (off-mode + shape
census over all 523²), `full_examples.csv` (matched-EV opposite-shape illustrations). See
`convert_full_dists.py` header for schemas. Response-process products (from
`dist_processes.py`, Report I §7): `proc_content.csv` (rating-entropy content/hedge budget),
`proc_signature.csv` (EV-conditioned category distribution), `proc_modepairs.csv` (mode-pair
census), `proc_endpoint_examples.csv` (coin-flip exemplars). All tables carry the **12 cohort
raters** in the npz's alphabetical adjective order (a different index space than the old
EV-only tables).

## Open / next

- **Raw per-pair distributions — complete (all 12).** The full 523²×7 distributions are the
  basis of the analysis (`convert_full_dists.py`). Report I §6 gives the honest category
  usage, off-mode-mass vs strict-bimodality census, and corrects the medoid-era verdict (the
  35-medoid sample, `data/medoids/`, actually tracked off-mode mass fine — even `FalconMamba`
  78.8%→78.3%; the earlier error was metric choice). Three distinct diffuseness profiles:
  `FalconMamba` (diffuse digit-prior), `Phi4` (broad-unimodal), `Aya` (genuinely bimodal).
- **Soft-evidence treatment (unblocked; full cohort in hand).** Distributional similarity (EMD
  between judgment distributions) vs EV-cosine, and real-distribution confidence weighting —
  most valuable for `FalconMamba`/`Aya`/`Phi4`, whose EV the two moments misread. Bridges the
  earlier soft-evidence IRT work; `FalconMamba`'s distribution-level human-match (EV r≈0.73)
  is the strongest single motivation.
- **Pass 2 — human criterion (deferred).** The in-repo human 525-PDA matrix
  (`results/adjectives/escs_525pda_corr_raw.json`, N=700) is the natural criterion, but
  its labels are UPPERCASE + 8-char-truncated (`ACCOMPLI`), so alignment needs a careful
  prefix/fuzzy match (collision risk on shared prefixes). Held as a scoped follow-up.
  **Caveat (ecb):** human correspondence is *one* strand of evidence, **not** validity in
  itself — a match supports, a mismatch does not disconfirm a coherent measurement of the
  models' own implicit personality theory.
- ~~Dissimilarity-frame experiment~~ **— DONE (Report VII, 2026-07-29).** α and β are *not* separately
  identified by the collected data (we recover only `(β−α)f`). Tversky's lever is to ask for
  **difference** instead of similarity: a base-rate account (`B[i,j]≈P(j|i)`) predicts **no
  reversal**, the contrast model predicts the prominence gradient **inverts**. Needs only a
  few thousand high-prominence-gap pairs + a low-gap control, ideally with a symmetric
  ("how similar are…") frame as the α=β condition. Four cohort models are on the Orin.
- **The diagonal is NOT the fix** (§8.1, corrected 2026-07-28). Collecting `s(a,a)` fails on
  three counts: the self-pair task is pragmatically degenerate (*also* presupposes a distinct
  predicate), Tversky read `s(a,a)` off confusion data rather than by asking, and — decisive —
  **ceiling**: 58% of adjectives already reach ≥6.5 and 11% ≥6.9 via their best near-synonym
  on a 7-point scale, so the diagonal would be a wall of 7s with no variance to read `f` from.
  Worth 523 prompts only as an **instrument-integrity check** (does the rater read a contrast
  presupposition into *also*?) and to calibrate per-rater ceilings.
- Directional (curl) channel as a substantive implicit-personality-theory construct.
- Ties to the model-as-unit Kane validity argument (`memory/project_validity_argument_program`).
