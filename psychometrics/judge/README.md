# JUDGE channel — psychometric exploration (ecb)

A **measurement-focused** re-analysis of rgb's JUDGE (`tom_likely`) data: each LLM's
conditional-endorsement Likert judgments over the 523 person-descriptive adjectives
(Saucier 525-PDA − 2 corrupted → 523). For an ordered pair *(i, j)* the model answers
*"how strongly would a person described as **i** also be **j**?"* as a full 7-category
rating distribution, giving a `523×523×7` tensor `dists` per model, from which we recompute
the expected-rating matrix `B` and entropy matrix `Hent`.

**Current basis: the full raw distributions** (`judge_dists_full` batch 1), which cover **9
of the 12-model cohort** (`Aya, Gemma3-4B/12B/27B, Llama3.2-3B, Llama3.1-8B, Phi4, Qwen2.5-3B/7B`);
`FalconMamba`, `Gemma-4`, `Qwen-32` are pending in batch 2. These supersede the earlier
EV-only 12-model release (bf16 numerics had drifted; bimodal cells' EV flipped up to ~4).
All reports and the companion narrative are the **9-model reanalysis**.

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

Regenerate everything (extract the batch-1 tarball to `data/dists_full/` first):

```bash
python3 convert_full_dists.py
cd reports && for f in 0*_*.Rmd; do Rscript -e "rmarkdown::render('$f')"; done
```

`convert_judge.py` (from the old EV-only 12-model `introspect_full` npz) and
`convert_medoids.py` (35-adjective raw sample) are retained for provenance; the current
pipeline is `convert_full_dists.py`.

## Reports (`reports/`)

| # | file | what |
|---|------|------|
| I   | `01_response_processes.Rmd` | scale usage (expected-rating distribution), spread-index shape (mean-detrended concentration), decisiveness (entropy), response-style typology, marginals/base-rates, **§6 full 523² raw distributions** (off-mode mass vs strict bimodality; corrects the medoid-era "faithful for 10/12" verdict — `Phi4` broad-unimodal, `Aya` genuinely bimodal) |
| II  | `02_asymmetry.Rmd`          | how much asymmetry, gradient(prevalence)/curl split, cross-rater reliability of the asymmetry, measurement verdict (symmetrize) |
| III | `03_reliability.Rmd`        | G-theory variance components, ICC(2/3), D-study, rater-quality anatomy, which pairs are reliably vs idiosyncratically judged, within-rater direction reliability |
| IV  | `04_structure.Rmd`          | dimensionality, 5-factor solution, **structural invariance across raters** (Tucker congruence), the evaluative-halo general factor as a discriminant-validity threat |
| V   | `05_consensus.Rmd`          | *reactive* — does weighting/trimming the ensemble beat the flat mean? (no); consensus growth curve |

Companion narrative (validity-argument framing): `ecb-reports/judge_psychometric_soundness.md`.

## Headline findings

- **Response styles differ ~4× across raters** (leniency, differentiation, decisiveness
  are ~independent). One rater (`Llama3.2-3B`) is range-restricted / low-information.
- **The two-moment (EV, entropy) summary fails for `Phi4` and `Aya`** on the full raw data:
  `Phi4` is diffuse but **broad-unimodal** (56% off-mode mass, 93% unimodal), `Aya` is
  **genuinely bimodal** (~15% empty-middle mixtures). Adequate for the other 7 raters.
- **Asymmetry is ~20% of the *centered* signal** (not the 0.2–3.5% a mean-inclusive
  figure suggests), ≈½ removable prevalence gradient, and ~2× less reliable across raters
  than the symmetric part → symmetrize for the similarity, keep directionality as a
  separate channel.
- **A single LLM is a moderately reliable rater** (ICC(2,1)≈0.62); the **cohort consensus
  is excellent** (ICC(2,k)≈0.94). ~6 raters suffice for G≥0.9; weighting/trimming buys ~nothing.
- **Between-rater disagreement peaks at the *low-similarity, confident* end** (not a
  "contested middle") and falls with entropy — raters converge on hedged mid-scale judgments.
- **Internal structure is coherent, low-rank and reproducible across raters** (mean factor
  congruence 0.86), resembling Big Five — but dominated by an **evaluative halo** (PC1 ≈
  55%) that is its main discriminant-validity threat.

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
`convert_full_dists.py` header for schemas. All tables carry the **9 batch-1 raters** in
the npz's alphabetical adjective order (a different index space than the old EV-only tables).

## Open / next

- **Raw per-pair distributions — arrived (batch 1).** The full 523²×7 distributions for 9
  of 12 models are now the basis of the analysis (`convert_full_dists.py`). Report I §6
  gives the honest category usage, off-mode-mass vs strict-bimodality census, and corrects
  the medoid-era verdict (the 35-medoid sample, `data/medoids/`, actually tracked off-mode
  mass fine; the earlier error was metric choice). **Batch 2** (`FalconMamba`, `Gemma-4`,
  `Qwen-32`) completes the cohort.
- **Soft-evidence treatment (unblocked by the raw data).** Distributional similarity (EMD
  between judgment distributions) vs EV-cosine, and real-distribution confidence weighting —
  most valuable for `Aya`/`Phi4`, whose EV the two moments misread. Bridges the earlier
  soft-evidence IRT work.
- **Pass 2 — human criterion (deferred).** The in-repo human 525-PDA matrix
  (`results/adjectives/escs_525pda_corr_raw.json`, N=700) is the natural criterion, but
  its labels are UPPERCASE + 8-char-truncated (`ACCOMPLI`), so alignment needs a careful
  prefix/fuzzy match (collision risk on shared prefixes). Held as a scoped follow-up.
  **Caveat (ecb):** human correspondence is *one* strand of evidence, **not** validity in
  itself — a match supports, a mismatch does not disconfirm a coherent measurement of the
  models' own implicit personality theory.
- Directional (curl) channel as a substantive implicit-personality-theory construct.
- Ties to the model-as-unit Kane validity argument (`memory/project_validity_argument_program`).
