# JUDGE channel — psychometric exploration (ecb)

A **measurement-focused** re-analysis of rgb's JUDGE (`tom_likely`) data: each of 12
LLMs' conditional-endorsement Likert judgments over the 523 person-descriptive adjectives
(Saucier 525-PDA − 2 corrupted → 523). For an ordered pair *(i, j)* the model answers
*"how strongly would a person described as **i** also be **j**?"* as an expected 1–7
rating (from the full token log-prob distribution), giving a `523×523` matrix `B` per
model plus an entropy matrix `Hent`.

rgb's own analysis (W15–W16, `rgb_reports/`) is **geometric/mechanistic** (read/write
dissociation, Helmholtz–Hodge asymmetry, factor congruence to humans). This track asks
the **psychometric** question the geometry brackets: *treated as measurements, are these
judgments sound?* — with the unit of analysis being **models as raters**.

Nothing here writes to `rgb_reports/` or `results/`; it only reads the source npz.

## Pipeline

```
results/adjectives/introspect_full/*_tom_likely_dir.npz   (rgb source, gitignored)
        │
        ▼  python convert_judge.py
data/   *.csv.gz  tidy tables (directional / symmetric / antisymmetric pairs,
        marginals, per-model summary + response-style indices)
        │
        ▼  reports/*.Rmd  (source reports/_common.R)
reports/*.html
```

Regenerate everything:

```bash
python3 convert_judge.py
cd reports && for f in 0*_*.Rmd; do Rscript -e "rmarkdown::render('$f')"; done
```

## Reports (`reports/`)

| # | file | what |
|---|------|------|
| I   | `01_response_processes.Rmd` | scale usage (expected-rating distribution), spread-index shape (mean-detrended concentration), decisiveness (entropy), response-style typology, marginals/base-rates, **§6 raw-distribution validation** (35-medoid sample: genuine mixtures vs digit-suppression) |
| II  | `02_asymmetry.Rmd`          | how much asymmetry, gradient(prevalence)/curl split, cross-rater reliability of the asymmetry, measurement verdict (symmetrize) |
| III | `03_reliability.Rmd`        | G-theory variance components, ICC(2/3), D-study, rater-quality anatomy, which pairs are reliably vs idiosyncratically judged, within-rater direction reliability |
| IV  | `04_structure.Rmd`          | dimensionality, 5-factor solution, **structural invariance across raters** (Tucker congruence), the evaluative-halo general factor as a discriminant-validity threat |
| V   | `05_consensus.Rmd`          | *reactive* — does weighting/trimming the ensemble beat the flat mean? (no); consensus growth curve |

Companion narrative (validity-argument framing): `ecb-reports/judge_psychometric_soundness.md`.

## Headline findings

- **Response styles differ ~4× across raters** (leniency, differentiation, decisiveness
  are ~independent). Two raters (`Llama3.2-3B`, `FalconMamba`) are range-restricted /
  low-information.
- **Asymmetry is ~21% of the *centered* signal** (not the 0.2–3.5% a mean-inclusive
  figure suggests), ≈½ removable prevalence gradient, and ~2× less reliable across raters
  than the symmetric part → symmetrize for the similarity, keep directionality as a
  separate channel.
- **A single LLM is a moderately reliable rater** (ICC(2,1)≈0.61); the **cohort consensus
  is excellent** (ICC(2,k)≈0.95). ~6–7 raters suffice; weighting/trimming buys ~nothing.
- **Internal structure is coherent, low-rank and reproducible across raters** (mean factor
  congruence 0.85), resembling Big Five — but dominated by an **evaluative halo** (PC1 ≈
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

`adjectives.csv`, `model_meta.csv`, `model_summary.csv`, `marginals.csv`, and wide
`i,j,<model...>` tables: `dir_ev`, `dir_hent` (ordered i≠j), `sym_ev`, `asym_ev`,
`sym_hent` (i<j). See `convert_judge.py` header for schemas.

## Open / next

- **Raw per-pair distributions.** The main release ships only `B` (mean) and `Hent`
  (entropy); the full 7-category distribution is computed but not persisted (see
  `REQUEST_raw_distributions_rgb.md`). rgb shared a **35-medoid sample** of the full
  `(35,35,7)` distributions (`data/medoids/`, gitignored; tidied by `convert_medoids.py`).
  Report I §6 uses it to **validate** the spread-index proxy (faithful for 10/12 models)
  and to characterize what it hides: `Aya` makes genuine ~11% no-OR-yes *mixtures*; a minor
  digit-2/6 *suppression* artifact runs through Aya/FalconMamba/Qwen; FalconMamba's breadth
  is a notched-unimodal, not bimodal. Full 523² is a cheap re-run for rgb if the Aya
  mixtures warrant it downstream.
- **Pass 2 — human criterion (deferred).** The in-repo human 525-PDA matrix
  (`results/adjectives/escs_525pda_corr_raw.json`, N=700) is the natural criterion, but
  its labels are UPPERCASE + 8-char-truncated (`ACCOMPLI`), so alignment needs a careful
  prefix/fuzzy match (collision risk on shared prefixes). Held as a scoped follow-up.
  **Caveat (ecb):** human correspondence is *one* strand of evidence, **not** validity in
  itself — a match supports, a mismatch does not disconfirm a coherent measurement of the
  models' own implicit personality theory.
- Directional (curl) channel as a substantive implicit-personality-theory construct.
- Ties to the model-as-unit Kane validity argument (`memory/project_validity_argument_program`).
