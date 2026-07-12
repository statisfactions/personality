# LLM-personality data release — rgb track

Underlying data from rgb's LLM-personality measurement work (fork of
Serapio-Garcia et al. 2023). It measures how open-weight LLMs "have" personality
across four channels, plus standard psychometric surveys and RepE trait vectors.
Each tarball is one track; grab whatever's useful.

## Orient here first — the four channels

| channel | what it measures | side | tarball |
|---|---|---|---|
| **HUMAN** | 525-PDA human self-report (Saucier) — behavioral ground truth | — | *not shipped; public, doi:10.7910/DVN/GHYMEV* |
| **REPRESENT** | residual-stream geometry: what the model *represents* about each adjective | read | `adjectives_*.tar` |
| **JUDGE** | implicit personality theory: conditional-endorsement Likert ("is an X person also Y?") | — | `judge_likert_*.tar.gz` |
| **ENACT** | persona-rollout direction: what the model *does* when told to be X | write | `persona_vectors_products_*.tar.gz` |

Plus **surveys** (IPIP/HEXACO via distributional logprobs) and **RepE** (contrast-pair extraction).

All adjective channels use the **523-adjective set** = Saucier 525-PDA minus 2
corrupted columns, in the same order everywhere (the `adjectives` array inside
each file is authoritative).

## Model name map (short → display)

`llama3.2`→Llama3.2-3B · `Llama8`→Llama3.1-8B · `qwen2.5`→Qwen2.5-3B ·
`Qwen7`→Qwen2.5-7B · `Qwen32`→Qwen2.5-32B · `gemma3`→Gemma3-4B ·
`Gemma12`→Gemma3-12B · `Gemma27`→Gemma3-27B · `Gemma4`→Gemma-4-31B ·
`phi4`→Phi4-3.8B · `Aya`→Aya-expanse-8B · `FalconMamba`→FalconMamba-7B.

---

## `judge_likert_*.tar.gz` (23 MB) — JUDGE
Has its own README inside. 12 models' `introspect_full/<Model>_tom_likely_dir.npz`:
`B` (523,523) float32 = row-conditioned expected Likert 1–7 ("given a person is
adjective i, how strongly also j"; NOT symmetric), `Hent` (523,523) entropy,
`adjectives` (523,).

## `persona_vectors_products_*.tar.gz` (134 MB) — ENACT + four-grid
- **`enact_mid/<model>.npz`** — packaged ENACT directions, one file per model
  (the going-forward format): `dir` (523, hidden) = mid-layer persona direction
  per adjective; `axis` (hidden,) = assistant axis; `grand` (hidden,) = grand
  mean; `adjectives` (523,). All **10 cohort models** (llama3.2, Llama8,
  qwen2.5, Qwen7, Qwen32, gemma3, Gemma12, Gemma27, phi4, Aya).
- **`enact_vectors_mid.npz`** — the same data as one legacy combined file
  (keys `dir__<m>` / `axis__<m>` / `grand__<m>`), frozen at the 2026-07-04
  10-model snapshot; prefer the per-model files.
- **`<model>_pda_report.json`** (10 models) — extraction report: `mid_layer`,
  `n_layers`, `massive_dims`, per-adjective `adjectives{dir_norm, boot_cos_mean/
  p05, norm_corr(_ablated), len_corr, top10_dim_share, leak_rate, placebo}`,
  `pairwise_cos_mid` (every adjective-pair cosine at the mid layer, raw +
  massive-dim-ablated), `assistant_axis{norm, cos_to_adjectives(_ablated)}`.
- **`four_grid_<model>.json`** — HUMAN/REPRESENT/JUDGE/ENACT cross-channel
  correlations (Spearman & Pearson × raw / pc1-removed / double-centered) +
  `enact_variants` (raw_nonablated, leak_filtered).
- **`<model>_pda_texts.json`** — the actual persona-rollout generations per adjective.
- `figs/` — html + png.
```python
z = np.load("enact_vectors_mid.npz", allow_pickle=True)
D, adj = z["dir__llama3.2"], z["adjectives"]      # (523, 3072), (523,)
```

## `judge_dists_full_batch1_*.tar.gz` (~40 MB) — JUDGE raw distributions, FULL 523-set
Nine models so far (llama3.2, qwen2.5, gemma3, phi4, Llama8, Qwen7, Aya,
Gemma12, Gemma27; Qwen32/FalconMamba/Gemma4 in flight). `dists` (523,523,7)
float16 + recomputed `ev`/`entropy`. **Supersedes the EV-only introspect_full
B/Hent** (regenerated 2026-07; README_full.md inside has the numerics-drift
and bimodality caveats — notably phi4's matrix is 56% bimodal, so its EV-only
B was substantially an average of disagreeing modes).

## `judge_dists_medoids_*.tar.gz` (0.5 MB) — JUDGE raw distributions (sample)
Full 7-digit Likert distributions behind the JUDGE matrix, for the 35
trait-cluster exemplars x 12 models (`judge_dists/<model>_tom_likely_dists.npz`:
`dists` (35,35,7), `ev`, `entropy`, `adjectives`, `prompt`). README inside.
The full 523-set stores only EV+entropy; this sample carries the raw
distributions (EV can mask bimodality). Other subsets are cheap re-runs
(`scripts/judge_distributions.py`) — ask.

## `adjectives_*.tar` (17 GB) — REPRESENT (read side) + JUDGE source
- **`acts/<repo>__pers.pt`** — REPRESENT residual-stream activations. dict:
  `acts` (523, n_layers+1, hidden) **float16** = per-adjective activation at every
  layer for the last token of the prompt `"My personality is {adj}"`; `adjectives`
  (523,); `framing`='pers'; `template`; `model`. 13 instruct models (`__pers.pt`)
  + 18 base/SFT/Instruct/DPO variants (`__pers_bare.pt`, incl the OLMo-2 ladder).
- **`introspect/`, `introspect_full/`** — the JUDGE data (same as the judge_likert
  tarball; see its README for the `B`/`Hent` format).
- other `*.npz/*.json/*.html/*.png` — derived geometry analyses (facet congruence,
  human-match, PC grids, over-extraction diagnostics).
```python
d = torch.load("acts/CohereLabs_aya-expanse-8b__pers.pt", weights_only=False)
X = d["acts"][:, MID, :].astype("float32")        # (523, hidden) at layer MID
# REPRESENT cosine geometry: normalize rows of X, then X @ X.T
```

## `repe_*.tar` (2.3 GB) — RepE contrast-pair extraction
- **`<repo>_<trait>_directions.pt`** per model × HEXACO trait (H/E/X/A/C/O). dict:
  `trait`, `trait_name`, `model`, `n_pairs` (50), `raw_diffs` (50, n_layers+1,
  hidden) = the (high-trait − low-trait) residual-stream difference for each of 50
  scenario contrast pairs, at every layer.
  **These are raw per-pair diffs** — derive a trait direction by mean-diff / LDA
  over the 50 pairs. Do NOT use PCA PC1 (it's a residual-norm artifact in pre-norm
  transformers; see report_week2 / week6). Pairs: `instruments/contrast_pairs.json`.

## `persona_*.tar.gz` (44 MB) — persona-conditioned experiments (W7/W11/W12)
Persona-internalization + fine-grained-framing (FG) experiments.
`persona_repr_mapping_*.json` = representation under sampled-z personas across
conditions (honest / fg / fgpfx × raw / reflowed × response-position);
`cohort_self_rating_*.json` = `{prompt_template, items, ratings}`. Many condition
combinations — see rgb_reports/report_week7, report_week13 §5, and the producing
scripts (`persona_instrument_response.py`, `analyze_rep_cohort_fg.py`,
`persona_w11_gfc_comparison.py`) for what each condition means.

## `surveys_misc_*.tar.gz` (28 MB) — psychometric surveys + misc
- **`surveys/<model>_<instrument>.json`** — distributional-logprob surveys
  (`ipip300`, `ipip120_english`, `ipip120_mandarin`, `hexaco100`). Schema:
  `{model, hf_repo, instrument, timestamp, n_items, n_variants, scale_scores,
  item_results` (per-item Likert log-prob distributions)`, variant_evs` (per-phrasing
  expected values, for the 4-phrasing ICC reliability)`}`. Producers:
  `run_ipip300.py`, `run_hexaco.py`.
- **`facets/`** — facet-geometry analyses (`embedding_facet_baseline.json` + figs).
- **`item_level/`** — `<model>_item_cos.csv` per-item cosine data.
- **`binary_choice_6trait.json`** — single-trait binary-choice results, keyed
  `<trait>_<ollama-model>` (e.g. `H_gemma3:4b`).

---

## Not in this release
- **Raw per-rollout ENACT vectors** (`*_pda.pt`, ~65 GB — the full activations
  behind `enact_vectors_mid.npz`): on request (the release script's `INCLUDE_RAW=1`).
- Regenerable extraction checkpoints.
- **HUMAN 525-PDA** — public: Harvard Dataverse doi:10.7910/DVN/GHYMEV (700
  respondents × 525 adjectives, 1–7 self-ratings; drop 2 corrupted cols → 523).

Integrity: `MANIFEST_<date>.sha256`. Regenerated by `scripts/make_data_release.sh`.
Questions → rgb.
