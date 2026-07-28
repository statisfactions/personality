# Handoff: running this stack anywhere (statisfactions edition)

The collaboration model is **push and handoff**: rgb pushes this repo; the
big binaries live in tarballs on the shared Drive
(`~/gdrive/projects/llm-personality/`, see its README.md = a copy of
`rgb_reports/data_release_readme.md`). Everything analysis-sized is
committed in-repo. Nothing requires rgb's laptop.

## Orientation (read in this order)

1. `rgb_reports/synthesis.md` — current truth; glossary up front; claims
   ledger at the bottom (LIVE/REV/DEAD/OPEN).
2. `rgb_reports/overview.md` — week-by-week index of the lab notebook.
3. `CLAUDE.md` — repo map, scripts table, model registry.

## Bootstrap on foreign hardware

```bash
git clone <repo> && cd personality
python -m venv .venv && .venv/bin/pip install torch transformers \
    accelerate scikit-learn scipy plotly sentence-transformers \
    huggingface_hub safetensors
bash scripts/fetch_dependencies.sh        # third-party clones -> dependencies/
export HF_TOKEN=...                        # gated models: Gemma, Llama
```

Model shortnames resolve via `scripts/hf_logprobs.py` (MODELS dict); any
script runs with `PYTHONPATH=scripts python scripts/<name>.py`. CUDA works
everywhere we use MPS (device auto-detected). Long jobs are
row/part-checkpointed and resume on rerun — kill freely.

## Analysis-only quests: inputs already in-repo (no GPU needed)

| quest | inputs (all committed) |
|---|---|
| model × readout variance decomposition | `results/steer_map/facet_channel_sims.npz` (cohort-mean 523×523 per channel); `results/adjectives/selfreport/*_self_full.json` (SELF, 6 framings × cohort); `results/adjectives/enactability/*.json`; four-grid JSONs in the persona_vectors tarball |
| referent-swap (observer vs direct framing) | `results/adjectives/selfreport/*_self_full.json` — framings `direct`, `observer`, etc. already run on the cohort |
| VP / generation-preference re-analysis | `results/vp_rescore/*.json` (+ `_bare`, `_base` variants), `labels.npy`; scoring in `scripts/vp_rescore.py` |
| TIDE congruence re-analysis | `results/tide_enact/Llama8_matrix.npz` (+ ablation npz); their matrices via `scripts/tide_preoblimin.py` (auto-downloads from persona-cartography/monorepo) |
| JUDGE raw distributions | Drive tarballs `judge_dists_full_batch{1,2}` (12 models, (523,523,7) each) |

## GPU quests: what inference costs

All 3–8B jobs run on a single 24GB card (or Apple silicon). Reference
timings (M5 Max): VP scoring 520 responses ≈ 12 min/model; SELF framing
sweep ≈ 1–2 h/model; v7 questionnaire on 1,056 rollouts ≈ 2 h (KV-cached);
full 523×523 JUDGE ≈ 2–4 days/small model. Scripts print resume state on
start.

## Conventions that keep the data comparable

- 523 adjective set, always in the order inside each file's `adjectives`
  array (authoritative per-file).
- EV + entropy from full logprob distributions; never argmax-only.
- Massive-dim winsorize before any activation geometry (dims per model in
  `results/persona_vectors/<m>_pda_meta.json`).
- Report human-match raw AND PC1-removed; report split-half reliability
  for any new instrument (see the recipe, synthesis Part III).
- Register predictions in writing before running; grade them after,
  misses included (the ledger depends on this).

Questions → rgb. Corrections to frozen paper-1 claims → errata section,
not silent edits.
