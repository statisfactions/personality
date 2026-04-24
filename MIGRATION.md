# Machine migration checklist

Notes for moving development to a new box with larger models. Generated on the move, not a formal doc.

## In git (already on origin/main — just `git clone`)

All code, instruments, reports, and the stratified contrast-pair JSON.

## NOT in git (carry over manually or regenerate)

### Sizable caches worth transferring (save ~1 hr of extraction)

- `results/phase_b_cache/` — **3.1 GB** — small-model pair + neutral activations (4B and under). Keep for cross-model-size comparisons vs new big-model extractions.
- `results/phase_b_cache_stratified/` — **2.2 GB** — stratified pair activations on small models. Same rationale.
- `results/repe/` — 515 MB — older LDA-era extraction files. Most downstream code has migrated away; low priority.

Rsync command:
```
rsync -avz --progress \
    results/phase_b_cache \
    results/phase_b_cache_stratified \
    results/repe \
    NEW_BOX:~/src/personality/results/
```

### Small results worth transferring (optional, small)

- `results/*.json`, `*.csv`, `*.txt`, `*.html` — summary stats, logs, and visualizations. Under 10 MB total. Quality-of-life so the existing `.log` / `.json` outputs are viewable without re-running analyses. Regeneratable.

### Do NOT transfer

- `.venv/` — 947 MB. Recreate on the new box (likely Linux): `python -m venv .venv && .venv/bin/pip install -r requirements.txt`. Mac `.venv` won't work on Linux anyway.
- `~/.cache/huggingface/` — model weights. Cheaper to re-download on the new box's faster disk/network; new box will also be pulling bigger models anyway.

## Environment setup on the new box

1. `git clone git@github.com:statisfactions/personality.git && cd personality`
2. `python -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Transfer API keys: `ANTHROPIC_API_KEY` (or `CLAUDE_DEFAULT`), `HF_TOKEN` for gated models (Gemma, Llama).
4. Verify HuggingFace access: `.venv/bin/huggingface-cli whoami`
5. Rsync caches (see above) if keeping small-model comparison data.

## Device notes

Current Mac-specific assumptions scattered in the scripts (`device="mps"`, `dtype="bfloat16"`):

- `scripts/phase_b_sweep.py`
- `scripts/extract_stratified.py`
- `scripts/extract_meandiff_vectors.py`
- `scripts/compare_probe_steering.py`
- `scripts/optimize_steering.py`
- `scripts/validate_protocol.py`

If the new box has CUDA, these need `--device cuda`. Consider a CLI default or env var if switching back and forth a lot.

## Models already configured

See `scripts/extract_stratified.py:29-34` and `CLAUDE.md` for the 4-model roster (Llama 3.2-3B, Gemma 3-4B, Phi4-mini, Qwen 2.5-3B). On the bigger box, consider adding:

- Gemma 3-12B, 27B (needs HF approval; already have HF_TOKEN)
- Llama 3.1-8B, 3.3-70B
- Qwen 2.5-14B, 32B
- Phi 4 (full 14B)

## Statisfactions' track (last 2 commits on main)

- `instruments/okada_gfc30.json` — 30 desirability-matched GFC Big Five pairs from Okada et al. (2026)
- `instruments/synthetic_personas.json` — 400 MVN-sampled synthetic personas (Goldberg markers × stanine intensity)
- `scripts/generate_trait_personas.py` — persona generator
- `scripts/run_gfc_ollama.py` — GFC inference for Ollama, 7-point bipolar responses + logprobs, `--neutral` flag for default-placement measurement

Worth a pass before reimplementing any of it on the new box.
