#!/usr/bin/env bash
# End-to-end GFC+TIRT pipeline for the W8 trajectory plot's third readout
# line. Runs 7 cohort models × 3 persona forms = 21 HF inferences, fits
# Okada Appendix-D-exact TIRT to each, then regenerates the trajectory plot.
#
# Requirements:
#   - CUDA-capable GPU with enough VRAM for the cohort (the 12B/8B/7B
#     models are the constraint; ~20 GB peak)
#   - bf16-compatible torch, transformers, rstan
#   - HF token in env for gated models (Llama 3.1/3.2 are gated)
#   - Run from repo root (paths are relative)
#
# Outputs:
#   psychometrics/gfc_tirt/<MODEL>_gfc30_hf_<FORM>.json   (raw inference)
#   psychometrics/gfc_tirt/<MODEL>_gfc30_hf_<FORM>_indep_fit.rds
#   results/persona_gfc_tirt_<MODEL>_<FORM>.json          (recovery sidecar)
#   results/persona_w8_trajectory.html                    (replot)
#
# Resume-safe: HF script checkpoints every 200 prompts; R fitter overwrites
# fits unless SKIP_EXISTING_FITS=1 is set. To re-fit just one model/form,
# delete the corresponding _indep_fit.rds and rerun.

set -euo pipefail

MODELS=(Gemma Llama Phi4 Qwen Gemma12 Llama8 Qwen7)
FORMS=(description ipip_raw ipip_reflowed)
N_PERSONAS="${N_PERSONAS:-50}"
SKIP_EXISTING_FITS="${SKIP_EXISTING_FITS:-0}"

mkdir -p psychometrics/gfc_tirt results

# ---------------------------------------------------------------------------
# Stage 1: HF inference (21 runs)
# ---------------------------------------------------------------------------
echo "=== Stage 1: HF GFC-30 inference (7 models × 3 forms) ==="
for M in "${MODELS[@]}"; do
  for FIELD in "${FORMS[@]}"; do
    if [[ "$FIELD" == "description" ]]; then
      PERS=instruments/synthetic_personas.json
    else
      PERS=instruments/synthetic_personas_ipip.json
    fi
    OUT="psychometrics/gfc_tirt/${M}_gfc30_hf_${FIELD}.json"
    echo
    echo "--- ${M} × ${FIELD} → ${OUT} ---"
    PYTHONPATH=scripts python3 scripts/run_gfc_hf.py \
      --model "$M" \
      --max-personas "$N_PERSONAS" \
      --synthetic-personas "$PERS" \
      --persona-field "$FIELD" \
      --output "$OUT"
  done
done

# ---------------------------------------------------------------------------
# Stage 2: TIRT fits (21 fits)
# ---------------------------------------------------------------------------
echo
echo "=== Stage 2: Okada Appendix-D TIRT fits ==="
for M in "${MODELS[@]}"; do
  for FIELD in "${FORMS[@]}"; do
    IN="psychometrics/gfc_tirt/${M}_gfc30_hf_${FIELD}.json"
    RDS="psychometrics/gfc_tirt/${M}_gfc30_hf_${FIELD}_indep_fit.rds"
    if [[ "$SKIP_EXISTING_FITS" == "1" && -f "$RDS" ]]; then
      echo "--- ${M} × ${FIELD}: fit exists, skipping ---"
      continue
    fi
    echo
    echo "--- ${M} × ${FIELD} → ${RDS} ---"
    Rscript psychometrics/gfc_tirt/fit_tirt_okada_indep.R \
      "$IN" "$RDS" "$N_PERSONAS"
  done
done

# ---------------------------------------------------------------------------
# Stage 3: Replot
# ---------------------------------------------------------------------------
echo
echo "=== Stage 3: Regenerate trajectory plot ==="
python3 scripts/persona_w8_summary_plot.py
echo
echo "Done. Open results/persona_w8_trajectory.html"
