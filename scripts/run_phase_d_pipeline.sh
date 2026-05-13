#!/usr/bin/env bash
# W11 Phase D pipeline:
#   1. Run scripts/run_gfc_hf.py against the IPIP-NEO-GFC-60 instrument
#      across 7 cohort models × 3 persona forms (description / ipip_raw /
#      ipip_reflowed). Naming: ipipneogfc60 to distinguish from gfc30.
#   2. Run psychometrics/gfc_tirt/fit_tirt_okada_indep.R on each result
#      to get per-persona theta recovery.
#
# Estimated runtime: ~3-4 hours for 21 inferences (P=60 → ~2x GFC-30
# inference per call), ~10-15 min for 21 TIRT fits.

set -uo pipefail
cd /Users/rgb/src/personality

INSTRUMENT="instruments/ipip_neo_gfc_P60.json"
INSTRUMENT_TAG="ipipneogfc60"

MODELS=(Gemma Llama Phi4 Qwen Gemma12 Llama8 Qwen7)
FORMS=(description ipip_raw ipip_reflowed)
N_PERSONAS=50

mkdir -p psychometrics/gfc_tirt results/persona

# ---------------------------------------------------------------------------
# Stage 1: HF inference (21 runs)
# ---------------------------------------------------------------------------
echo "=== Stage 1: HF inference (P=60, 7 models × 3 forms) ==="
N_DONE=0; N_FAILED=0
for M in "${MODELS[@]}"; do
  for FIELD in "${FORMS[@]}"; do
    if [[ "$FIELD" == "description" ]]; then
      PERS=instruments/synthetic_personas.json
    else
      PERS=instruments/synthetic_personas_ipip.json
    fi
    OUT="psychometrics/gfc_tirt/${M}_${INSTRUMENT_TAG}_hf_${FIELD}.json"
    if [[ -f "$OUT" ]]; then
      N_DONE=$((N_DONE + 1))
      echo "--- $(date '+%H:%M:%S')  ${M} × ${FIELD}: output exists, skipping ---"
      continue
    fi
    echo
    echo "=== $(date '+%H:%M:%S')  ${M} × ${FIELD} → ${OUT} ==="
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
      --model "$M" \
      --max-personas "$N_PERSONAS" \
      --synthetic-personas "$PERS" \
      --persona-field "$FIELD" \
      --instrument "$INSTRUMENT" \
      --output "$OUT" \
      && { N_DONE=$((N_DONE + 1)); echo "  OK (${N_DONE} done)"; } \
      || { N_FAILED=$((N_FAILED + 1)); echo "  FAILED (${N_FAILED} fails)"; }
  done
done
echo
echo "=== Stage 1 done: ${N_DONE} ok, ${N_FAILED} failed ==="

# ---------------------------------------------------------------------------
# Stage 2: TIRT fits (21 fits)
# ---------------------------------------------------------------------------
echo
echo "=== Stage 2: Okada Appendix-D TIRT fits ==="
N_DONE=0; N_FAILED=0
for M in "${MODELS[@]}"; do
  for FIELD in "${FORMS[@]}"; do
    IN="psychometrics/gfc_tirt/${M}_${INSTRUMENT_TAG}_hf_${FIELD}.json"
    RDS="psychometrics/gfc_tirt/${M}_${INSTRUMENT_TAG}_hf_${FIELD}_indep_fit.rds"
    if [[ ! -f "$IN" ]]; then
      echo "--- ${M} × ${FIELD}: inference output missing, skipping ---"
      N_FAILED=$((N_FAILED + 1))
      continue
    fi
    if [[ -f "$RDS" ]]; then
      N_DONE=$((N_DONE + 1))
      echo "--- ${M} × ${FIELD}: fit exists, skipping ---"
      continue
    fi
    echo
    echo "=== $(date '+%H:%M:%S')  TIRT fit ${M} × ${FIELD} → ${RDS} ==="
    Rscript psychometrics/gfc_tirt/fit_tirt_okada_indep.R "$IN" "$RDS" "$N_PERSONAS" >/dev/null 2>&1 \
      && { N_DONE=$((N_DONE + 1)); echo "  OK (${N_DONE} done)"; } \
      || { N_FAILED=$((N_FAILED + 1)); echo "  FAILED (${N_FAILED} fails)"; }
  done
done
echo
echo "=== Stage 2 done at $(date): ${N_DONE} ok, ${N_FAILED} failed ==="
echo "=== Phase D pipeline complete ==="
