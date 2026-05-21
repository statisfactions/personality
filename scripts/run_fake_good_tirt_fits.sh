#!/bin/bash
# Fit TIRT on each of the 21 fake-good inference outputs.
# Mirror scripts/run_phase_d_pipeline.sh stage 2 structure but with
# the _fake_good filename suffix.

set -e
cd "$(dirname "$0")/.."

MODELS=(Gemma Gemma12 Llama Llama8 Phi4 Qwen Qwen7 Gemma27 Qwen32 Gemma4)
FORMS=(description ipip_raw ipip_reflowed)

LOG_FILE="psychometrics/gfc_tirt/fake_good_tirt_fits.log"
: > "$LOG_FILE"

i=0
total=$((${#MODELS[@]} * ${#FORMS[@]}))
for m in "${MODELS[@]}"; do
  for f in "${FORMS[@]}"; do
    i=$((i + 1))
    RESP="psychometrics/gfc_tirt/${m}_ipipneogfc60_hf_${f}_fake_good.json"
    RDS="psychometrics/gfc_tirt/${m}_ipipneogfc60_hf_${f}_fake_good_indep_fit.rds"
    REC="results/persona/persona_gfc_tirt_${m}_ipipneogfc60_hf_${f}_fake_good.json"
    if [[ ! -f "$RESP" ]]; then
      echo "[$i/$total] SKIP $RESP (missing)" | tee -a "$LOG_FILE"
      continue
    fi
    if [[ -f "$RDS" && -f "$REC" ]]; then
      echo "[$i/$total] DONE $m / $f / fake_good" | tee -a "$LOG_FILE"
      continue
    fi
    echo "[$i/$total] FIT  $m / $f / fake_good" | tee -a "$LOG_FILE"
    Rscript psychometrics/gfc_tirt/fit_tirt_okada_indep.R \
      "$RESP" "$RDS" 50 700 4 "$REC" \
      >> "$LOG_FILE" 2>&1
  done
done

echo "Done." | tee -a "$LOG_FILE"
