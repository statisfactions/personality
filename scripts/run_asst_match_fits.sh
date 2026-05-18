#!/bin/bash
# Run TIRT fits on the 42 assistant-matched / mismatched filtered JSONs.
set -e
cd "$(dirname "$0")/.."

SRC_DIR="psychometrics/gfc_tirt/ablation_assistant_subsets"
OUT_JSON_DIR="results/persona/ablation_assistant"
mkdir -p "$OUT_JSON_DIR"

LOG_FILE="psychometrics/gfc_tirt/asst_match_fits.log"
: > "$LOG_FILE"

i=0
total=$(ls "$SRC_DIR"/*.json | wc -l | tr -d ' ')
for resp in "$SRC_DIR"/*.json; do
  i=$((i + 1))
  base=$(basename "$resp" .json)
  rds_out="$SRC_DIR/${base}_indep_fit.rds"
  rec_out="$OUT_JSON_DIR/persona_gfc_tirt_${base}.json"

  if [ -f "$rds_out" ] && [ -f "$rec_out" ]; then
    echo "[$i/$total] SKIP $base" | tee -a "$LOG_FILE"
    continue
  fi
  echo "[$i/$total] FIT  $base" | tee -a "$LOG_FILE"
  Rscript psychometrics/gfc_tirt/fit_tirt_okada_indep.R \
    "$resp" "$rds_out" 50 700 4 "$rec_out" \
    >> "$LOG_FILE" 2>&1
done

echo "Done." | tee -a "$LOG_FILE"
