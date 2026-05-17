#!/bin/bash
# Run the 63 W12 ablation TIRT fits (7 models x 3 forms x 3 subsets).
# Sequential — each fit uses 4 cores internally.

set -e
cd "$(dirname "$0")/.."

SRC_DIR="psychometrics/gfc_tirt/ablation_subsets"
OUT_RDS_DIR="psychometrics/gfc_tirt/ablation_subsets"  # RDS next to JSON
OUT_JSON_DIR="results/persona/ablation"
mkdir -p "$OUT_JSON_DIR"

LOG_FILE="psychometrics/gfc_tirt/ablation_run.log"
: > "$LOG_FILE"

i=0
total=$(ls "$SRC_DIR"/*.json | wc -l | tr -d ' ')
for resp in "$SRC_DIR"/*.json; do
  i=$((i + 1))
  base=$(basename "$resp" .json)        # e.g., Gemma12_ipipneogfc60_hf_description_top30
  rds_out="$OUT_RDS_DIR/${base}_indep_fit.rds"
  rec_out="$OUT_JSON_DIR/persona_gfc_tirt_${base}.json"

  if [ -f "$rds_out" ] && [ -f "$rec_out" ]; then
    echo "[$i/$total] SKIP $base (already done)" | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$i/$total] FIT  $base" | tee -a "$LOG_FILE"
  Rscript psychometrics/gfc_tirt/fit_tirt_okada_indep.R \
    "$resp" "$rds_out" 50 700 4 "$rec_out" \
    >> "$LOG_FILE" 2>&1
done

echo "Done." | tee -a "$LOG_FILE"
