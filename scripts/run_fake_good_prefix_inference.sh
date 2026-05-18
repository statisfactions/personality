#!/bin/bash
# W12 §5d cohort: FG-prefix inference (mirror of run_fake_good_inference.sh
# but with FG instruction placed BEFORE persona in the system message).
# Output filenames carry a _fgpfx suffix per scripts/run_gfc_hf.py.

set -e
cd "$(dirname "$0")/.."

MODELS=(Gemma Gemma12 Llama Llama8 Phi4 Qwen Qwen7)
FORMS=(description ipip_raw ipip_reflowed)
INSTRUMENT="instruments/ipip_neo_gfc_P60.json"

LOG_FILE="psychometrics/gfc_tirt/fake_good_prefix_inference.log"
: > "$LOG_FILE"

i=0
total=$((${#MODELS[@]} * ${#FORMS[@]}))
for m in "${MODELS[@]}"; do
  for f in "${FORMS[@]}"; do
    i=$((i + 1))
    if [[ "$f" == "description" ]]; then
      PERSONAS="instruments/synthetic_personas.json"
    else
      PERSONAS="instruments/synthetic_personas_ipip.json"
    fi
    OUT="psychometrics/gfc_tirt/${m}_ipipneogfc60_hf_${f}_fake_good_fgpfx.json"
    if [[ -f "$OUT" ]]; then
      # Allow skip if already done (Llama8 description was the smoke).
      DONE=$(.venv/bin/python -c "import json; d=json.load(open('$OUT')); print(d.get('n_completed',0))" 2>/dev/null || echo 0)
      if [[ "$DONE" == "3000" ]]; then
        echo "[$i/$total] SKIP $m / $f (already done)" | tee -a "$LOG_FILE"
        continue
      fi
    fi
    echo "[$i/$total] $m / $f / fake_good (prefix; personas=$PERSONAS → $OUT)" | tee -a "$LOG_FILE"
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
      --model "$m" \
      --persona-field "$f" \
      --synthetic-personas "$PERSONAS" \
      --instrument "$INSTRUMENT" \
      --condition fake_good \
      --fg-position prefix \
      --max-personas 50 \
      --output "$OUT" \
      >> "$LOG_FILE" 2>&1
  done
done

echo "Done." | tee -a "$LOG_FILE"
