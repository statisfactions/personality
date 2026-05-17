#!/bin/bash
# W12 fake-good inference batch: 7 models x 3 persona forms x P=60 instrument.
# Outputs land at psychometrics/gfc_tirt/<MODEL>_ipipneogfc60_hf_<FORM>_fake_good.json
# (mirroring the honest filenames with _fake_good suffix).
#
# Total: 7 * 3 * 50 personas * 60 pairs = 63,000 prompts.
# At ~6 prompts/s on M5 Max (Gemma 4B reference), expect ~3 hr serial.

set -e
cd "$(dirname "$0")/.."

MODELS=(Gemma Gemma12 Llama Llama8 Phi4 Qwen Qwen7)
FORMS=(description ipip_raw ipip_reflowed)
INSTRUMENT="instruments/ipip_neo_gfc_P60.json"

# Mirror W11 phase-D persona-file selection: description lives in
# synthetic_personas.json (Goldberg markers); ipip_raw and ipip_reflowed
# live in synthetic_personas_ipip.json.
LOG_FILE="psychometrics/gfc_tirt/fake_good_inference.log"
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
    # Force the ipipneogfc60_hf filename pattern (matching W11 phase-D
    # convention); the script's default hardcodes "gfc30_hf" regardless
    # of which instrument is loaded.
    OUT="psychometrics/gfc_tirt/${m}_ipipneogfc60_hf_${f}_fake_good.json"
    echo "[$i/$total] $m / $f / fake_good (personas=$PERSONAS → $OUT)" | tee -a "$LOG_FILE"
    PYTHONPATH=scripts .venv/bin/python scripts/run_gfc_hf.py \
      --model "$m" \
      --persona-field "$f" \
      --synthetic-personas "$PERSONAS" \
      --instrument "$INSTRUMENT" \
      --condition fake_good \
      --max-personas 50 \
      --output "$OUT" \
      >> "$LOG_FILE" 2>&1
  done
done

echo "Done." | tee -a "$LOG_FILE"
