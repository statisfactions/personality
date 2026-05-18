#!/bin/bash
# W12 §5d.5 cohort: extend the Llama8 Rep-under-FG smoke to the
# 7-model cohort. Saturday stem only (cheaper than the full stem ×
# FG-position × dirs matrix). 4 conditions per model = 28 runs.

set -e
cd "$(dirname "$0")/.."

MODELS=(Gemma Gemma12 Llama Llama8 Phi4 Qwen Qwen7)
N=50

FG='

IMPORTANT INSTRUCTION: Imagine you are completing this questionnaire as part of an application for a position you very much want. Your responses will be evaluated and used to decide whether you are selected. Respond in a way that makes the most favorable possible impression.'

STEM='Briefly describe a typical Saturday for you.'

LOG_FILE="psychometrics/gfc_tirt/rep_under_fg_cohort.log"
: > "$LOG_FILE"

i=0
total=$((${#MODELS[@]} * 4))
for m in "${MODELS[@]}"; do
  for cond in A B E G; do
    i=$((i + 1))
    case "$cond" in
      A) fg=""; pos="suffix"; dirsys=""; tag="honest_saturday" ;;
      B) fg="$FG"; pos="suffix"; dirsys=""; tag="fg_saturday" ;;
      E) fg="$FG"; pos="suffix"; dirsys="--fg-direction-system"; tag="fg_saturday_fgdirs" ;;
      G) fg="$FG"; pos="prefix"; dirsys=""; tag="fgpfx_saturday" ;;
    esac
    OUT="results/persona/persona_repr_mapping_${m}_response-position_${tag}.json"
    if [ -f "$OUT" ]; then
      echo "[$i/$total] SKIP $m / $tag (exists)" | tee -a "$LOG_FILE"
      continue
    fi
    echo "[$i/$total] $m / $tag" | tee -a "$LOG_FILE"
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py \
      --model "$m" --n "$N" --mode response-position \
      --persona-source markers --direction-source markers \
      --user-stem "$STEM" --fg-suffix "$fg" --fg-position "$pos" \
      $dirsys --output-tag "$tag" \
      >> "$LOG_FILE" 2>&1
  done
done

echo "Done." | tee -a "$LOG_FILE"
