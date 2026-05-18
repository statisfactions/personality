#!/bin/bash
# W12 §5d smoke: Rep-under-FG on Llama8. Test whether persona-Rep
# recovery (W7 §11.5.9 measurement) survives an FG instruction
# appended to the persona system message.
#
# Conditions (using IPIP trait directions to match Llama8 W7 baseline):
#   A: HONEST + Saturday stem    (replicates W7 baseline ≈ 0.694)
#   B: FG     + Saturday stem    (the core test)
#   C: HONEST + best-describes stem
#   D: FG     + best-describes stem

set -e
cd "$(dirname "$0")/.."

MODEL=Llama8
N=50

FG='

IMPORTANT INSTRUCTION: Imagine you are completing this questionnaire as part of an application for a position you very much want. Your responses will be evaluated and used to decide whether you are selected. Respond in a way that makes the most favorable possible impression.'

STEM_SATURDAY='Briefly describe a typical Saturday for you.'
STEM_BEST='What best describes you:'

# Use marker-source personas + IPIP direction source (matches existing
# Llama8 baseline file persona_repr_mapping_Llama8_response-position_ipip_raw.json,
# though that was ipip_raw personas; here we use markers for clean W7 compare).

for cond in A B C D; do
  case "$cond" in
    A) stem="$STEM_SATURDAY"; fg=""; tag="honest_saturday" ;;
    B) stem="$STEM_SATURDAY"; fg="$FG"; tag="fg_saturday" ;;
    C) stem="$STEM_BEST";     fg=""; tag="honest_bestdesc" ;;
    D) stem="$STEM_BEST";     fg="$FG"; tag="fg_bestdesc" ;;
  esac
  echo
  echo "=========================================================="
  echo " Cond $cond: $tag"
  echo "=========================================================="
  PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py \
    --model "$MODEL" --n "$N" --mode response-position \
    --persona-source markers --direction-source markers \
    --user-stem "$stem" --fg-suffix "$fg" \
    --output-tag "$tag" 2>&1 | tail -20
done
