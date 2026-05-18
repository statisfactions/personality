#!/bin/bash
# W12 §5d round 2: rotation test + prompt-ordering test on Llama8.
#
# Round 1 (run_rep_under_fg_smoke.sh) established:
#   honest+Saturday  → diag r = +0.774
#   FG-suffix+Saturday → +0.612 (Δ -0.16)
#   honest+best-desc → +0.701
#   FG-suffix+best-desc → +0.300 (Δ -0.40, with A/N sign-flips)
#
# Round 2 conditions (all on Llama8 × 50 marker personas):
#   E: FG-suffix  + Saturday  + FG-directions  (rotation test)
#   F: FG-suffix  + best-desc + FG-directions  (rotation test)
#   G: FG-prefix  + Saturday  + honest-dirs    (ordering test)
#   H: FG-prefix  + best-desc + honest-dirs    (ordering test)
#   I: FG-prefix  + Saturday  + FG-directions  (combined)
#   J: FG-prefix  + best-desc + FG-directions  (combined)

set -e
cd "$(dirname "$0")/.."

MODEL=Llama8
N=50

FG='

IMPORTANT INSTRUCTION: Imagine you are completing this questionnaire as part of an application for a position you very much want. Your responses will be evaluated and used to decide whether you are selected. Respond in a way that makes the most favorable possible impression.'

STEM_SATURDAY='Briefly describe a typical Saturday for you.'
STEM_BEST='What best describes you:'

for cond in E F G H I J; do
  case "$cond" in
    E) stem="$STEM_SATURDAY"; fg="$FG"; pos="suffix"; dirsys="--fg-direction-system"; tag="fg_saturday_fgdirs" ;;
    F) stem="$STEM_BEST";     fg="$FG"; pos="suffix"; dirsys="--fg-direction-system"; tag="fg_bestdesc_fgdirs" ;;
    G) stem="$STEM_SATURDAY"; fg="$FG"; pos="prefix"; dirsys=""; tag="fgpfx_saturday" ;;
    H) stem="$STEM_BEST";     fg="$FG"; pos="prefix"; dirsys=""; tag="fgpfx_bestdesc" ;;
    I) stem="$STEM_SATURDAY"; fg="$FG"; pos="prefix"; dirsys="--fg-direction-system"; tag="fgpfx_saturday_fgdirs" ;;
    J) stem="$STEM_BEST";     fg="$FG"; pos="prefix"; dirsys="--fg-direction-system"; tag="fgpfx_bestdesc_fgdirs" ;;
  esac
  echo
  echo "=========================================================="
  echo " Cond $cond: $tag"
  echo "=========================================================="
  PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py \
    --model "$MODEL" --n "$N" --mode response-position \
    --persona-source markers --direction-source markers \
    --user-stem "$stem" --fg-suffix "$fg" --fg-position "$pos" \
    $dirsys --output-tag "$tag" 2>&1 | tail -8
done
