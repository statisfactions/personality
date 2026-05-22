#!/bin/bash
# Fill the 270-cell cube: 10 models x 3 persona forms x 3 conditions x 3 methods.
# TIRT is already complete from the W12 scaleup (Stages A/C/E); this script
# fills Likert (persona_instrument_response.py) and Repr (persona_repr_mapping.py
# Saturday-stem) for all 180 method-cells.
#
# Saturday-stem chosen for Repr per the option-1 decision in conversation
# 2026-05-20; instrument-item Repr is a separate matched-stimulus cache built
# later if the inverse-FG steering work needs it (see memory).
#
# Skip-if-exists per cell; safe to interrupt and resume.

set -e
cd "$(dirname "$0")/.."

MODELS=(Gemma Gemma12 Llama Llama8 Phi4 Qwen Qwen7 Gemma27 Qwen32 Gemma4)
# Internal persona-source values match persona_instrument_response.py /
# persona_repr_mapping.py: markers, ipip_raw, ipip_reflowed.
FORMS=(markers ipip_raw ipip_reflowed)
# Three conditions: HONEST, FG-suffix, FG-prefix.
CONDITIONS=(honest fg_suffix fg_prefix)

N=50
STEM='Briefly describe a typical Saturday for you.'
# Canonical FG instruction (must match run_gfc_hf.py / persona_repr_mapping.py
# / persona_instrument_response.py FAKE_GOOD_SUFFIX, with the leading "\n\n"
# preserved so the suffix-position result is the same).
FG='

IMPORTANT INSTRUCTION: Imagine you are completing this questionnaire as part of an application for a position you very much want. Your responses will be evaluated and used to decide whether you are selected. Respond in a way that makes the most favorable possible impression.'

LOG_FILE="psychometrics/gfc_tirt/fill_270_cube.log"
: > "$LOG_FILE"

# Map (form, condition) -> per-method output suffix.
# Likert (persona_instrument_response.py):
#   markers + honest          ->  ""
#   markers + fg_suffix       ->  "_fake_good"
#   markers + fg_prefix       ->  "_fake_good_fgpfx"
#   ipip_raw + honest         ->  "_ipip_raw"
#   ipip_raw + fg_suffix      ->  "_ipip_raw_fake_good"
#   ipip_raw + fg_prefix      ->  "_ipip_raw_fake_good_fgpfx"
#   ipip_reflowed + ...       ->  "_ipip_reflowed[...]"
# Repr (persona_repr_mapping.py):
#   markers + honest          ->  "_response-position_honest_saturday"
#   markers + fg_suffix       ->  "_response-position_fg_saturday"
#   markers + fg_prefix       ->  "_response-position_fgpfx_saturday"
#   ipip_raw + honest         ->  "_response-position_ipip_raw_honest_saturday"
#   ipip_raw + fg_suffix      ->  "_response-position_ipip_raw_fg_saturday"
#   ipip_raw + fg_prefix      ->  "_response-position_ipip_raw_fgpfx_saturday"
#   ipip_reflowed + ...       ->  "_response-position_ipip_reflowed_..."

run_likert_cell() {
  local m="$1" form="$2" cond="$3"
  local form_arg cond_arg pos_arg
  if [[ "$form" == "markers" ]]; then form_arg="markers"; form_suffix=""; else form_arg="$form"; form_suffix="_$form"; fi
  case "$cond" in
    honest)    cond_arg="honest";    pos_arg="suffix"; cond_suffix="" ;;
    fg_suffix) cond_arg="fake_good"; pos_arg="suffix"; cond_suffix="_fake_good" ;;
    fg_prefix) cond_arg="fake_good"; pos_arg="prefix"; cond_suffix="_fake_good_fgpfx" ;;
  esac
  local OUT="results/persona/persona_instrument_response_${m}${form_suffix}${cond_suffix}.json"
  if [[ -f "$OUT" ]]; then
    echo "  SKIP likert $m / $form / $cond -> $OUT" | tee -a "$LOG_FILE"
    return
  fi
  echo "  RUN  likert $m / $form / $cond -> $OUT" | tee -a "$LOG_FILE"
  PYTHONPATH=scripts .venv/bin/python scripts/persona_instrument_response.py \
    --model "$m" --n "$N" \
    --persona-source "$form_arg" --rating-target markers \
    --condition "$cond_arg" --fg-position "$pos_arg" \
    >> "$LOG_FILE" 2>&1
}

run_repr_cell() {
  local m="$1" form="$2" cond="$3"
  local form_arg form_tag fg_arg pos_arg cond_tag
  if [[ "$form" == "markers" ]]; then form_arg="markers"; form_tag=""; else form_arg="$form"; form_tag="${form}_"; fi
  case "$cond" in
    honest)    fg_arg="";     pos_arg="suffix"; cond_tag="honest_saturday" ;;
    fg_suffix) fg_arg="$FG";  pos_arg="suffix"; cond_tag="fg_saturday" ;;
    fg_prefix) fg_arg="$FG";  pos_arg="prefix"; cond_tag="fgpfx_saturday" ;;
  esac
  # persona_repr_mapping.py builds: _response-position[_<form>]_<output_tag>
  # We always want the form first when non-markers; pass form via persona-source
  # and use output-tag="<cond_tag>" so the cell-key in the filename is
  # ipip_raw_honest_saturday / fg_saturday / fgpfx_saturday for non-markers and
  # honest_saturday / fg_saturday / fgpfx_saturday for markers.
  local OUT="results/persona/persona_repr_mapping_${m}_response-position_${form_tag}${cond_tag}.json"
  if [[ -f "$OUT" ]]; then
    echo "  SKIP repr   $m / $form / $cond -> $OUT" | tee -a "$LOG_FILE"
    return
  fi
  echo "  RUN  repr   $m / $form / $cond -> $OUT" | tee -a "$LOG_FILE"
  PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py \
    --model "$m" --n "$N" --mode response-position \
    --persona-source "$form_arg" --direction-source markers \
    --user-stem "$STEM" --fg-suffix "$fg_arg" --fg-position "$pos_arg" \
    --output-tag "$cond_tag" \
    >> "$LOG_FILE" 2>&1
}

total=$((${#MODELS[@]} * ${#FORMS[@]} * ${#CONDITIONS[@]} * 2))
echo "=== fill 270-cube start at $(date) ===" | tee -a "$LOG_FILE"
echo "  target: $total cells (10 models x 3 forms x 3 conds x 2 methods)" | tee -a "$LOG_FILE"

# Iterate (model, form, condition); for each cell run BOTH methods so a model
# is loaded twice per cell pair. We could batch better by holding a model
# loaded across all its cells, but separate subprocesses give us clean MPS
# state per call (already proven robust by Stage F's per-cell subprocess
# pattern; Stage G's single-process pattern caused the OS crash).
i=0
for m in "${MODELS[@]}"; do
  for form in "${FORMS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
      i=$((i + 1))
      echo "[$i/$((total / 2))] $m / $form / $cond" | tee -a "$LOG_FILE"
      run_likert_cell "$m" "$form" "$cond"
      run_repr_cell   "$m" "$form" "$cond"
    done
  done
done

echo "=== fill 270-cube done at $(date) ===" | tee -a "$LOG_FILE"
