#!/bin/bash
# W12 cohort scale-up: run all baseline experiments on the 3 new large
# models (Gemma27, Qwen32, Gemma4). Plus W9 facet activation extraction.
#
# Existing 7-model cohort runs are skip-protected by each downstream
# wrapper's skip-if-exists logic, so re-running these wrappers should
# only do work on the new models.
#
# Stages (serial):
#   A. W11 phase-D HONEST inference + TIRT fits (run_phase_d_pipeline.sh)
#   B. W12 §5b FG-suffix inference (run_fake_good_inference.sh)
#   C. W12 §5b TIRT fits (run_fake_good_tirt_fits.sh)
#   D. W12 §5e FG-prefix inference (run_fake_good_prefix_inference.sh)
#   E. W12 §5e TIRT fits (run_fake_good_prefix_tirt_fits.sh)
#   F. W12 §5d.5 Rep cohort (run_rep_under_fg_cohort.sh)
#   G. W12 §5c.1 self-rating (run_no_persona_self_rating.py)
#   H. W9 facet activation extraction + per-model analysis
#
# Estimated runtime: ~21-25 hr inference + ~30 min TIRT + ~30 min facets.

set -e
cd "$(dirname "$0")/.."

LOG="psychometrics/gfc_tirt/scaleup_cohort.log"
: > "$LOG"

stage() {
  echo
  echo "==========================================================="
  echo " STAGE $1: $2 ($(date))"
  echo "==========================================================="
  echo "[$1] $2 starting at $(date)" >> "$LOG"
}

NEW_MODELS=(Gemma27 Qwen32 Gemma4)

stage A "W11 HONEST P=60 inference + TIRT fits"
bash scripts/run_phase_d_pipeline.sh >> "$LOG" 2>&1

stage B "W12 §5b FG-suffix P=60 inference"
bash scripts/run_fake_good_inference.sh >> "$LOG" 2>&1

stage C "W12 §5b FG-suffix TIRT fits"
bash scripts/run_fake_good_tirt_fits.sh >> "$LOG" 2>&1

stage D "W12 §5e FG-prefix P=60 inference"
bash scripts/run_fake_good_prefix_inference.sh >> "$LOG" 2>&1

stage E "W12 §5e FG-prefix TIRT fits"
bash scripts/run_fake_good_prefix_tirt_fits.sh >> "$LOG" 2>&1

stage F "W12 §5d.5 Rep cohort (Saturday stem, 4 conditions)"
bash scripts/run_rep_under_fg_cohort.sh >> "$LOG" 2>&1

stage G "W12 §5c.1 no-persona self-rating"
PYTHONPATH=scripts .venv/bin/python scripts/run_no_persona_self_rating.py \
  >> "$LOG" 2>&1

stage H "W9 facet activation extraction (ipip_facet_cluster.py)"
PYTHONPATH=scripts .venv/bin/python scripts/ipip_facet_cluster.py \
  --models "${NEW_MODELS[@]}" --extraction meandiff-pcs \
  >> "$LOG" 2>&1

echo "=== scaleup complete at $(date) ===" | tee -a "$LOG"
