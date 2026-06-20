#!/bin/bash
# Full 525-PDA persona-vector extraction across the cohort, small -> large.
#
# Budget per model: 523 adjectives x 5 sys prompts x 12 questions x 1 rollout
# = ~31.4k generations (+60 default-assistant) — roughly a day per small
# model on MPS, multi-day for 27B+.
#
# Per-rollout acts are saved at every 4th layer (+ mid) so cleaning/analysis
# decisions can be revisited without re-running inference; condition means
# (hence directions) are full-layer regardless. ~3-11GB per model.
#
# Resumable: models whose _pda_report.json already exists are skipped, and a
# model failure doesn't kill the loop. Launch under caffeinate so the Mac
# doesn't sleep mid-run:
#   caffeinate -dims scripts/run_persona_cohort.sh
#   COHORT="llama3.2 gemma3" caffeinate -dims scripts/run_persona_cohort.sh
set -uo pipefail
cd "$(dirname "$0")/.."

COHORT=${COHORT:-"llama3.2 qwen2.5 gemma3 phi4 Llama8 Qwen7 Aya Gemma12 Gemma27 Qwen32"}

for m in $COHORT; do
  report="results/persona_vectors/${m}_pda_report.json"
  if [ -f "$report" ]; then
    echo "=== $m already done ($report exists), skipping ==="
    continue
  fi
  echo "=== $m  $(date) ==="
  if ! PYTHONPATH=scripts .venv/bin/python -u scripts/extract_persona_vectors.py \
      --model "$m" --pda \
      --n-sys 5 --n-questions 12 --rollouts 1 \
      --avg-window 60 --acts-stride 4 \
      --out results/persona_vectors \
      2>&1 | tee "results/persona_vectors_${m}_pda.log"; then
    echo "!!! $m FAILED — continuing with next model"
  fi
done
echo "=== cohort sweep done $(date) ==="
