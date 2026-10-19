#!/bin/bash
# JUDGE on the wide-ENACT models (rgb, 2026-10-18: "start the same (long)
# process for Judge"): full directional tom_likely matrix (525 x 524 =
# 275,100 prompts, resumable), then the direct base-rate query and the
# base-rate fit -- the same three steps as the 12-model cohort
# (run_judge525.sh). Gemma-4-31B already has JUDGE. Smallest first, so the
# first complete model lands early. Outputs go to introspect_full under the
# repo-id stem (org_name); pkit.channels.JUDGE_ROSTER keeps them out of the
# paper's channel until they are added deliberately.
# Edit tmp/judge_full_todo.txt to reorder or stop after the current model.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; OUT=results/adjectives/introspect_full
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTHONPATH=scripts HF_HUB_OFFLINE=1
echo "=== judge full arm start $(date)" >> $L/gpu_chain.log
while true; do
  m=$(grep -v '^\s*$' tmp/judge_full_todo.txt | while read -r x; do fs=$(echo "$x" | tr / _); [ -f "$OUT/${fs}_base_rate_fit.npz" ] || [ -f "$OUT/${fs}.judge_failed" ] || { echo "$x"; break; }; done)
  [ -z "$m" ] && break
  fs=$(echo "$m" | tr / _)
  echo "--- judge-full $m $(date)" >> $L/gpu_chain.log
  { caffeinate -i $PY -u scripts/adjective_judge_full.py --models "$m" --mode tom_likely --both-directions \
      && caffeinate -i $PY -u scripts/base_rate_query.py --model "$m" \
      && $PY -u scripts/judge_base_rate_fit.py --model "$m"; } >> $L/judge_full_${fs}.log 2>&1 \
    || { echo "--- judge-full FAILED ($?) $m $(date)" >> $L/gpu_chain.log; touch "$OUT/${fs}.judge_failed"; }
done
echo "=== judge full arm done $(date)" >> $L/gpu_chain.log
