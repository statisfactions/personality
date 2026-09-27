#!/bin/bash
# Full-cohort battery arm (rgb, 2026-09-27): k=20, 3 contexts, seed 0 (same
# contexts as the pilot), core SELF roster minus the 7 pilot models, small
# -> large, one model at a time under the MPS caps, 1 h timeout per model,
# offline hub so a missing shard fails fast instead of hanging.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTHONPATH=scripts HF_HUB_OFFLINE=1
echo "=== battery full cohort start $(date)" >> $L/gpu_chain.log
while read -r m; do
  [ -z "$m" ] && continue
  out="results/adjectives/battery/$(echo $m | tr / _)_battery.json"
  [ -f "$out" ] && { echo "--- battery skip (exists) $m" >> $L/gpu_chain.log; continue; }
  echo "--- battery $m $(date)" >> $L/gpu_chain.log
  timeout 3600 caffeinate -i $PY scripts/battery_pilot.py --model "$m" --ks 20 --contexts 3 >> $L/battery_pilot.log 2>&1 || echo "--- battery FAILED ($?) $m $(date)" >> $L/gpu_chain.log
done < tmp/battery_todo.txt
echo "=== battery full cohort done $(date)" >> $L/gpu_chain.log
