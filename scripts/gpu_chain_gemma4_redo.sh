#!/bin/bash
# Gemma4 censored-item rerun with a SHORT recycle (40 items): its footprint starts
# high (82 GB after load on 2026-09-17) and grows ~5 GB/h; 100-item cycles would
# approach the reboot zone. Checkpoint-safe; resumes from the .part.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"
echo "=== gemma4 redo relaunched with 40-item recycle $(date)" >> $L/gpu_chain.log
while true; do
  PYTHONPATH=scripts caffeinate -i $PY scripts/self_adjective_report.py --model google/gemma-4-31B-it --full --think --force-close --max-new 1024 --redo-unclosed results/adjectives/selfreport/google_gemma-4-31B-it_self_full_think.json --exit-after-items 40 >> $L/think_redo.log 2>&1
  rc=$?; [ $rc -eq 3 ] && { echo "--- recycle $(date) (gemma4 redo)" >> $L/gpu_chain.log; continue; }; break
done
echo "=== gemma4 redo done $(date)" >> $L/gpu_chain.log
