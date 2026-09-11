#!/bin/bash
# Waiter (2026-09-11): the Glimmer @1024 redo was OOM-killed at 65%
# (3 framings + 320/525 pda done; .part checkpoint intact). Wait for the
# think-redo chain to finish, then relaunch — the script resumes from
# the checkpoint. Fresh process resets the multi-day memory creep.
cd "$(dirname "$0")/.."
L=tmp/logs
while pgrep -f run_think_redo.sh > /dev/null; do sleep 300; done
echo "=== glimmer resume start $(date)" >> $L/gpu_chain.log
caffeinate -i .venv/bin/python scripts/self_adjective_report.py \
  --model meta-models/Muse-Glimmer-30B --full --think --force-close \
  --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== glimmer resume done $(date)" >> $L/gpu_chain.log
