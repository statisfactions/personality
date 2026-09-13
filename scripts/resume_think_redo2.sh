#!/bin/bash
# Relaunch after the 2026-09-13 04:54 reboot killed the chain mid-smoke
# and the Glimmer waiter. Remaining: two hybrid damage smokes, then the
# Glimmer full resume (from its .part checkpoint, ~31h). Each step is
# checkpoint-safe; a further reboot resumes.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; export PYTHONPATH=scripts
run() { caffeinate -i $PY "$@"; }
echo "=== resume2 start $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model google/gemma-4-31B-it --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== gemma4 smoke done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model Qwen/Qwen3-14B --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== qwen3-14b smoke done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model meta-models/Muse-Glimmer-30B --full --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== glimmer resume done $(date)" >> $L/gpu_chain.log
