#!/bin/bash
# Think-arm redo chain (2026-09-07): protocol v2 (1024 budget + force-close)
# for the known-damaged arms, GLIMMER FIRST (the person-frame anomaly needs
# it), then Qwen3-8B; then @1024 damage smokes for the untested hybrids
# (Gemma4, Qwen3-14B) to decide their redos. Qwen3.8-27B needs no redo
# (old-arm r=.997). Outputs land as *_self_full_think_fc_b1024.json —
# validate against the old arm, then swap the loader-visible file.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; export PYTHONPATH=scripts
run() { caffeinate -i $PY "$@"; }
echo "=== think redo start $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model meta-models/Muse-Glimmer-30B --full --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== glimmer redo done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model Qwen/Qwen3-8B --full --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== qwen3-8b redo done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model google/gemma-4-31B-it --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
run scripts/self_adjective_report.py --model Qwen/Qwen3-14B --think --force-close --max-new 1024 >> $L/think_redo.log 2>&1
echo "=== think redo chain done $(date)" >> $L/gpu_chain.log
