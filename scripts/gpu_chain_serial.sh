#!/bin/bash
# Serial GPU chain after the 2026-09-14 14:38 reboot (Glimmer recycle-load +
# concurrent pair-run model load exceeded memory). ONE model-loading job at a
# time, always. Order per rgb's priority: comparative pair on the core roster
# (small repos -> short-name 9 -> big 10; each skips existing outputs), then
# the think-redo chain (gemma4/qwen3-14b smokes already exist -> skipped;
# Glimmer resumes from its .part).
cd "$(dirname "$0")/.."
L=tmp/logs; export PYTHONPATH=.
V="direct pda more_assistant less_assistant more_ai less_ai more_lm less_lm"
echo "=== serial chain start $(date)" >> $L/gpu_chain.log
SMALL=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_small_repos.json'))))")
BIG=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_big_repos.json'))))")
caffeinate -i .venv/bin/python scripts/reference_group_probe.py --tag _pair --variants $V --models $SMALL >> $L/refgroup_pair.log 2>&1
caffeinate -i .venv/bin/python scripts/reference_group_probe.py --tag _pair --variants $V --models Aya Qwen Llama FalconMamba Gemma Gemma12 Llama8 Phi4 Qwen7 >> $L/refgroup_pair.log 2>&1
caffeinate -i .venv/bin/python scripts/reference_group_probe.py --tag _pair --variants $V --models $BIG Qwen32 Gemma27 >> $L/refgroup_pair.log 2>&1
echo "=== pair run complete $(date)" >> $L/gpu_chain.log
echo "wrote PAIR-RUN-COMPLETE" >> $L/refgroup_pair.log
PYTHONPATH=scripts bash scripts/resume_think_redo3.sh
