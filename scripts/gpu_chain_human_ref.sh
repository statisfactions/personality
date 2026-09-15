#!/bin/bash
# P19: pause Glimmer (checkpoint-safe), run the human-referenced pair alone on all core models, resume the chain.
cd "$(dirname "$0")/.."
L=tmp/logs; export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SMALL=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_small_repos.json'))))")
BIG=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_big_repos.json'))))")
echo "=== human-ref pair start $(date)" >> $L/gpu_chain.log
caffeinate -i .venv/bin/python scripts/reference_group_probe.py --tag _human --variants more_person less_person --models $SMALL Aya Qwen Llama FalconMamba Gemma Gemma12 Llama8 Phi4 Qwen7 $BIG Qwen32 Gemma27 >> $L/refgroup_human.log 2>&1
echo "=== human-ref pair complete $(date)" >> $L/gpu_chain.log
echo "wrote HUMAN-REF-COMPLETE" >> $L/refgroup_human.log
PYTHONPATH=scripts bash scripts/resume_think_redo3.sh
