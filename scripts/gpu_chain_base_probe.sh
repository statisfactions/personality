#!/bin/bash
# P20 baseline probe: pause the redo chain (checkpoint-safe), run alone, resume the chain.
cd "$(dirname "$0")/.."
L=tmp/logs; export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SMALL=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_small_repos.json'))))")
BIG=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('tmp/refgroup_big_repos.json'))))")
echo "=== base probe start $(date)" >> $L/gpu_chain.log
caffeinate -i .venv/bin/python scripts/reference_group_probe.py --tag _base --variants base_assistant base_ai base_lm base_person --models $SMALL Aya Qwen Llama FalconMamba Gemma Gemma12 Llama8 Phi4 Qwen7 $BIG Qwen32 Gemma27 google/gemma-2b-it >> $L/refgroup_base.log 2>&1
echo "=== base probe complete $(date)" >> $L/gpu_chain.log
echo "wrote BASE-PROBE-COMPLETE" >> $L/refgroup_base.log
bash scripts/gpu_chain_redo.sh
