#!/bin/bash
# JUDGE 525 backfill, take 2 (2026-09-06): the Sep-3 run no-op'd because
# adjective_judge_full.load_adjectives() reads the corr-json labels, which
# were still 523 until the human-matrix regen. Now 525 -> the backfill
# branch actually fires (+2 adjectives, ~2k pairs/model). Then re-fit
# base rates on the 525 matrices.
cd "$(dirname "$0")/.."
L=tmp/logs; export PYTHONPATH=scripts
caffeinate -i .venv/bin/python scripts/backfill_525.py --steps judge >> $L/backfill_525.log 2>&1
for m in Aya FalconMamba Gemma Gemma12 Gemma27 Gemma4 Llama Llama8 Phi4 Qwen Qwen32 Qwen7; do
  .venv/bin/python scripts/judge_base_rate_fit.py --model "$m" >> $L/base_rate_fit.log 2>&1
done
echo "=== judge 525 + refits done $(date)" >> $L/gpu_chain.log
