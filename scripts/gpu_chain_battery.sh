#!/bin/bash
# Battery-size sensitivity pilot ladder (2026-09-27): small -> large, one model
# at a time, under the MPS watermark caps. k in {5,20,75}, 3 contexts each.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTHONPATH=scripts
echo "=== battery pilot ladder start $(date)" >> $L/gpu_chain.log
for m in Qwen/Qwen2.5-0.5B-Instruct google/gemma-3-1b-it meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B google/gemma-3-12b-it google/gemma-3-27b-it; do
  echo "--- battery $m $(date)" >> $L/gpu_chain.log
  caffeinate -i $PY scripts/battery_pilot.py --model $m --ks 5,20,75 --contexts 3 >> $L/battery_pilot.log 2>&1 || echo "--- battery FAILED $m $(date)" >> $L/gpu_chain.log
done
echo "=== battery pilot ladder done $(date)" >> $L/gpu_chain.log
