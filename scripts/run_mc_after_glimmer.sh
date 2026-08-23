#!/bin/bash
# MC-over-chains SELF arm: waits for the GPU to be free of capture jobs,
# then runs the R1 chaos-poster and the Glimmer validation subset.
cd "$(dirname "$0")/.."
while pgrep -f "self_adjective_report.py.*--full" > /dev/null \
   || pgrep -f "default_enact_capture|extract_adjectives|run_ipip300|run_hexaco" > /dev/null; do
  sleep 120
done
sleep 60
PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/self_adjective_report.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --think-mc 8 \
  --framings direct person pda observer >> tmp/logs/mc_r1qwen.log 2>&1
PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/self_adjective_report.py \
  --model meta-models/Muse-Glimmer-30B --think-mc 5 \
  --framings direct pda >> tmp/logs/mc_glimmer.log 2>&1
echo "MC runs complete $(date)" >> tmp/logs/mc_r1qwen.log
