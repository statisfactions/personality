#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_mass_audit_after_smoke.sh|run_forced_close_smoke.sh|cohort_queue.py|run_post_think_pipeline.sh" > /dev/null; do sleep 300; done
for m in llama3.2 qwen2.5 gemma3 phi4; do
  PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/prefill_placement_check.py --model "$m" --device mps >> tmp/logs/prefill_placement.log 2>&1
done
echo "PLACEMENT CHECK DONE $(date)" >> tmp/logs/prefill_placement.log
