#!/bin/bash
# Registered experiment: mid-thought last-mention EV (existing think arms)
# vs forced-close decision EV, smoke set, Qwen3-8B + Glimmer. Waits for GPU.
cd "$(dirname "$0")/.."
while pgrep -f "cohort_queue.py|run_post_think_pipeline.sh|self_adjective_report.py|default_enact_capture|extract_adjectives|adjective_judge_full|extract_persona_vectors|base_rate_query" > /dev/null; do sleep 300; done
for m in Qwen/Qwen3-8B meta-models/Muse-Glimmer-30B; do
  PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/self_adjective_report.py --model "$m" --think --force-close >> tmp/logs/forced_close_smoke.log 2>&1
done
echo "FORCED-CLOSE SMOKE DONE $(date)" >> tmp/logs/forced_close_smoke.log
