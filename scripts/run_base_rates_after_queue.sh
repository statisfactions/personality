#!/bin/bash
# Direct base-rate queries for the 12 JUDGE models, after the think arms.
cd "$(dirname "$0")/.."
while pgrep -f "cohort_queue.py" > /dev/null; do sleep 300; done
for m in $(ls results/adjectives/introspect_full/*_tom_likely_dir.npz | xargs -n1 basename | sed 's/_tom_likely_dir.npz//'); do
  PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/base_rate_query.py --model "$m" >> tmp/logs/base_rates.log 2>&1
done
echo "BASE RATES DONE $(date)" >> tmp/logs/base_rates.log
