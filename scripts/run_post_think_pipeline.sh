#!/bin/bash
# The single post-think-arms GPU pipeline (rgb 2026-08-28: "a queue of queues"):
#   1. 525 backfill (SELF -> REPRESENT -> ENACT -> JUDGE)
#   2. direct base rates (525) for the JUDGE models
#   3. joint least-squares base-rate fit (CPU)
cd "$(dirname "$0")/.."
while pgrep -f "cohort_queue.py" > /dev/null; do sleep 300; done
PYTHONPATH=scripts .venv/bin/python scripts/backfill_525.py >> tmp/logs/backfill_525.log 2>&1
for m in $(ls results/adjectives/introspect_full/*_tom_likely_dir.npz | xargs -n1 basename | sed 's/_tom_likely_dir.npz//'); do
  PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/base_rate_query.py --model "$m" >> tmp/logs/base_rates.log 2>&1
  PYTHONPATH=scripts .venv/bin/python scripts/judge_base_rate_fit.py --model "$m" >> tmp/logs/base_rate_fit.log 2>&1
done
echo "POST-THINK PIPELINE DONE $(date)" >> tmp/logs/backfill_525.log
