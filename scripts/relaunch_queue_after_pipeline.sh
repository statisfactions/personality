#!/bin/bash
# After the post-think pipeline finishes, relaunch the cohort queue so the
# timed-out Qwen3.8 think arm resumes from its checkpoints (~12h left).
cd "$(dirname "$0")/.."
while pgrep -f "run_post_think_pipeline.sh" > /dev/null; do sleep 300; done
PYTHONPATH=scripts .venv/bin/python scripts/cohort_queue.py --min-free-gb 300 >> tmp/logs/queue_restart13.log 2>&1
