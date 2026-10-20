#!/bin/bash
# Pause the JUDGE chain and free the GPU/RAM (rgb wants the machine). The
# chain stays alive and waits; the model job is killed and resumes from its
# last checkpoint (<= 1600 pairs lost). Undo: scripts/judge_resume.sh
cd "$(dirname "$0")/.."
touch tmp/judge_pause
pkill -f "scripts/adjective_judge_full.py|scripts/base_rate_query.py|scripts/judge_base_rate_fit.py" && echo "model job stopped"
pgrep -f gpu_chain_judge_full.sh >/dev/null && echo "chain waiting on tmp/judge_pause" || echo "chain not running (start: nohup bash scripts/gpu_chain_judge_full.sh &)"
