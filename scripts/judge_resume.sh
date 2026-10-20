#!/bin/bash
# Resume the JUDGE chain paused by scripts/judge_pause.sh (picks up within 30 s).
cd "$(dirname "$0")/.."
rm -f tmp/judge_pause
pgrep -f gpu_chain_judge_full.sh >/dev/null && echo "resuming within 30 s" || { nohup bash scripts/gpu_chain_judge_full.sh >/dev/null 2>&1 & echo "chain restarted"; }
