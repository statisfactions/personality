#!/bin/bash
# Censored-item reruns (rgb go, 2026-09-16): --redo-unclosed keeps the closed
# items of the @384 arms (exact, r 1.000) and reruns only the censored ones at
# @1024 + force-close. Process-recycled every 100 items (Gemma4 leaks ~5 GB/h).
# Front: the gemma-2b-it online sweep for the pair run (3 min). One job at a time.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"
run() {
  while true; do
    PYTHONPATH=scripts caffeinate -i $PY "$@" --exit-after-items 100 >> $L/think_redo.log 2>&1
    rc=$?; [ $rc -eq 3 ] && { echo "--- recycle $(date) ($2)" >> $L/gpu_chain.log; continue; }; return $rc
  done
}
echo "=== redo chain start $(date)" >> $L/gpu_chain.log
V="direct pda more_assistant less_assistant more_ai less_ai more_lm less_lm"
PYTHONPATH=. caffeinate -i $PY scripts/reference_group_probe.py --tag _pair --variants $V --models google/gemma-2b-it >> $L/refgroup_pair.log 2>&1
PYTHONPATH=. caffeinate -i $PY scripts/reference_group_probe.py --tag _human --variants more_person less_person --models google/gemma-2b-it >> $L/refgroup_human.log 2>&1
echo "=== gemma-2b-it sweep done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model Qwen/Qwen3-14B --full --think --force-close --max-new 1024 --redo-unclosed results/adjectives/selfreport/Qwen_Qwen3-14B_self_full_think.json
echo "=== qwen3-14b redo done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model google/gemma-4-31B-it --full --think --force-close --max-new 1024 --redo-unclosed results/adjectives/selfreport/google_gemma-4-31B-it_self_full_think.json
echo "=== gemma4 redo done $(date)" >> $L/gpu_chain.log
