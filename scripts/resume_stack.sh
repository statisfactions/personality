#!/bin/bash
# One-command recovery after a reboot (or `pause`): restarts the wide-n
# queue + blinkenlights. Everything is checkpointed, so this is always safe.
#   bash scripts/resume_stack.sh          # restart everything
#   bash scripts/resume_stack.sh pause    # stop GPU work before unplugging
cd "$(dirname "$0")/.." || exit 1
if [ "$1" = "pause" ]; then
    pkill -f cohort_queue.py
    pkill -f "self_adjective_report|extract_adjectives|default_enact_capture"
    echo "queue paused (state safe; resume with: bash scripts/resume_stack.sh)"
    exit 0
fi
pgrep -f cohort_queue.py >/dev/null || {
    nohup .venv/bin/python scripts/cohort_queue.py --min-free-gb 300 \
        >> results/cohort100/queue.log 2>&1 &
    echo "queue restarted ($!)"; }
pgrep -f cohort_status.py >/dev/null || {
    nohup .venv/bin/python scripts/cohort_status.py \
        > tmp/logs/status_gen.log 2>&1 &
    echo "status generator restarted ($!)"; }
pgrep -f "http.server 8765" >/dev/null || {
    nohup .venv/bin/python -m http.server 8765 -d tmp/status --bind 127.0.0.1 \
        > tmp/logs/status_http.log 2>&1 &
    echo "blinkenlights back at http://127.0.0.1:8765"; }
