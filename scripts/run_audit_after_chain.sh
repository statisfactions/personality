#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_gpu_chain.sh" > /dev/null; do sleep 300; done
PYTHONPATH=scripts caffeinate -i .venv/bin/python scripts/digit_mass_audit.py >> tmp/logs/digit_mass_audit.log 2>&1
echo "MASS AUDIT (rerun) DONE $(date)" >> tmp/logs/digit_mass_audit.log
