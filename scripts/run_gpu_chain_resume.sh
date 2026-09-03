#!/bin/bash
# Resume the GPU chain after a reboot/pause (2026-09-03).
# The original run_gpu_chain.sh is linear; this entry point skips the
# completed one-shot stages (forced-close smokes, digit-mass audit,
# placement check — all done 2026-08-30) and re-enters at the backfill,
# which is idempotent per file: self/represent no-op instantly, enact
# redoes at most the interrupted model's 2-adjective run, judge resumes
# per model. Then base rates -> LS fits -> cohort queue, verbatim.
#   pause:  bash scripts/run_gpu_chain_resume.sh pause
#   resume: nohup bash scripts/run_gpu_chain_resume.sh &   (or setsid)
cd "$(dirname "$0")/.."
L=tmp/logs
PY=".venv/bin/python"; export PYTHONPATH=scripts
if [ "$1" = "pause" ]; then
    pkill -f run_gpu_chain.sh
    pkill -f run_gpu_chain_resume.sh
    pkill -f "backfill_525|extract_persona_vectors|adjective_judge_full|finalize_from_checkpoints"
    pkill -f "base_rate_query|self_adjective_report|cohort_queue|run_audit_after_chain|digit_mass_audit"
    echo "chain paused (everything checkpointed; resume with this script)"
    exit 0
fi
run() { caffeinate -i $PY "$@"; }
echo "=== chain RESUME $(date)" >> $L/gpu_chain.log
$PY scripts/backfill_525.py >> $L/backfill_525.log 2>&1
for m in $(ls results/adjectives/introspect_full/*_tom_likely_dir.npz | xargs -n1 basename | sed 's/_tom_likely_dir.npz//'); do
  run scripts/base_rate_query.py --model "$m" >> $L/base_rates.log 2>&1
  $PY scripts/judge_base_rate_fit.py --model "$m" >> $L/base_rate_fit.log 2>&1
done
echo "=== backfill + base rates done $(date)" >> $L/gpu_chain.log
run scripts/digit_mass_audit.py >> $L/digit_mass_audit.log 2>&1
echo "=== mass audit rerun done $(date)" >> $L/gpu_chain.log
$PY scripts/cohort_queue.py --min-free-gb 300 >> $L/queue_restart13.log 2>&1
echo "=== chain done $(date)" >> $L/gpu_chain.log
