#!/bin/bash
# THE single GPU chain (rgb 2026-08-29: audits first, in case they force rework).
#   1. forced-close smoke: Qwen3-8B @384 + @1024, Qwen3.8-27B @384, Glimmer @384
#   2. digit-mass audit (wide cohort, smoke x direct)
#   3. cue-placement check (4 deep models)
#   4. 525 backfill (idempotent; resumes ENACT from checkpoints) -> base rates -> LS fit
#   5. cohort queue (Qwen3.8 think-arm resume)
cd "$(dirname "$0")/.."
L=tmp/logs
PY=".venv/bin/python"; export PYTHONPATH=scripts
run() { caffeinate -i $PY "$@"; }
echo "=== chain start $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model Qwen/Qwen3-8B --think --force-close >> $L/forced_close_smoke.log 2>&1
run scripts/self_adjective_report.py --model Qwen/Qwen3-8B --think --force-close --max-new 1024 >> $L/forced_close_smoke.log 2>&1
run scripts/self_adjective_report.py --model Qwen/Qwen3.8-27B --think --force-close --max-new 1024 >> $L/forced_close_smoke.log 2>&1
run scripts/self_adjective_report.py --model meta-models/Muse-Glimmer-30B --think --force-close --max-new 1024 >> $L/forced_close_smoke.log 2>&1
echo "=== forced-close smoke done $(date)" >> $L/gpu_chain.log
run scripts/digit_mass_audit.py >> $L/digit_mass_audit.log 2>&1
echo "=== mass audit done $(date)" >> $L/gpu_chain.log
for m in llama3.2 qwen2.5 gemma3 phi4; do run scripts/prefill_placement_check.py --model "$m" --device mps >> $L/prefill_placement.log 2>&1; done
echo "=== placement check done $(date)" >> $L/gpu_chain.log
$PY scripts/backfill_525.py >> $L/backfill_525.log 2>&1
for m in $(ls results/adjectives/introspect_full/*_tom_likely_dir.npz | xargs -n1 basename | sed 's/_tom_likely_dir.npz//'); do
  run scripts/base_rate_query.py --model "$m" >> $L/base_rates.log 2>&1
  $PY scripts/judge_base_rate_fit.py --model "$m" >> $L/base_rate_fit.log 2>&1
done
echo "=== backfill + base rates done $(date)" >> $L/gpu_chain.log
$PY scripts/cohort_queue.py --min-free-gb 300 >> $L/queue_restart13.log 2>&1
echo "=== chain done $(date)" >> $L/gpu_chain.log
