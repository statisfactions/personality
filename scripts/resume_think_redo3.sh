#!/bin/bash
# Third relaunch (2026-09-14 03:50). The Gemma4 @1024 think arm leaks
# ~5 GB/h of process footprint (28 GB MALLOC_SMALL + graphics) that
# torch.mps.empty_cache() does not return; RSS looked flat because the
# compressor was absorbing it (96 GB footprint, 73 GB compressed, stuck).
# Guard: every stage recycles its process every 100 items (exit 3 ->
# resume from the .part checkpoint). Remaining: gemma4 smoke (outputs
# 40/58), qwen3-14b smoke, glimmer full resume.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; export PYTHONPATH=scripts
run() {  # loop while the script asks to be recycled (exit 3)
  while true; do
    caffeinate -i $PY "$@" --exit-after-items 100 >> $L/think_redo.log 2>&1
    rc=$?
    [ $rc -eq 3 ] && { echo "--- recycle $(date) ($*)" >> $L/gpu_chain.log; continue; }
    return $rc
  done
}
echo "=== resume5 start $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model google/gemma-4-31B-it --think --force-close --max-new 1024
echo "=== gemma4 smoke done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model Qwen/Qwen3-14B --think --force-close --max-new 1024
echo "=== qwen3-14b smoke done $(date)" >> $L/gpu_chain.log
run scripts/self_adjective_report.py --model meta-models/Muse-Glimmer-30B --full --think --force-close --max-new 1024
echo "=== glimmer resume done $(date)" >> $L/gpu_chain.log
