#!/bin/bash
# Full-adjective ENACT (paper recipe: 523 PDA adjectives x 5 framings x 12
# questions x 1 rollout, 60-token averaging window, acts stride 4) on the
# big/new models first (rgb, 2026-10-10: "while the GPU is free"). Same
# recipe as the cohort-10 run (scripts/run_persona_cohort.sh) except
# generation is batched (--gen-batch 20: left-padded sampling, unpadded
# re-forward; validated against the legacy path on Qwen2.5-7B medoids).
# Resumable per condition via checkpoints; edit tmp/enact_full_todo.txt to
# reorder or stop after the current model.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; OUT=results/persona_vectors/pda_wide
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTHONPATH=scripts HF_HUB_OFFLINE=1
mkdir -p $OUT
echo "=== enact full arm start $(date)" >> $L/gpu_chain.log
while true; do
  m=$(grep -v '^\s*$' tmp/enact_full_todo.txt | while read -r x; do fs=$(echo "$x" | tr / _); [ -f "$OUT/${fs}_pda_report.json" ] || [ -f "$OUT/${fs}.failed" ] || { echo "$x"; break; }; done)
  [ -z "$m" ] && break
  fs=$(echo "$m" | tr / _)
  echo "--- enact-full $m $(date)" >> $L/gpu_chain.log
  caffeinate -i $PY -u scripts/extract_persona_vectors.py --model "$m" --pda \
    --n-sys 5 --n-questions 12 --rollouts 1 --avg-window 60 --acts-stride 4 \
    --gen-batch 20 --out $OUT >> $L/enact_full_${fs}.log 2>&1 \
    || { echo "--- enact-full FAILED ($?) $m $(date)" >> $L/gpu_chain.log; touch "$OUT/${fs}.failed"; }
done
echo "=== enact full arm done $(date)" >> $L/gpu_chain.log
