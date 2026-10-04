#!/bin/bash
# Medoid ENACT arm over the core cohort (rgb, 2026-10-03): 44 blocks44 medoids
# x 15 rollouts (5 questions x 3 induction templates) + a 15-rollout default,
# matching the cohort-10 recipe otherwise (performance induction, temp .8,
# top-p .95, 100 new tokens, acts stride 4). ~675 generations per model.
# Resumable per condition via the script's own checkpoints.
cd "$(dirname "$0")/.."
L=tmp/logs; PY=".venv/bin/python"; OUT=results/persona_vectors/medoids15
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTHONPATH=scripts HF_HUB_OFFLINE=1
mkdir -p $OUT
echo "=== enact medoid arm start $(date)" >> $L/gpu_chain.log
while read -r m; do
  [ -z "$m" ] && continue
  fs=$(echo "$m" | tr / _)
  [ -f "$OUT/${fs}_medoids15.pt" ] && { echo "--- enact skip (exists) $m" >> $L/gpu_chain.log; continue; }
  echo "--- enact $m $(date)" >> $L/gpu_chain.log
  timeout 14400 caffeinate -i $PY scripts/extract_persona_vectors.py --model "$m" --adjectives $(cat tmp/medoids44.txt) \
    --n-questions 5 --n-sys 3 --rollouts 1 --acts-stride 4 --bootstrap 50 --tag medoids15 --out $OUT \
    >> $L/enact_medoid.log 2>&1 || echo "--- enact FAILED ($?) $m $(date)" >> $L/gpu_chain.log
done < tmp/enact_medoid_todo.txt
echo "=== enact medoid arm done $(date)" >> $L/gpu_chain.log
