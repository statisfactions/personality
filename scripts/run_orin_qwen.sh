#!/bin/bash
# Just qwen2.5:3b — gemma3:4b runs separately once pull completes.
set -e
export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\"")
PERSONAS_FILE="instruments/synthetic_personas.json"
N_PERSONAS=50

run_one () {
  local model="$1"; local mode="$2"; local outfile="$3"
  if [ -f "$outfile" ]; then
    valid=$(python3 -c "import json; d=json.load(open('$outfile')); print(len([r for r in d['results'] if r.get('response_argmax')]))" 2>/dev/null || echo 0)
    expected=$([ "$mode" = "honest" ] || [ "$mode" = "fakegood" ] && echo $((N_PERSONAS*30)) || echo 30)
    if [ "$valid" -ge "$expected" ]; then echo "[$model:$mode] complete — skipping"; return; fi
    echo "[$model:$mode] resuming ($valid/$expected done)"
  fi
  case "$mode" in
    honest) python3 scripts/run_gfc_ollama.py --remote --model "$model" --synthetic-personas "$PERSONAS_FILE" --max-personas "$N_PERSONAS" --output "$outfile" --checkpoint-every 100 ;;
    fakegood) python3 scripts/run_gfc_ollama.py --remote --model "$model" --synthetic-personas "$PERSONAS_FILE" --max-personas "$N_PERSONAS" --fake-good --output "$outfile" --checkpoint-every 100 ;;
    bare) python3 scripts/run_gfc_ollama.py --remote --model "$model" --neutral bare --output "$outfile" ;;
    respondent) python3 scripts/run_gfc_ollama.py --remote --model "$model" --neutral respondent --output "$outfile" ;;
  esac
}

run_one "qwen2.5:3b" honest     "results/qwen2.5-3b_gfc30_synthetic.json"
run_one "qwen2.5:3b" fakegood   "results/qwen2.5-3b_gfc30_synthetic-fakegood.json"
run_one "qwen2.5:3b" bare       "results/qwen2.5-3b_gfc30_neutral-bare.json"
run_one "qwen2.5:3b" respondent "results/qwen2.5-3b_gfc30_neutral-respondent.json"

run_one "phi4-mini" fakegood     "results/phi4-mini_gfc30_synthetic-fakegood.json"
run_one "llama3.2:3b" honest     "results/llama3.2-3b_gfc30_synthetic.json"
run_one "llama3.2:3b" fakegood   "results/llama3.2-3b_gfc30_synthetic-fakegood.json"

echo "Done qwen+phi+llama runs."
