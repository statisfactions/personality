#!/bin/bash
# Sequential Orin inference for the pooled-TIRT replication.
# Runs all needed conditions for the 4 open RGB-lineup models, one model
# at a time, with checkpointing so we can resume on failure.
#
# Conditions (per model): honest (50 personas), fake-good (50 personas),
# bare (no persona), respondent (no persona).
# Each persona run = 50 * 30 = 1500 prompts. Each neutral run = 30 prompts.
#
# We skip conditions that already have complete output files.

set -e
export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\"")
export PYTHONUNBUFFERED=1

PERSONAS_FILE="instruments/synthetic_personas.json"
N_PERSONAS=50

run_one () {
  local model="$1"
  local mode="$2"   # honest | fakegood | bare | respondent
  local outfile="$3"

  if [ -f "$outfile" ]; then
    valid=$(python3 -c "import json; d=json.load(open('$outfile')); print(len([r for r in d['results'] if r.get('response_argmax')]))" 2>/dev/null || echo 0)
    expected=$([ "$mode" = "honest" ] || [ "$mode" = "fakegood" ] && echo $((N_PERSONAS*30)) || echo 30)
    if [ "$valid" -ge "$expected" ]; then
      echo "[$model:$mode] complete ($valid/$expected) — skipping"
      return
    fi
    echo "[$model:$mode] resuming ($valid/$expected done)"
  fi

  case "$mode" in
    honest)
      python3 scripts/run_gfc_ollama.py --remote --model "$model" \
        --synthetic-personas "$PERSONAS_FILE" --max-personas "$N_PERSONAS" \
        --output "$outfile" --checkpoint-every 100
      ;;
    fakegood)
      python3 scripts/run_gfc_ollama.py --remote --model "$model" \
        --synthetic-personas "$PERSONAS_FILE" --max-personas "$N_PERSONAS" \
        --fake-good --output "$outfile" --checkpoint-every 100
      ;;
    bare)
      python3 scripts/run_gfc_ollama.py --remote --model "$model" \
        --neutral bare --output "$outfile"
      ;;
    respondent)
      python3 scripts/run_gfc_ollama.py --remote --model "$model" \
        --neutral respondent --output "$outfile"
      ;;
  esac
}

# Slug helper: ollama "gemma3:4b" → "gemma3-4b"
slug () { echo "${1//:/-}"; }

# === Per-model conditions ===
# Format: model | conditions to run (others assumed done already)
#
# We track which (model, condition) pairs are needed for the pooled fit.

# Gemma3-4B-IT (RGB exact) — all 4 conditions new
run_one "gemma3:4b" honest     "results/$(slug gemma3:4b)_gfc30_synthetic.json"
run_one "gemma3:4b" fakegood   "results/$(slug gemma3:4b)_gfc30_synthetic-fakegood.json"
run_one "gemma3:4b" bare       "results/$(slug gemma3:4b)_gfc30_neutral-bare.json"
run_one "gemma3:4b" respondent "results/$(slug gemma3:4b)_gfc30_neutral-respondent.json"

# Qwen2.5-3B (new on Orin) — all 4 conditions new
run_one "qwen2.5:3b" honest     "results/$(slug qwen2.5:3b)_gfc30_synthetic.json"
run_one "qwen2.5:3b" fakegood   "results/$(slug qwen2.5:3b)_gfc30_synthetic-fakegood.json"
run_one "qwen2.5:3b" bare       "results/$(slug qwen2.5:3b)_gfc30_neutral-bare.json"
run_one "qwen2.5:3b" respondent "results/$(slug qwen2.5:3b)_gfc30_neutral-respondent.json"

# Phi4-mini (have honest at 400 personas + bare/respondent already)
# Just need fake-good on first 50
run_one "phi4-mini" fakegood   "results/$(slug phi4-mini)_gfc30_synthetic-fakegood.json"

# Llama3.2-3B (have bare/respondent already; need honest + fake-good)
run_one "llama3.2:3b" honest     "results/$(slug llama3.2:3b)_gfc30_synthetic.json"
run_one "llama3.2:3b" fakegood   "results/$(slug llama3.2:3b)_gfc30_synthetic-fakegood.json"

echo
echo "All Orin runs complete."
