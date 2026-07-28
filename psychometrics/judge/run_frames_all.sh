#!/usr/bin/env bash
# Sequential frame-experiment collection for the four Orin cohort models
# (concurrent runs thrash the Orin's GPU memory as models swap in and out).
# Resumable: re-run to pick up where an interrupted pass left off.
set -u
cd "$(dirname "$0")"
: "${OLLAMA_API_KEY:?set OLLAMA_API_KEY first}"
for m in gemma3:4b qwen2.5:3b llama3.2:3b phi4-mini:latest; do
  echo "=== $m  $(date -Is) ==="
  python3 run_frames_ollama.py --model "$m" || echo "!! $m exited $?"
done
echo "=== all done $(date -Is) ==="
