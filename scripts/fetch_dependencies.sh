#!/usr/bin/env bash
# Restore the automatable third-party dependencies into dependencies/.
# Manual datasets (Dataverse/PsychArchives registration walls) are listed in
# dependencies/README.md and go into data/ by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dependencies

clone() {
  local url=$1 dir=$2
  if [ -d "dependencies/$dir/.git" ]; then
    echo "[skip] dependencies/$dir exists"
  else
    git clone --depth 1 "$url" "dependencies/$dir"
  fi
}

clone https://github.com/holi-lab/ValuePortrait.git ValuePortrait
clone https://github.com/anthropics/jacobian-lens.git jacobian-lens
clone https://github.com/persona-cartography/persona-cartography.git persona-cartography

echo
echo "Done. HF-hub artifacts (models, neuronpedia lenses, sentence encoders)"
echo "download themselves on first script use — set HF_TOKEN for gated models."
echo "Manual datasets: see dependencies/README.md."
