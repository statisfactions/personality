#!/usr/bin/env bash
# make_data_release.sh — package rgb's Tier-1 "analysis-ready" data for
# statisfactions and drop it onto the mounted Google Drive filesystem.
#
# Tier 1 = extracted products + acts + survey/JUDGE results. It EXCLUDES:
#   - raw per-rollout *_pda.pt      (~65 GB; set INCLUDE_RAW=1 to also copy them)
#   - *_pda_ckpt checkpoint dirs    (~26 GB; regenerable via finalize_from_checkpoints.py — never shipped)
#   - phase_b_cache*                (statisfactions' own caches)
#
# Builds tarballs LOCALLY (fast disk, checksummed) then copies to the Drive
# mount, so a slow/failed upload never leaves a corrupt half-tar on Drive.
# Small/text tracks -> .tar.gz; big binary (.pt) tracks -> plain .tar (torch
# tensors don't gzip). Re-runnable: same-day stamp overwrites.
#
# Usage:
#   scripts/make_data_release.sh                 # build + copy to Drive mount
#   DRY_RUN=1   scripts/make_data_release.sh     # print the plan, touch nothing
#   DO_COPY=0   scripts/make_data_release.sh     # build tarballs locally only
#   INCLUDE_RAW=1 scripts/make_data_release.sh   # also copy the ~65 GB raw *_pda.pt
#   DEST=/path  scripts/make_data_release.sh     # override Drive destination
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d results ] || { echo "!! run from the repo (no results/ here)"; exit 1; }

DEST="${DEST:-$HOME/gdrive/projects/llm-personality}"
STAGE="${STAGE:-results/releases}"
STAMP="$(date +%Y%m%d)"
DO_COPY="${DO_COPY:-1}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_RAW="${INCLUDE_RAW:-0}"

say () { printf '\n=== %s ===\n' "$*"; }
run () { if [ "$DRY_RUN" = 1 ]; then printf '  DRY: %s\n' "$*"; else eval "$*"; fi; }

mkdir -p "$STAGE"
say "Tier-1 data release  stamp=$STAMP  stage=$STAGE  dest=$DEST"

# 1. persona_vectors PRODUCTS: npz vectors, *_report.json, four_grid_*.json,
#    *_texts.json, figs. Excludes the 65 GB raw *_pda.pt and *_pda_ckpt dirs.
run "tar -C results -czf '$STAGE/persona_vectors_products_$STAMP.tar.gz' --exclude='*.pt' --exclude='*_pda_ckpt' persona_vectors"

# 2. adjectives: REPRESENT residual-stream acts (.pt) + JUDGE (introspect*) —
#    big binaries, plain tar.
run "tar -C results -cf '$STAGE/adjectives_$STAMP.tar' adjectives"

# 3. repe: RepE trait directions (.pt) — plain tar.
run "tar -C results -cf '$STAGE/repe_$STAMP.tar' repe"

# 4. persona: persona-conditioned Likert/representation data — small files.
run "tar -C results -czf '$STAGE/persona_$STAMP.tar.gz' persona"

# 5. surveys + misc products + root-level result JSONs (skips *.log).
run "tar -C results -czf '$STAGE/surveys_misc_$STAMP.tar.gz' surveys facets item_level desirability sweeps stimuli \$(cd results && ls *.json 2>/dev/null)"

# checksums + human manifest (globs also pick up judge_likert_$STAMP.tar.gz if present)
say "checksums + manifest"
run "( cd '$STAGE' && shasum -a 256 *_$STAMP.tar *_$STAMP.tar.gz > 'MANIFEST_$STAMP.sha256' )"
if [ "$DRY_RUN" != 1 ]; then
  { echo "# LLM-personality data release (Tier 1) — $STAMP"
    echo "# rgb track. EXCLUDES raw *_pda.pt (~65 GB) and *_pda_ckpt checkpoints."
    echo "# Each *.tar.gz = small/text products; each *.tar = big .pt binaries."
    echo "# See README inside persona_vectors_products / judge_likert for formats."
    echo
    ( cd "$STAGE" && du -h *_"$STAMP".tar *_"$STAMP".tar.gz 2>/dev/null )
  } > "$STAGE/RELEASE_$STAMP.txt"
  cat "$STAGE/RELEASE_$STAMP.txt"
fi

# optional Tier 2: the raw per-rollout vectors, copied individually (not tarred —
# they're multi-GB dense binaries; a per-model object lets him grab one at a time).
if [ "$INCLUDE_RAW" = 1 ]; then
  say "INCLUDE_RAW=1 — copying raw *_pda.pt (~65 GB) to $DEST/raw_vectors/"
  run "mkdir -p '$DEST/raw_vectors'"
  run "cp -v results/persona_vectors/*_pda.pt '$DEST/raw_vectors/'"
fi

if [ "$DO_COPY" = 1 ]; then
  say "copy to Drive mount: $DEST"
  if [ ! -d "$(dirname "$DEST")" ]; then
    echo "!! Drive path parent missing ($DEST) — is the Drive mount up? Aborting copy."
    echo "   Tarballs are built in $STAGE; re-run with the mount up, or DO_COPY=0."
    exit 1
  fi
  run "mkdir -p '$DEST'"
  run "cp -v '$STAGE'/*_$STAMP.tar '$STAGE'/*_$STAMP.tar.gz '$STAGE/MANIFEST_$STAMP.sha256' '$STAGE/RELEASE_$STAMP.txt' '$DEST/'"
  run "cp -v rgb_reports/data_release_readme.md '$DEST/README.md'"
  say "done — Drive for Desktop will upload in the background. Verify sync in the Drive UI, then the local $STAGE copies can be removed."
else
  say "DO_COPY=0 — tarballs left in $STAGE (not copied to Drive)"
fi
