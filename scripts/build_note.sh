#!/bin/bash
# Standalone-HTML build for the self-perception progress note.
# Figures are base64-inlined (--embed-resources) so the output is one
# shareable file. Citation keys (@X) stay literal until a .bib exists;
# when task-5's bib lands, add: --citeproc --bibliography=note.bib
set -e
cd "$(dirname "$0")/../rgb_reports"
pandoc progress_note_selfperception.md \
  --standalone --embed-resources \
  --css=note_assets/note.css \
  --metadata pagetitle="Models Vary Widely in Strength of Self-Image" \
  -o progress_note_selfperception.html
echo "wrote rgb_reports/progress_note_selfperception.html"
