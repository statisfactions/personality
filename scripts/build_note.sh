#!/bin/bash
# Standalone-HTML build for the self-perception progress note.
# Figures are base64-inlined (--embed-resources) so the output is one
# shareable file. Citations resolve author-date in-text (citeproc default).
set -e
cd "$(dirname "$0")/../rgb_reports"
pandoc progress_note_selfperception.md \
  --standalone --embed-resources \
  --citeproc --bibliography=note_assets/note.bib \
  --metadata link-citations=true \
  --css=note_assets/note.css \
  --metadata pagetitle="Models Vary Widely in Strength of Self-Image" \
  -o progress_note_selfperception.html
echo "wrote rgb_reports/progress_note_selfperception.html"

# PDF via headless Chrome from the styled HTML — preserves the lab-note
# CSS (the pandoc→LaTeX route ignores it and restyles formal).
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf=progress_note_selfperception.pdf \
  "file://$(pwd)/progress_note_selfperception.html" 2>/dev/null
echo "wrote rgb_reports/progress_note_selfperception.pdf"
