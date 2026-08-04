#!/bin/bash
# Standalone-HTML build for the self-perception progress note.
# Figures are base64-inlined (--embed-resources) so the output is one
# shareable file. Citations resolve author-date in-text (citeproc default).
set -e
cd "$(dirname "$0")/../rgb_reports"

# lint: table captions must be sequential from 1; prose refs must exist
python3 - <<'PYEOF'
import re, sys
s = open("progress_note_selfperception.md").read()
caps = [int(n) for n in re.findall(r"\*\*Table (\d+)\*\*", s)]
refs = {int(n) for n in re.findall(r"(?<!\*)Table (\d+)(?!\*)", s)}
ok = True
if caps != list(range(1, len(caps) + 1)):
    print(f"TABLE NUMBERING BROKEN: captions are {caps}", file=sys.stderr)
    ok = False
dangling = refs - set(caps)
if dangling:
    print(f"DANGLING TABLE REFS (no caption): {sorted(dangling)}",
          file=sys.stderr)
    ok = False
figs = [int(n) for n in re.findall(r"Figure (\d+)", s)]
if figs and sorted(set(figs)) != list(range(1, max(figs) + 1)):
    print(f"FIGURE NUMBERING GAPPY: {sorted(set(figs))}", file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
PYEOF
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
