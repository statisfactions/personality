cd /Users/rgb/src/personality; PYTHONPATH=scripts .venv/bin/python scripts/facet_slides.py >/dev/null && .venv/bin/python - <<'EOF'
from PIL import Image
from pathlib import Path
d = Path("results/persona_vectors/figs/slides")
files = sorted(d.glob("slide*.png"))
imgs = [Image.open(f).convert("RGB") for f in files]
imgs[0].save(d / "facet_slides.pdf", save_all=True, append_images=imgs[1:], resolution=200)
print("rebuilt")
EOF

