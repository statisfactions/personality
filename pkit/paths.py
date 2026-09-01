"""Canonical repo paths.

ROOT defaults to the repository containing this package; override with the
PERSONALITY_ROOT environment variable (useful from a worktree that lacks
gitignored data/results, or from a notebook living elsewhere).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PERSONALITY_ROOT",
                           Path(__file__).resolve().parent.parent))
RESULTS = ROOT / "results"
ADJ = RESULTS / "adjectives"
SELF_DIR = ADJ / "selfreport"
INTROSPECT = ADJ / "introspect_full"
ENACT_MID = RESULTS / "persona_vectors" / "enact_mid"
HUMAN_CORR = ADJ / "escs_525pda_corr_raw.json"
HUMAN_POR = ROOT / "data" / "escs_525pda" / "525_PDA-1.por"
MANIFEST = ROOT / "instruments" / "cohort100_manifest.json"
