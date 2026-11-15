"""Canonical repo paths.

ROOT defaults to the repository containing this package; override with the
PERSONALITY_ROOT environment variable (useful from a worktree that lacks
gitignored data/results, or from a notebook living elsewhere).

PERSONALITY_DATA instead points the loaders at a curated paper-data
directory (../personality-paper/data, built by scripts/build_paper_data.py):
only the files the paper reads, under readable paths. Paths the curated
layout lacks are left pointing into ROOT.
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
ENACT_META = RESULTS / "persona_vectors"            # <model>_pda_meta.json
REPRESENT_GRIDS = ADJ / "represent_model_grids.npz"
HUMAN_CORR = ADJ / "escs_525pda_corr_raw.json"
HUMAN_POR = ROOT / "data" / "escs_525pda" / "525_PDA-1.por"
MANIFEST = ROOT / "instruments" / "cohort100_manifest.json"
HAMPSON_SDY = ROOT / "data" / "hampson_1987" / "sdy_british_pda525.csv"
JOINT_DIAG = RESULTS / "persona_vectors" / "joint_diag"

# Curated layout: keep in sync with scripts/build_paper_data.py
PAPER_LAYOUT = {
    "HUMAN_CORR": "human/escs_525pda_corr_raw.json",
    "HUMAN_POR": "human/525_PDA-1.por",
    "HAMPSON_SDY": "human/hampson_1987_sdy_british_pda525.csv",
    "SELF_DIR": "self",
    "INTROSPECT": "judge",
    "REPRESENT_GRIDS": "represent/represent_model_grids.npz",
    "ENACT_MID": "enact",
    "ENACT_META": "enact",
    "JOINT_DIAG": "joint_diag",
}
DATA = os.environ.get("PERSONALITY_DATA")
if DATA:
    for _name, _rel in PAPER_LAYOUT.items():
        globals()[_name] = Path(DATA) / _rel
