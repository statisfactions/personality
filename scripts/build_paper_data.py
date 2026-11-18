"""Copy the files the paper reads into a curated data directory (rgb 2026-11-14).

The paper (../personality-paper) renders through pkit; with PERSONALITY_DATA
set, pkit.paths maps every loader onto the layout below instead of results/.
Selection reuses pkit's own logic, so the curated set is exactly what the
loaders open:

  human/       HUMAN ratings (.por), its correlation matrix, Hampson 1987 norms
  self/        SELF cohort files (load.self_paths("cohort"): EXCLUDE, dedupe,
               THINK_PREFER swap already applied, so a re-glob finds the same set)
  judge/       JUDGE roster (channels.JUDGE_ROSTER) tom_likely_dir.npz
  represent/   per-model REPRESENT cosine grids
  enact/       ENACT persona vectors + their pda_meta (massive dims)
  joint_diag/  the K=6 joint-axes fit

Writes MANIFEST.json (source path, size, sha256, analysis-repo commit). Files
listed in a previous MANIFEST.json but no longer selected are removed.
The staging dir is uploaded to the HF dataset repo status-luporum/personality-
artifacts, which the paper DVC-imports (../personality-paper/data/README.md).

Small paper files skip the upload and go straight into the paper repo, which
tracks them in git, so a tweak is regenerate + copy + commit there:
  figures/            the PDFs the paper includes (built by scripts/fig_*.py)
  _data/prompts.json  the JUDGE prompt templates (adjective_introspection.PROMPT;
                      importing that module needs torch, the render should not)

Usage: .venv/bin/python scripts/build_paper_data.py [--dest results/paper_data]
       .venv/bin/python scripts/build_paper_data.py --paper-only
"""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
from pkit import paths, load, channels

ap = argparse.ArgumentParser()
ap.add_argument("--dest", default=str(paths.RESULTS / "paper_data"))
ap.add_argument("--paper", default=str(paths.ROOT.parent / "personality-paper"))
ap.add_argument("--paper-only", action="store_true")
args = ap.parse_args()
assert not os.environ.get("PERSONALITY_DATA"), "unset PERSONALITY_DATA: the sources must be results/"
DEST = Path(args.dest)
L = paths.PAPER_LAYOUT
FIGS = ["fig_cluster_grids_level", "fig_cluster_grids_top", "fig_channel_similarity_level",
        "fig_channel_similarity_pc1", "fig_channel_similarity_top", "fig_joint_axes"]

sel = {L["HUMAN_CORR"]: paths.HUMAN_CORR, L["HUMAN_POR"]: paths.HUMAN_POR,
       L["HAMPSON_SDY"]: paths.HAMPSON_SDY,
       L["REPRESENT_GRIDS"]: paths.REPRESENT_GRIDS,
       f'{L["JOINT_DIAG"]}/joint_diag_k6.npz': paths.JOINT_DIAG / "joint_diag_k6.npz"}
for p in load.self_paths("cohort").values():
    sel[f'{L["SELF_DIR"]}/{Path(p).name}'] = Path(p)
for m in channels.JUDGE_ROSTER:
    sel[f'{L["INTROSPECT"]}/{m}_tom_likely_dir.npz'] = paths.INTROSPECT / f"{m}_tom_likely_dir.npz"
for p in sorted(paths.ENACT_MID.glob("*.npz")):
    sel[f'{L["ENACT_MID"]}/{p.name}'] = p
    sel[f'{L["ENACT_META"]}/{p.stem}_pda_meta.json'] = paths.ENACT_META / f"{p.stem}_pda_meta.json"
figs = {f"{f}.pdf": paths.RESULTS / "persona_vectors" / "figs" / f"{f}.pdf" for f in FIGS}

missing = [str(s) for s in [*sel.values(), *figs.values()] if not s.exists()]
assert not missing, f"missing sources: {missing}"

PAPER = Path(args.paper)
if PAPER.exists():
    (PAPER / "figures").mkdir(exist_ok=True)
    for name, src in figs.items():
        shutil.copy2(src, PAPER / "figures" / name)
    import sys; sys.path.insert(0, str(paths.ROOT / "scripts"))
    import adjective_introspection
    json.dump(adjective_introspection.PROMPT, open(PAPER / "_data" / "prompts.json", "w"), indent=1)
    print(f"{len(figs)} figures + prompts.json -> {PAPER}")
else:
    print(f"no paper repo at {PAPER}; figures and prompts not copied")
if args.paper_only:
    raise SystemExit


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


old = DEST / "MANIFEST.json"
stale = set(json.load(open(old))["files"]) - set(sel) if old.exists() else set()
for rel in sorted(stale):
    (DEST / rel).unlink(missing_ok=True)
    print("removed", rel)

files, copied = {}, 0
for rel, src in sorted(sel.items()):
    dst = DEST / rel
    digest = sha256(src)
    if not (dst.exists() and sha256(dst) == digest):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); copied += 1
    files[rel] = {"source": str(src.relative_to(paths.ROOT)), "bytes": src.stat().st_size, "sha256": digest}

git = lambda *a: subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True).stdout.strip()
json.dump({"analysis_commit": git("rev-parse", "HEAD"),
           "analysis_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "files": files}, open(old, "w"), indent=1)
print(f"{len(files)} files ({sum(f['bytes'] for f in files.values()) / 1e6:.0f} MB), {copied} copied -> {DEST}")
