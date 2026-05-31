#!/usr/bin/env python3
"""Fetch external research data into data/ (gitignored), reproducibly.

`data/` holds copyrighted / large external datasets that we cannot check into
git. This script re-downloads them from their canonical sources so a fresh
clone can be hydrated with one command. Each source is a declarative entry in
SOURCES below — add a dict, get a downloader for free.

Provenance for every URL is documented in rgb_reports/bibliography.md.

Usage:
  .venv/bin/python scripts/fetch_external_data.py --list
  .venv/bin/python scripts/fetch_external_data.py                 # all ACTIVE sources
  .venv/bin/python scripts/fetch_external_data.py --only escs_525pda escs_360pda
  .venv/bin/python scripts/fetch_external_data.py --include-legacy # also legacy mirrors
  .venv/bin/python scripts/fetch_external_data.py --force          # re-download

Idempotent: an already-fetched source is skipped (a .fetched marker is written
into its dir). Nothing here parses/normalizes the data — inspect formats by hand
before trusting them (cf. the IPIP300.por pre-reversed-keys lesson, W9 §7.5.1).
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_ROOT = Path("data")

# Harvard Dataverse: download ALL files of a dataset as a zip bundle.
def dataverse(doi):
    return ("https://dataverse.harvard.edu/api/access/dataset/:persistentId/"
            f"?persistentId=doi:{doi}")

SOURCES = {
    # ---- ACTIVE: human IPIP-NEO primary source (the facet-correlation matrix) ----
    "kajonius_johnson_2019": {
        "status": "active",
        "about": "Johnson IPIP-NEO-300/120 raw item data + scoring key. PRIMARY "
                 "source for instruments/ipip300_human_facet_correlations.json. "
                 "PsychArchives deposit e42a4531 (Kajonius & Johnson 2019 suppl). "
                 "SPSS .por (1-5 Likert, 0=missing) + ASCII .dat. ~170 MB zip.",
        "downloads": [
            {"url": "https://pada.psycharchives.org/bitstream/"
                    "82c844a8-8f45-471f-87d7-59a4dfcb56de",
             "out": "Kajonius_Johnson_2019_SUPPL.zip", "extract": "unzip"},
        ],
        "verify": ["IPIP300.por", "IPIP300.por.zip"],  # appears nested after unzip
    },
    # ---- ACTIVE: Cutler & Condon 2022 deep-lexical-hypothesis supplement ----
    "cutler_condon_2022": {
        "status": "active",
        "about": "Cutler & Condon (2022) 'Deep Lexical Hypothesis' (arXiv 2203.02092) "
                 "OSF deposit xm7hg. The 251 MB zip contains the S&G-1996 human "
                 "435-col x 899-row IPSATIZED adjective ratings + R analysis scripts "
                 "(the human matrix behind their Fig 1). PDF is the JPSP supplement.",
        "downloads": [
            {"url": "https://osf.io/download/9vkb7/",
             "out": "SupplementalMaterials_compressed.zip", "extract": "unzip"},
            {"url": "https://osf.io/download/pes25/",
             "out": "SupplementalMaterials_JPSP.pdf", "extract": None},
        ],
    },
    # ---- ACTIVE: ESCS public adjective ratings (theory-neutral re-clustering) ----
    # Eugene-Springfield Community Sample, self-ratings, Harvard Dataverse.
    # Independent of C&C (different sample); we control any ipsatization ourselves.
    "escs_525pda": {
        "status": "active",
        "about": "ESCS 525 Person-Descriptive Adjectives, self-ratings (Dataverse "
                 "doi:10.7910/DVN/GHYMEV). Public, ~742 KB .tab. NOTE: different "
                 "sample from C&C's S&G-1996 students; usable as an independent, "
                 "re-clusterable human adjective substrate.",
        "downloads": [
            {"url": dataverse("10.7910/DVN/GHYMEV"),
             "out": "escs_525pda_dataverse.zip", "extract": "unzip"},
        ],
    },
    "escs_360pda": {
        "status": "active",
        "about": "ESCS 360 Person-Descriptive Adjectives, self-ratings (Dataverse "
                 "doi:10.7910/DVN/ZNGS1K). Same family as 525-PDA, smaller item set.",
        "downloads": [
            {"url": dataverse("10.7910/DVN/ZNGS1K"),
             "out": "escs_360pda_dataverse.zip", "extract": "unzip"},
        ],
    },
    # ---- LEGACY: NeuroQuest mirror (dependency eliminated in favor of PA primary) ----
    "neuroquest_ipip": {
        "status": "legacy",
        "about": "NeuroQuestAi/ipip-neo-data GitHub LFS mirror of Johnson's "
                 "pre-scored IPIP-NEO-300 (N=145,388). Superseded by the "
                 "kajonius_johnson_2019 PsychArchives primary; kept for provenance.",
        "downloads": [
            {"url": "https://media.githubusercontent.com/media/NeuroQuestAi/"
                    "ipip-neo-data/main/datasets/refined/IPIP300-SCORES.csv.gz",
             "out": "IPIP300-SCORES.csv.gz", "extract": None},
        ],
    },
}


def curl(url, dest):
    """Download url -> dest with curl (-L follow redirects, -C - resume, --fail)."""
    print(f"    GET {url}")
    rc = subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-C", "-",
         "-o", str(dest), url],
    ).returncode
    if rc != 0:
        # -C - errors if the file is already complete; treat that as success.
        if dest.exists() and dest.stat().st_size > 0:
            print(f"    (curl rc={rc}; file present, continuing)")
            return
        raise RuntimeError(f"curl failed (rc={rc}) for {url}")


def extract(path, kind, dest_dir):
    if kind == "unzip":
        print(f"    unzip {path.name}")
        with zipfile.ZipFile(path) as z:
            z.extractall(dest_dir)
    elif kind == "gunzip":
        target = dest_dir / path.stem
        print(f"    gunzip {path.name} -> {target.name}")
        with gzip.open(path, "rb") as fi, open(target, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def fetch_source(key, spec, force):
    dest_dir = DATA_ROOT / key
    marker = dest_dir / ".fetched"
    if marker.exists() and not force:
        print(f"  [skip] {key} (already fetched; --force to redo)")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for dl in spec["downloads"]:
        out = dest_dir / dl["out"]
        if out.exists() and out.stat().st_size > 0 and not force:
            print(f"    have {out.name}")
        else:
            curl(dl["url"], out)
        if dl.get("extract"):
            extract(out, dl["extract"], dest_dir)
    missing = [v for v in spec.get("verify", []) if not list(dest_dir.rglob(v))]
    if missing:
        print(f"  WARN {key}: expected files not found after fetch: {missing}")
    marker.write_text("fetched by scripts/fetch_external_data.py\n")
    print(f"  [done] {key} -> {dest_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="fetch just these source keys")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also fetch sources marked legacy")
    ap.add_argument("--force", action="store_true", help="re-download/extract")
    args = ap.parse_args()

    if args.list:
        for k, s in SOURCES.items():
            print(f"\n[{s['status']:7}] {k}")
            print(f"    {s['about']}")
            for dl in s["downloads"]:
                print(f"    - {dl['out']}  ({dl['url']})")
        return

    if args.only:
        keys = [k for k in args.only if k in SOURCES]
        for k in args.only:
            if k not in SOURCES:
                print(f"unknown source: {k}", file=sys.stderr)
    else:
        keys = [k for k, s in SOURCES.items()
                if s["status"] == "active" or args.include_legacy]

    print(f"Fetching {len(keys)} source(s) into {DATA_ROOT}/: {', '.join(keys)}")
    for k in keys:
        print(f"\n== {k} ==")
        try:
            fetch_source(k, SOURCES[k], args.force)
        except Exception as e:
            print(f"  ERROR fetching {k}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
