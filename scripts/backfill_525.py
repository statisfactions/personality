"""525 backfill driver (2026-08-28): append the two reinstated adjectives
(Inspirational/Insensitive) to every extraction still at 523, in cost order
SELF -> REPRESENT -> ENACT -> JUDGE. Each step only touches files that lack
the adjectives; safe to re-run. Logs to tmp/logs/backfill_525.log.

Usage: PYTHONPATH=scripts python scripts/backfill_525.py [--steps self represent enact judge]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

import hf_logprobs as hf

MISSING = {"inspirational", "insensitive"}
PY = [sys.executable]
ENV = dict(os.environ, PYTHONPATH="scripts")


def run(cmd, tag):
    print(f"[{tag}] {' '.join(cmd)}", flush=True)
    rc = subprocess.call(["caffeinate", "-i"] + cmd, env=ENV)
    print(f"[{tag}] rc={rc}", flush=True)
    return rc


def step_self():
    for p in sorted(glob.glob("results/adjectives/selfreport/*_self_full*.json")):
        if "ARTIFACT" in p or "PRERETRY" in p or "thinkmc" in p:
            continue
        d = json.load(open(p))
        adj = {a.lower() for a in d.get("adjectives", [])} or \
              {a.lower() for a in next(iter(d["results"].values()))}
        if MISSING <= adj:
            continue
        base = os.path.basename(p)
        think = base.endswith("_self_full_think.json")
        name = base.replace("_self_full_think.json", "").replace("_self_full.json", "")
        model = name if name in hf.MODELS else name.replace("_", "/", 1)
        run(PY + ["scripts/self_adjective_report.py", "--model", model, "--full",
                  "--backfill"] + (["--think"] if think else []), f"self {name}")


def step_represent():
    for p in sorted(glob.glob("results/adjectives/acts/*__pers.pt")):
        b = torch.load(p, map_location="cpu", weights_only=False)
        if MISSING <= {str(a).lower() for a in b["adjectives"]}:
            continue
        repo = os.path.basename(p).replace("__pers.pt", "").replace("_", "/", 1)
        run(PY + ["scripts/extract_adjectives.py", "--models", repo, "--backfill"],
            f"represent {repo}")


def step_enact():
    for p in sorted(glob.glob("results/persona_vectors/*_pda_texts.json")):
        short = os.path.basename(p).replace("_pda_texts.json", "")
        if MISSING <= {k.lower() for k in json.load(open(p))}:
            continue
        # separate tag so the 2-adjective run cannot overwrite the full
        # aggregate; then move its checkpoints into the pda ckpt dir and
        # rebuild the aggregate model-free
        run(PY + ["scripts/extract_persona_vectors.py", "--model", short, "--pda",
                  "--adjectives", *sorted(MISSING), "--tag", "pda_backfill",
                  "--no-save-acts"], f"enact {short}")
        src = f"results/persona_vectors/{short}_pda_backfill_ckpt"
        dst = f"results/persona_vectors/{short}_pda_ckpt"
        moved = 0
        for a in sorted(MISSING):
            f = f"{src}/{a}.pt"
            if os.path.exists(f) and os.path.isdir(dst):
                shutil.copy2(f, f"{dst}/{a}.pt"); moved += 1
        print(f"[enact {short}] moved {moved} checkpoints -> {dst}", flush=True)
        if moved:
            run(PY + ["scripts/finalize_from_checkpoints.py", "--model", short],
                f"finalize {short}")


def step_judge():
    for p in sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz")):
        z = np.load(p, allow_pickle=True)
        if MISSING <= {str(a).lower() for a in z["adjectives"]}:
            continue
        short = os.path.basename(p).replace("_tom_likely_dir.npz", "")
        run(PY + ["scripts/adjective_judge_full.py", "--models", short,
                  "--both-directions", "--mode", "tom_likely", "--backfill"],
            f"judge {short}")


STEPS = {"self": step_self, "represent": step_represent,
         "enact": step_enact, "judge": step_judge}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    for s in args.steps:
        print(f"===== {s} =====", flush=True)
        STEPS[s]()
    print("BACKFILL 525 DONE", flush=True)
