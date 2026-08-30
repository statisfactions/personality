"""Digit-mass audit of the prefill SELF arm across the wide cohort (2026-08-29).

For every model with a SELF file, run the smoke set x 'direct' framing and
record the total probability on digit tokens at the read position, plus the
generation-fallback read where mass < MASS_FLOOR. Answers rgb's worry:
how often is the renormalized prefill dist a sliver of non-answer mass?
Skips models whose snapshot isn't cached (no downloads).

Usage: PYTHONPATH=scripts python scripts/digit_mass_audit.py
Out:   results/adjectives/digit_mass_audit.json (+ stdout table)
"""
import glob
import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download

import facet_slides_wide as fw
import hf_logprobs as hf
from self_adjective_report import (AGREE_SCALE, DIGITS, FRAMINGS, MASS_FLOOR,
                                   SMOKE_ADJS, ev_of, think_distribution)

OUT = "results/adjectives/digit_mass_audit.json"


def cached(repo):
    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def main():
    from extract_adjectives import load_adjectives
    allset = {a.lower() for a in load_adjectives()}
    adjs = [a for a in SMOKE_ADJS if a in allset]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    repos = set()
    for p in glob.glob("results/adjectives/selfreport/*_self_full.json"):
        name = os.path.basename(p).replace("_self_full.json", "")
        if fw.EXCLUDE.search(name):
            continue
        repos.add(hf.resolve(name) if name in hf.MODELS else name.replace("_", "/", 1))
    for repo in sorted(repos):
        if repo in done or not cached(repo):
            continue
        try:
            model, tok, device = hf.load_model(repo, dtype=torch.bfloat16)
        except Exception as e:
            done[repo] = {"error": type(e).__name__}
            continue
        masses, low, gen_ok, gen_ev_shift = [], 0, 0, []
        for a in adjs:
            prompt = AGREE_SCALE.format(statement=FRAMINGS["direct"].format(adj=a))
            dist, _, _, mass = hf.likert_distribution(
                model, tok, prompt, device, digits=DIGITS,
                use_chat_template=tok.chat_template is not None, return_mass=True)
            masses.append(mass)
            if mass < MASS_FLOOR:
                low += 1
                d2, _, _, _, _ = think_distribution(model, tok, prompt, device,
                                                    max_new=16, enable_thinking=False,
                                                    model_name=repo)
                if d2:
                    gen_ok += 1
                    gen_ev_shift.append(ev_of(d2) - ev_of(dist))
        done[repo] = {"median_mass": float(np.median(masses)),
                      "p10_mass": float(np.percentile(masses, 10)),
                      "frac_below_floor": low / len(adjs),
                      "fallback_parsed": gen_ok / max(low, 1),
                      "mean_ev_shift_on_fallback": (float(np.mean(gen_ev_shift))
                                                    if gen_ev_shift else None)}
        print(f"{repo:48s} median mass {done[repo]['median_mass']:.2f}  p10 {done[repo]['p10_mass']:.2f}  "
              f"below floor {100*done[repo]['frac_below_floor']:.0f}%  fallback parsed "
              f"{100*done[repo]['fallback_parsed']:.0f}%", flush=True)
        json.dump(done, open(OUT, "w"), indent=1)
        del model, tok
        import gc; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
