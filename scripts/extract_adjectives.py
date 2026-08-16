#!/usr/bin/env python3
"""Extract LLM representations of the 523 clean 525-PDA adjectives.

Analogue of the IPIP item extraction, for single adjectives. For each model and
each framing we run a carrier sentence ENDING in the adjective and read the
hidden states of the adjective span only (via split_prefix) — so the adjective's
representation carries the carrier's personality context through causal
attention, without being diluted by the (identical-across-adjectives) carrier
tokens. All layers cached so analysis can sweep depth later.

The open question this addresses: how much does "this is personality" framing
matter for single words? Three framings, read-span held fixed at the adjective:
  self : "I am {adj}"             (first-person, matches IPIP)
  pers : "My personality is {adj}" (explicit personality label)
  desc : "Someone who is {adj}"   (third-person descriptor; encoder-style)

Idempotent: one cache file per (model, framing); existing files are skipped, so
the job is resumable. Models run small->large so early results land first.

Cache: results/adjectives/acts/<safe_model>__<framing>.pt
  {acts: float32 (n_adj, n_layers+1, hidden), adjectives: [...], framing, template}
  (fp32, not fp16 — Gemma-3 massive activations overflow fp16 to inf.)

Usage:
  smoke:  PYTHONPATH=scripts .venv/bin/python scripts/extract_adjectives.py --models Qwen --limit 5
  full :  PYTHONPATH=scripts .venv/bin/python scripts/extract_adjectives.py
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pyreadstat
import torch

import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS as ALL_MODELS, load_model
from adjective_corr_cluster import POR, DENY_LABELS

OUT_DIR = Path("results/adjectives/acts")

# (template, split_prefix) — split_prefix is everything before {adj}, so the
# read span is the adjective (+ trailing tokens). Carrier ends with the adjective.
FRAMINGS = {
    "self": ("I am {adj}", "I am"),
    "pers": ("My personality is {adj}", "My personality is"),
    "desc": ("Someone who is {adj}", "Someone who is"),
    # No-context floor: bare word, no carrier. split_prefix=None means the read
    # span is the whole chat-wrapped word (incl. the constant template tokens,
    # removable by itempc1) — so bare differs from the others in read-span as
    # well as context; treat it as the floor, not a matched-span condition.
    "bare": ("{adj}", None),
}

# small -> large so early results land first
COHORT = ["Qwen", "Llama", "Phi4", "Gemma", "FalconMamba", "Qwen7", "Aya",
          "Llama8", "Gemma12", "Gemma27", "Gemma4", "Qwen32"]


def safe(s):
    return s.replace("/", "_")


def load_adjectives():
    df, meta = pyreadstat.read_por(POR)
    deny = {c for c in df.columns
            if (meta.column_names_to_labels.get(c) or c) in DENY_LABELS}
    cols = [c for c in df.columns if c.upper() != "ID" and c not in deny]
    labels = [meta.column_names_to_labels.get(c) or c for c in cols]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=COHORT)
    ap.add_argument("--framings", nargs="+", default=list(FRAMINGS),
                    choices=list(FRAMINGS))
    ap.add_argument("--limit", type=int, default=None,
                    help="only first N adjectives (smoke test)")
    ap.add_argument("--bare", action="store_true",
                    help="no chat template (base/staged ckpts w/o one); '_bare' suffix")
    args = ap.parse_args()
    suf = "_bare" if args.bare else ""

    adjectives = load_adjectives()
    if args.limit:
        adjectives = adjectives[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(adjectives)} adjectives x {len(args.framings)} framings x "
          f"{len(args.models)} models")

    for model_name in args.models:
        repo = ALL_MODELS.get(model_name, model_name)  # unknown = raw HF repo
        # Skip model entirely if all its framing caches exist.
        todo = [f for f in args.framings
                if not (OUT_DIR / f"{safe(repo)}__{f}{suf}.pt").exists()
                or args.limit]
        if not todo:
            print(f"[skip] {model_name}: all framings cached")
            continue

        print(f"\n===== {model_name} ({repo}) =====")
        model, tok, device = load_model(model_name)
        for framing in todo:
            template, prefix = FRAMINGS[framing]
            out = OUT_DIR / f"{safe(repo)}__{framing}{suf}.pt"
            if out.exists() and not args.limit:
                print(f"  [have] {framing}")
                continue
            print(f"  {framing}: {template!r} (read span after {prefix!r}), "
                  f"chat_template={not args.bare}")
            acts = []
            for adj in adjectives:
                text = template.format(adj=adj.lower())
                a = mdx.hidden_states_for_text(
                    model, tok, text, device,
                    split_prefix=prefix, chat_template=not args.bare)
                # fp32 (already returned by hidden_states_for_text). NOT fp16:
                # Gemma-3's massive activations (~1e5, §3.10) overflow fp16's
                # 65504 ceiling to inf, corrupting the whole cache.
                acts.append(a.numpy())
            acts = np.stack(acts)  # (n_adj, n_layers+1, hidden)
            torch.save({"acts": acts, "adjectives": adjectives,
                        "framing": framing, "template": template,
                        "model": repo}, out)
            print(f"    saved {acts.shape} -> {out}")
        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print("\ndone")


if __name__ == "__main__":
    main()
