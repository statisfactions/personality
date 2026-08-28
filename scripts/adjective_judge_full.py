#!/usr/bin/env python3
"""Full 523x523 pairwise JUDGMENT matrix over the adjective set (W16 §1 follow-up).

W16's regress z-scored the eval-antonym block against the 26-word probe set — a
curated, evaluation-enriched yardstick (rgb's concern). This computes the model's
JUDGED similarity for ALL pairs of the full 523 525-PDA adjectives, so the antonym
block can be z-scored against the *full empirical distribution* of personality-
adjective pairs, and so the behavioral (write) side is available at the same
523-adjective resolution as the representation (read) side (cached acts).

Same elicitation as W15 introspection, semantic mode, valence-neutral anchor:
  "How similar in meaning are these two words, as descriptions of a person? ...
   1 = completely different in meaning, 7 = nearly the same. Number:"
read as the EV of the 1-7 digit distribution at the final token, chat-templated.

Pragmatics for the 523-word scale:
  - UPPER TRIANGLE only (i<j), mirrored — 136,503 pairs not 273,006. The W15
    directional asymmetry was a minor side-finding and the 26-word matrix was
    symmetrized anyway. (--both-directions to override.)
  - BATCHED left-padded forward passes (the single-prompt helper re-templates and
    re-encodes per call — far too slow here).
  - RESUMABLE: checkpoints the partial matrix + a progress pointer every
    --ckpt-batches batches, so a sleep/interrupt continues cleanly.

Output: results/adjectives/introspect_full/<short>.npz
  {B: (n,n) symmetrized EV, Hent: (n,n) entropy, adjectives, done, n_pairs}
plus a small <short>.json with the headline stats once complete.

Usage:
  smoke: PYTHONPATH=scripts .venv/bin/python scripts/adjective_judge_full.py \
             --models Qwen7 --limit 40
  full : PYTHONPATH=scripts .venv/bin/python scripts/adjective_judge_full.py \
             --models Qwen7 Gemma12
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from hf_logprobs import MODELS, load_model
from adjective_introspection import PROMPT, GROUPS

OUT_DIR = "results/adjectives/introspect_full"
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
DIGIT_VAL = np.array([int(d) for d in DIGITS], float)


def load_adjectives():
    H = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    return list(H["labels"])


def digit_token_ids(tok):
    """Per-digit list of single-token ids among {d, ' '+d}. Mirrors hf_logprobs."""
    out = []
    for d in DIGITS:
        ids = []
        for c in (d, " " + d):
            enc = tok(c, add_special_tokens=False).input_ids
            if len(enc) == 1:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"digit {d!r} not single-token")
        out.append(ids)
    return out  # list[len 7] of list[ids]


def batch_ev_entropy(model, tok, device, texts, digit_ids):
    """For a batch of chat-templated prompt strings, return (ev[], entropy[]).

    Left-padded so the final real token is at position -1 for every row. Digit
    distribution = softmax over the selected digit-token logits (variant ids
    summed per digit), renormalized within the 7 digits — matching the
    single-prompt helper's _prob_per_label."""
    enc = tok(texts, return_tensors="pt", add_special_tokens=False,
              padding=True).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    logits = out.logits[:, -1, :].float()             # (B, vocab); left-pad => -1 is last real
    # gather selected logits, summing variant probs per digit
    flat_ids, owner = [], []
    for di, ids in enumerate(digit_ids):
        for tid in ids:
            flat_ids.append(tid); owner.append(di)
    sel = logits[:, torch.tensor(flat_ids, device=logits.device)]      # (B, n_sel)
    probs = torch.softmax(sel, dim=-1)                                  # renorm within selected
    # sum variant columns into 7 digit buckets
    P = torch.zeros((logits.shape[0], len(DIGITS)), device=logits.device)
    owner_t = torch.tensor(owner, device=logits.device)
    P.index_add_(1, owner_t, probs)
    Pn = P.cpu().numpy()
    ev = Pn @ DIGIT_VAL
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(np.where(Pn > 0, Pn * np.log(Pn), 0.0), axis=1)
    return ev, H


def run_model(short, both_dir, batch_size, ckpt_batches, limit, mode,
              backfill=False):
    adjectives = load_adjectives()
    if limit:
        adjectives = adjectives[:limit]
    n = len(adjectives)
    pairs = ([(i, j) for i in range(n) for j in range(n) if i != j] if both_dir
             else [(i, j) for i in range(n) for j in range(i + 1, n)])
    os.makedirs(OUT_DIR, exist_ok=True)
    mtag = "" if mode == "semantic" else f"_{mode}"   # keep semantic runs unsuffixed
    path = f"{OUT_DIR}/{short}{mtag}{'_dir' if both_dir else ''}{'_smoke' if limit else ''}.npz"

    B = np.full((n, n), np.nan, np.float32)
    Hent = np.full((n, n), np.nan, np.float32)
    done = 0
    full_pairs = pairs
    if os.path.exists(path) and not limit:
        z = np.load(path, allow_pickle=True)
        old_adj = [str(a) for a in z["adjectives"]]
        if old_adj == adjectives and int(z["done"]) <= len(pairs):
            B, Hent, done = z["B"].copy(), z["Hent"].copy(), int(z["done"])
            print(f"[resume] {short}: {done}/{len(pairs)} pairs done")
        elif backfill and set(old_adj) < set(adjectives):
            # 525 extension: keep every old cell, run only pairs touching
            # the new adjectives (both directions), then mark complete
            pos = {a: adjectives.index(a) for a in old_adj}
            oi = [pos[a] for a in old_adj]
            B[np.ix_(oi, oi)] = z["B"]; Hent[np.ix_(oi, oi)] = z["Hent"]
            new_idx = {i for i, a in enumerate(adjectives) if a not in pos}
            pairs = [(i, j) for (i, j) in pairs if i in new_idx or j in new_idx]
            print(f"[backfill] {short}: +{len(new_idx)} adjectives -> "
                  f"{len(pairs)} pairs")
        else:
            print(f"[restart] {short}: checkpoint mismatch, starting over")

    if done >= len(pairs):
        print(f"[skip] {short}: already complete")
        return path, adjectives, B, Hent

    model, tok, device = load_model(short)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    digit_ids = digit_token_ids(tok)

    # Qwen3-family templates default to reasoning-on; that emits <think> before
    # the answer and destroys digit-mass at the final token. Force it off.
    ct_kw = {"enable_thinking": False} if short.startswith("Qwen3") else {}

    def templ(i, j):
        msgs = [{"role": "user", "content":
                 PROMPT[mode].format(a=adjectives[i].lower(),
                                     b=adjectives[j].lower())}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, **ct_kw)

    t0 = time.time(); nb = 0
    for start in range(done, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        texts = [templ(i, j) for i, j in chunk]
        ev, H = batch_ev_entropy(model, tok, device, texts, digit_ids)
        for (i, j), e, h in zip(chunk, ev, H):
            B[i, j] = e; Hent[i, j] = h
            if not both_dir:
                B[j, i] = e; Hent[j, i] = h
        done = start + len(chunk); nb += 1
        if nb % ckpt_batches == 0 or done >= len(pairs):
            np.savez(path, B=B, Hent=Hent, adjectives=np.array(adjectives),
                     done=done, n_pairs=len(pairs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(pairs) - done) / max(rate, 1e-9)
            print(f"  {short}: {done}/{len(pairs)} ({100*done/len(pairs):.1f}%) "
                  f"{rate:.0f} pair/s  eta {eta/60:.0f} min", flush=True)

    import gc
    del model, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if pairs is not full_pairs:   # backfill: mark complete for the full list
        np.savez(path, B=B, Hent=Hent, adjectives=np.array(adjectives),
                 done=len(full_pairs), n_pairs=len(full_pairs))

    return path, adjectives, B, Hent


def headline(short, adjectives, B):
    """z-score the eval-antonym block against the FULL distribution, vs the
    26-word-set baseline — the whole point of going to full resolution."""
    idx = {w: k for k, w in enumerate(adjectives)}
    pos = [idx[w] for w in GROUPS["pos_eval"] if w in idx]
    neg = [idx[w] for w in GROUPS["neg_eval"] if w in idx]
    off = ~np.eye(len(adjectives), dtype=bool)        # all off-diagonal (both-dir safe)
    full = B[off]; full = full[~np.isnan(full)]
    m, s = np.mean(full), np.std(full)
    # both antonym directions (identical for symmetric runs, distinct for both-dir)
    ant = float(np.nanmean(np.concatenate([B[np.ix_(pos, neg)].ravel(),
                                           B[np.ix_(neg, pos)].ravel()])))
    z_full = (ant - m) / s
    # 26-word-set baseline: restrict the reference distribution to the probe set
    probe = [idx[w] for ws in GROUPS.values() for w in ws if w in idx]
    sub = B[np.ix_(probe, probe)]
    pm = ~np.eye(len(probe), dtype=bool)
    subv = sub[pm]; subv = subv[~np.isnan(subv)]
    m2, s2 = np.mean(subv), np.std(subv)
    z_probe = (ant - m2) / s2
    return {"antonym_ev": float(ant), "full_mean": float(m), "full_std": float(s),
            "antonym_z_full": float(z_full), "antonym_z_probeset": float(z_probe)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen7", "Gemma12"])
    ap.add_argument("--both-directions", action="store_true")
    ap.add_argument("--mode", default="semantic", choices=list(PROMPT),
                    help="question form (semantic | tom | tom_likely)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ckpt-batches", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="first N adjectives (smoke)")
    ap.add_argument("--backfill", action="store_true",
                    help="extend existing matrices to the 525 list: run only "
                         "pairs touching adjectives the file lacks")
    args = ap.parse_args()
    for short in args.models:
        if short not in MODELS:
            print(f"skip unknown {short}"); continue
        print(f"\n===== {short} ({MODELS[short]}) mode={args.mode} "
              f"both_dir={args.both_directions} =====")
        path, adj, B, Hent = run_model(short, args.both_directions,
                                       args.batch_size, args.ckpt_batches,
                                       args.limit, args.mode, backfill=args.backfill)
        if not args.limit:
            stats = headline(short, adj, B)
            stats["mean_entropy"] = float(np.nanmean(Hent))
            json.dump(stats, open(path.replace(".npz", ".json"), "w"), indent=2)
            print(f"  antonym EV {stats['antonym_ev']:.2f}  "
                  f"z_full {stats['antonym_z_full']:+.2f}  "
                  f"z_probeset {stats['antonym_z_probeset']:+.2f}  "
                  f"(full mean {stats['full_mean']:.2f} sd {stats['full_std']:.2f})")
    print("\ndone")


if __name__ == "__main__":
    main()
