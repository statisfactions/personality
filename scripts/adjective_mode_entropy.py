#!/usr/bin/env python3
"""Is the ToM judgment less single-pole than semantic? (W16 §1 cont.)

The full-pair SEMANTIC run is 79% single-pole (entropy ~0): unrelated words are a
confident "completely different". rgb's hypothesis: the ToM framing ("a person who
is very A — how accurately does B describe them?") has no reason to floor on
unrelated traits — uncorrelated != anti-correlated, so the model should spread its
mass. The floor may be a property of the meaning-similarity frame, not the model.

Cheap test: sample K random pairs from the full 523 (mostly unrelated, where the
floor lives), run BOTH modes (one direction, with entropy), compare the single-pole
fraction and entropy/EV distributions. Doesn't need the full 80-min ToM matrix.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_mode_entropy.py --model Qwen7 --k 600
"""
import argparse

import numpy as np

from hf_logprobs import load_model
from adjective_introspection import PROMPT
from adjective_judge_full import load_adjectives, digit_token_ids, batch_ev_entropy
from adjective_judge_symmetry import sample_pairs


def run_mode(model, tok, device, adj, pairs, mode, digit_ids, bs=48):
    texts = []
    for i, j in pairs:
        msgs = [{"role": "user",
                 "content": PROMPT[mode].format(a=adj[i].lower(), b=adj[j].lower())}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    ev, H = [], []
    for s in range(0, len(texts), bs):
        e, h = batch_ev_entropy(model, tok, device, texts[s:s + bs], digit_ids)
        ev.append(e); H.append(h)
    return np.concatenate(ev), np.concatenate(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--k", type=int, default=600)
    args = ap.parse_args()
    adj = load_adjectives(); n = len(adj)
    pairs = sample_pairs(n, args.k)
    print(f"{args.model}: {args.k} random pairs from {n} adjectives, semantic vs ToM\n")

    model, tok, device = load_model(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    digit_ids = digit_token_ids(tok)

    print(f"{'mode':<9}{'mean H':>8}{'med H':>8}{'%single-pole':>14}{'%H>1':>7}"
          f"{'mean EV':>9}{'%EV-floor':>11}{'r(H,EV)':>9}")
    for mode in ("semantic", "tom"):
        ev, H = run_mode(model, tok, device, adj, pairs, mode, digit_ids)
        floor = np.mean(ev <= ev.min() + 0.1)
        print(f"{mode:<9}{H.mean():>8.3f}{np.median(H):>8.3f}{100*np.mean(H<0.01):>13.0f}%"
              f"{100*np.mean(H>1.0):>6.0f}%{ev.mean():>9.2f}{100*floor:>10.0f}%"
              f"{np.corrcoef(H, ev)[0,1]:>+9.3f}")


if __name__ == "__main__":
    main()
