#!/usr/bin/env python3
"""How load-bearing is "very" in the tom_likely asymmetry? (W16 §5)

The full run uses "Consider a person who is VERY {a}..." — conditioning on an
EXTREME, which injects its own base-rate/variance effects into the directional
asymmetry P(neg|pos) > P(pos|neg). rgb: the base-rate control (3) is more
indicative WITHOUT "very". So run the eval antonym block + base-rate marginals
under both forms and compare the raw and base-rate-controlled asymmetry.

Base rate of a target trait B = mean P(B|A) over a broad anchor sample (eval words
+ N random anchors). Residual asymmetry = [P(neg|pos)-base(neg)] - [P(pos|neg)-base(pos)].

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_very_test.py --models Qwen7 Gemma12
"""
import argparse

import numpy as np

from hf_logprobs import load_model
from adjective_introspection import GROUPS
from adjective_judge_full import load_adjectives, digit_token_ids, batch_ev_entropy

FORMS = {
    "very":  ("Consider a person who is very {a}.\n"
              "How likely is this person to also be {b}?\n"
              "Answer with one number from 1 to 7, where 1 = very unlikely and "
              "7 = very likely.\nNumber:"),
    "plain": ("Consider a person who is {a}.\n"
              "How likely is this person to also be {b}?\n"
              "Answer with one number from 1 to 7, where 1 = very unlikely and "
              "7 = very likely.\nNumber:"),
}


def run(model, tok, device, tmpl, pairs, words, digit_ids, bs=48):
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": tmpl.format(a=words[i].lower(), b=words[j].lower())}],
        tokenize=False, add_generation_prompt=True) for i, j in pairs]
    ev = []
    for s in range(0, len(texts), bs):
        e, _ = batch_ev_entropy(model, tok, device, texts[s:s + bs], digit_ids)
        ev.append(e)
    return np.concatenate(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen7", "Gemma12"])
    ap.add_argument("--n-anchor", type=int, default=40, help="random anchors for base rate")
    args = ap.parse_args()
    full = load_adjectives()
    pos = [full.index(w) for w in GROUPS["pos_eval"]]
    neg = [full.index(w) for w in GROUPS["neg_eval"]]
    evalset = set(pos + neg)
    rng = np.random.default_rng(0)
    cand = [i for i in range(len(full)) if i not in evalset]
    anchors = list(rng.choice(cand, size=args.n_anchor, replace=False))
    targets = pos + neg                      # the 8 eval words as targets
    all_anchors = pos + neg + anchors        # for base-rate marginals

    # ordered pairs (anchor a, target b), a != b
    pairs = [(a, b) for a in all_anchors for b in targets if a != b]

    for short in args.models:
        model, tok, device = load_model(short)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        digit_ids = digit_token_ids(tok)
        print(f"\n===== {short} =====")
        print(f"{'form':<7}{'P(neg|pos)':>11}{'P(pos|neg)':>11}{'raw gap':>9}"
              f"{'base[neg]':>10}{'base[pos]':>10}{'resid gap':>11}")
        for name, tmpl in FORMS.items():
            ev = run(model, tok, device, tmpl, pairs, full, digit_ids)
            V = {(a, b): e for (a, b), e in zip(pairs, ev)}
            pn = np.mean([V[(a, b)] for a in pos for b in neg])      # P(neg | pos-anchor)
            np_ = np.mean([V[(a, b)] for a in neg for b in pos])     # P(pos | neg-anchor)
            base = {b: np.mean([V[(a, b)] for a in all_anchors if a != b]) for b in targets}
            bneg = np.mean([base[b] for b in neg]); bpos = np.mean([base[b] for b in pos])
            rpn = pn - bneg; rnp = np_ - bpos
            print(f"{name:<7}{pn:>11.2f}{np_:>11.2f}{pn-np_:>+9.2f}"
                  f"{bneg:>10.2f}{bpos:>10.2f}{rpn-rnp:>+11.2f}")
        import gc, torch
        del model, tok; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
