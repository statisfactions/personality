#!/usr/bin/env python3
"""Is the JUDGED similarity symmetric? — cross-check before the full run (W16 §1).

Before committing the 523x523 full-pair judgment to UPPER-TRIANGLE-ONLY (which
assumes judge(a,b) == judge(b,a)), verify the assumption. rgb's prediction: the
semantic anchor ("how similar in meaning are a and b") should be near-symmetric
(similarity is a symmetric relation), whereas the ToM anchor ("consider a person
who is very a; how well does b describe them") is intrinsically DIRECTIONAL and
should show real asymmetry.

Samples K random pairs from the full 523 adjectives, runs BOTH directions for
both modes (batched), and reports:
  - bias            : mean(fwd - rev)            (systematic order effect)
  - |asymmetry|     : mean|fwd - rev|            (magnitude, on the 1-7 scale)
  - asym / spread   : |asymmetry| / std(all EVs) (is it big vs pair-to-pair range?)
  - r(fwd, rev)     : Pearson                    (1.0 = perfectly symmetric)
Plus, for context, the 26-word ToM cached directional asymmetry (zero-cost).

If semantic |asymmetry| is small vs the EV spread and r ~ 1, upper-triangle is
justified for the full run; otherwise pass --both-directions there.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_judge_symmetry.py \
           --model Qwen7 --k 400
"""
import argparse
import json

import numpy as np

from hf_logprobs import load_model
from adjective_introspection import PROMPT
from adjective_judge_full import load_adjectives, digit_token_ids, batch_ev_entropy


def sample_pairs(n, k, seed=0):
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < k:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and (i, j) not in seen:
            seen.add((i, j)); out.append((i, j))
    return out


def both_dir_ev(model, tok, device, adj, pairs, mode, digit_ids, bs=48):
    """Return fwd[], rev[] EV arrays for the sampled pairs in `mode`."""
    def texts_for(order):
        out = []
        for i, j in pairs:
            a, b = (adj[i], adj[j]) if order == "fwd" else (adj[j], adj[i])
            msgs = [{"role": "user",
                     "content": PROMPT[mode].format(a=a.lower(), b=b.lower())}]
            out.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        return out

    res = {}
    for order in ("fwd", "rev"):
        texts = texts_for(order)
        ev = np.concatenate([batch_ev_entropy(model, tok, device, texts[s:s + bs],
                                              digit_ids)[0]
                             for s in range(0, len(texts), bs)])
        res[order] = ev
    return res["fwd"], res["rev"]


def report(mode, fwd, rev):
    asym = fwd - rev
    spread = np.std(np.concatenate([fwd, rev]))
    r = float(np.corrcoef(fwd, rev)[0, 1])
    print(f"  {mode:<9} bias {np.mean(asym):+.3f}  |asym| {np.mean(np.abs(asym)):.3f}"
          f"  (= {np.mean(np.abs(asym))/spread:.2f}× EV-spread {spread:.2f})  "
          f"r(fwd,rev) {r:.3f}")
    return {"mode": mode, "bias": float(np.mean(asym)),
            "mean_abs_asym": float(np.mean(np.abs(asym))),
            "ev_spread": float(spread), "r_fwd_rev": r}


def cached_tom_asym(short):
    """26-word ToM cached directional |B - B.T| off-diagonal mean (zero-cost)."""
    try:
        d = json.load(open(f"results/adjectives/introspect_tom/{short}.json"))
    except FileNotFoundError:
        return None
    B = np.array(d["directional"], float)
    off = ~np.eye(B.shape[0], dtype=bool)
    return float(np.nanmean(np.abs(B - B.T)[off]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--k", type=int, default=400)
    args = ap.parse_args()

    adj = load_adjectives(); n = len(adj)
    pairs = sample_pairs(n, args.k)
    print(f"{args.model}: {args.k} random pairs from {n} adjectives, both directions\n")

    model, tok, device = load_model(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    digit_ids = digit_token_ids(tok)

    rows = []
    for mode in ("semantic", "tom"):
        fwd, rev = both_dir_ev(model, tok, device, adj, pairs, mode, digit_ids)
        rows.append(report(mode, fwd, rev))

    tc = cached_tom_asym(args.model)
    print(f"\n  [context] 26-word ToM cached |B-B.T| off-diag mean: "
          f"{tc:.3f}" if tc is not None else "  [no cached ToM]")

    sem = next(r for r in rows if r["mode"] == "semantic")
    tom = next(r for r in rows if r["mode"] == "tom")
    print(f"\n  verdict: semantic |asym| {sem['mean_abs_asym']:.3f} vs ToM "
          f"{tom['mean_abs_asym']:.3f} "
          f"({'semantic MORE symmetric, as predicted' if sem['mean_abs_asym'] < tom['mean_abs_asym'] else 'NOT more symmetric — investigate'})")
    print(f"  -> upper-triangle for the full run is "
          f"{'JUSTIFIED' if sem['r_fwd_rev'] > 0.97 and sem['mean_abs_asym']/sem['ev_spread'] < 0.25 else 'QUESTIONABLE — consider --both-directions'}")

    json.dump({"model": args.model, "k": args.k, "modes": rows,
               "tom_cached_26word_asym": tc},
              open(f"results/adjectives/introspect_full/symmetry_{args.model}.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
