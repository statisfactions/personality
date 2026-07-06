"""Is raw-REPRESENT steerability prompt-dependent? (W17 §14)

qwen2.5's raw read vectors steer conduct (~enact potency); llama3.2's barely
steer (~1/4, topic-not-conduct). All prior repr vectors came from the 'pers'
framing ("My personality is {adj}"). Here we steer with raw centered repr
vectors from all four extraction framings (pers / self / desc / bare) on the
same 12 clean held-out adjectives, same dose, same judge protocol. Offline
geometry predicts framing-INVARIANCE (ENACT-in-span moves <5 points across
framings within a model).

Runs generate -> judge -> analyze in one process (gen model freed before the
judge loads). Usage (PYTHONPATH=scripts):
  python scripts/steer_repr_framings.py --model llama3.2 --judge Qwen7
  python scripts/steer_repr_framings.py --model qwen2.5 --judge Llama8
"""
import argparse
import gc
import json

import numpy as np
import torch

import hf_logprobs as hf
from steer_mapped_vs_recorded import (EVAL_QUESTIONS, N_STRATUM, OUT_DIR,
                                      Steerer, _judge_ev, _seed, _vec_path,
                                      generate_one, residual_norm)

FRAMINGS = ["pers", "self", "desc", "bare"]


def build_vectors(model_name):
    """Raw centered repr vector per (framing, clean test adjective), massive
    dims zeroed (matching the original repr steering condition)."""
    z = np.load(_vec_path(model_name), allow_pickle=True)
    adjs = [str(a) for a in z["adjs"]][N_STRATUM:]      # clean stratum
    mid, massive = int(z["mid"]), list(z["massive"])
    repo = hf.resolve(model_name).replace("/", "_")
    vecs = {}
    for fr in FRAMINGS:
        dr = torch.load(f"results/adjectives/acts/{repo}__{fr}.pt",
                        map_location="cpu", weights_only=False)
        adjR = [str(a).lower() for a in dr["adjectives"]]
        R = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
        R[:, massive] = 0.0
        Rc = R - R.mean(0)
        for a in adjs:
            v = Rc[adjR.index(a)]
            vecs[(fr, a)] = (v / np.linalg.norm(v)).astype(np.float32)
    return adjs, vecs, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--rollouts", type=int, default=2)
    args = ap.parse_args()

    adjs, vecs, mid = build_vectors(args.model)
    out_path = (f"{OUT_DIR}/{args.model}_framings_"
                f"f{int(round(args.frac * 100)):02d}.json")

    # --- generate ---
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    rn = residual_norm(model, tok, device, mid)
    alpha = args.frac * rn
    print(f"alpha = {args.frac} * {rn:.1f} = {alpha:.2f}")
    steer = Steerer(model, mid, device, torch.bfloat16)
    gens = []
    for adj in adjs:
        for fr in FRAMINGS:
            steer.set(vecs[(fr, adj)], alpha)
            for qi, q in enumerate(EVAL_QUESTIONS):
                for r in range(args.rollouts):
                    gens.append({"adj": adj, "cond": fr, "q": qi,
                                 "rollout": r,
                                 "text": generate_one(model, tok, device, q,
                                                      _seed(adj, fr, qi, r))})
        print(f"  {adj} done ({len(gens)} gens)", flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    steer.set(None, 0.0)
    for qi, q in enumerate(EVAL_QUESTIONS):
        for r in range(args.rollouts):
            gens.append({"adj": None, "cond": "baseline", "q": qi,
                         "rollout": r,
                         "text": generate_one(model, tok, device, q,
                                              _seed("__baseline__", qi, r))})
    steer.remove()
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # --- judge ---
    jm, jtok, jdev = hf.load_model(args.judge, dtype=torch.bfloat16)

    def target_prompt(q, text, adj):
        return (f'Someone was asked: "{q}"\n\nThey replied:\n"""{text}"""\n\n'
                f"On a scale from 1 to 7, how {adj} does this person seem? "
                f"Answer with a single digit from 1 to 7.")

    def fluency_prompt(q, text):
        return (f'Someone was asked: "{q}"\n\nThey replied:\n"""{text}"""\n\n'
                f"On a scale from 1 (garbled nonsense) to 7 (perfectly clear "
                f"and coherent), how coherent is this reply? "
                f"Answer with a single digit from 1 to 7.")

    for i, g in enumerate(gens):
        q = EVAL_QUESTIONS[g["q"]]
        g["fluency"] = _judge_ev(jm, jtok, jdev, fluency_prompt(q, g["text"]))
        if g["adj"]:
            g["target_ev"] = _judge_ev(jm, jtok, jdev,
                                       target_prompt(q, g["text"], g["adj"]))
        else:
            g["target_ev"] = {a: _judge_ev(jm, jtok, jdev,
                                           target_prompt(q, g["text"], a))
                              for a in adjs}
        if (i + 1) % 100 == 0:
            print(f"  judged {i + 1}/{len(gens)}", flush=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "judge": args.judge,
                   "frac": args.frac, "gens": gens}, f, indent=1)

    # --- analyze ---
    base = [g for g in gens if g["cond"] == "baseline"]
    base_ev = {a: np.mean([g["target_ev"][a] for g in base]) for a in adjs}
    base_flu = np.mean([g["fluency"] for g in base])
    print(f"\nbaseline fluency {base_flu:.2f}")
    print(f"{'adj':>16}" + "".join(f"{fr:>16}" for fr in FRAMINGS))
    for a in adjs:
        cells = ""
        for fr in FRAMINGS:
            sel = [g for g in gens if g["adj"] == a and g["cond"] == fr]
            dd = np.mean([g["target_ev"] for g in sel]) - base_ev[a]
            fl = np.mean([g["fluency"] for g in sel])
            cells += f"  {dd:+5.2f}/{fl:4.1f}  "
        print(f"{a:>16}" + cells)
    for fr in FRAMINGS:
        sel = [g for g in gens if g["cond"] == fr]
        dd = np.mean([g["target_ev"] - base_ev[g["adj"]] for g in sel])
        fl = np.mean([g["fluency"] for g in sel])
        print(f"{'mean ' + fr:>16}: delta {dd:+.2f}  fluency {fl:.2f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
