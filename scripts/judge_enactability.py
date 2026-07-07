"""Judge-based enactability scores per adjective (W17 §15.9 follow-up).

For each adjective, judge a sample of the ORIGINAL persona-extraction rollouts
("how {adj} does this person seem?", distributional Likert EV) and the same
number of default-assistant baseline rollouts on the same adjective.
Enactability(a) = mean judged EV on a's persona rollouts − mean judged EV on
baseline rollouts. Words whose roleplay never reaches the text (disabled,
employed, slim...) should score ~0; style words (rude, cheerful...) high.

Use: mask the says-vs-is analyses to the enactable stratum; validate the
§15.9 unenactability-collapse account (enactability should be NEGATIVELY
related to assistant-axis cos percentile among identity words).

Usage (PYTHONPATH=scripts):
  python scripts/judge_enactability.py --model llama3.2 --judge Qwen7
  python scripts/judge_enactability.py --model qwen2.5 --judge Llama8
"""
import argparse
import json
import os

import numpy as np
import torch

import hf_logprobs as hf
from steer_mapped_vs_recorded import _judge_ev

OUT_DIR = "results/adjectives/enactability"
K = 4  # rollouts judged per (adjective, condition)


def target_prompt(q, text, adj):
    return (f'Someone was asked: "{q}"\n\nThey replied:\n"""{text}"""\n\n'
            f"On a scale from 1 to 7, how {adj} does this person seem? "
            f"Answer with a single digit from 1 to 7.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{args.model}_enactability.json"
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return
    part = out + ".part"
    scores = json.load(open(part))["scores"] if os.path.exists(part) else {}

    texts = json.load(open(f"results/persona_vectors/{args.model}_pda_texts.json"))
    adjs = sorted(a for a in texts if a != "__default__")
    base = texts["__default__"]
    rng = np.random.default_rng(0)

    model, tok, device = hf.load_model(args.judge, dtype=torch.bfloat16)
    for n, a in enumerate(adjs):
        if a in scores:
            continue
        pers = [texts[a][i] for i in rng.choice(len(texts[a]), K, replace=False)]
        bl = [base[i] for i in rng.choice(len(base), K, replace=False)]
        pe = [_judge_ev(model, tok, device,
                        target_prompt(e["question"], e["text"], a))
              for e in pers]
        be = [_judge_ev(model, tok, device,
                        target_prompt(e["question"], e["text"], a))
              for e in bl]
        scores[a] = {"persona_ev": float(np.mean(pe)),
                     "baseline_ev": float(np.mean(be)),
                     "enactability": float(np.mean(pe) - np.mean(be))}
        if (n + 1) % 50 == 0:
            with open(part, "w") as f:
                json.dump({"model": args.model, "scores": scores}, f)
            print(f"  {n+1}/{len(adjs)}", flush=True)

    with open(out, "w") as f:
        json.dump({"model": args.model, "judge": args.judge, "k": K,
                   "scores": scores}, f, indent=1)
    if os.path.exists(part):
        os.remove(part)
    top = sorted(adjs, key=lambda a: -scores[a]["enactability"])[:8]
    bot = sorted(adjs, key=lambda a: scores[a]["enactability"])[:8]
    print(f"most enactable: {top}")
    print(f"least enactable: {bot}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
