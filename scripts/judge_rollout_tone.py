"""Text-grounded halo check: judge the tone of saved persona rollouts.

The write-side halo claim ("a handsome persona writes more positively") should
ground in the generated text, not in activation geometry — geometry filtering
(massive-dim ablation) can't be allowed to decide an *absence* claim. A judge
model scores each rollout's tone 1-7 (logprob EV); halo = tone(handsome)
above tone(neutral-physical)/tone(default). wonderful/awful anchor the range.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/judge_rollout_tone.py \
      --judge qwen2.5 --texts results/persona_vectors/llama3.2_smoke_texts.json \
                              results/persona_vectors/gemma3_smoke_texts.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import hf_logprobs as hf

PROMPT = (
    'Here is a response someone wrote to an everyday question:\n\n"{text}"\n\n'
    "Rate the overall emotional tone of this response on a scale from 1 to 7, "
    "where 1 = very negative and 7 = very positive. "
    "Respond with a single digit.\n\nRating:"
)
DIGITS = tuple(str(i) for i in range(1, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen2.5")
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out", default="results/persona_vectors/tone_judgments.json")
    args = ap.parse_args()

    model, tok, device = hf.load_model(args.judge)
    results = {}
    for path in args.texts:
        name = Path(path).stem
        data = json.load(open(path))
        results[name] = {}
        print(f"\n=== {name} (judge: {args.judge}) ===")
        for cond, rollouts in data.items():
            evs = []
            for r in rollouts:
                text = r["text"].strip()
                if not text:
                    continue
                dist, _, _ = hf.likert_distribution(
                    model, tok, PROMPT.format(text=text), device, digits=DIGITS)
                evs.append(sum(int(k) * p for k, p in dist.items()))
            evs = np.array(evs)
            results[name][cond] = {
                "mean": float(evs.mean()), "se": float(evs.std() / len(evs) ** 0.5),
                "n": len(evs),
            }
            print(f"  {cond:>12s}: tone EV = {evs.mean():.2f} ± {evs.std() / len(evs) ** 0.5:.2f}  (n={len(evs)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"judge": args.judge, "results": results}, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
