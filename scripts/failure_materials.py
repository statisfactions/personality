"""Failure-arm materials: the model's OWN genuine errors on verifiable tasks.

Not persona playacting (rgb's concern 4), and not synthetic problems —
GSM8K test items (rgb: credible failure is crucial, and an off-the-shelf
benchmark at the right difficulty for this scale beats home-made
arithmetic). 7-8B models sit around 70-85% on GSM8K, so genuine failures
are plentiful without being pathological, and the `####` gold answers
remove parser ambiguity on the ground truth.

The model attempts each item at temperature until it produces a WRONG
answer (failure material) and a RIGHT one (success control — the
asymmetry arm).

Each item stores the problem, the true answer, the model's own wrong
attempt and its own correct attempt, so dose contexts can be built in
matched failure/success pairs with identical task content.

Usage:
  PYTHONPATH=scripts python scripts/failure_materials.py --model Llama8
Output: results/selfperception/failure_materials_<model>.json
"""
import argparse
import json
import os
import random
import re

import torch

import hf_logprobs as hf

OUT = "results/selfperception"
MAX_TRIES = 12
MAX_NEW = 200


def make_problems(n, seed=0):
    """GSM8K test items, shuffled deterministically."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        ans = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        try:
            out.append((ds[i]["question"], int(ans)))
        except ValueError:
            continue
    return out


def _unused_synthetic(n, seed=0):
    rng = random.Random(seed)
    probs = []
    while len(probs) < n:
        kind = rng.choice(["rate", "pct", "multi", "unit"])
        if kind == "rate":
            r, h, b = rng.randint(13, 47), rng.randint(3, 9), rng.randint(20, 90)
            probs.append((f"A machine produces {r} widgets per hour and runs "
                          f"for {h} hours. {b} widgets are rejected. How many "
                          f"good widgets remain?", r * h - b))
        elif kind == "pct":
            base, p, add = rng.randrange(200, 900, 20), rng.choice([15, 20, 25, 40]), rng.randint(30, 120)
            probs.append((f"A shop has {base} items. It sells {p}% of them, "
                          f"then receives {add} more. How many items does it "
                          f"have now?", base - base * p // 100 + add)) if base * p % 100 == 0 else None
            if base * p % 100:
                continue
        elif kind == "multi":
            a, b2, c, d = (rng.randint(12, 40), rng.randint(3, 12),
                           rng.randint(5, 30), rng.randint(2, 8))
            probs.append((f"A team of {a} people each collect {b2} samples. "
                          f"They discard {c} damaged samples, then split the "
                          f"rest evenly among {d} labs (any remainder is kept "
                          f"aside). How many samples does each lab get?",
                          (a * b2 - c) // d))
        else:
            km, m, extra = rng.randint(3, 18), rng.randint(2, 9), rng.randint(50, 400)
            probs.append((f"A route is {km} km long. A runner covers it {m} "
                          f"times, then runs {extra} more metres. How many "
                          f"metres in total?", km * 1000 * m + extra))
    return probs[:n]


def final_int(text):
    """Parse the model's stated final answer. Requires the FINAL: marker the
    prompt asks for — free-text tails end on incidental numbers (remainders,
    unit notes) and silently mislabel correct work as wrong, which would turn
    the failure arm into a gaslighting experiment."""
    m = re.findall(r"FINAL:\s*\$?(-?\d[\d,]*)", text.replace("*", ""))
    if not m:
        return None
    try:
        return int(m[-1].replace(",", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama8")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/failure_materials_{args.model}.json"
    data = json.load(open(out)) if os.path.exists(out) else {"items": []}
    done = {it["problem"] for it in data["items"]}

    probs = make_problems(args.n * 3)
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    model.eval()

    def attempt(prob, temp):
        s = tok.apply_chat_template(
            [{"role": "user", "content": prob + "\n\nShow your reasoning, "
              "then end with a line of exactly this form:\nFINAL: <number>"}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(s, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(device)
        g = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=temp > 0,
                           temperature=max(temp, 1e-5), top_p=0.95,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()

    with torch.no_grad():
        for prob, ans in probs:
            if len(data["items"]) >= args.n:
                break
            if prob in done:
                continue
            wrong = right = None
            for t in range(MAX_TRIES):
                txt = attempt(prob, 1.0 if t else 0.0)
                got = final_int(txt)
                if got is None:
                    continue
                if got != ans and wrong is None:
                    wrong = dict(text=txt, given=got)
                elif got == ans and right is None:
                    right = dict(text=txt, given=got)
                if wrong and right:
                    break
            if wrong and right:
                data["items"].append(dict(problem=prob, answer=ans,
                                          wrong=wrong, right=right))
                print(f"  {len(data['items'])}/{args.n}: "
                      f"wrong={wrong['given']} true={ans}", flush=True)
                json.dump(data, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(data['items'])} matched failure/success items)")


if __name__ == "__main__":
    main()
