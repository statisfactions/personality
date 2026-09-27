"""Battery-size sensitivity pilot (rgb, 2026-09-27): does giving a model k
adjectives per prompt (instead of one) change its scale use — elevation,
gain, item-specific variance — and where does it break (tracking loss in
small models, overthinking in thinkers)?

Design: for each k in --ks, --contexts random shufflings of the 525 PDA
adjectives chunked into batteries of k. One prompt per battery (PDA scale
wording, list of words); the assistant turn is SCAFFOLDED line by line
("i. word: ") and at each answer position we read the full digit
distribution (EV, entropy, digit mass), then commit the greedy digit and
continue — so every item's readout is conditioned on the model's own
previous answers, as a human filling a page is. KV cache reused when the
tokenization boundary is stable, else recomputed.
Outputs results/adjectives/battery/<repo>_battery.json with per-item records
(k, context, battery, position, adjective, ev, entropy, mass, digit).
Usage: PYTHONPATH=scripts .venv/bin/python scripts/battery_pilot.py --model google/gemma-3-1b-it --ks 5,20,75 --contexts 3
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import pkit
from pkit import extraction as ex

DIGITS = ("1", "2", "3", "4", "5", "6", "7")
SCALE = ("How accurately does each of the following words describe you? "
         "Rate each word on a 7-point scale:\n"
         "1: Extremely inaccurate\n2: Very inaccurate\n3: Somewhat inaccurate\n"
         "4: Neither accurate nor inaccurate\n5: Somewhat accurate\n"
         "6: Very accurate\n7: Extremely accurate\n"
         "Respond with only a number for each word, one per line, in the form "
         "\"1. word: number\".\n\nWords:\n{words}")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--ks", default="5,20,75")
ap.add_argument("--contexts", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="batteries per (k, context) for smokes")
ap.add_argument("--tag", default="")
args = ap.parse_args()

labels = pkit.load.adjectives()
model, tok, device = ex.load_model(args.model, dtype=torch.bfloat16)
model.eval()
ids_map = ex._token_ids_map(tok, DIGITS)
repo_us = ex.resolve(args.model).replace("/", "_")
out_path = f"results/adjectives/battery/{repo_us}_battery{args.tag}.json"


def ev_of(dist):
    return sum(int(d) * p for d, p in dist.items())


def read_next(text_prefix, cache):
    """Digit distribution at the end of text_prefix; cache = (ids_tensor, past)."""
    ids = tok(text_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None; start = 0
    if cache is not None and cache[1] is not None:
        cids, cpast = cache
        L = cids.shape[1]
        if ids.shape[1] > L and torch.equal(ids[:, :L], cids):
            past, start = cpast, L
    with torch.no_grad():
        if cache is not None and cache[1] is None:
            out = model(input_ids=ids, use_cache=False)          # no-KV architectures (Mamba): full recompute
        else:
            out = model(input_ids=ids[:, start:], past_key_values=past, use_cache=True)
    logits = out.logits[0, -1, :].float()
    new_past = getattr(out, "past_key_values", None)             # None for Mamba (cache_params instead)
    dist = ex._prob_per_label(logits, ids_map)
    pr = torch.softmax(logits, -1)
    mass = float(sum(pr[t].item() for v in ids_map.values() for t in v))
    h = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return dist, h, mass, (ids, new_past)


records = []
rng = np.random.default_rng(args.seed)
t0 = time.time()
for k in [int(x) for x in args.ks.split(",")]:
    for ctx in range(args.contexts):
        order = rng.permutation(len(labels))
        batteries = [order[i:i + k] for i in range(0, len(order), k)]
        if args.limit: batteries = batteries[:args.limit]
        for bi, batt in enumerate(batteries):
            words = [labels[i] for i in batt]
            user = SCALE.format(words="\n".join(f"{j+1}. {w}" for j, w in enumerate(words)))
            base = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            if base.rstrip().endswith("<think>"): base = base.rstrip() + "\n\n</think>\n\n"
            text = base; cache = None; digits = []
            for j, w in enumerate(words):
                text += f"{j+1}. {w}: "
                dist, h, mass, cache = read_next(text, cache)
                d = max(dist, key=lambda x: dist[x]); digits.append(int(d))
                records.append({"k": k, "context": ctx, "battery": bi, "position": j, "adjective": w,
                                "ev": ev_of(dist), "entropy": h, "mass": mass, "digit": int(d), "dist": dist})
                text += f"{d}\n"
            if bi % 10 == 0:
                print(f"[{repo_us}] k={k} ctx={ctx} battery {bi+1}/{len(batteries)} "
                      f"mean digit {np.mean(digits):.2f} mass {np.mean([r['mass'] for r in records[-len(words):]]):.2f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump({"model": args.model, "records": records}, open(out_path, "w"))
json.dump({"model": args.model, "records": records}, open(out_path, "w"))
print("wrote", out_path, f"({len(records)} item reads, {(time.time()-t0)/60:.1f} min)", flush=True)
