#!/usr/bin/env python3
"""Smoke test for Qwen3.6-35B-A3B (MoE) as a tom_likely judge (W16).

Three questions before committing to a full 273k-pair run:
  (1) Does the 35B-A3B MoE load + run on MPS at all (OOM / dtype)?
  (2) Speed: pair/s with 3B active — or does MPS hit an expert-routing cliff
      like FalconMamba (~5 pair/s)?
  (3) Is enable_thinking=False actually taking? Reasoning-on emits <think> and
      scatters probability off the answer digits. We check RAW digit-mass (sum
      of softmax prob on the 1-7 tokens over the FULL vocab) — instruct judges
      sit ~1.0; a low value means thinking leaked or format non-compliance.

Also sanity-checks discrimination: synonym/antonym/unrelated tom_likely EVs.
Usage: PYTHONPATH=scripts .venv/bin/python scripts/qwen36_smoke.py
"""
import time

import numpy as np
import torch

from hf_logprobs import load_model
from adjective_introspection import PROMPT

DIGITS = ("1", "2", "3", "4", "5", "6", "7")
DVAL = np.array([int(d) for d in DIGITS], float)

# a few labeled probes (a -> b): expect high for synonyms, low for antonyms,
# mid/uncertain for unrelated. tom_likely = "how likely is a very-A person to
# also be B".
PROBES = [
    ("kind", "warm", "syn"),
    ("cruel", "harsh", "syn"),
    ("honest", "dishonest", "ant"),
    ("brave", "cowardly", "ant"),
    ("wonderful", "awful", "ant-eval"),
    ("talkative", "punctual", "unrel"),
    ("generous", "tall", "unrel"),
]


def main():
    short = "Qwen36"
    t_load = time.time()
    model, tok, device = load_model(short)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[load] {short} on {device} in {time.time()-t_load:.0f}s")

    # digit token ids (single-token {d, ' '+d}); plus a flat set for raw mass
    digit_ids, flat = [], []
    for d in DIGITS:
        ids = [tok(c, add_special_tokens=False).input_ids[0]
               for c in (d, " " + d)
               if len(tok(c, add_special_tokens=False).input_ids) == 1]
        digit_ids.append(ids); flat += ids

    ct_kw = {"enable_thinking": False}

    def templ(a, b):
        msgs = [{"role": "user",
                 "content": PROMPT["tom_likely"].format(a=a, b=b)}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, **ct_kw)

    texts = [templ(a, b) for a, b, _ in PROBES]
    enc = tok(texts, return_tensors="pt", add_special_tokens=False,
              padding=True).to(device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
    dt = time.time() - t0
    probs = torch.softmax(logits, -1).cpu().numpy()

    print(f"\n[speed] {len(PROBES)} prompts in {dt:.1f}s "
          f"= {len(PROBES)/dt:.1f} pair/s (batched, first pass)\n")
    print(f"{'a':>11} {'b':>11} {'kind':>9} {'rawmass':>8} {'EV':>5}")
    for k, (a, b, lab) in enumerate(PROBES):
        p = probs[k]
        rawmass = float(p[flat].sum())
        pd = np.array([p[ids].sum() for ids in digit_ids])
        pd = pd / (pd.sum() + 1e-9)
        ev = float((pd * DVAL).sum())
        print(f"{a:>11} {b:>11} {lab:>9} {rawmass:8.4f} {ev:5.2f}")

    # decode the top token to eyeball whether a <think> or text leaked
    top = logits.argmax(-1).cpu().tolist()
    print("\n[top-token per probe]",
          [tok.decode([t]).strip()[:6] or "·" for t in top])


if __name__ == "__main__":
    main()
