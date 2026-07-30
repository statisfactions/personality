"""Does the questionnaire read the INSTRUCTION or the CONDUCT? (W19 §3.5)

tide_on_enact.py administered the v7 items with both the persona system
prompt AND the rollout in context. Ablation on ~1/5 of adjectives (every
5th, coworker question):
  B rollout-only:     drop sys, keep user question + assistant rollout
  C instruction-only: keep sys, drop the conversation (item is 1st turn)
Compare each to A (full, from the main run): if C≈A the instrument reads
the standing order (SELF in costume); if B≈A it reads enacted conduct —
and B alone is the self-perception condition (self-description from own
observed behavior, no instruction).

Usage: PYTHONPATH=scripts python scripts/tide_sys_ablation.py
Output: results/tide_enact/Llama8_ablate.npz
"""
import copy
import json
import os

import numpy as np
import torch

import hf_logprobs as hf
from tide_on_enact import QPICK, load_items

OUT = "results/tide_enact"
MODEL = "Llama8"


def main():
    items, prefill = load_items()
    d = json.load(open(f"results/persona_vectors/{MODEL}_pda_texts.json"))
    adjs = sorted(a for a in d if a != "__default__")[::5]
    print(f"{len(adjs)} adjectives x {len(items)} items x 2 conditions")

    model, tok, device = hf.load_model(MODEL, dtype=torch.bfloat16)
    AB = {lab: [t[0] for s in (lab, " " + lab)
                for t in [tok.encode(s, add_special_tokens=False)]
                if len(t) == 1] for lab in ("A", "B")}

    part = f"{OUT}/{MODEL}_ablate_part.jsonl"
    done = set()
    if os.path.exists(part):
        done = {(r["adj"], r["cond"]) for r in map(json.loads, open(part))}
    fout = open(part, "a")

    def administer(msgs):
        pre_s = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=False) \
            if msgs and msgs[-1]["role"] == "assistant" else None
        row = {}
        if pre_s is not None:
            pre = tok(pre_s, add_special_tokens=False).input_ids
            base = model(torch.tensor([pre], device=device), use_cache=True)
        for it in items:
            full_s = tok.apply_chat_template(
                msgs + [{"role": "user", "content": it["prompt"]}],
                tokenize=False, add_generation_prompt=True) + prefill
            full = tok(full_s, add_special_tokens=False).input_ids
            if pre_s is not None and full[:len(pre)] == pre:
                cache = copy.deepcopy(base.past_key_values)
                logits = model(torch.tensor([full[len(pre):]], device=device),
                               past_key_values=cache).logits[0, -1].float()
            else:
                logits = model(torch.tensor([full], device=device)
                               ).logits[0, -1].float()
            p = logits.softmax(-1)
            mA = sum(p[t].item() for t in AB["A"])
            mB = sum(p[t].item() for t in AB["B"])
            hi, lo = (mA, mB) if it["high"] == "A" else (mB, mA)
            row[it["id"]] = (hi - lo) / (hi + lo + 1e-12)
        return row

    q = QPICK[0]
    with torch.no_grad():
        for ai, adj in enumerate(adjs):
            match = [r for r in d[adj] if r["question"] == q]
            if not match:
                continue
            r = match[0]
            for cond, msgs in (
                ("B", [{"role": "user", "content": r["question"]},
                       {"role": "assistant", "content": r["text"]}]),
                ("C", [{"role": "system", "content": r["sys"]}]),
            ):
                if (adj, cond) in done:
                    continue
                row = administer(msgs)
                fout.write(json.dumps(dict(adj=adj, cond=cond, scores=row))
                           + "\n")
                fout.flush()
            if (ai + 1) % 10 == 0:
                print(f"  {ai+1}/{len(adjs)} ({adj})", flush=True)
    fout.close()

    rows = [json.loads(l) for l in open(part)]
    ids = [it["id"] for it in items]
    out = {}
    for cond in ("B", "C"):
        rs = [r for r in rows if r["cond"] == cond]
        out[f"R_{cond}"] = np.array([[r["scores"][i] for i in ids] for r in rs])
        out[f"adj_{cond}"] = np.array([r["adj"] for r in rs])
    np.savez_compressed(f"{OUT}/{MODEL}_ablate.npz", item_ids=np.array(ids),
                        **out)
    print(f"wrote {OUT}/{MODEL}_ablate.npz")


if __name__ == "__main__":
    main()
