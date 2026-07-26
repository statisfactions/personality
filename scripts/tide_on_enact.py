"""Administer Persona Cartography's v7 FC-pair questionnaire (72 items) to
OUR persona-enactment rollouts (W19 §3: the two-doors congruence test).

Their TIDE matrix: 72 items appended to 2,500 default-assistant rollouts
whose variance comes from deployment roles + user archetypes. Ours: same 72
items appended to Llama8 persona-enactment rollouts (523 adjectives x 2
fixed questions + default-assistant baselines) — labeled respondents, no
scenario soup (question is a controlled 2-level factor). If the item
covariance / factor axes match theirs, instruction-enactment and
role-accommodation share one conduct basis.

Readout matches theirs: signed choice-mass toward the high option after
prefill "Answer:" ((m_hi - m_lo)/(m_hi + m_lo)). Prefix KV-caching per
respondent (context computed once, 72 short continuations).

Usage:
  PYTHONPATH=scripts python scripts/tide_on_enact.py            # inference
Output: results/tide_enact/Llama8_matrix.npz (+ .part.jsonl checkpoint)
"""
import copy
import json
import os

import numpy as np
import torch

import hf_logprobs as hf

OUT = "results/tide_enact"
MODEL = "Llama8"
QPICK = ["My coworker keeps taking credit for my ideas in meetings. "
         "How should I handle it?",
         "I have a free Saturday and no plans. What should I do with it?"]
V7 = ("dependencies/persona-cartography/datasets/psychometric_questionnaires/"
      "psychometric_questionnaire_v7_fc_pair.json")


def load_items():
    d = json.load(open(V7))["block_fc_pairs"]
    tmpl, prefill = d["prompt_template"], d["prefill"]
    items = []
    for it in d["items"]:
        oa, ob = it["options"][0]["text"], it["options"][1]["text"]
        items.append(dict(id=it["id"], dim=it["axis"], high=it["high_option"],
                          prompt=tmpl.format(stem=it["stem"], option_a=oa,
                                             option_b=ob)))
    return items, prefill


def respondents():
    d = json.load(open(f"results/persona_vectors/{MODEL}_pda_texts.json"))
    out = []
    for adj, rolls in d.items():
        for qi, q in enumerate(QPICK):
            match = [r for r in rolls if r["question"] == q]
            if adj == "__default__":
                for j, r in enumerate(match):
                    out.append((f"__default__{j}", qi, r))
            elif match:
                out.append((adj, qi, match[0]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    items, prefill = load_items()
    resp = respondents()
    print(f"{len(resp)} respondents x {len(items)} items")

    part = f"{OUT}/{MODEL}_part.jsonl"
    done = set()
    if os.path.exists(part):
        for l in open(part):
            r = json.loads(l)
            done.add((r["adj"], r["qi"]))
        print(f"resuming: {len(done)} respondents done")

    model, tok, device = hf.load_model(MODEL, dtype=torch.bfloat16)
    AB = {}
    for lab in ("A", "B"):
        AB[lab] = [t[0] for s in (lab, " " + lab)
                   for t in [tok.encode(s, add_special_tokens=False)]
                   if len(t) == 1]
    print("A tokens:", AB["A"], "B tokens:", AB["B"])

    fout = open(part, "a")
    with torch.no_grad():
        for ri, (adj, qi, r) in enumerate(resp):
            if (adj, qi) in done:
                continue
            msgs = ([{"role": "system", "content": r["sys"]}] if r["sys"]
                    else [])
            msgs += [{"role": "user", "content": r["question"]},
                     {"role": "assistant", "content": r["text"]}]
            pre_s = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
            pre = tok(pre_s, add_special_tokens=False).input_ids
            pre_t = torch.tensor([pre], device=device)
            base = model(pre_t, use_cache=True)
            row = {}
            for it in items:
                full_s = tok.apply_chat_template(
                    msgs + [{"role": "user", "content": it["prompt"]}],
                    tokenize=False, add_generation_prompt=True) + prefill
                full = tok(full_s, add_special_tokens=False).input_ids
                if full[:len(pre)] != pre:
                    logits = model(torch.tensor([full], device=device)
                                   ).logits[0, -1].float()
                else:
                    cache = copy.deepcopy(base.past_key_values)
                    cont = torch.tensor([full[len(pre):]], device=device)
                    logits = model(cont, past_key_values=cache
                                   ).logits[0, -1].float()
                p = logits.softmax(-1)
                mA = sum(p[t].item() for t in AB["A"])
                mB = sum(p[t].item() for t in AB["B"])
                hi, lo = (mA, mB) if it["high"] == "A" else (mB, mA)
                row[it["id"]] = (hi - lo) / (hi + lo + 1e-12)
            fout.write(json.dumps(dict(adj=adj, qi=qi, scores=row)) + "\n")
            fout.flush()
            if (ri + 1) % 20 == 0:
                print(f"  {ri+1}/{len(resp)} ({adj})", flush=True)

    fout.close()
    # assemble matrix
    rows = [json.loads(l) for l in open(part)]
    ids = [it["id"] for it in items]
    Rm = np.array([[r["scores"][i] for i in ids] for r in rows])
    np.savez_compressed(f"{OUT}/{MODEL}_matrix.npz", R=Rm,
                        adjectives=np.array([r["adj"] for r in rows]),
                        qi=np.array([r["qi"] for r in rows]),
                        item_ids=np.array(ids))
    print(f"wrote {OUT}/{MODEL}_matrix.npz  {Rm.shape}")


if __name__ == "__main__":
    main()
