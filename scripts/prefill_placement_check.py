"""Does the cue's location matter for non-thinkers? (2026-08-29, rgb)

Three readouts of the same SELF items:
  A user-cue        user turn ends "Rating: "; read the FIRST assistant token
                    (the standing prefill arm)
  B assistant-cue   user turn without the cue; assistant turn prefilled with
                    "Rating:" (continue_final_message); read the next token
  C generated       from A's prompt, greedy-generate <= 8 tokens and read the
                    digit distribution at the first emitted digit (the think
                    arm's readout, minus the thinking)
Smoke set x 6 framings; runs on CPU so it can go while the GPU is busy.

Usage: PYTHONPATH=scripts python scripts/prefill_placement_check.py --model llama3.2
"""
import argparse
import json
import math

import numpy as np
import torch

import hf_logprobs as hf
from self_adjective_report import AGREE_SCALE, PDA_SCALE, FRAMINGS, SMOKE_ADJS, ev_of
from selfperception_dose import digit_tokens

DIGITS = ("1", "2", "3", "4", "5", "6", "7")


def dist_at(logits, dt):
    p = torch.softmax(logits.float(), -1)
    d = {}
    for tid, dg in dt.items():
        d[dg] = d.get(dg, 0.0) + p[tid].item()
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model, tok, device = hf.load_model(args.model, device=args.device,
                                       dtype=torch.bfloat16)
    dt = {tid: d for d, tids in digit_tokens(tok).items() for tid in tids}
    from extract_adjectives import load_adjectives
    allset = {a.lower() for a in load_adjectives()}
    adjs = [a for a in SMOKE_ADJS if a in allset]
    res = {"A": {}, "B": {}, "C": {}}
    for f, template in FRAMINGS.items():
        for a in adjs:
            prompt = (PDA_SCALE.format(adj=a) if f == "pda"
                      else AGREE_SCALE.format(statement=template.format(adj=a)))
            key = f"{f}|{a}"
            # A
            dA, _, hA = hf.likert_distribution(model, tok, prompt, device, digits=DIGITS)
            res["A"][key] = {"ev": ev_of(dA), "entropy": hA}
            # B: strip the trailing cue line from the user turn, prefill it
            user = prompt.rsplit("\n", 1)[0]          # drop 'Rating: '
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": "Rating:"}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           continue_final_message=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**ids, use_cache=False).logits[0, -1]
            dB = dist_at(lg, dt)
            res["B"][key] = {"ev": ev_of(dB),
                             "entropy": -sum(p * math.log(p) for p in dB.values() if p > 0)}
            # C: generate from A's prompt, read at the first digit
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=tok.eos_token_id)
            seq = out.sequences[0, ids.shape[1]:].tolist()
            hits = [i for i, t in enumerate(seq) if t in dt]
            if hits:
                dC = dist_at(out.scores[hits[0]][0], dt)
                dL = dist_at(out.scores[hits[-1]][0], dt)
                res["C"][key] = {"ev": ev_of(dC), "pos": hits[0],
                                 "ev_last": ev_of(dL), "pos_last": hits[-1],
                                 "n_digits": len(hits), "text": tok.decode(seq),
                                 "entropy": -sum(p * math.log(p) for p in dC.values() if p > 0)}
            else:
                res["C"][key] = {"ev": None, "pos": None, "tail": tok.decode(seq)}
        print(f"  {f} done", flush=True)
    keys = list(res["A"])
    def vec(c):
        return np.array([res[c][k]["ev"] if res[c][k]["ev"] is not None else np.nan for k in keys])
    A, B, C = vec("A"), vec("B"), vec("C")
    okC = ~np.isnan(C)
    print(f"\n{args.model}: n={len(keys)}")
    print(f"  A vs B (user-cue vs assistant-cue): r={np.corrcoef(A, B)[0,1]:.3f}  mean|dEV|={np.abs(A-B).mean():.2f}  "
          f"mean EV A {A.mean():.2f} B {B.mean():.2f}  mean H A {np.mean([res['A'][k]['entropy'] for k in keys]):.2f} B {np.mean([res['B'][k]['entropy'] for k in keys]):.2f}")
    print(f"  A vs C (user-cue vs generated-digit): r={np.corrcoef(A[okC], C[okC])[0,1]:.3f}  mean|dEV|={np.abs(A-C)[okC].mean():.2f}  "
          f"C parsed {okC.mean()*100:.0f}%  first-digit position median {np.median([res['C'][k]['pos'] for k in keys if res['C'][k]['pos'] is not None]):.0f}")
    json.dump(res, open(f"results/adjectives/prefill_placement_{args.model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
