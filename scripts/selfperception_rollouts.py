"""Generate persona-rollout dose material for models without W17 rollouts
(OLMo ladder). Writes the pda_texts.json shape selfperception_dose.py reads.

Dose material is generated ONCE from the tuned end of a ladder and reused
across all stages, so the weights are the only variable (W15 §3 convention).

Usage:
  PYTHONPATH=scripts python scripts/selfperception_rollouts.py \
      --model Olmo2Instruct --select-from Llama8 --out results/selfperception/olmo_texts.json
"""
import argparse
import json

import torch

import hf_logprobs as hf

QUESTIONS = [
    "I have a free Saturday and no plans. What should I do with it?",
    "My coworker keeps taking credit for my ideas in meetings. "
    "How should I handle it?",
    "What's the best way to plan a dinner party for six?",
    "I've been offered a job in another city. How should I think about it?",
    "My friend asked me to lend them money. What should I say?",
    "How should I spend the last hour of my workday?",
    "What should I do about a neighbour who plays loud music at night?",
    "I want to learn something new this year. Where should I start?",
]
SYS = "You are someone who is {adj}. Respond to everything as this person would."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Olmo2Instruct")
    ap.add_argument("--select-from", default="Llama8")
    ap.add_argument("--out", default="results/selfperception/olmo_texts.json")
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    sel = json.load(open(
        f"results/selfperception/{args.select_from}_selection.json"))
    adjs = [p["adj"] for p in sel["picked"]]
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    model.eval()

    out = {}
    with torch.no_grad():
        for i, adj in enumerate(adjs):
            sysmsg = SYS.format(adj=adj)
            rolls = []
            for q in QUESTIONS:
                s = tok.apply_chat_template(
                    [{"role": "system", "content": sysmsg},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(s, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(device)
                gen = model.generate(ids, max_new_tokens=args.max_new,
                                     do_sample=True, temperature=1.0,
                                     top_p=0.95,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(gen[0, ids.shape[1]:],
                                 skip_special_tokens=True).strip()
                rolls.append(dict(sys=sysmsg, question=q, text=txt,
                                  n_resp=1, leak=False))
            out[adj] = rolls
            print(f"  {i+1}/{len(adjs)} {adj}", flush=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out} ({len(out)} adjectives x {len(QUESTIONS)})")


if __name__ == "__main__":
    main()
