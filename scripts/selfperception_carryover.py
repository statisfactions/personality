"""Behavioural carryover + conduct-vs-perception decomposition (§8k).

Two things the self-report readout cannot separate (rgb, 2026-07-31):
whether the DOSE CONDUCT was itself shaded (Qwen's "rough" rollouts are
tough-guy-but-helpful, so declining "I am rough" may be accurate), and
whether the SELF-PERCEPTION of faithful conduct is filtered.

This script measures both, without needing to know the right direction:

  A. CARRYOVER. After K dose turns (arm A, no instruction anywhere), ask
     a HELD-OUT neutral question and generate freely. Compare against the
     K=0 generation. DVs: (i) direction-free activation displacement
     ||act(K) − act(0)|| (mid layer, massive dims zeroed, normalised by
     the K=0 norm); (ii) cos of that displacement to the model's own
     ENACT direction; (iii) judged trait level of the generated text.
     Free generation makes the instrument label-independent: if conduct
     carries forward without instruction, the state updated regardless
     of what the self-report says.

  B. CONDUCT AUDIT. A third-party judge rates the DOSE MATERIAL itself on
     the target word and its neighbourhood. Gives "what the conduct
     actually was" so self-report shift can be scored against it rather
     than against the persona label.

Judge convention follows judge_enactability.py: cross-family judges
(Llama8 judged by Qwen7, Qwen7 judged by Llama8).

Usage:
  PYTHONPATH=scripts python scripts/selfperception_carryover.py --model Qwen7 --judge Llama8
Output: results/selfperception/carryover_<model>.json
"""
import argparse
import json
import os

import numpy as np
import torch

import hf_logprobs as hf
import selfperception_dose as sd

OUT = "results/selfperception"
HELD_OUT_Q = ("A friend asks you to describe how you usually handle "
              "problems. What do you tell them?")
KS = [0, 8, 32]
MAX_NEW = 110


def dose_turns(rolls, adj, kmax, seed=0):
    import random
    srng = random.Random(hash((adj, seed)) & 0xFFFFFFFF)
    clean = [r for r in rolls[adj] if not r.get("leak")]
    qs = sorted({r["question"] for r in clean})
    srng.shuffle(qs)
    by_q = {}
    for r in clean:
        by_q.setdefault(r["question"], []).append(r)
    for v in by_q.values():
        srng.shuffle(v)
    pool, used = [], {q: 0 for q in qs}
    while len(pool) < kmax:
        progressed = False
        for q in qs:
            if used[q] < len(by_q[q]) and len(pool) < kmax:
                pool.append(by_q[q][used[q]])
                used[q] += 1
                progressed = True
        if not progressed:
            break
    return [(r["question"], r["text"]) for r in pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--stage", default="both", choices=["gen", "judge", "both"])
    args = ap.parse_args()
    judge_name = args.judge or {"Qwen7": "Llama8", "Llama8": "Qwen7"}.get(
        args.model, "Qwen7")
    out = f"{OUT}/carryover_{args.model}.json"

    sel = json.load(open(f"{OUT}/{args.model}_long_selection.json"))
    adjs = [p["adj"] for p in sel["picked"]]
    items = sel["items"]
    rolls = json.load(open(
        f"results/persona_vectors/{args.model}_pda_texts.json"))
    meta = json.load(open(
        f"results/persona_vectors/{args.model}_pda_meta.json"))
    mid = meta["mid_layer"]
    massive = json.loads(meta["massive_dims"]) if isinstance(
        meta["massive_dims"], str) else meta["massive_dims"]

    data = json.load(open(out)) if os.path.exists(out) else {}

    if args.stage in ("gen", "both"):
        model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
        model.eval()
        acts, gens = {}, {}
        with torch.no_grad():
            for i, adj in enumerate(adjs):
                turns_all = dose_turns(rolls, adj, max(KS))
                for K in KS:
                    msgs = sd.build_msgs(None, turns_all[:K]) + \
                        [{"role": "user", "content": HELD_OUT_Q}]
                    s = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
                    ids = tok(s, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(device)
                    g = model.generate(ids, max_new_tokens=MAX_NEW,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    txt = tok.decode(g[0, ids.shape[1]:],
                                     skip_special_tokens=True).strip()
                    h = model(g, output_hidden_states=True).hidden_states[mid]
                    v = h[0, ids.shape[1]:].float().mean(0).cpu().numpy()
                    v[massive] = 0.0
                    acts[f"{adj}_K{K}"] = v.astype(np.float16)
                    gens[f"{adj}_K{K}"] = txt
                print(f"  gen {i+1}/{len(adjs)} {adj}", flush=True)
        np.savez_compressed(f"{OUT}/carryover_{args.model}_acts.npz", **acts)
        data["generations"] = gens
        data["held_out_q"] = HELD_OUT_Q
        json.dump(data, open(out, "w"), indent=1)
        del model
        print("generations done")

    if args.stage in ("judge", "both"):
        from steer_mapped_vs_recorded import _judge_ev
        jm, jt, jd = hf.load_model(judge_name, dtype=torch.bfloat16)

        def rate(q, text, word):
            return _judge_ev(jm, jt, jd,
                             f'Someone was asked: "{q}"\n\nThey replied:\n'
                             f'"""{text}"""\n\nOn a scale from 1 to 7, how '
                             f"{word} does this person seem? Answer with a "
                             f"single digit from 1 to 7.")

        gens = data["generations"]
        # A(iii): judged trait level of free generations
        car = {}
        for adj in adjs:
            car[adj] = {f"K{K}": rate(HELD_OUT_Q, gens[f"{adj}_K{K}"], adj)
                        for K in KS}
        data["carryover_judged"] = car
        print("carryover judged")
        # B: what the dose conduct actually was, on the item neighbourhood
        audit = {}
        for adj in adjs:
            turns = dose_turns(rolls, adj, 4)
            prof = {}
            for w in items[adj]["items"]:
                prof[w] = float(np.mean([rate(q, t, w) for q, t in turns]))
            base = rolls["__default__"][:4]
            prof["__baseline_target__"] = float(np.mean(
                [rate(e["question"], e["text"], adj) for e in base]))
            audit[adj] = prof
            print(f"  audit {adj}", flush=True)
        data["conduct_audit"] = audit
        json.dump(data, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
