"""Is the dose displacement output-potent or output-null? (§8l, rgb's question)

Qwen's residual stream moves substantially under dose (‖Δ‖/‖h‖ = 0.71 at
K=32) while neither its behaviour nor its self-report follows. Two
accounts:

  (a) COMPLEMENT — the displacement lies in directions the output map is
      blind to (the complement of the Jacobian/output-relevant subspace).
      Inert geometry: nothing downstream needs to suppress it.
  (b) GATING — the displacement IS output-potent, and something damps it
      in the full-context computation.

They differ under injection. Take δ = act(K=32) − act(K=0), the model's
OWN dose displacement, and add it to the residual stream at the mid
layer during an UNDOSED generation:

  (a) predicts no output change at α=1 (and none until the text breaks);
  (b) predicts the trait appears — the direction works fine when nothing
      is suppressing it, so the suppression is contextual/downstream.

Controls: a random vector of matched norm (potency floor), and the
model's own ENACT direction scaled to the same norm (potency ceiling —
a direction known to steer).

Usage:
  PYTHONPATH=scripts python scripts/selfperception_jspace.py --model Qwen7
  PYTHONPATH=scripts python scripts/selfperception_jspace.py --model Qwen7 --stage judge
Output: results/selfperception/jspace_<model>.json
"""
import argparse
import json
import os

import numpy as np
import torch

import hf_logprobs as hf

OUT = "results/selfperception"
HELD_OUT_Q = ("A friend asks you to describe how you usually handle "
              "problems. What do you tell them?")
ALPHAS = [0.0, 1.0, 2.0]
MAX_NEW = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--stage", default="both", choices=["gen", "judge", "both"])
    ap.add_argument("--n-adj", type=int, default=12)
    args = ap.parse_args()
    judge_name = args.judge or {"Qwen7": "Llama8", "Llama8": "Qwen7"}.get(
        args.model, "Qwen7")
    out = f"{OUT}/jspace_{args.model}.json"
    data = json.load(open(out)) if os.path.exists(out) else {}

    sel = json.load(open(f"{OUT}/{args.model}_long_selection.json"))
    adjs = [p["adj"] for p in sel["picked"]][:args.n_adj]
    meta = json.load(open(
        f"results/persona_vectors/{args.model}_pda_meta.json"))
    mid = meta["mid_layer"]
    massive = json.loads(meta["massive_dims"]) if isinstance(
        meta["massive_dims"], str) else meta["massive_dims"]
    z = np.load(f"{OUT}/carryover_{args.model}_acts.npz")

    if args.stage in ("gen", "both"):
        pda = torch.load(f"results/persona_vectors/{args.model}_pda.pt",
                         map_location="cpu", weights_only=False)
        model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
        model.eval()
        layers = model.model.layers
        state = {"vec": None}

        def hook(mod, inp, output):
            if state["vec"] is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h = h + state["vec"].to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        handle = layers[mid - 1].register_forward_hook(hook)
        rng = np.random.default_rng(0)
        gens, kls, norms = {}, {}, {}
        s = tok.apply_chat_template(
            [{"role": "user", "content": HELD_OUT_Q}], tokenize=False,
            add_generation_prompt=True)
        ids = tok(s, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            state["vec"] = None
            base_logits = model(ids).logits[0, -1].float().log_softmax(-1)
            for adj in adjs:
                d = (z[f"{adj}_K32"].astype(np.float32)
                     - z[f"{adj}_K0"].astype(np.float32))
                variants = {"dose": d}
                r = rng.normal(size=d.shape).astype(np.float32)
                r[massive] = 0.0
                variants["random"] = r / np.linalg.norm(r) * np.linalg.norm(d)
                en = np.asarray(pda["directions"][adj])[mid].astype(np.float32)
                en[massive] = 0.0
                variants["enact"] = en / np.linalg.norm(en) * np.linalg.norm(d)
                norms[adj] = float(np.linalg.norm(d))
                for name, vec in variants.items():
                    for a in ALPHAS:
                        if a == 0.0 and name != "dose":
                            continue
                        key = f"{adj}|{name}|{a}"
                        state["vec"] = (None if a == 0.0 else
                                        torch.tensor(vec * a, device=device))
                        lg = model(ids).logits[0, -1].float().log_softmax(-1)
                        kls[key] = float((lg.exp() * (lg - base_logits)).sum())
                        g = model.generate(ids, max_new_tokens=MAX_NEW,
                                           do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                        gens[key] = tok.decode(g[0, ids.shape[1]:],
                                               skip_special_tokens=True).strip()
                        state["vec"] = None
                print(f"  {adj} done", flush=True)
        handle.remove()
        data.update(generations=gens, kl=kls, dose_norms=norms,
                    alphas=ALPHAS, mid_layer=mid, adjs=adjs)
        json.dump(data, open(out, "w"), indent=1)
        del model
        print("generation done")

    if args.stage in ("judge", "both"):
        from steer_mapped_vs_recorded import _judge_ev
        jm, jt, jd = hf.load_model(judge_name, dtype=torch.bfloat16)
        judged = {}
        for key, txt in data["generations"].items():
            adj = key.split("|")[0]
            judged[key] = _judge_ev(
                jm, jt, jd,
                f'Someone was asked: "{HELD_OUT_Q}"\n\nThey replied:\n'
                f'"""{txt}"""\n\nOn a scale from 1 to 7, how {adj} does this '
                f"person seem? Answer with a single digit from 1 to 7.")
        data["judged"] = judged
        json.dump(data, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
