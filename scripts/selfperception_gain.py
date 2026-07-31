"""ENACT steering gain curve: soft gain reduction or hard ceiling? (§8n)

§8m left a second, separable effect: Qwen damps even its OWN
trait-carrying ENACT direction relative to Llama at matched norm
(judged +0.76 vs +1.14 at alpha=1). Two shapes distinguish the
mechanisms:

  SOFT GAIN   — Qwen's curve is Llama's scaled down; both keep rising
                with alpha until text degrades. A smaller coefficient
                on the same pathway.
  HARD CEILING— Qwen's curve saturates at a trait level Llama passes,
                while its KL keeps climbing (steering "spends" without
                buying trait). A guard that clamps the expressed
                persona.

Sweep alpha over a wide range with the model's own ENACT direction,
scaled in units of the mean residual-stream norm at the injection layer
(so alpha is comparable across models, per W17 dose conventions — NOT
the raw vector norm, which differs by family).

DVs per alpha: judged trait (cross-family judge), KL of the next-token
distribution vs alpha=0, and a text-quality guard (mean per-token
logprob of the generation under the generating model itself) so
saturation can be distinguished from degradation.

Usage:
  PYTHONPATH=scripts python scripts/selfperception_gain.py --model Qwen7
Output: results/selfperception/gain_<model>.json
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
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]   # frac of ||h||
MAX_NEW = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--stage", default="both", choices=["gen", "judge", "both"])
    ap.add_argument("--n-adj", type=int, default=10)
    args = ap.parse_args()
    judge_name = args.judge or {"Qwen7": "Llama8", "Llama8": "Qwen7"}.get(
        args.model, "Qwen7")
    out = f"{OUT}/gain_{args.model}.json"
    data = json.load(open(out)) if os.path.exists(out) else {}

    sel = json.load(open(f"{OUT}/{args.model}_long_selection.json"))
    adjs = [p["adj"] for p in sel["picked"]][:args.n_adj]
    meta = json.load(open(
        f"results/persona_vectors/{args.model}_pda_meta.json"))
    mid = meta["mid_layer"]
    massive = json.loads(meta["massive_dims"]) if isinstance(
        meta["massive_dims"], str) else meta["massive_dims"]

    if args.stage in ("gen", "both"):
        pda = torch.load(f"results/persona_vectors/{args.model}_pda.pt",
                         map_location="cpu", weights_only=False)
        model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
        model.eval()
        state = {"vec": None}

        def hook(mod, inp, output):
            if state["vec"] is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h = h + state["vec"].to(h.dtype)
            return ((h,) + tuple(output[1:])) if isinstance(output, tuple) \
                else h

        handle = model.model.layers[mid - 1].register_forward_hook(hook)
        s = tok.apply_chat_template(
            [{"role": "user", "content": HELD_OUT_Q}], tokenize=False,
            add_generation_prompt=True)
        ids = tok(s, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(device)
        gens, kls, ppl = {}, {}, {}
        with torch.no_grad():
            state["vec"] = None
            o = model(ids, output_hidden_states=True)
            base_lp = o.logits[0, -1].float().log_softmax(-1)
            hnorm = float(o.hidden_states[mid][0].float().norm(dim=-1).mean())
            print(f"mean ||h|| at layer {mid}: {hnorm:.2f}")
            for adj in adjs:
                en = np.asarray(pda["directions"][adj])[mid].astype(np.float32)
                en[massive] = 0.0
                en = en / np.linalg.norm(en)
                for a in ALPHAS:
                    key = f"{adj}|{a}"
                    state["vec"] = (None if a == 0.0 else torch.tensor(
                        en * a * hnorm, device=device))
                    lg = model(ids).logits[0, -1].float().log_softmax(-1)
                    kls[key] = float((lg.exp() * (lg - base_lp)).sum())
                    g = model.generate(ids, max_new_tokens=MAX_NEW,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    txt = tok.decode(g[0, ids.shape[1]:],
                                     skip_special_tokens=True).strip()
                    gens[key] = txt
                    # quality guard: unsteered logprob of the steered text
                    state["vec"] = None
                    full = torch.cat([ids, g[:, ids.shape[1]:]], dim=1)
                    lp = model(full).logits[0].float().log_softmax(-1)
                    tgt = full[0, 1:]
                    ppl[key] = float(lp[:-1].gather(
                        -1, tgt[:, None]).squeeze(-1)[ids.shape[1] - 1:].mean())
                print(f"  {adj} done", flush=True)
        handle.remove()
        data.update(generations=gens, kl=kls, logprob=ppl, alphas=ALPHAS,
                    adjs=adjs, hnorm=hnorm, layer=mid)
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

    if "judged" in data:
        print(f"\n{'alpha':>6s} {'judged':>7s} {'KL':>7s} {'logprob':>8s}")
        for a in data["alphas"]:
            js = np.mean([data["judged"][f"{x}|{a}"] for x in data["adjs"]])
            kl = np.mean([data["kl"][f"{x}|{a}"] for x in data["adjs"]])
            lp = np.mean([data["logprob"][f"{x}|{a}"] for x in data["adjs"]])
            print(f"{a:6.2f} {js:7.2f} {kl:7.3f} {lp:8.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
