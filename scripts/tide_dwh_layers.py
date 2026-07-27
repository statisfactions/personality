"""Layer-resolved ΔW·h (W19 §4.5, rgb's point: LoRA applies at EVERY layer,
steering at one).

(a) All-layer Δh profiles for the 10 adapters: |Δh_l| growth, per-text
    consistency per layer, direction persistence (cos of Δh_l with Δh_16).
(b) Block-wise application for agreeableness amp/sup: ΔW restricted to
    layers 0-7 / 8-15 / 16-23 / 24-31; layer-16 and layer-32 Δh; trait-
    matched sign test per block — where does trait content enter?

Usage: PYTHONPATH=scripts python scripts/tide_dwh_layers.py
Output: results/tide_enact/Llama8_dwh_layers.npz
"""
import json
import os
import re

import numpy as np
import torch

import hf_logprobs as hf
from tide_dwh_bridge import (TRAITS, build_deltas, capture, fetch_adapter,
                             get_module)

OUT = "results/tide_enact"
BLOCKS = [(0, 8), (8, 16), (16, 24), (24, 32)]


@torch.no_grad()
def apply_deltas(model, deltas, sign, layers=None):
    for path, dW in deltas.items():
        if layers is not None:
            m = re.search(r"layers\.(\d+)\.", path)
            if m and not (layers[0] <= int(m.group(1)) < layers[1]):
                continue
        w = get_module(model, path).weight
        w += sign * dW.to(w.device, w.dtype)


@torch.no_grad()
def capture_all(model, tok, device, texts):
    outs = []
    for r in texts:
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["text"]}]
        pre_s = tok.apply_chat_template(msgs[:1], tokenize=False,
                                        add_generation_prompt=True)
        full_s = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False)
        pre = tok(pre_s, add_special_tokens=False).input_ids
        full = tok(full_s, add_special_tokens=False).input_ids
        hs = model(torch.tensor([full], device=device),
                   output_hidden_states=True).hidden_states
        outs.append(np.stack([h[0, len(pre):].float().mean(0).cpu().numpy()
                              for h in hs]))          # (n_layers+1, d)
    return np.array(outs)                              # (n_texts, L+1, d)


def main():
    texts = json.load(open("results/persona_vectors/Llama8_pda_texts.json")
                      )["__default__"]
    model, tok, device = hf.load_model("Llama8", dtype=torch.bfloat16)
    print("capturing base (all layers)...")
    H0 = capture_all(model, tok, device, texts)
    print("base", H0.shape, flush=True)

    save = {"base_mean": H0.mean(0)}
    # (a) all-layer profiles, 10 adapters
    for trait in TRAITS:
        for pole in ("amplifier", "suppressor"):
            name = f"{trait[:4]}_{pole[:3]}"
            cfg, st = fetch_adapter(trait, pole)
            deltas = build_deltas(cfg, st)
            apply_deltas(model, deltas, +1)
            H1 = capture_all(model, tok, device, texts)
            apply_deltas(model, deltas, -1)
            save[f"prof_{name}"] = (H1 - H0).mean(0)   # (L+1, d)
            dl = (H1 - H0)
            cons = []
            for l in range(dl.shape[1]):
                mu = dl[:, l].mean(0)
                cons.append(np.mean([
                    np.dot(dl[i, l], mu) /
                    (np.linalg.norm(dl[i, l]) * np.linalg.norm(mu) + 1e-9)
                    for i in range(len(dl))]))
            save[f"cons_{name}"] = np.array(cons)
            print(f"  {name} done", flush=True)
            del deltas

    # (b) block-wise, agreeableness pair
    for pole in ("amplifier", "suppressor"):
        name = f"agre_{pole[:3]}"
        cfg, st = fetch_adapter("agreeableness", pole)
        deltas = build_deltas(cfg, st)
        for b0, b1 in BLOCKS:
            apply_deltas(model, deltas, +1, layers=(b0, b1))
            H1 = capture_all(model, tok, device, texts)
            apply_deltas(model, deltas, -1, layers=(b0, b1))
            save[f"block_{name}_{b0}_{b1}"] = (H1 - H0).mean(0)
            print(f"  {name} block {b0}-{b1} done", flush=True)
        del deltas

    np.savez_compressed(f"{OUT}/Llama8_dwh_layers.npz", **save)
    print(f"saved {OUT}/Llama8_dwh_layers.npz")


if __name__ == "__main__":
    main()
