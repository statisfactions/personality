#!/usr/bin/env python3
"""Steady-state speed probe for Qwen3.6 MoE as a judge (W16).

The smoke test's 1.1 pair/s was a 7-prompt FIRST pass (MPS kernel compile). This
loads once, runs a warmup batch, then times several full-size batches to get the
real throughput that decides whether a 273k-pair run is feasible (or whether the
linear-attention torch fallback caps it near FalconMamba's ~5 pair/s).
Usage: PYTHONPATH=scripts .venv/bin/python scripts/qwen36_speed.py
"""
import time

import torch

from hf_logprobs import load_model
from adjective_introspection import PROMPT
from adjective_judge_full import load_adjectives

BATCH = 32
N_TIMED = 4


def main():
    model, tok, device = load_model("Qwen36")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    adj = [a.lower() for a in load_adjectives()]

    def batch_texts(start):
        out = []
        for k in range(BATCH):
            i = (start + k) % len(adj)
            j = (start + k + 7) % len(adj)
            msgs = [{"role": "user",
                     "content": PROMPT["tom_likely"].format(a=adj[i], b=adj[j])}]
            out.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        return out

    def run(start):
        enc = tok(batch_texts(start), return_tensors="pt",
                  add_special_tokens=False, padding=True).to(device)
        with torch.no_grad():
            model(**enc).logits[:, -1, :].float()

    print("[warmup] compiling kernels...")
    run(0)
    if device == "mps":
        torch.mps.synchronize()

    times = []
    for b in range(N_TIMED):
        t0 = time.time()
        run((b + 1) * BATCH)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"[batch {b+1}] {BATCH} pairs in {dt:.1f}s = {BATCH/dt:.1f} pair/s")

    rate = BATCH * N_TIMED / sum(times)
    print(f"\n[steady-state] {rate:.1f} pair/s")
    print(f"[full-run ETA] 273006 pairs / {rate:.1f} = "
          f"{273006/rate/3600:.1f} h  (upper-tri 136503 = {136503/rate/3600:.1f} h)")


if __name__ == "__main__":
    main()
