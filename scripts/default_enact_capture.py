"""__default__ ENACT capture for the wide-n cohort (rollouts + activations).

The behavioral complement to the SELF channel: how does each model's DEFAULT
assistant conduct vary across the population? Matches the cohort-10
__default__ arm of extract_persona_vectors.py exactly: empty system prompt,
the 12 generic everyday QUESTIONS, 5 sampled rollouts each (60 total),
temp 0.8 / top-p 0.95 / 100 new tokens, per-rollout response-token-mean
hidden states (fp32 cast before the token mean, fp64 accumulation).

Stores ALL layers per rollout (~40 MB/model) so layer choice stays open at
analysis time; massive-dim winsorization is an analysis-time decision too.

Usage: PYTHONPATH=scripts python scripts/default_enact_capture.py --model REPO
Out:   results/cohort100/default_enact/<repo-sanitized>_default.pt
       {acts (60, L+1, H) fp32, texts, sum fp64, config}
"""
import argparse
import os

import numpy as np
import torch

import hf_logprobs as hf
from extract_persona_vectors import (QUESTIONS, build_inputs,
                                     rollout_mean_states)

torch.set_grad_enabled(False)

OUT_DIR = "results/cohort100/default_enact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts-per-q", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{args.model.replace('/', '_')}_default.pt"
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return

    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    acts, texts = [], []
    ssum = 0.0
    seed = args.seed
    for q in QUESTIONS:
        ids = build_inputs(tok, "", q, device)
        for _ in range(args.rollouts_per_q):
            seed += 1
            states, text, n_resp = rollout_mean_states(
                model, tok, ids, device, args.max_new_tokens,
                args.temperature, args.top_p, seed)
            if states is None:
                continue
            arr = states.numpy()
            ssum = ssum + arr.astype(np.float64)
            acts.append(arr)
            texts.append({"sys": "", "question": q, "text": text,
                          "n_resp": n_resp})
        print(f"  {args.model} {q[:40]:40s} {len(acts)} rollouts", flush=True)
    tmp = out + ".tmp"
    torch.save({"acts": np.stack(acts).astype(np.float32), "texts": texts,
                "sum": ssum,
                "config": vars(args) | {"questions": QUESTIONS}}, tmp)
    os.replace(tmp, out)
    print(f"wrote {out}  ({len(acts)} rollouts, "
          f"{acts[0].shape[0]} layers x {acts[0].shape[1]} dims)")


if __name__ == "__main__":
    main()
