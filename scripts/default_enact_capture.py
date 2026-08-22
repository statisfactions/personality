"""__default__ ENACT capture for the wide-n cohort (rollouts + activations).

The behavioral complement to the SELF channel: how does each model's DEFAULT
assistant conduct vary across the population? Matches the cohort-10
__default__ arm of extract_persona_vectors.py: empty system prompt, the 12
generic everyday QUESTIONS, 5 sampled rollouts each (60 total), temp 0.8 /
top-p 0.95 / 100 new tokens (thinking models get a larger budget so the
answer span exists), fp32 cast before token means, fp64 accumulation.

CoT/answer SPLIT (rgb 2026-08-11): for models that emit think blocks, the
reasoning voice and the answer voice are captured as SEPARATE reads —
mean states over the CoT span and over the answer span, plus the full-span
mean for cohort-10 comparability. Raw texts keep the think block verbatim
(BOW/TF-IDF on CoT is a first-class analysis: LW tDJWZLQNN7poqCwKa).
Stores ALL layers per rollout; layer choice / winsorization stay
analysis-time decisions.

Usage: PYTHONPATH=scripts python scripts/default_enact_capture.py --model REPO
Out:   results/cohort100/default_enact/<repo-sanitized>_default.pt
       {acts, acts_cot, acts_answer (60, L+1, H) fp32 (cot/answer NaN-padded
        where absent), texts, config}
"""
import argparse
import os
import re

import numpy as np
import torch

import hf_logprobs as hf
from extract_persona_vectors import QUESTIONS, build_inputs

torch.set_grad_enabled(False)

OUT_DIR = "results/cohort100/default_enact"
THINK_CLOSE = re.compile(r"</think(?:ing)?>|<\|end_of_thought\|>")


def rollout_split_states(model, tok, input_ids, device, max_new_tokens,
                         temperature, top_p, seed):
    """One sampled rollout; re-forward; return per-span layer-mean states.
    Spans: full response, CoT (up to & incl. think-close), answer (after).
    No think-close found -> cot=None, answer=full. Returns
    (full, cot, answer, text_with_specials, n_resp)."""
    torch.manual_seed(seed)
    prompt_len = input_ids.shape[1]
    gen_kw = dict(do_sample=temperature > 0,
                  temperature=temperature or None,
                  top_p=top_p if temperature > 0 else None,
                  max_new_tokens=max_new_tokens,
                  pad_token_id=tok.eos_token_id)
    try:
        out_ids = model.generate(input_ids, **gen_kw)
    except TypeError:
        # legacy-cache custom code (InternLM2.5): regenerate cache-free
        out_ids = model.generate(input_ids, use_cache=False, **gen_kw)
    seq = out_ids[0]
    end = seq.shape[0]
    while end > prompt_len and seq[end - 1].item() == tok.eos_token_id:
        end -= 1
    if end - prompt_len < 1:
        return None, None, None, "", 0
    text = tok.decode(seq[prompt_len:end], skip_special_tokens=False)
    m = THINK_CLOSE.search(text)
    split = None
    if m:
        # token index of the think-close boundary within the response span
        split = prompt_len + len(tok(text[:m.end()],
                                     add_special_tokens=False).input_ids)
        split = min(split, end - 1)
    outputs = model(seq[:end].unsqueeze(0), output_hidden_states=True)
    if outputs.hidden_states is None:
        # some remote-code models only honor the config flag, not the kwarg
        model.config.output_hidden_states = True
        outputs = model(seq[:end].unsqueeze(0))

    def span_mean(a, b):
        if b - a < 1:
            return None
        return torch.stack([hs[0, a:b, :].float().mean(dim=0)
                            for hs in outputs.hidden_states]).cpu().numpy()

    full = span_mean(prompt_len, end)
    cot = span_mean(prompt_len, split) if split else None
    ans = span_mean(split, end) if split else full
    return full, cot, ans, text, end - prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts-per-q", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--think-max-new-tokens", type=int, default=400,
                    help="budget when the model emits a think block (probed "
                         "on the first rollout)")
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

    # upfront thinker probe: greedy 48 tokens on q0 (or a template that
    # auto-opens a think block) -> pick the budget BEFORE capturing, so
    # early rollouts aren't answer-truncated
    ids0 = build_inputs(tok, "", QUESTIONS[0], device)
    tmpl = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                   tokenize=False, add_generation_prompt=True)
    probe_ids = model.generate(ids0, max_new_tokens=48, do_sample=False,
                               pad_token_id=tok.eos_token_id)
    probe = tok.decode(probe_ids[0, ids0.shape[1]:],
                       skip_special_tokens=False)
    thinker = "<think" in probe or "<think" in tmpl[-40:] or \
        "<|begin_of_thought|>" in probe
    budget = args.think_max_new_tokens if thinker else args.max_new_tokens
    print(f"thinker={thinker} budget={budget}", flush=True)

    acts, acts_cot, acts_ans, texts = [], [], [], []
    seed = args.seed
    n_think = 0
    for qi, q in enumerate(QUESTIONS):
        ids = build_inputs(tok, "", q, device)
        for _ in range(args.rollouts_per_q):
            seed += 1
            full, cot, ans, text, n_resp = rollout_split_states(
                model, tok, ids, device, budget, args.temperature,
                args.top_p, seed)
            if full is None:
                continue
            if cot is not None:
                n_think += 1
            acts.append(full)
            acts_cot.append(cot)
            acts_ans.append(ans)
            texts.append({"sys": "", "question": q, "text": text,
                          "n_resp": n_resp, "has_think": cot is not None})
        print(f"  {args.model} q{qi:02d} {len(acts)} rollouts "
              f"({n_think} with think)", flush=True)
    shape = acts[0].shape
    nanpad = np.full(shape, np.nan, np.float32)
    blob = {"acts": np.stack(acts).astype(np.float32),
            "acts_cot": np.stack([a if a is not None else nanpad
                                  for a in acts_cot]).astype(np.float32),
            "acts_answer": np.stack([a if a is not None else nanpad
                                     for a in acts_ans]).astype(np.float32),
            "texts": texts,
            "config": vars(args) | {"questions": QUESTIONS,
                                    "n_think": n_think}}
    tmp = out + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, out)
    print(f"wrote {out}  ({len(acts)} rollouts, {n_think} with think, "
          f"layers x dims = {shape[0]} x {shape[1]})")


if __name__ == "__main__":
    main()
