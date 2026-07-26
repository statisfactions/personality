"""Replicate arXiv:2509.10078 Appendix E's sampling-validity check and test
its length confound (rgb's catch: their rank uses total log P, and canned
candidates may simply be shorter than temperature-1 samples).

For --model (default qwen2.5), --n-scenarios (default 20, stratified by
source): sample 10 responses at temp 1.0 (assistant turn, same prompt
framing as vp_rescore), then rank the 5 VP candidates among the 15 texts
by (a) total log P — their metric; (b) per-token log P. Report candidate
vs sample lengths and best-VP-candidate rank under both.

Usage: PYTHONPATH=scripts python scripts/vp_appendixE_check.py
"""
import argparse
import json

import numpy as np
import torch

import hf_logprobs as hf
from vp_rescore import load_vp, resp_logprob

OUT = "results/vp_rescore"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    rows = load_vp()
    scen = {}
    for r in rows:
        scen.setdefault(r["q"], {"scen": r["scen"], "resps": []})
        scen[r["q"]]["resps"].append(r["resp"])
    tags = json.load(open(f"{OUT}/scenario_sources.json"))
    rng = np.random.default_rng(71)
    pick = []
    for src in ("ShareGPT", "LMSYS", "Reddit", "DearAbby"):
        qs = [q for q in scen if tags[q] == src]
        pick += list(rng.choice(qs, args.n_scenarios // 4, replace=False))

    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    results = []
    for qi, q in enumerate(pick):
        msgs = [{"role": "user", "content": scen[q]["scen"]}]
        pre = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
        ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(device)
        samples = []
        for _ in range(args.samples):
            with torch.no_grad():
                out = model.generate(**ids, do_sample=True, temperature=1.0,
                                     top_p=1.0, max_new_tokens=400,
                                     pad_token_id=tok.eos_token_id)
            samples.append(tok.decode(out[0, ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
        texts = scen[q]["resps"] + samples          # 5 VP + 10 samples
        lps, nts = [], []
        for t in texts:
            lp, nt = resp_logprob(model, tok, device, scen[q]["scen"], t)
            lps.append(lp)
            nts.append(nt)
        lps, nts = np.array(lps), np.array(nts, float)
        rank_tot = (np.argsort(np.argsort(-lps)) + 1)[:5].min()
        rank_ppt = (np.argsort(np.argsort(-(lps / nts))) + 1)[:5].min()
        results.append(dict(q=int(q), src=tags[q],
                            vp_len=nts[:5].tolist(), s_len=nts[5:].tolist(),
                            rank_tot=int(rank_tot), rank_ppt=int(rank_ppt)))
        print(f"  [{qi+1}/{len(pick)}] {tags[q]:9s} best-VP rank: "
              f"total {rank_tot:2d}  per-tok {rank_ppt:2d}   "
              f"len VP {np.median(nts[:5]):.0f} vs samples "
              f"{np.median(nts[5:]):.0f}", flush=True)

    vp_len = np.concatenate([r["vp_len"] for r in results])
    s_len = np.concatenate([r["s_len"] for r in results])
    rt = np.array([r["rank_tot"] for r in results])
    rp = np.array([r["rank_ppt"] for r in results])
    print(f"\nlengths: VP median {np.median(vp_len):.0f} tokens, samples "
          f"median {np.median(s_len):.0f} (ratio {np.median(s_len)/np.median(vp_len):.1f}x)")
    print(f"best-VP-candidate rank /15: TOTAL lp median {np.median(rt):.0f} "
          f"(their metric; they report 4)   PER-TOKEN median {np.median(rp):.0f}")
    json.dump(results, open(f"{OUT}/{args.model}_appendixE.json", "w"))
    print(f"saved {OUT}/{args.model}_appendixE.json")


if __name__ == "__main__":
    main()
