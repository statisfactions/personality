"""Cross-judge agreement: Gemma4 re-scores texts Qwen32 already judged.

Blocking check for the full-523 cross-quadrant table: Qwen7's cells are
Gemma4-judged (cross-family constraint), everything else Qwen32-judged.
Sample subjects from families NEITHER judge belongs to (Llama8, Aya), have
Gemma4 score the same texts with the SAME candidate menus (stored in the
judged jsonl), and compare per axis: r, mean offset, and the drift-scale
mapping. Any Qwen7-vs-others comparison inherits these calibration curves.

Usage: PYTHONPATH=scripts python scripts/judge_agreement.py [--wait-pid P]
       [--n-adj 30]
Out:   results/selfperception/judge_agreement.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

import hf_logprobs as hf
import selfperception_judge as sj

torch.set_grad_enabled(False)

RUN = "results/selfperception/framing_run523"
OUT = "results/selfperception/judge_agreement.json"
CONT_Q = "I'm trying to pick something to do this evening. Any suggestions?"
KEYS = ["probe", "probe_drop", "neutral", "drop_force", "be_yourself",
        "as_you_want"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-adj", type=int, default=30)
    ap.add_argument("--wait-pid", type=int, default=None)
    args = ap.parse_args()

    if args.wait_pid:
        print(f"waiting on pid {args.wait_pid}", flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
                time.sleep(60)
            except OSError:
                break
        print("pid gone, starting", flush=True)

    rng = random.Random(7)
    items = []
    for subj in ("Llama8", "Aya"):
        gen = {json.loads(l)["adj"]: json.loads(l)
               for l in open(f"{RUN}/{subj}_gen.jsonl")}
        jd = [json.loads(l) for l in open(f"{RUN}/{subj}_judged.jsonl")]
        for r in rng.sample(jd, args.n_adj):
            items.append((subj, r, gen[r["adj"]]))

    model, tok, device = hf.load_model("Gemma4", dtype=torch.bfloat16)
    out = []
    for subj, r, g in items:
        cands = r["scores"]["cands"]
        rec = {"subj": subj, "adj": r["adj"]}
        for k in KEYS:
            text = (g[k] if k in ("probe", "probe_drop")
                    else g["framings"][k])
            mode = "reflection" if k.startswith("probe") else "continuation"
            j, _ = sj.score_one(model, tok, device, text, mode, cands,
                                None if mode == "reflection" else CONT_Q)
            rec[k] = {"qwen32": {a: r["scores"][k].get(a)
                                 for a in ("drift", "disavowal", "awareness")},
                      "gemma4": {a: (j or {}).get(a)
                                 for a in ("drift", "disavowal", "awareness")}}
        out.append(rec)
        print(f"  {subj:7s} {r['adj']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)

    print("\n===== AGREEMENT (paired, all texts) =====")
    for axis in ("drift", "disavowal"):
        q, gm = [], []
        for rec in out:
            for k in KEYS:
                a, b = rec[k]["qwen32"].get(axis), rec[k]["gemma4"].get(axis)
                if a is not None and b is not None:
                    q.append(a)
                    gm.append(b)
        q, gm = np.array(q, float), np.array(gm, float)
        print(f"  {axis:10s} r={np.corrcoef(q, gm)[0, 1]:.3f}  "
              f"mean Qwen32={q.mean():.2f}  Gemma4={gm.mean():.2f}  "
              f"offset={gm.mean() - q.mean():+.2f}  n={len(q)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
