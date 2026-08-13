"""Full v3 re-judge of the framing_run523 texts (single clean rubric).

Ablation verdict (judge_drift_ablation.json): v3 = wording exclusion +
__default__ calibration refs kills the drop_force false-positive tail
(100% -> 1% >=5) at acceptable sensitivity cost (POS mean 6.12 -> 5.78).
Mixing rubrics across framings would contaminate difference scores, so ALL
probes/framings are re-scored under v3 for all four subjects. Candidate
menus are reused verbatim from the v1 judged files; refs are two clean
__default__ rollout excerpts of the SUBJECT model (the judge calibrates to
that subject's baseline voice). Cross-family judges as in the main run.

Checkpointed per adjective to <subj>_judged_v3.jsonl (resume-safe).

Usage: PYTHONPATH=scripts python scripts/rejudge_v3.py [--wait-pid P]
"""
import argparse
import gc
import json
import os
import time

import torch

import hf_logprobs as hf
import selfperception_judge as sj
from selfperception_framing_run import judge_for

torch.set_grad_enabled(False)

RUN = "results/selfperception/framing_run523"
CONT_Q = "I'm trying to pick something to do this evening. Any suggestions?"
SUBJECTS = ["Gemma12", "Llama8", "Aya", "Qwen7"]
KEYS = ["probe", "probe_drop", "neutral", "drop_force", "be_yourself",
        "as_you_want"]


def refs_for(subj, n=2, clip=300):
    import random
    rolls = json.load(open(f"results/persona_vectors/{subj}_pda_texts.json"))
    clean = [r["text"] for r in rolls["__default__"] if not r.get("leak")]
    return [t[:clip] for t in random.Random(11).sample(clean, n)]


def main():
    ap = argparse.ArgumentParser()
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

    by_judge = {}
    for s in SUBJECTS:
        by_judge.setdefault(judge_for(s), []).append(s)
    for judge, subs in by_judge.items():
        model = tok = device = None
        for subj in subs:
            gen = {json.loads(l)["adj"]: json.loads(l)
                   for l in open(f"{RUN}/{subj}_gen.jsonl")}
            jd = [json.loads(l) for l in open(f"{RUN}/{subj}_judged.jsonl")]
            ckpt = f"{RUN}/{subj}_judged_v3.jsonl"
            done = set()
            if os.path.exists(ckpt):
                done = {json.loads(l)["adj"] for l in open(ckpt) if l.strip()}
            todo = [r for r in jd if r["adj"] not in done]
            if not todo:
                print(f"[skip] {subj}: all done", flush=True)
                continue
            if model is None:
                model, tok, device = hf.load_model(judge,
                                                   dtype=torch.bfloat16)
            refs = refs_for(subj)
            fout = open(ckpt, "a")
            for r in todo:
                g = gen[r["adj"]]
                cands = r["scores"]["cands"]
                sc = {"cands": cands, "judge": judge, "rubric": "v3"}
                for k in KEYS:
                    text = (g[k] if k in ("probe", "probe_drop")
                            else g["framings"][k])
                    mode = ("reflection" if k.startswith("probe")
                            else "continuation")
                    sc[k] = sj.score_one(
                        model, tok, device, text, mode, cands,
                        None if mode == "reflection" else CONT_Q,
                        drift_v2=True, ref_texts=refs)[0]
                fout.write(json.dumps(dict(subj=subj, adj=r["adj"],
                                           target=r["target"], k0=r["k0"],
                                           k8=r["k8"], scores=sc)) + "\n")
                fout.flush()
                print(f"  v3[{judge}] {subj:8s} {r['adj']}", flush=True)
            fout.close()
        if model is not None:
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print("=== REJUDGE V3 DONE ===", flush=True)


if __name__ == "__main__":
    main()
