"""Are base models assistant-shaped, merely desirable, or shapeless? (rgb)

523-adjective SELF (direct framing, bare text) on base models vs their
tuned siblings and the cohort. Reports EV mean/SD, entropy, and
correlations with the cohort-mean profile and its PC1 (the
desirability/assistant axis, W18). Entropy and spread are reported for
EVERY model, not just the bases: a flat distribution has EV 4.0 by
construction, so EV-based correlations are uninterpretable without them.

Usage: PYTHONPATH=scripts python scripts/base_self_shape.py
"""
import glob
import json

import numpy as np

SR = "results/adjectives/selfreport"
TUNED = ["llama3.2", "Llama8", "gemma3", "Gemma12", "Gemma27", "qwen2.5",
         "Qwen7", "Qwen32", "phi4", "Aya", "FalconMamba", "Olmo2Inst"]
BASES = [("Llama8Base", "Llama8"), ("Qwen7Base", "Qwen7"),
         ("Olmo2Base", "Olmo2Inst")]


def prof(m):
    d = json.load(open(f"{SR}/{m}_self_full.json"))
    adjs = d["adjectives"]
    r = d["results"]["direct"]
    return (adjs, np.array([r[a]["ev"] for a in adjs]),
            np.array([r[a]["entropy"] for a in adjs]))


def main():
    have = [m for m in TUNED if glob.glob(f"{SR}/{m}_self_full.json")]
    adjs = prof(have[0])[0]
    M = np.stack([prof(m)[1] for m in have])
    cohort = M.mean(0)
    X = M - M.mean(1, keepdims=True)
    X = X - X.mean(0)
    vt = np.linalg.svd(X, full_matrices=False)[2]
    pc1 = vt[0] * np.sign(np.corrcoef(vt[0], cohort)[0, 1])

    print(f"{'model':14s} {'meanEV':>7s} {'SD':>6s} {'H':>6s} "
          f"{'r(sibling)':>11s} {'r(cohort)':>10s} {'r(PC1)':>8s}")
    for m in have:
        _, ev, ent = prof(m)
        print(f"{m:14s} {ev.mean():7.2f} {ev.std():6.2f} {ent.mean():6.2f} "
              f"{'':>11s} {np.corrcoef(ev, cohort)[0,1]:+10.2f} "
              f"{np.corrcoef(ev, pc1)[0,1]:+8.2f}")
    print("--- bases ---")
    for b, sib in BASES:
        if not glob.glob(f"{SR}/{b}_self_full.json"):
            print(f"{b:14s} {'pending':>7s}")
            continue
        _, ev, ent = prof(b)
        rs = (np.corrcoef(ev, prof(sib)[1])[0, 1]
              if glob.glob(f"{SR}/{sib}_self_full.json") else float("nan"))
        print(f"{b:14s} {ev.mean():7.2f} {ev.std():6.2f} {ent.mean():6.2f} "
              f"{rs:+11.2f} {np.corrcoef(ev, cohort)[0,1]:+10.2f} "
              f"{np.corrcoef(ev, pc1)[0,1]:+8.2f}")
        o = np.argsort(-ev)
        print(f"   top: {[adjs[i] for i in o[:6]]}")
        print(f"   bot: {[adjs[i] for i in o[-6:]]}")
    print(f"\ncohort reference: mean SD {np.mean([prof(m)[1].std() for m in have]):.2f}, "
          f"mean H {np.mean([prof(m)[2].mean() for m in have]):.2f}")


if __name__ == "__main__":
    main()
