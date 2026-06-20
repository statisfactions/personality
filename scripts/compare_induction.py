"""Compare induction registers (performance / subtle / plain) for the
persona-vector extraction (W17).

All instruction is roleplay and the default assistant is itself a persona, so
there is no zero — but we can measure the GRADIENT and ask:
  1. theater drop: does naturalistic/downscaled framing shed the *...* stage
     directions (the smirk register)?
  2. rotation vs attenuation: are subtle/plain directions the SAME axis as
     performance (high cos) just weaker (small magnitude), or a DIFFERENT
     axis (low cos)?
  3. is there a consistent PERFORMANCE AXIS = mean(perf_dir - subtle_dir),
     shared across adjectives (high projection) -> the thing to subtract for
     clean assistant-drift measurement.
  4. drift vs default: ||induced(X) - default_assistant|| per register — how
     far each register moves the assistant baseline, and whether the eval
     antonyms stay over-separated.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/compare_induction.py \
      --model llama3.2 --set smoke
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

PV = Path("results/persona_vectors")
# single-asterisk stage directions only — NOT markdown **bold** (which the
# default assistant emits in lists, inflating false positives)
STAGE = re.compile(r"(?<!\*)\*[^*\n]{1,60}\*(?!\*)")
REGISTERS = ["performance", "subtle", "plain", "assistant"]


def tag_for(base, reg):
    return base if reg == "performance" else f"{base}_{reg}"


def theater(texts):
    rows = [r for c, rs in texts.items() if c != "__default__" for r in rs]
    rate = np.mean([bool(STAGE.search(r["text"])) for r in rows])
    length = np.mean([r["n_resp"] for r in rows])
    return float(rate), float(length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--set", default="smoke")
    args = ap.parse_args()

    data, texts = {}, {}
    for reg in REGISTERS:
        p = PV / f"{args.model}_{tag_for(args.set, reg)}.pt"
        if not p.exists():
            print(f"  (missing {reg}: {p.name})")
            continue
        data[reg] = torch.load(p, weights_only=False)
        texts[reg] = json.load(open(PV / f"{args.model}_{tag_for(args.set, reg)}_texts.json"))
    regs = list(data)
    mid = data[regs[0]]["report"]["mid_layer"]
    adjs = sorted(set.intersection(*[set(d["directions"]) for d in data.values()]))

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-12)

    print(f"model={args.model} set={args.set} mid_layer={mid} | {len(adjs)} adjectives\n")
    print(f"{'register':>12s} {'theater':>8s} {'resp_len':>9s} {'mean||dir||':>11s} "
          f"{'cos(W,A)':>9s} {'cos(dir,asst)':>13s}")
    dirs = {}
    for reg in regs:
        d = data[reg]
        dirs[reg] = {a: d["directions"][a][mid].astype(np.float64) for a in adjs}
        rate, length = theater(texts[reg])
        mag = np.mean([np.linalg.norm(dirs[reg][a]) for a in adjs])
        wa = (unit(dirs[reg].get("wonderful", np.zeros(1)))
              @ unit(dirs[reg].get("awful", np.zeros(1)))
              if "wonderful" in dirs[reg] and "awful" in dirs[reg] else float("nan"))
        # mean |cos| of each trait direction with this register's assistant axis
        # — does the framing keep traits ON the assistant axis (HHH) or off it?
        aa = unit(d["assistant_axis"][mid].astype(np.float64))
        casst = np.mean([abs(unit(dirs[reg][a]) @ aa) for a in adjs])
        print(f"{reg:>12s} {rate:8.2f} {length:9.0f} {mag:11.2f} {wa:+9.3f} "
              f"{casst:13.3f}")

    # rotation vs attenuation: each register's directions vs performance
    if "performance" in dirs:
        print(f"\nvs performance (per-adjective, mid layer):")
        print(f"{'register':>12s} {'mean cos':>9s} {'mag ratio':>10s}")
        for reg in regs:
            if reg == "performance":
                continue
            cos = np.mean([unit(dirs[reg][a]) @ unit(dirs["performance"][a])
                           for a in adjs])
            mr = np.mean([np.linalg.norm(dirs[reg][a]) /
                          (np.linalg.norm(dirs["performance"][a]) + 1e-12)
                          for a in adjs])
            print(f"{reg:>12s} {cos:9.3f} {mr:10.2f}")

        # Shared performance subspace? Use SVD of the difference vectors, NOT
        # projection onto their mean — the residual is a sign-flipping axis
        # (theatrical-positive for good traits, -negative for bad) that
        # CANCELS in the mean but survives in the SVD. Report PC1 explained
        # variance vs a random-direction null; compare to the directions' own
        # concentration (a separable low-rank theater axis would be MORE
        # concentrated than the signal — it isn't).
        def pc_evr(X, k=2):
            s = np.linalg.svd(X, compute_uv=False)
            return (s[:k] ** 2 / (s ** 2).sum())

        rng = np.random.default_rng(0)
        n, d = len(adjs), next(iter(dirs["performance"].values())).shape[0]
        null = np.mean([pc_evr(rng.standard_normal((n, d)))[0]
                        for _ in range(100)])
        sig = pc_evr(np.stack([dirs["performance"][a] for a in adjs]))[0]
        print(f"\nperformance-residual structure (PC1 explained var, "
              f"null={null:.2f}, signal PC1={sig:.2f}):")
        for reg in regs:
            if reg == "performance":
                continue
            U = np.stack([unit(dirs[reg][a]) for a in adjs])
            R = np.stack([dirs["performance"][a] -
                          (dirs["performance"][a] @ U[i]) * U[i]
                          for i, a in enumerate(adjs)])
            print(f"  perf ⊥ {reg}: PC1={pc_evr(R)[0]:.2f}  "
                  f"({'shared subspace' if pc_evr(R)[0] > 2 * null else 'diffuse'}"
                  f", but {'NOT ' if pc_evr(R)[0] <= sig else ''}more "
                  f"concentrated than signal)")

    # drift vs default: ||induced(X) - default|| via directions + assistant_axis
    # (adj - default) = directions[adj] - assistant_axis
    print(f"\ndrift from default-assistant (mean ||induced(X) − default||):")
    for reg in regs:
        aa = data[reg]["assistant_axis"][mid].astype(np.float64)
        drift = [np.linalg.norm(dirs[reg][a] - aa) for a in adjs]
        print(f"  {reg:>12s}: {np.mean(drift):.2f}")


if __name__ == "__main__":
    main()
