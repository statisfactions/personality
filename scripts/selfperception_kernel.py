"""How much of the dose displacement lands in the J-lens kernel? (§8m, rgb)

rgb's mechanism question in its analytic form: take the PRE-FITTED
Jacobian lens J_l for the model (Neuronpedia / Anthropic "Verbalizable
Workspace"), and decompose the dose displacement δ = act(K=32) − act(K=0)
against J_l's spectrum. Variance share (proportion of squared length,
as in PCA) in the small-singular-value subspace = directions the output
map is (nearly) blind to.

  complement account -> δ sits disproportionately in the kernel
                        relative to a matched-norm random vector
  gating account     -> δ's spectral profile looks like ENACT's
                        (output-potent), and suppression is elsewhere

Reported per model: variance share of δ in the bottom-p singular subspace,
the "potency ratio" ‖J δ‖/(σ_mean‖δ‖), and the same for random and ENACT
controls. Lens is d_model×d_model per layer; no forward passes needed.

Usage: PYTHONPATH=scripts python scripts/selfperception_kernel.py --model Qwen7
Output: results/selfperception/kernel_<model>.json
"""
import argparse
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import hf_logprobs as hf
from jlens_enact_probe import LENS_FILES, LENS_REPO

OUT = "results/selfperception"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    ap.add_argument("--layer", type=int, default=None,
                    help="lens layer (default: model's mid layer)")
    args = ap.parse_args()
    m = args.model

    meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
    mid = args.layer if args.layer is not None else meta["mid_layer"]
    massive = json.loads(meta["massive_dims"]) if isinstance(
        meta["massive_dims"], str) else meta["massive_dims"]

    J = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[m]),
                   map_location="cpu", weights_only=False)["J"]
    if isinstance(J, dict):                    # {layer: (d,d)} on this lens
        avail = sorted(J.keys())
        L = mid if mid in J else min(avail, key=lambda x: abs(x - mid))
        Jl = J[L].float()
        print(f"lens layers {avail[0]}..{avail[-1]}; using L{L} (mid={mid})")
    else:
        Jl = J[mid].float() if J.dim() == 3 else J.float()
    print(f"lens J[{mid}] shape {tuple(Jl.shape)}")
    U, S, Vh = torch.linalg.svd(Jl)
    S = S.numpy()
    V = Vh.numpy()                      # rows = right singular directions
    eff = float((S.sum() ** 2) / (S ** 2).sum())   # participation ratio
    print(f"spectrum: sigma max {S[0]:.3f} min {S[-1]:.3e} "
          f"effective rank {eff:.1f} / {len(S)}")

    z = np.load(f"{OUT}/carryover_{m}_acts.npz")
    pda = torch.load(f"results/persona_vectors/{m}_pda.pt",
                     map_location="cpu", weights_only=False)
    adjs = sorted({k.rsplit("_K", 1)[0] for k in z.files})
    rng = np.random.default_rng(0)

    def profile(v):
        v = v.astype(np.float32).copy()
        v[massive] = 0.0
        v = v / (np.linalg.norm(v) + 1e-9)
        c = V @ v                                  # coords in singular basis
        e = c ** 2                    # variance share per direction (Σ=1)
        n = len(S)
        bot = {p: float(e[int(n * (1 - p)):].sum()) for p in (0.5, 0.25, 0.1)}
        top = {p: float(e[:int(n * p)].sum()) for p in (0.1, 0.25)}
        gain = float(np.linalg.norm(S * c) / (S.mean() + 1e-12))
        return bot, top, gain

    res = {}
    for adj in adjs:
        d = z[f"{adj}_K32"].astype(np.float32) - z[f"{adj}_K0"].astype(np.float32)
        r = rng.normal(size=d.shape).astype(np.float32)
        en = np.asarray(pda["directions"][adj])[mid].astype(np.float32)
        row = {}
        for name, vec in (("dose", d), ("random", r), ("enact", en)):
            bot, top, gain = profile(vec)
            row[name] = dict(bottom50=bot[0.5], bottom25=bot[0.25],
                             bottom10=bot[0.1], top10=top[0.1],
                             top25=top[0.25], gain=gain)
        res[adj] = row

    def agg(name, key):
        return float(np.mean([res[a][name][key] for a in res]))

    print(f"\n{'vector':8s} {'E_bottom50':>11s} {'E_bottom25':>11s} "
          f"{'E_top10':>8s} {'gain':>7s}")
    for name in ("dose", "random", "enact"):
        print(f"{name:8s} {agg(name,'bottom50'):11.3f} "
              f"{agg(name,'bottom25'):11.3f} {agg(name,'top10'):8.3f} "
              f"{agg(name,'gain'):7.3f}")

    out = f"{OUT}/kernel_{m}.json"
    json.dump(dict(model=m, layer=mid, eff_rank=eff,
                   sigma=[float(x) for x in S[:20]],
                   sigma_tail=[float(x) for x in S[-5:]],
                   per_adjective=res,
                   summary={n: {k: agg(n, k) for k in
                                ("bottom50", "bottom25", "bottom10",
                                 "top10", "top25", "gain")}
                            for n in ("dose", "random", "enact")}),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
