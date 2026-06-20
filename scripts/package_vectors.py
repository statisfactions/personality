"""Package the cohort's valuable, committable artifacts from the big (gitignored)
*_pda.pt files (W17):

  1. <model>_pda_meta.json — the report WITHOUT pairwise_cos_mid[_ablated]
     (those are 11.7/12 MB and regenerable from the vectors). Keeps per-
     adjective diagnostics, assistant_axis info, massive_dims, mid_layer, etc.
  2. enact_vectors_mid.npz — mid-layer ENACT directions for every completed
     model (the persona vectors themselves), + assistant_axis & grand_mean at
     mid, + the shared adjective order. ~6.4 MB/model fp32.

Re-run as cohort models complete; globs whatever *_pda.pt exist.
Usage:  PYTHONPATH=scripts .venv/bin/python scripts/package_vectors.py
"""
import json
from pathlib import Path

import numpy as np
import torch

PV = Path("results/persona_vectors")
DROP = ("pairwise_cos_mid", "pairwise_cos_mid_ablated")


def main():
    pts = sorted(PV.glob("*_pda.pt"))
    arrays, axes, grands, ref_adj = {}, {}, {}, None
    for p in pts:
        model = p.stem.replace("_pda", "")
        d = torch.load(p, weights_only=False)
        mid = d["report"]["mid_layer"]
        adj = sorted(d["directions"])           # canonical order
        if ref_adj is None:
            ref_adj = adj
        assert adj == ref_adj, f"{model}: adjective set differs"
        arrays[model] = np.stack([d["directions"][a][mid] for a in adj]).astype(np.float32)
        axes[model] = d["assistant_axis"][mid].astype(np.float32)
        grands[model] = d["grand_mean"][mid].astype(np.float32)

        meta = {k: v for k, v in d["report"].items() if k not in DROP}
        mp = PV / f"{model}_pda_meta.json"
        json.dump(meta, open(mp, "w"), indent=1)
        print(f"  {model}: meta {mp.stat().st_size/1e3:.0f} KB, "
              f"vectors {arrays[model].shape}")

    assert ref_adj is not None, "no *_pda.pt files found"
    npz = PV / "enact_vectors_mid.npz"
    payload = {"adjectives": np.array(ref_adj)}
    for m in arrays:
        payload[f"dir__{m}"] = arrays[m]
        payload[f"axis__{m}"] = axes[m]
        payload[f"grand__{m}"] = grands[m]
    np.savez_compressed(npz, allow_pickle=False, **payload)
    print(f"\nsaved {npz} ({npz.stat().st_size/1e6:.1f} MB, "
          f"{len(arrays)} models, {len(ref_adj)} adjectives)")
    print("load: z=np.load(...); z['dir__llama3.2'] -> (523, hidden); "
          "z['adjectives']")


if __name__ == "__main__":
    main()
