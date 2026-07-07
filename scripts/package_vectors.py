"""Package the cohort's valuable, committable artifacts from the big (gitignored)
*_pda.pt files (W17):

  1. <model>_pda_meta.json — the report WITHOUT pairwise_cos_mid[_ablated]
     (those are 11.7/12 MB and regenerable from the vectors). Keeps per-
     adjective diagnostics, assistant_axis info, massive_dims, mid_layer, etc.
  2. enact_mid/<model>.npz — per-model mid-layer ENACT directions (the
     persona vectors themselves), + assistant_axis & grand_mean at mid, + the
     adjective order. ~7 MB fp32 each. ONE FILE PER MODEL so a cohort refresh
     only commits the changed model's blob (np.savez output is
     byte-deterministic, so unchanged models don't churn git history).

The legacy combined enact_vectors_mid.npz (all models in one ~72 MB file) is
frozen at the 2026-07-04 10-model snapshot in git; regenerate it only with
--combined, and think twice before committing it — each revision is a new
full-size blob (GitHub soft limit 50 MB).

Re-run as cohort models complete; globs whatever *_pda.pt exist.
Usage:  PYTHONPATH=scripts .venv/bin/python scripts/package_vectors.py [--combined]
"""
import json
from pathlib import Path

import numpy as np
import torch

PV = Path("results/persona_vectors")
DROP = ("pairwise_cos_mid", "pairwise_cos_mid_ablated")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", action="store_true",
                    help="also regenerate the legacy combined npz (large blob)")
    args = ap.parse_args()
    (PV / "enact_mid").mkdir(exist_ok=True)
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
    for m in arrays:
        mp = PV / "enact_mid" / f"{m}.npz"
        np.savez_compressed(mp, allow_pickle=False,
                            dir=arrays[m], axis=axes[m], grand=grands[m],
                            adjectives=np.array(ref_adj))
        print(f"  saved {mp} ({mp.stat().st_size/1e6:.1f} MB)")
    if args.combined:
        npz = PV / "enact_vectors_mid.npz"
        payload = {"adjectives": np.array(ref_adj)}
        for m in arrays:
            payload[f"dir__{m}"] = arrays[m]
            payload[f"axis__{m}"] = axes[m]
            payload[f"grand__{m}"] = grands[m]
        np.savez_compressed(npz, allow_pickle=False, **payload)
        print(f"\nsaved legacy combined {npz} "
              f"({npz.stat().st_size/1e6:.1f} MB)")
    print(f"\n{len(arrays)} models, {len(ref_adj)} adjectives")
    print("load: z=np.load('enact_mid/<model>.npz'); z['dir'] -> "
          "(523, hidden); z['adjectives']")


if __name__ == "__main__":
    main()
