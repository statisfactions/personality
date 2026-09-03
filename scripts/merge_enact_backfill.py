"""Merge a pda_backfill ENACT run's new-adjective vectors into enact_mid.

The extractor's directions are cond_mean - grand(role means of ITS OWN
run), so a subset run's vectors are referenced to a subset grand. This
script re-references: raw cond_mean = direction + backfill grand, then
new_dir = cond_mean - ORIGINAL grand (enact_mid npz 'grand'), so merged
vectors share the original 523-run reference frame. Also appends the new
adjectives' rollout texts to {model}_pda_texts.json. Idempotent.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/merge_enact_backfill.py --model Aya
"""
import argparse
import json
import os

import numpy as np
import torch

MISSING = {"inspirational", "insensitive"}
PV = "results/persona_vectors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="pda_backfill")
    args = ap.parse_args()
    m = args.model

    npz_path = f"{PV}/enact_mid/{m}.npz"
    z = np.load(npz_path, allow_pickle=True)
    adj = [str(a).lower() for a in z["adjectives"]]
    need = sorted(MISSING - set(adj))
    if not need:
        print(f"[merge {m}] already complete ({len(adj)} adjectives)")
        return

    bf = torch.load(f"{PV}/{m}_{args.tag}.pt", map_location="cpu",
                    weights_only=False)
    mid_bf = bf["report"]["mid_layer"]
    meta = json.load(open(f"{PV}/{m}_pda_meta.json"))
    mid_orig = meta["mid_layer"]
    assert mid_bf == mid_orig, (m, mid_bf, mid_orig)

    D = np.asarray(z["dir"], float)
    grand_orig = np.asarray(z["grand"], float)
    grand_bf = np.asarray(bf["grand_mean"], float)[mid_bf]
    assert grand_bf.shape == grand_orig.shape

    rows = []
    for a in need:
        cond_mean = np.asarray(bf["directions"][a], float)[mid_bf] + grand_bf
        rows.append(cond_mean - grand_orig)
        # sanity: new vector norm within the existing distribution
        norms = np.linalg.norm(D, axis=1)
        nn = np.linalg.norm(rows[-1])
        print(f"[merge {m}] {a}: |v|={nn:.1f} (existing p5-p95 "
              f"{np.percentile(norms, 5):.1f}-{np.percentile(norms, 95):.1f})")

    all_adj = adj + need
    order = np.argsort(all_adj)
    D_new = np.vstack([D, np.array(rows)])[order]
    adj_new = np.array(all_adj, dtype=object)[order]
    np.savez(npz_path, dir=D_new.astype(z["dir"].dtype),
             axis=z["axis"], grand=z["grand"], adjectives=adj_new)
    print(f"[merge {m}] enact_mid updated: {len(adj)} -> {len(adj_new)}")

    tx_path = f"{PV}/{m}_pda_texts.json"
    if os.path.exists(tx_path):
        tx = json.load(open(tx_path))
        bft = json.load(open(f"{PV}/{m}_{args.tag}_texts.json"))
        added = 0
        for a in need:
            if a not in {k.lower() for k in tx} and a in bft:
                tx[a] = bft[a]
                added += 1
        if added:
            json.dump(tx, open(tx_path, "w"), indent=1)
        print(f"[merge {m}] texts: +{added}")


if __name__ == "__main__":
    main()
