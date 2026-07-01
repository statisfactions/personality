"""Finalize a persona-vector run from its per-condition checkpoints WITHOUT
reloading the model (W17 rescue tool).

Big models (Gemma27/Qwen32) complete all 523+default conditions but the 54GB
model stays resident through the final aggregation+save, which np.stacks ~11GB
of acts and OOM-crashes right at the finish. This script does the identical
aggregation + diagnostics + save reading only the checkpoints (acts/sum/texts/
keep per condition) — no model — so the save runs with the model's memory freed.

Replicates extract_persona_vectors.main()'s aggregation block verbatim.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/finalize_from_checkpoints.py \
      --model Gemma27
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from extract_persona_vectors import (bootstrap_coherence, PLACEBOS,
                                      DEFAULT_KEY, INDUCTION, QUESTIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="pda")
    ap.add_argument("--out", default="results/persona_vectors")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sys", type=int, default=5)
    ap.add_argument("--n-questions", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out)
    ckpt_dir = out_dir / f"{args.model}_{args.tag}_ckpt"
    files = sorted(ckpt_dir.glob("*.pt"))
    assert files, f"no checkpoints in {ckpt_dir}"

    acts, sums, texts, keep = {}, {}, {}, None
    for p in files:
        cond = p.stem
        b = torch.load(p, weights_only=False)
        acts[cond] = b["acts"]
        sums[cond] = b["sum"]
        texts[cond] = b["texts"]
        keep = b["keep"]
    adjectives = sorted(c for c in acts if c != DEFAULT_KEY)
    conditions = [DEFAULT_KEY] + adjectives
    n_layers = sums[DEFAULT_KEY].shape[0] - 1
    mid = (n_layers + 1) // 2
    mid_idx = keep.index(mid)
    print(f"{args.model}: {len(conditions)} conditions, n_layers={n_layers}, "
          f"mid={mid} (keep idx {mid_idx})")

    # --- Directions (fp64, all layers) — verbatim from extract_persona_vectors
    cond_means = {c: sums[c] / len(acts[c]) for c in conditions}
    role_means = [cond_means[a] for a in adjectives]
    grand_mean = np.mean(role_means, axis=0)
    directions = {a: cond_means[a] - grand_mean for a in adjectives}
    assistant_axis = cond_means[DEFAULT_KEY] - grand_mean

    all_mid = np.concatenate([np.stack(acts[c])[:, mid_idx, :]
                              for c in conditions], axis=0).astype(np.float64)
    mean_abs = np.abs(all_mid).mean(axis=0)
    massive = np.where(mean_abs >= 20 * np.median(mean_abs))[0]
    mask = np.ones(all_mid.shape[1]); mask[massive] = 0.0
    del all_mid
    if len(massive):
        print(f"  !! {len(massive)} massive dims at layer {mid}: "
              f"{[int(i) for i in massive[:10]]}")

    rng = np.random.default_rng(args.seed)
    report = {"model": args.model, "mid_layer": mid, "n_layers": n_layers,
              "massive_dims": [int(i) for i in massive],
              "adjectives": {}, "pairwise_cos_mid": {}, "assistant_axis": {}}

    def unit(v):
        return v / np.linalg.norm(v)

    for a in adjectives:
        d = directions[a][mid]
        X = np.stack([r[mid_idx] for r in acts[a]]).astype(np.float64)
        cos = bootstrap_coherence(X, grand_mean[mid], d, args.bootstrap, rng)
        proj = X @ unit(d)
        norms = np.linalg.norm(X, axis=1)
        lens = np.array([t["n_resp"] for t in texts[a]], dtype=np.float64)
        norm_r = float(np.corrcoef(proj, norms)[0, 1])
        len_r = (float(np.corrcoef(proj, lens)[0, 1]) if lens.std() > 0 else 0.0)
        Xa = X * mask
        proj_a = Xa @ unit(d * mask)
        norm_r_abl = float(np.corrcoef(proj_a, np.linalg.norm(Xa, axis=1))[0, 1])
        dom = float(np.sort(d ** 2)[-10:].sum() / (d ** 2).sum())
        report["adjectives"][a] = {
            "n_rollouts": len(acts[a]), "placebo": a in PLACEBOS,
            "dir_norm": float(np.linalg.norm(d)),
            "boot_cos_mean": float(cos.mean()),
            "boot_cos_p05": float(np.percentile(cos, 5)),
            "norm_corr": norm_r, "norm_corr_ablated": norm_r_abl,
            "len_corr": len_r, "top10_dim_share": dom,
            "leak_rate": float(np.mean([t["leak"] for t in texts[a]])),
        }

    report["pairwise_cos_mid_ablated"] = {}
    for i, a in enumerate(adjectives):
        ua, uam = unit(directions[a][mid]), unit(directions[a][mid] * mask)
        for b in adjectives[i + 1:]:
            report["pairwise_cos_mid"][f"{a}|{b}"] = float(ua @ unit(directions[b][mid]))
            report["pairwise_cos_mid_ablated"][f"{a}|{b}"] = float(uam @ unit(directions[b][mid] * mask))

    aa = assistant_axis[mid]
    report["assistant_axis"] = {
        "norm": float(np.linalg.norm(aa)),
        "cos_to_adjectives": {a: float(unit(aa) @ unit(directions[a][mid]))
                              for a in adjectives},
        "cos_to_adjectives_ablated": {
            a: float(unit(aa * mask) @ unit(directions[a][mid] * mask))
            for a in adjectives},
    }
    print(f"  mid: ||assistant_axis||={np.linalg.norm(aa):.2f}, "
          f"wonderful|awful={report['pairwise_cos_mid'].get('awful|wonderful', report['pairwise_cos_mid'].get('wonderful|awful')):.3f}")

    base = out_dir / f"{args.model}_{args.tag}"
    torch.save({
        "acts": {c: np.stack(acts[c]).astype(np.float32) for c in conditions},
        "acts_layers": keep,
        "directions": {a: directions[a].astype(np.float32) for a in adjectives},
        "assistant_axis": assistant_axis.astype(np.float32),
        "grand_mean": grand_mean.astype(np.float32),
        "report": report,
        "config": {"model": args.model, "pda": True, "n_sys": args.n_sys,
                   "n_questions": args.n_questions, "rollouts": 1,
                   "avg_window": 60, "acts_stride": 4, "induction": "performance",
                   "bootstrap": args.bootstrap, "seed": args.seed,
                   "no_save_acts": False, "tag": args.tag,
                   "finalized_from_checkpoints": True},
        "questions": QUESTIONS[:args.n_questions],
        "sys_templates": INDUCTION["performance"][:args.n_sys],
    }, f"{base}.pt")
    json.dump(texts, open(f"{base}_texts.json", "w"), indent=1)
    json.dump(report, open(f"{base}_report.json", "w"), indent=1)
    print(f"saved {base}.pt / _texts.json / _report.json")


if __name__ == "__main__":
    main()
