#!/usr/bin/env python3
"""Persona × representation mapping (W7 §11.5.4 #4, rgb's twist).

Takes synthetic Big Five personas (z-sampled, statisfactions's
generate_trait_personas), runs each persona description through the model,
projects the resulting last-token activation onto Big Five marker-derived
trait directions, and asks: does the projection magnitude track the
sampled z-score, per trait, across personas?

Cross-correlation matrix tests:
- Diagonal: per-trait reconstruction. r(sampled[t], projected[t])
- Off-diagonal: trait confusion. If sampled E predicts projected C, the
  marker directions aren't independent.

Caveat: persona descriptions literally use the Goldberg marker adjectives
that defined the trait directions, so this is partly a tautology check
(can the model linearly reconstruct trait composition of a marker-rich
prompt). Still informative as a baseline. The harder test (prompt with
persona, ask a neutral question, read activations at the response
position) is queued as a follow-up.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py
    PYTHONPATH=scripts .venv/bin/python scripts/persona_repr_mapping.py \
        --model Qwen7 --n 50
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS as ALL_MODELS, load_model
from generate_trait_personas import MARKERS, TRAITS


PERSONA_FILE = "instruments/synthetic_personas.json"
CACHE_DIR = Path("results/phase_b_cache")
DEFAULT_MODEL = "Qwen7"
DEFAULT_N = 50
SEED = 42


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def extract_marker_directions(model, tok, device, common_layer, neutral_layer):
    """Return {trait: unit-norm direction (hidden_dim,)} via mean(high)-mean(low),
    neutral-PC-projected. Same recipe as markers_as_stimuli.py."""
    pcs, _, _ = mdx.compute_pc_projection(neutral_layer, 0.5)

    print(f"  extracting {sum(len(v['high']) + len(v['low']) for v in MARKERS.values())} marker activations...")
    by_trait_pole = {(t, p): [] for t in TRAITS for p in ("high", "low")}
    for trait in TRAITS:
        for pole in ("high", "low"):
            for adj in MARKERS[trait][pole]:
                a = mdx.hidden_states_for_text(
                    model, tok, adj, device,
                    split_prefix=None, chat_template=True,
                )
                by_trait_pole[(trait, pole)].append(a[common_layer].float().numpy())

    directions = {}
    for trait in TRAITS:
        high_mean = np.mean(by_trait_pole[(trait, "high")], axis=0)
        low_mean = np.mean(by_trait_pole[(trait, "low")], axis=0)
        directions[trait] = unit(mdx.project_out_pcs(high_mean - low_mean, pcs))
    return directions


def extract_persona_projections(model, tok, device, personas, directions,
                                common_layer, neutral_baseline):
    """For each persona, chat-template the description, get last-token
    activation at common_layer, subtract neutral baseline, project on each
    trait direction. Returns list of dicts."""
    out = []
    for i, p in enumerate(personas):
        a = mdx.hidden_states_for_text(
            model, tok, p["description"], device,
            split_prefix=None, chat_template=True,
        )
        act = a[common_layer].float().numpy()
        act_centered = act - neutral_baseline
        projs = {t: float(np.dot(act_centered, directions[t])) for t in TRAITS}
        out.append({
            "persona_id": p["persona_id"],
            "z_scores": p["z_scores"],
            "stanines": p["stanines"],
            "projections": projs,
        })
        if (i + 1) % 10 == 0:
            print(f"    persona {i + 1}/{len(personas)} done")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.model not in ALL_MODELS:
        raise ValueError(f"unknown model: {args.model}")

    with open(PERSONA_FILE) as f:
        all_personas = json.load(f)["personas"]

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(all_personas), size=args.n, replace=False)
    selected = [all_personas[int(i)] for i in idxs]
    print(f"Selected {len(selected)} personas (seed={args.seed}) from {len(all_personas)}")

    repo = ALL_MODELS[args.model]
    tag = safe(repo)
    print(f"Model: {args.model} ({repo})")

    neutral_path = CACHE_DIR / f"{tag}_neutral_chat.pt"
    if not neutral_path.exists():
        raise FileNotFoundError(f"missing neutral cache: {neutral_path}")
    neutral = torch.load(neutral_path, weights_only=False)
    if isinstance(neutral, torch.Tensor):
        neutral_np = neutral.numpy()
    else:
        neutral_np = neutral
    n_layers = neutral_np.shape[1]
    common_layer = int(round(n_layers * 2 / 3))
    neutral_layer_t = torch.from_numpy(neutral_np[:, common_layer, :])
    neutral_baseline = neutral_np[:, common_layer, :].mean(axis=0)
    print(f"Common layer: {common_layer}/{n_layers}")

    model, tok, device = load_model(args.model)

    print("\n--- Extracting Big Five marker directions ---")
    directions = extract_marker_directions(
        model, tok, device, common_layer, neutral_layer_t,
    )

    print("\n--- Extracting persona activations ---")
    persona_data = extract_persona_projections(
        model, tok, device, selected, directions, common_layer, neutral_baseline,
    )

    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ----- Per-trait diagonal correlation -----
    print("\n" + "=" * 72)
    print(f"  Per-trait diagonal: r(sampled z, projection)")
    print("=" * 72)
    diag = {}
    for t in TRAITS:
        zs = [pd["z_scores"][t] for pd in persona_data]
        prs = [pd["projections"][t] for pd in persona_data]
        r = float(np.corrcoef(zs, prs)[0, 1])
        diag[t] = r
        print(f"  {t}: r = {r:+.3f}  (n={len(persona_data)})")
    print(f"\n  Mean diagonal r: {np.mean(list(diag.values())):+.3f}")

    # ----- Full 5×5 cross-correlation -----
    print("\n" + "=" * 72)
    print(f"  Full 5x5 cross-correlation: sampled z (rows) -> projection (cols)")
    print("=" * 72)
    print(f"\n  {'':5s}" + "  ".join(f"{t:>6s}" for t in TRAITS))
    cross = np.zeros((5, 5))
    for i, ti in enumerate(TRAITS):
        zs = [pd["z_scores"][ti] for pd in persona_data]
        row = []
        for j, tj in enumerate(TRAITS):
            prs = [pd["projections"][tj] for pd in persona_data]
            r = float(np.corrcoef(zs, prs)[0, 1])
            cross[i, j] = r
            row.append(f"{r:+6.3f}")
        print(f"  {ti:5s}" + "  ".join(row))

    # Diagonal vs off-diagonal mean
    diag_vals = np.diag(cross)
    off_vals = cross[~np.eye(5, dtype=bool)]
    print(f"\n  Mean diagonal:    {np.mean(diag_vals):+.3f}")
    print(f"  Mean off-diagonal: {np.mean(off_vals):+.3f}")
    print(f"  Difference:       {np.mean(diag_vals) - np.mean(off_vals):+.3f}")

    # ----- Save -----
    payload = {
        "model": args.model,
        "hf_repo": repo,
        "n_personas": len(persona_data),
        "seed": args.seed,
        "common_layer": common_layer,
        "n_layers": n_layers,
        "traits": TRAITS,
        "diagonal_correlations": diag,
        "cross_correlation": cross.tolist(),
        "persona_data": persona_data,
    }
    out_path = Path(f"results/persona_repr_mapping_{args.model}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
