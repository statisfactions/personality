#!/usr/bin/env python3
"""Persona × instrument response track (W7 §11.5.4 #4 prereg check).

Prereg from §11.5.9: if the model internalizes personas at r ≈ 0.74
representationally, instrument responses under those personas should
recover sampled z's at a *higher* r (instruments are designed to measure
the trait dimension head-on, not via a representational projection that
linearly aggregates many directions).

Method:
- Pick N personas from synthetic_personas.json (matched seed for direct
  comparison with persona_repr_mapping.py).
- For each (persona, marker), build chat-template prompt:
    system: persona description
    user:   "How well does the following adjective describe you?
             1 = not at all, 5 = very well. Respond with only a number.
             Adjective: <marker>
             Rating:"
- Read logprob distribution over {1, 2, 3, 4, 5} at the assistant
  generation position. Compute expected value (EV).
- For each persona, score per trait: mean(EV on high-pole markers)
  − mean(EV on low-pole markers). Convert to z-scaled scoring.
- Per-trait Pearson r(sampled z, scored trait) across personas. Compare
  to the representation r from §11.5.9.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/persona_instrument_response.py
    PYTHONPATH=scripts .venv/bin/python scripts/persona_instrument_response.py \
        --model Qwen7 --n 50
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from hf_logprobs import MODELS as ALL_MODELS, load_model
from generate_trait_personas import MARKERS, TRAITS


PERSONA_FILE = "instruments/synthetic_personas.json"
DEFAULT_MODEL = "Qwen7"
DEFAULT_N = 50
SEED = 42

LIKERT_TEMPLATE = (
    "How well does the following adjective describe you?\n"
    "1 = not at all, 2 = slightly, 3 = somewhat, 4 = quite well, 5 = very well.\n"
    "Respond with only a number.\n\n"
    'Adjective: "{marker}"\n'
    "Rating: "
)


def likert_with_persona(model, tok, device, persona_desc, marker):
    """Likert distribution over {1..5} for `marker`, conditioned on persona
    as a system message. Returns (distribution dict, EV)."""
    user_text = LIKERT_TEMPLATE.format(marker=marker)
    messages = [
        {"role": "system", "content": persona_desc},
        {"role": "user", "content": user_text},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(device)

    # Token IDs for "1".."5"
    ids = []
    for n in range(1, 6):
        toks = tok.encode(str(n), add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"token '{n}' didn't tokenize to a single id: {toks}")
        ids.append(toks[0])

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    selected = logits[ids]
    probs = torch.softmax(selected, dim=0).float().cpu().numpy()
    dist = {str(n): float(probs[n - 1]) for n in range(1, 6)}
    ev = float(sum(n * probs[n - 1] for n in range(1, 6)))
    return dist, ev


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
    print(f"Model: {args.model} ({repo})")

    n_markers_total = sum(len(MARKERS[t]["high"]) + len(MARKERS[t]["low"]) for t in TRAITS)
    print(f"Markers: {n_markers_total} total ({n_markers_total // 5} avg per trait)")

    model, tok, device = load_model(args.model)

    # Persona × marker grid
    persona_data = []
    n_calls_total = len(selected) * n_markers_total
    print(f"\nRunning {n_calls_total} forward passes...")
    call_idx = 0
    for pi, p in enumerate(selected):
        per_marker_evs = {}  # (trait, pole, marker) -> EV
        for trait in TRAITS:
            for pole in ("high", "low"):
                for marker in MARKERS[trait][pole]:
                    _, ev = likert_with_persona(
                        model, tok, device, p["description"], marker,
                    )
                    per_marker_evs[(trait, pole, marker)] = ev
                    call_idx += 1
        # Score per trait: mean(high EV) − mean(low EV) → "trait-strength" score
        scored = {}
        for trait in TRAITS:
            high_evs = [per_marker_evs[(trait, "high", m)] for m in MARKERS[trait]["high"]]
            low_evs = [per_marker_evs[(trait, "low", m)] for m in MARKERS[trait]["low"]]
            scored[trait] = float(np.mean(high_evs) - np.mean(low_evs))
        persona_data.append({
            "persona_id": p["persona_id"],
            "z_scores": p["z_scores"],
            "stanines": p["stanines"],
            "scored_trait": scored,
            "high_means": {t: float(np.mean([per_marker_evs[(t, "high", m)] for m in MARKERS[t]["high"]])) for t in TRAITS},
            "low_means": {t: float(np.mean([per_marker_evs[(t, "low", m)] for m in MARKERS[t]["low"]])) for t in TRAITS},
        })
        print(f"  persona {pi + 1}/{len(selected)} done ({call_idx}/{n_calls_total} forward passes)")

    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ----- Per-trait diagonal -----
    print("\n" + "=" * 72)
    print(f"  Per-trait diagonal: r(sampled z, scored trait via Likert markers)")
    print("=" * 72)
    diag = {}
    for t in TRAITS:
        zs = [pd["z_scores"][t] for pd in persona_data]
        scored = [pd["scored_trait"][t] for pd in persona_data]
        r = float(np.corrcoef(zs, scored)[0, 1])
        diag[t] = r
        print(f"  {t}: r = {r:+.3f}  (n={len(persona_data)})")
    print(f"\n  Mean diagonal r: {np.mean(list(diag.values())):+.3f}")

    # ----- Full 5×5 cross-correlation -----
    print("\n" + "=" * 72)
    print(f"  Full 5x5: sampled z (rows) -> scored trait (cols)")
    print("=" * 72)
    print(f"\n  {'':5s}" + "  ".join(f"{t:>6s}" for t in TRAITS))
    cross = np.zeros((5, 5))
    for i, ti in enumerate(TRAITS):
        zs = [pd["z_scores"][ti] for pd in persona_data]
        row = []
        for j, tj in enumerate(TRAITS):
            scored = [pd["scored_trait"][tj] for pd in persona_data]
            r = float(np.corrcoef(zs, scored)[0, 1])
            cross[i, j] = r
            row.append(f"{r:+6.3f}")
        print(f"  {ti:5s}" + "  ".join(row))

    diag_vals = np.diag(cross)
    off_vals = cross[~np.eye(5, dtype=bool)]
    print(f"\n  Mean diagonal:    {np.mean(diag_vals):+.3f}")
    print(f"  Mean off-diagonal: {np.mean(off_vals):+.3f}")
    print(f"  Difference:       {np.mean(diag_vals) - np.mean(off_vals):+.3f}")

    # Compare to representation result if present
    repr_path = Path(f"results/persona_repr_mapping_{args.model}_response-position.json")
    if repr_path.exists():
        with open(repr_path) as f:
            repr_data = json.load(f)
        repr_diag = repr_data["diagonal_correlations"]
        print("\n" + "=" * 72)
        print(f"  PREREG CHECK: behavioral r (Likert) vs representational r")
        print("=" * 72)
        print(f"\n  {'Trait':5s}   {'Repr r':>10s}   {'Likert r':>10s}   {'Δ (L-R)':>10s}")
        for t in TRAITS:
            r_repr = repr_diag[t]
            r_lik = diag[t]
            print(f"  {t:5s}   {r_repr:>+10.3f}   {r_lik:>+10.3f}   {r_lik - r_repr:>+10.3f}")
        mean_r = np.mean(list(repr_diag.values()))
        mean_l = np.mean(list(diag.values()))
        print(f"  {'Mean':5s}   {mean_r:>+10.3f}   {mean_l:>+10.3f}   {mean_l - mean_r:>+10.3f}")
        print(f"\n  Prediction: Likert r > Representation r (instruments score higher than activations).")
        if mean_l > mean_r:
            print(f"  → Confirmed (mean Δ = {mean_l - mean_r:+.3f}).")
        else:
            print(f"  → NOT confirmed (mean Δ = {mean_l - mean_r:+.3f}). Interesting.")

    # ----- Save -----
    payload = {
        "model": args.model,
        "hf_repo": repo,
        "n_personas": len(persona_data),
        "n_markers": n_markers_total,
        "seed": args.seed,
        "traits": TRAITS,
        "diagonal_correlations": diag,
        "cross_correlation": cross.tolist(),
        "persona_data": persona_data,
    }
    out_path = Path(f"results/persona_instrument_response_{args.model}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
