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
import random
from pathlib import Path

import numpy as np
import torch

from hf_logprobs import MODELS as ALL_MODELS, load_model
from generate_trait_personas import MARKERS, TRAITS


PERSONA_FILES = {
    "markers": "instruments/synthetic_personas.json",
    "ipip_raw": "instruments/synthetic_personas_ipip.json",
}
PERSONA_TEXT_KEYS = {
    "markers": "description",
    "ipip_raw": "ipip_raw",
}
ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
ANNOTATIONS = "instruments/ipip300_annotations.json"
SHORT = {"A": "AGR", "C": "CON", "E": "EXT", "N": "NEU", "O": "OPE"}

DEFAULT_MODEL = "Qwen7"
DEFAULT_N = 50
SEED = 42

LIKERT_TEMPLATE_MARKER = (
    "How well does the following adjective describe you?\n"
    "1 = not at all, 2 = slightly, 3 = somewhat, 4 = quite well, 5 = very well.\n"
    "Respond with only a number.\n\n"
    'Adjective: "{target}"\n'
    "Rating: "
)
LIKERT_TEMPLATE_IPIP = (
    "How well does the following statement describe you?\n"
    "1 = not at all, 2 = slightly, 3 = somewhat, 4 = quite well, 5 = very well.\n"
    "Respond with only a number.\n\n"
    'Statement: "{target}"\n'
    "Rating: "
)


def likert_with_persona(model, tok, device, persona_desc, target_text, template):
    """Likert distribution over {1..5} for `target_text`, conditioned on persona
    as a system message. Returns (distribution dict, EV)."""
    user_text = template.format(target=target_text)
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


def load_ipip_resources():
    """Load IPIP-300 items, scale defs, and annotations."""
    with open(ADMIN_SESSION) as f:
        ipip = json.load(f)["measures"]["IPIP300"]
    items = ipip["items"]
    scales = ipip["scales"]
    with open(ANNOTATIONS) as f:
        ann = json.load(f)
    return items, scales, ann


def build_ipip_rating_set(persona_picks, scales, ann, items, fixes,
                          k_per_pole, rng):
    """Per-persona IPIP rating items: K_per_pole forward + K_per_pole reverse
    per trait, mild only, no deny, no overlap with this persona's composition.
    Facet-stratified when possible.

    Returns: {trait: {"high": [(iid, text), ...], "low": [(iid, text), ...]}}
    Note: "high" = forward-keyed (high-pole), "low" = reverse-keyed (low-pole),
    matching the marker convention.
    """
    used_iids = {p["iid"] for p in persona_picks}
    strong = set(ann["strong"])
    deny = set(ann["deny"].keys())

    rating = {}
    for trait in TRAITS:
        sc = scales[f"IPIP300-{SHORT[trait]}"]
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])

        # Per-facet pools, mild-only, no deny, no persona overlap
        facets = []
        for fi in range(6):
            f_iids = iids[fi::6]
            avail = [i for i in f_iids
                     if i not in deny and i not in strong
                     and i not in used_iids]
            facets.append({
                "fwd": [i for i in avail if i not in rev],
                "rev": [i for i in avail if i in rev],
            })

        # Stratification: shuffle facet order, first k_per_pole get forward,
        # next k_per_pole get reverse. Fall back to other facets if a slot
        # is empty.
        facet_order = list(range(6))
        rng.shuffle(facet_order)

        fwd_picks, rev_picks = [], []
        all_picks_so_far = set()
        for slot, fi in enumerate(facet_order):
            target = "fwd" if slot < k_per_pole else "rev"
            target_list = fwd_picks if target == "fwd" else rev_picks
            if len(target_list) >= k_per_pole:
                continue  # already filled this polarity
            pool = [i for i in facets[fi][target] if i not in all_picks_so_far]
            if pool:
                iid = rng.choice(pool)
                target_list.append(iid)
                all_picks_so_far.add(iid)
            else:
                # fall back: try other facets at the same polarity
                for other_fi in facet_order:
                    other_pool = [i for i in facets[other_fi][target]
                                  if i not in all_picks_so_far]
                    if other_pool:
                        iid = rng.choice(other_pool)
                        target_list.append(iid)
                        all_picks_so_far.add(iid)
                        break

        rating[trait] = {
            "high": [(iid, fixes.get(iid, items[iid])) for iid in fwd_picks],
            "low":  [(iid, fixes.get(iid, items[iid])) for iid in rev_picks],
        }
    return rating


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--markers-per-pole", type=int, default=0,
                        help="If >0, use only first K markers per (trait, pole). "
                             "When --rating-target=ipip, controls K items per pole "
                             "(default: 3).")
    parser.add_argument("--persona-source", default="markers",
                        choices=list(PERSONA_FILES.keys()),
                        help="markers = original Goldberg-marker descriptions; "
                             "ipip_raw = IPIP-NEO-300 behavioral composition")
    parser.add_argument("--rating-target", default="markers",
                        choices=["markers", "ipip"],
                        help="markers = Goldberg adjective Likert template "
                             "(W7/W8 §3); ipip = IPIP behavioral statements as "
                             "rating targets, with per-persona exclusion of "
                             "the persona's own composition items")
    args = parser.parse_args()

    if args.model not in ALL_MODELS:
        raise ValueError(f"unknown model: {args.model}")
    if args.rating_target == "ipip" and args.persona_source != "ipip_raw":
        raise ValueError("--rating-target=ipip requires --persona-source=ipip_raw "
                         "(needs persona pick provenance for exclusion).")

    persona_file = PERSONA_FILES[args.persona_source]
    text_key = PERSONA_TEXT_KEYS[args.persona_source]
    print(f"Persona source: {args.persona_source} ({persona_file}, key='{text_key}')")
    print(f"Rating target: {args.rating_target}")
    with open(persona_file) as f:
        all_personas = json.load(f)["personas"]
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(all_personas), size=args.n, replace=False)
    selected = [all_personas[int(i)] for i in idxs]
    print(f"Selected {len(selected)} personas (seed={args.seed}) from {len(all_personas)}")

    repo = ALL_MODELS[args.model]
    print(f"Model: {args.model} ({repo})")

    # Set up rating targets ----------------------------------------------------
    k_per_pole = args.markers_per_pole if args.markers_per_pole > 0 else 3
    if args.rating_target == "markers":
        # Original W7/W8§3 path: shared Goldberg-marker set, possibly sliced
        per_pole_count = args.markers_per_pole if args.markers_per_pole > 0 else len(MARKERS["A"]["high"])
        shared_targets = {
            t: {pole: [(None, m) for m in MARKERS[t][pole][:per_pole_count]]
                for pole in ("high", "low")}
            for t in TRAITS
        }
        likert_template = LIKERT_TEMPLATE_MARKER
        n_targets_per_persona = sum(
            len(shared_targets[t]["high"]) + len(shared_targets[t]["low"])
            for t in TRAITS
        )
        ipip_items = ipip_scales = ipip_ann = ipip_fixes = None  # unused
    else:
        # IPIP rating-target path: per-persona stratified set, excludes persona's picks
        ipip_items, ipip_scales, ipip_ann = load_ipip_resources()
        ipip_fixes = ipip_ann["fix"]
        shared_targets = {}  # unused; per-persona targets built in loop
        likert_template = LIKERT_TEMPLATE_IPIP
        n_targets_per_persona = 5 * 2 * k_per_pole

    print(f"Targets per persona: {n_targets_per_persona} "
          f"({n_targets_per_persona // 5} avg per trait)", flush=True)

    model, tok, device = load_model(args.model)

    # Persona × target grid
    persona_data = []
    n_calls_total = len(selected) * n_targets_per_persona
    print(f"\nRunning {n_calls_total} forward passes...", flush=True)
    call_idx = 0
    rating_set_rng = random.Random(args.seed + 1)  # deterministic per-persona rating sets
    for pi, p in enumerate(selected):
        # Build rating set for this persona
        if args.rating_target == "ipip":
            assert ipip_scales is not None and ipip_ann is not None \
                and ipip_items is not None and ipip_fixes is not None
            targets = build_ipip_rating_set(
                p["picks"], ipip_scales, ipip_ann, ipip_items, ipip_fixes,
                k_per_pole, rating_set_rng,
            )
        else:
            targets = shared_targets

        per_target_evs = {}  # (trait, pole, target_text) -> EV
        for trait in TRAITS:
            for pole in ("high", "low"):
                for _, target_text in targets[trait][pole]:
                    _, ev = likert_with_persona(
                        model, tok, device, p[text_key], target_text, likert_template,
                    )
                    per_target_evs[(trait, pole, target_text)] = ev
                    call_idx += 1
        # Score per trait: mean(high EV) − mean(low EV)
        scored = {}
        for trait in TRAITS:
            high_evs = [per_target_evs[(trait, "high", t[1])] for t in targets[trait]["high"]]
            low_evs  = [per_target_evs[(trait, "low",  t[1])] for t in targets[trait]["low"]]
            scored[trait] = float(np.mean(high_evs) - np.mean(low_evs))
        persona_data.append({
            "persona_id": p["persona_id"],
            "z_scores": p["z_scores"],
            "stanines": p["stanines"],
            "scored_trait": scored,
            "high_means": {t: float(np.mean([per_target_evs[(t, "high", tg[1])] for tg in targets[t]["high"]])) for t in TRAITS},
            "low_means":  {t: float(np.mean([per_target_evs[(t, "low",  tg[1])] for tg in targets[t]["low"]]))  for t in TRAITS},
            "rating_targets": {
                t: {pole: [{"iid": iid, "text": txt} for iid, txt in targets[t][pole]]
                    for pole in ("high", "low")}
                for t in TRAITS
            },
        })
        print(f"  persona {pi + 1}/{len(selected)} done ({call_idx}/{n_calls_total} forward passes)", flush=True)

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

    # Compare to representation result if present (matched persona-source)
    repr_suffix = "_response-position"
    if args.persona_source != "markers":
        repr_suffix += f"_{args.persona_source}"
    repr_path = Path(f"results/persona_repr_mapping_{args.model}{repr_suffix}.json")
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
        "n_targets_per_persona": n_targets_per_persona,
        "rating_target": args.rating_target,
        "persona_source": args.persona_source,
        "seed": args.seed,
        "traits": TRAITS,
        "diagonal_correlations": diag,
        "cross_correlation": cross.tolist(),
        "persona_data": persona_data,
    }
    out_suffix = "" if args.persona_source == "markers" else f"_{args.persona_source}"
    if args.rating_target != "markers":
        out_suffix += f"_target-{args.rating_target}"
    out_path = Path(f"results/persona_instrument_response_{args.model}{out_suffix}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
