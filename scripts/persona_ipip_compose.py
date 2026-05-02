#!/usr/bin/env python3
"""Compose persona descriptions from IPIP-NEO-300 behavioral items (W8).

Replaces the Goldberg-marker descriptions in `instruments/synthetic_personas.json`
(format: "You are very extraverted, very energetic, ..." — adjective lists)
with naturalistic first-person behavioral self-descriptions assembled from
validated IPIP items.

Two output forms per persona:
  - ipip_raw    : items concatenated as-is, master-shuffled across traits.
                  Stilted ("I... I... I...") but content is faithful by
                  construction (drawn from validated IPIP behavioral items
                  with published trait/facet loadings).
  - ipip_(reflowed) : OPTIONAL Sonnet-paraphrased version of the raw form
                       (not produced by this script — generated separately
                       and merged in). The raw-vs-reflow contrast isolates
                       stylistic naturalness with content held constant.

Selection rule (see instruments/ipip300_annotations.json _method block):
  - K = 6 items per trait (one per facet, stratified)
  - Polarity ratio by z-band (band_K6):
      z >= +1.0:    6F / 0R
      +0.3 <= z:    4F / 2R
      |z| < 0.3:    3F / 3R
      z <= -0.3:    2F / 4R
      z <= -1.0:    0F / 6R
  - Tier by stanine (intensity rule):
      stanines 3-7 (|z| roughly <= 1): MILD items only
      stanines 1-2, 8-9: MILD + STRONG mixed
  - Master-shuffle the 30 items at output so traits aren't grouped.
  - Typo overrides applied.
  - Deny-listed items skipped.
  - Fallback: if a (facet, polarity, tier) cell is empty, fall back first to
    same polarity any-tier, then to a different facet's same-polarity pool.

Usage:
    .venv/bin/python scripts/persona_ipip_compose.py             # all 400 personas
    .venv/bin/python scripts/persona_ipip_compose.py --persona-ids s1,s6,s50
    .venv/bin/python scripts/persona_ipip_compose.py --preview 3 # show 3 to stdout, no write
"""

import argparse
import json
import random
from pathlib import Path

ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
ANNOTATIONS = "instruments/ipip300_annotations.json"
PERSONAS_IN = "instruments/synthetic_personas.json"
PERSONAS_OUT = "instruments/synthetic_personas_ipip.json"

TRAITS = ["A", "C", "E", "N", "O"]
SHORT = {"A": "AGR", "C": "CON", "E": "EXT", "N": "NEU", "O": "OPE"}
FACETS = {
    "N": ["Anxiety", "Anger", "Depression", "Self-Consciousness",
          "Immoderation", "Vulnerability"],
    "E": ["Friendliness", "Gregariousness", "Assertiveness", "Activity Level",
          "Excitement-Seeking", "Cheerfulness"],
    "O": ["Imagination", "Artistic Interests", "Emotionality", "Adventurousness",
          "Intellect", "Liberalism"],
    "A": ["Trust", "Morality", "Altruism", "Cooperation", "Modesty", "Sympathy"],
    "C": ["Self-Efficacy", "Orderliness", "Dutifulness", "Achievement-Striving",
          "Self-Discipline", "Cautiousness"],
}


def band_K6(z):
    """Map z-score to (n_forward, n_reverse) — sums to 6."""
    if z >= 1.0:   return 6
    if z >= 0.3:   return 4
    if z > -0.3:   return 3
    if z > -1.0:   return 2
    return 0


def stanine_uses_strong(stanine):
    """stanines 3-7 use MILD only; 1-2 and 8-9 use MILD+STRONG."""
    return stanine <= 2 or stanine >= 8


def load_resources():
    with open(ADMIN_SESSION) as f:
        scales = json.load(f)["measures"]["IPIP300"]
        items = scales["items"]
        scale_defs = scales["scales"]
    with open(ANNOTATIONS) as f:
        ann = json.load(f)

    strong = set(ann["strong"])
    deny = set(ann["deny"].keys())
    fixes = ann["fix"]

    # Build per-facet item pools, tier-keyed.
    # facet_pools[trait][facet_idx] = {'mF': [...], 'mR': [...], 'sF': [...], 'sR': [...]}
    facet_pools = {}
    for t in TRAITS:
        sc = scale_defs[f"IPIP300-{SHORT[t]}"]
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])
        per_facet = []
        for fi in range(6):
            facet_iids = [i for i in iids[fi::6] if i not in deny]
            pool = {
                "mF": [i for i in facet_iids if i not in rev and i not in strong],
                "mR": [i for i in facet_iids if i in rev and i not in strong],
                "sF": [i for i in facet_iids if i not in rev and i in strong],
                "sR": [i for i in facet_iids if i in rev and i in strong],
            }
            per_facet.append(pool)
        facet_pools[t] = per_facet

    return items, fixes, facet_pools


def render_item(iid, items, fixes):
    return fixes.get(iid, items[iid])


def pick_for_trait(trait, z, stanine, facet_pools, rng):
    """Pick 6 items for one trait — one per facet — given band + tier rule."""
    nF = band_K6(z)
    use_strong = stanine_uses_strong(stanine)

    facet_order = list(range(6))
    rng.shuffle(facet_order)

    picks = []
    for slot, fi in enumerate(facet_order):
        polarity = "F" if slot < nF else "R"
        # Build candidate pool: mild always; add strong if extreme stanine.
        tier_keys = [f"m{polarity}"]
        if use_strong:
            tier_keys.append(f"s{polarity}")
        candidates = [iid for k in tier_keys for iid in facet_pools[trait][fi][k]]

        if not candidates:
            # Fallback 1: same polarity, any tier (covers strong-only facets
            # at non-extreme stanines, which shouldn't happen post-validation
            # but defensive).
            candidates = (facet_pools[trait][fi][f"m{polarity}"]
                          + facet_pools[trait][fi][f"s{polarity}"])

        if not candidates:
            # Fallback 2: pull from another facet at same polarity. Mark this
            # so we know stratification was relaxed.
            for other_fi in facet_order:
                if other_fi == fi:
                    continue
                pool = (facet_pools[trait][other_fi][f"m{polarity}"]
                        + (facet_pools[trait][other_fi][f"s{polarity}"]
                           if use_strong else []))
                if pool:
                    iid = rng.choice(pool)
                    picks.append({
                        "trait": trait, "facet": FACETS[trait][fi],
                        "polarity": polarity, "iid": iid,
                        "fallback": "cross-facet",
                    })
                    break
            else:
                raise RuntimeError(
                    f"No items available for {trait}.{FACETS[trait][fi]} polarity={polarity}"
                )
            continue

        iid = rng.choice(candidates)
        picks.append({
            "trait": trait, "facet": FACETS[trait][fi],
            "polarity": polarity, "iid": iid, "fallback": None,
        })
    return picks


def compose_persona(persona, items, fixes, facet_pools, rng_seed):
    """Compose one persona's IPIP description.

    Returns:
        {
          'ipip_raw': str (master-shuffled flat prose),
          'picks': list of pick dicts (provenance, in trait-grouped order),
        }
    """
    # Per-persona deterministic RNG: seeded from persona_id + global seed for
    # reproducibility. Same persona always produces same composition.
    seed = (rng_seed * 100003 + sum(ord(c) for c in persona["persona_id"])) & 0xFFFFFFFF
    rng = random.Random(seed)

    all_picks = []
    for t in TRAITS:
        z = persona["z_scores"][t]
        st = persona["stanines"][t]
        all_picks.extend(pick_for_trait(t, z, st, facet_pools, rng))

    # Master shuffle for output prose
    shuffled = all_picks.copy()
    rng.shuffle(shuffled)
    sentences = [render_item(p["iid"], items, fixes) for p in shuffled]
    raw_text = ". ".join(sentences) + "."

    return {"ipip_raw": raw_text, "picks": all_picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-ids", default=None,
                    help="Comma-separated persona IDs (e.g. s1,s6,s50). Default: all.")
    ap.add_argument("--preview", type=int, default=0,
                    help="Preview N personas to stdout, do not write file.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global RNG seed (combined with persona_id for per-persona).")
    ap.add_argument("--output", default=PERSONAS_OUT)
    args = ap.parse_args()

    items, fixes, facet_pools = load_resources()
    with open(PERSONAS_IN) as f:
        sp = json.load(f)

    if args.persona_ids:
        wanted = set(args.persona_ids.split(","))
        target_personas = [p for p in sp["personas"] if p["persona_id"] in wanted]
    else:
        target_personas = sp["personas"]

    composed = []
    fallback_count = 0
    for p in target_personas:
        out = compose_persona(p, items, fixes, facet_pools, args.seed)
        fallback_count += sum(1 for pick in out["picks"] if pick["fallback"])
        composed.append({
            "persona_id": p["persona_id"],
            "z_scores": p["z_scores"],
            "stanines": p["stanines"],
            "ipip_raw": out["ipip_raw"],
            "picks": out["picks"],
        })

    if args.preview:
        for c in composed[:args.preview]:
            print(f"\n=== {c['persona_id']} ===")
            print(f"z={c['z_scores']}")
            print(f"stanines={c['stanines']}")
            print(f"\nipip_raw ({len(c['ipip_raw'].split())} words):")
            print(c["ipip_raw"])
        print(f"\n[fallback events: {fallback_count} / {sum(len(c['picks']) for c in composed)} picks]")
        return

    out = {
        "_method": {
            "source": "instruments/ipip300_annotations.json",
            "composer": "scripts/persona_ipip_compose.py",
            "seed": args.seed,
            "n_personas": len(composed),
            "fallback_events": fallback_count,
        },
        "personas": composed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(composed)} personas to {args.output}")
    print(f"Fallback events: {fallback_count}")


if __name__ == "__main__":
    main()
