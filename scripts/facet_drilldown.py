#!/usr/bin/env python3
"""Facet- and item-level EN-vs-ZH drilldown from cached IPIP-120 survey data.

(1) Qwen32 Openness: which of the 6 O facets carries the -1.04 trait drop?
(2) Cross-model Neuroticism: is the systematic N drop clustered in a few
    facets/items or spread across the whole trait?

No new inference — uses results/surveys/<model>_ipip120_{english,mandarin}.json
plus the facet maps. Item k in English == item k in Mandarin (same Johnson item).
"""

import json
from collections import defaultdict
from pathlib import Path

S = Path("results/surveys")
EN_MAP = json.load(open("instruments/ipip120_english_facet_map.json"))["items"]
ZH_MAP = json.load(open("instruments/ipip120_mandarin_facet_map.json"))["items"]

OPINIONATED = ["Qwen", "Qwen7", "Qwen32", "Aya", "Gemma", "Gemma12", "Gemma27", "Gemma4"]


def scored_items(model):
    """Return {k: (en_scored, zh_scored, trait, facet, en_text_en)} for item index k."""
    en = json.load(open(S / f"{model}_ipip120_english.json"))["item_results"]
    zh = json.load(open(S / f"{model}_ipip120_mandarin.json"))["item_results"]
    out = {}
    for k in range(1, 121):
        e_id, z_id = f"ipip120e_{k}", f"ipip120m_{k}"
        if e_id not in en or z_id not in zh:
            continue
        meta = EN_MAP[e_id]
        rev = meta["pole"] == "rev"
        e_ev = en[e_id]["expected_value"]
        z_ev = zh[z_id]["expected_value"]
        e_s = (6.0 - e_ev) if rev else e_ev
        z_s = (6.0 - z_ev) if rev else z_ev
        out[k] = (e_s, z_s, meta["trait"], meta["facet"], meta["text"])
    return out


def facet_means(items):
    """{(trait,facet): (en_mean, zh_mean, delta, n)}"""
    by_facet = defaultdict(lambda: [[], []])
    for k, (e, z, trait, facet, _) in items.items():
        by_facet[(trait, facet)][0].append(e)
        by_facet[(trait, facet)][1].append(z)
    res = {}
    for key, (es, zs) in by_facet.items():
        em, zm = sum(es)/len(es), sum(zs)/len(zs)
        res[key] = (em, zm, zm - em, len(es))
    return res


# ============ (1) Qwen32 Openness drilldown ============
print("=" * 70)
print("(1) Qwen32 — all 6 Openness facets, EN vs ZH (trait O dropped -1.04)")
print("=" * 70)
q32 = scored_items("Qwen32")
fm = facet_means(q32)
print(f"  {'facet':<22s} {'EN':>6s} {'ZH':>6s} {'Δ':>7s}")
for (trait, facet), (em, zm, d, n) in sorted(fm.items()):
    if trait == "O":
        print(f"  O:{facet:<20s} {em:>6.2f} {zm:>6.2f} {d:>+7.2f}")
# Item-level within Openness
print("\n  Item-level (Openness only), sorted by |Δ|:")
o_items = [(k, e, z, facet, txt) for k, (e, z, t, facet, txt) in q32.items() if t == "O"]
for k, e, z, facet, txt in sorted(o_items, key=lambda x: -abs(x[2]-x[1])):
    print(f"    [{k:>3d}] O:{facet:<16s} EN={e:.2f} ZH={z:.2f} Δ={z-e:+.2f}  {txt[:42]}")


# ============ (2) Cross-model Neuroticism breakdown ============
print()
print("=" * 70)
print("(2) Neuroticism EN→ZH shift: clustered or widespread?")
print("=" * 70)
# Per-model, per-N-facet delta
N_FACETS = ["Anxiety", "Anger", "Depression", "Self-consciousness", "Immoderation", "Vulnerability"]
print(f"  {'model':<9s} " + " ".join(f"{f[:5]:>6s}" for f in N_FACETS) + f" {'N tot':>7s}")
facet_deltas_accum = defaultdict(list)
for m in OPINIONATED:
    items = scored_items(m)
    fm = facet_means(items)
    row = []
    for f in N_FACETS:
        d = fm.get(("N", f), (0, 0, float("nan"), 0))[2]
        row.append(d)
        facet_deltas_accum[f].append(d)
    ntot = sum(row) / len(row)
    print(f"  {m:<9s} " + " ".join(f"{d:>+6.2f}" for d in row) + f" {ntot:>+7.2f}")
print(f"  {'MEAN':<9s} " + " ".join(f"{sum(facet_deltas_accum[f])/len(facet_deltas_accum[f]):>+6.2f}" for f in N_FACETS))

# Which N items move most across models (mean signed delta)?
print("\n  N items ranked by mean signed Δ across opinionated models:")
item_deltas = defaultdict(list)
item_info = {}
for m in OPINIONATED:
    items = scored_items(m)
    for k, (e, z, t, facet, txt) in items.items():
        if t == "N":
            item_deltas[k].append(z - e)
            item_info[k] = (facet, txt)
ranked = sorted(item_deltas.items(), key=lambda kv: sum(kv[1])/len(kv[1]))
for k, ds in ranked:
    facet, txt = item_info[k]
    md = sum(ds)/len(ds)
    print(f"    [{k:>3d}] N:{facet:<18s} meanΔ={md:>+.2f}  {txt[:44]}")
