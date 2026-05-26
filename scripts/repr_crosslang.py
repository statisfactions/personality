#!/usr/bin/env python3
"""W13 §3.8: cross-language facet-geometry (repr) on IPIP-NEO-120 EN vs ZH.

Extracts per-facet directions (meandiff-itempc1: mean(fwd) − mean(rev) with the
top-1 item PC projected out — no neutral corpus, language-internal by
construction) from the English and Mandarin IPIP-120 items, per model, and
compares the resulting 30×30 facet cosine geometry across languages.

The pre-registration (memory/project_w13_mandarin_preregistration.md) predicts:
  - median per-facet EN↔ZH row-r > 0.85 (geometry mostly invariant)
  - O:Liberalism in the bottom-3 for drift (CCP bright-line / political items)
  - median row-r across non-political facets > 0.90

Reuses the math from ipip_facet_cluster.py but builds the item pool from an
arbitrary facet map (so it works for the ipip120m_/ipip120e_ id schemes) and
caches activations per (model, language). The English IPIP-300 pipeline is
untouched.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/repr_crosslang.py --models Qwen Qwen7 ...
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS as ALL_MODELS, load_model
from ipip_facet_cluster import (_CANON_TO_ABBREV, FACETS, TRAIT_ORDER, unit,
                                build_directions, safe)

CACHE_DIR = Path("results/phase_b_cache_ipip")
OUT_PATH = Path("results/facets/ipip_facet_crosslang.json")

FACET_MAPS = {
    "en": "instruments/ipip120_english_facet_map.json",
    "zh": "instruments/ipip120_mandarin_facet_map.json",
}

HUMAN = json.load(open("instruments/ipip300_human_facet_correlations.json"))
H = np.array(HUMAN["correlation_matrix"])
HF = HUMAN["facet_order"]
IU = np.triu_indices(30, k=1)

# IPIP-300 facet matrices (meandiff-itempc1) for the within-language form-change
# floor: EN-120-vs-EN-300 tells us how much the 4-item short form alone costs,
# so EN-120-vs-ZH-120 (language) can be read against it rather than against 1.0.
IPIP300 = json.load(open("results/facets/ipip_facet_cluster.json"))


def ipip300_matrix(model_name):
    if model_name not in IPIP300:
        return None
    e = IPIP300[model_name]
    fn = e["facet_names"]
    cos = np.array(e["cosine_matrix"])
    idx = [fn.index(lbl) for lbl in HF]
    return cos[np.ix_(idx, idx)]


def build_pool(facet_map_path):
    """{(trait, facet_abbrev): {fwd: [(iid,text)], rev: [(iid,text)]}} from any
    facet map whose items carry trait/facet(full)/pole/text. Facet-name lookup
    is case-insensitive — the IPIP-120 sheet lowercases the second word of
    hyphenated facets ("Self-efficacy" vs canonical "Self-Efficacy")."""
    canon_ci = {k.lower(): v for k, v in _CANON_TO_ABBREV.items()}
    items = json.load(open(facet_map_path))["items"]
    pool = {(t, f): {"fwd": [], "rev": []} for t in TRAIT_ORDER for f in FACETS[t]}
    for iid, info in items.items():
        abbrev = canon_ci[info["facet"].lower()]
        pool[(info["trait"], abbrev)][info["pole"]].append((iid, info["text"]))
    return pool


def extract_or_load(model_name, lang):
    """Per-(model,lang) activation cache; mean-token, chat template."""
    repo = ALL_MODELS[model_name]
    cache = CACHE_DIR / f"{safe(repo)}_ipip120_{lang}_chat.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        acts = blob["acts"]
        return (acts if isinstance(acts, np.ndarray) else acts.numpy()), blob["meta"]

    pool = build_pool(FACET_MAPS[lang])
    model, tok, device = load_model(model_name)
    all_acts, meta = [], []
    for (t, f), polled in pool.items():
        for pole in ("fwd", "rev"):
            for iid, text in polled[pole]:
                a = mdx.hidden_states_for_text(model, tok, text, device,
                                               split_prefix=None, chat_template=True)
                all_acts.append(a.float().numpy())
                meta.append({"trait": t, "facet": f, "pole": pole, "item_id": iid})
    acts = np.stack(all_acts)
    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": acts, "meta": meta, "model": repo, "lang": lang}, cache)
    return acts, meta


def build_directions_crosslang(acts, meta, layer):
    """meandiff-itempc1 generalized to the IPIP-120 short form, where 9/30
    facets have all 4 items keyed one way (Johnson balanced keying at the
    trait level, not the facet level). The global item centroid stands in as
    the implicit opposite pole:
      both poles : mean(fwd) - mean(rev)      (identical to meandiff-itempc1)
      fwd only   : mean(fwd) - centroid
      rev only   : centroid  - mean(rev)
    then project out the top-1 item PC (anisotropy axis) and unit-normalize.
    """
    by = defaultdict(lambda: {"fwd": [], "rev": []})
    for i, m in enumerate(meta):
        by[(m["trait"], m["facet"])][m["pole"]].append(acts[i, layer])
    centroid = acts[:, layer, :].mean(axis=0)
    all_pcs, _, _ = mdx.compute_pc_projection(torch.from_numpy(acts[:, layer, :]), 0.5)
    pc1 = all_pcs[:1]

    names, rows = [], []
    for t in TRAIT_ORDER:
        for f in FACETS[t]:
            p = by[(t, f)]
            fwd = np.mean(p["fwd"], axis=0) if p["fwd"] else None
            rev = np.mean(p["rev"], axis=0) if p["rev"] else None
            if fwd is not None and rev is not None:
                d = fwd - rev
            elif fwd is not None:
                d = fwd - centroid
            elif rev is not None:
                d = centroid - rev
            else:
                continue
            names.append(f"{t}:{f}")
            rows.append(unit(mdx.project_out_pcs(d, pc1)))
    return names, np.vstack(rows)


def cosine_matrix(acts, meta, common_layer):
    facet_names, D = build_directions_crosslang(acts, meta, common_layer)
    return facet_names, D @ D.T


def reorder_to_human(facet_names, cos):
    idx = [facet_names.index(lbl) for lbl in HF]
    return cos[np.ix_(idx, idx)]


def row_r(A, B, i):
    mask = np.ones(30, dtype=bool); mask[i] = False
    return float(np.corrcoef(A[i, mask], B[i, mask])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    results = {}
    for m in args.models:
        if m not in ALL_MODELS:
            print(f"skip unknown {m}"); continue
        print(f"\n===== {m} =====")
        per_lang = {}
        for lang in ("en", "zh"):
            acts, meta = extract_or_load(m, lang)
            n_layers = acts.shape[1] - 1
            layer = int(round(n_layers * 2 / 3))
            names, cos = cosine_matrix(acts, meta, layer)
            per_lang[lang] = reorder_to_human(names, cos)
            print(f"  {lang}: acts {acts.shape}, layer {layer}/{n_layers}")

        en_m, zh_m = per_lang["en"], per_lang["zh"]
        # Overall EN<->ZH geometry agreement (flattened upper-tri).
        overall = float(np.corrcoef(en_m[IU], zh_m[IU])[0, 1])
        # Per-facet EN<->ZH row correlation (self-cell excluded).
        per_facet = {HF[i]: row_r(en_m, zh_m, i) for i in range(30)}
        ranked = sorted(per_facet.items(), key=lambda kv: kv[1])  # most-drifted first
        # r vs human for each language (sanity / magnitude).
        r_en = float(np.corrcoef(H[IU], en_m[IU])[0, 1])
        r_zh = float(np.corrcoef(H[IU], zh_m[IU])[0, 1])

        # Form-change floor: EN-120 vs the model's EN-IPIP-300 geometry. If the
        # language agreement (EN-120 vs ZH-120) matches this, language adds no
        # drift beyond the short form's own noise.
        m300 = ipip300_matrix(m)
        form_floor = float(np.corrcoef(en_m[IU], m300[IU])[0, 1]) if m300 is not None else None
        both = float(np.corrcoef(zh_m[IU], m300[IU])[0, 1]) if m300 is not None else None

        results[m] = {
            "en_zh_overall_r": overall,
            "form_floor_en120_vs_en300_r": form_floor,
            "form_plus_lang_zh120_vs_en300_r": both,
            "en_vs_human_r": r_en, "zh_vs_human_r": r_zh,
            "per_facet_en_zh_row_r": per_facet,
        }
        lib_rank = [k for k, _ in ranked].index("O:Liberal") + 1
        ff = f"{form_floor:+.3f}" if form_floor is not None else "n/a"
        verdict = ""
        if form_floor is not None:
            verdict = ("  -> language ≈ form-noise (invariant)" if overall >= form_floor - 0.05
                       else f"  -> language drifts {form_floor - overall:+.3f} beyond form floor")
        print(f"  EN↔ZH geom r = {overall:+.3f}  vs form-floor(EN120-EN300) {ff}{verdict}")
        print(f"  (r_en={r_en:+.3f} r_zh={r_zh:+.3f} vs human; form+lang ZH120-EN300 = "
              f"{both:+.3f})" if both is not None else "")
        print(f"  O:Liberal row-r = {per_facet['O:Liberal']:+.3f}  (drift rank {lib_rank}/30)"
              f"   most-drifted: " + ", ".join(f"{k} {v:+.2f}" for k, v in ranked[:4]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
