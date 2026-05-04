#!/usr/bin/env python3
"""IPIP-NEO-300 facet-level cosine cluster analysis (W8 §11).

Mirrors `scripts/facet_cluster.py` but for the 30 IPIP-NEO-300 facets
(5 Big Five traits × 6 facets each, 10 items per facet) instead of the
24 HEXACO facets.

Per facet, builds a direction:
    d_facet = mean(forward-keyed item activations)
            − mean(reverse-keyed item activations)
neutral-PC projected at 50% variance, unit-normed. 30 directions per
model. 30×30 cosine similarity matrix.

Per-model analysis (parallel to facet_cluster.py):
- within-trait vs across-trait mean cosine
- nearest-neighbor within-trait purity (does the closest other facet
  share parent trait?)
- 5-cluster hierarchical purity (do the 5 clusters recover the 5 Big
  Five traits?)

Compare against HEXACO facet results (24 facets / 6 traits): if Big
Five facets cluster as cleanly as HEXACO facets, the chunking-
granularity hypothesis (to_try #18) gets weaker support; if they're
messier, support strengthens.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/ipip_facet_cluster.py
    PYTHONPATH=scripts .venv/bin/python scripts/ipip_facet_cluster.py \
        --models Qwen7
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


ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"
ANNOTATIONS = "instruments/ipip300_annotations.json"
CACHE_DIR = Path("results/phase_b_cache")

TRAIT_ORDER = ["A", "C", "E", "N", "O"]
SHORT = {"A": "AGR", "C": "CON", "E": "EXT", "N": "NEU", "O": "OPE"}
FACETS = {
    "N": ["Anxiety", "Anger", "Depression", "Self-Cons",
          "Immoder", "Vulner"],
    "E": ["Friend", "Gregar", "Assert", "Activity",
          "Excite", "Cheerf"],
    "O": ["Imagin", "Artist", "Emotion", "Advent", "Intell", "Liberal"],
    "A": ["Trust", "Moral", "Altru", "Cooper", "Modest", "Sympath"],
    "C": ["Self-Eff", "Order", "Dutiful", "Achieve", "Discipl",
          "Caution"],
}


def safe(s): return s.replace("/", "_")
def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def build_facet_pool():
    """Build {(trait, facet_name): {fwd: [text], rev: [text]}} from IPIP-300,
    excluding deny-listed items, applying typo fixes."""
    with open(ADMIN_SESSION) as f:
        ipip = json.load(f)["measures"]["IPIP300"]
    items = ipip["items"]
    scales = ipip["scales"]
    with open(ANNOTATIONS) as f:
        ann = json.load(f)
    deny = set(ann["deny"].keys())
    fixes = ann["fix"]

    pool = {}
    for t in TRAIT_ORDER:
        sc = scales[f"IPIP300-{SHORT[t]}"]
        iids = sc["item_ids"]
        rev = set(sc["reverse_keyed_item_ids"])
        for fi, fname in enumerate(FACETS[t]):
            facet_iids = [i for i in iids[fi::6] if i not in deny]
            fwd_iids = [i for i in facet_iids if i not in rev]
            rev_iids = [i for i in facet_iids if i in rev]
            pool[(t, fname)] = {
                "fwd": [fixes.get(i, items[i]) for i in fwd_iids],
                "rev": [fixes.get(i, items[i]) for i in rev_iids],
            }
    return pool


def extract_facet_directions(model_name):
    """Run model, extract one direction per (trait, facet)."""
    repo = ALL_MODELS[model_name]
    tag = safe(repo)

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
    print(f"  Common layer: {common_layer}/{n_layers}")

    pcs, _, _ = mdx.compute_pc_projection(neutral_layer_t, 0.5)

    pool = build_facet_pool()
    n_total = sum(len(v["fwd"]) + len(v["rev"]) for v in pool.values())
    print(f"  Extracting {n_total} IPIP item activations across {len(pool)} facets...")

    model, tok, device = load_model(model_name)

    # Cache activations to avoid recomputation per (trait, facet)
    facet_acts = {}  # (trait, facet) -> {fwd: [act], rev: [act]}
    for (trait, facet), texts in pool.items():
        acts = {"fwd": [], "rev": []}
        for pole in ("fwd", "rev"):
            for text in texts[pole]:
                a = mdx.hidden_states_for_text(
                    model, tok, text, device,
                    split_prefix=None, chat_template=True,
                )
                acts[pole].append(a[common_layer].float().numpy())
        facet_acts[(trait, facet)] = acts

    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # Per-facet direction
    facet_names = []
    dir_rows = []
    for t in TRAIT_ORDER:
        for fname in FACETS[t]:
            acts = facet_acts[(t, fname)]
            fwd_mean = np.mean(acts["fwd"], axis=0) if acts["fwd"] else None
            rev_mean = np.mean(acts["rev"], axis=0) if acts["rev"] else None
            if fwd_mean is None or rev_mean is None:
                print(f"  WARNING: facet {t}.{fname} missing fwd or rev items, skipping")
                continue
            d = unit(mdx.project_out_pcs(fwd_mean - rev_mean, pcs))
            facet_names.append(f"{t}:{fname}")
            dir_rows.append(d)

    return facet_names, np.vstack(dir_rows), common_layer


def analyze(model_name, facet_names, D, common_layer):
    n = len(facet_names)
    cos = D @ D.T
    parent = [f.split(":")[0] for f in facet_names]

    # Within-trait vs across-trait mean cosines
    within, across = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if parent[i] == parent[j] else across).append(cos[i, j])
    print(f"\n--- {model_name} (layer {common_layer}, {n} facets) ---")
    print(f"  within-trait cos:  mean={np.mean(within):+.3f}  median={np.median(within):+.3f}  "
          f"n={len(within)}")
    print(f"  across-trait cos:  mean={np.mean(across):+.3f}  median={np.median(across):+.3f}  "
          f"n={len(across)}")
    ratio_denom = max(abs(np.mean(across)), 1e-3)
    print(f"  ratio (within/across): {np.mean(within) / ratio_denom:+.2f}x")

    # Nearest-neighbor within-trait purity
    right = 0
    misgrouped = []
    for i in range(n):
        sims = cos[i].copy()
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        if parent[i] == parent[j]:
            right += 1
        else:
            misgrouped.append((facet_names[i], facet_names[j], cos[i, j]))
    print(f"  nearest-neighbor within-trait: {right}/{n}")
    if misgrouped:
        print(f"  sample mis-groupings (facet -> nearest, cos):")
        for a, b, c in misgrouped[:8]:
            print(f"    {a:>22s} -> {b:<22s} ({c:+.3f})")

    # Hierarchical clustering, cut to 5 clusters (Big Five)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1 - cos
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=5, criterion="maxclust")
    cluster_to_traits = defaultdict(list)
    for c, par in zip(clusters, parent):
        cluster_to_traits[int(c)].append(par)
    purity_num = 0
    print("  5-cluster hierarchical assignment:")
    for c, trs in sorted(cluster_to_traits.items()):
        counts = {t: trs.count(t) for t in set(trs)}
        top_t, top_n = max(counts.items(), key=lambda x: x[1])
        purity_num += top_n
        print(f"    cluster {c}: n={len(trs):2d}  top={top_t}({top_n})  mix={counts}")
    purity = purity_num / n
    print(f"  5-cluster purity: {purity:.3f} (chance with 5 clusters ~ 1/5 = 0.200)")

    return {
        "model": model_name,
        "common_layer": common_layer,
        "facet_names": facet_names,
        "within_mean": float(np.mean(within)),
        "across_mean": float(np.mean(across)),
        "ratio": float(np.mean(within) / ratio_denom),
        "nn_within_trait": right,
        "n_facets": n,
        "purity_5": float(purity),
        "cosine_matrix": cos.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen7"],
                        help="Model names from hf_logprobs.MODELS. Default: Qwen7.")
    args = parser.parse_args()

    out = {}
    for model in args.models:
        if model not in ALL_MODELS:
            print(f"Skipping unknown model: {model}")
            continue
        print(f"\n========== Model: {model} ==========")
        try:
            facet_names, D, common_layer = extract_facet_directions(model)
        except Exception as e:
            print(f"  ERROR extracting directions for {model}: {e}")
            continue
        result = analyze(model, facet_names, D, common_layer)
        out[model] = result

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ipip_facet_cluster.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    if len(out) > 1:
        print(f"\n========== Summary ==========")
        print(f"  {'Model':<12} {'within':>8} {'across':>8} {'ratio':>7} "
              f"{'NN-within':>11} {'5-clust purity':>16}")
        for m, s in out.items():
            print(f"  {m:<12} {s['within_mean']:+8.3f} {s['across_mean']:+8.3f} "
                  f"{s['ratio']:+7.2f}x {s['nn_within_trait']:>4d}/{s['n_facets']:<4d} "
                  f"{s['purity_5']:>16.3f}")


if __name__ == "__main__":
    main()
