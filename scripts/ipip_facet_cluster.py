#!/usr/bin/env python3
"""IPIP-NEO-300 facet-level cosine cluster analysis (W8 §11, W9 §1).

Per facet, build a direction from forward-keyed and (optionally) reverse-keyed
IPIP item activations. Project onto a 30×30 cosine similarity matrix.

W8 §9 used a single extraction method (`meandiff-pcs`):
    d_facet = project_out_pcs(mean(fwd) - mean(rev), neutral_pcs), unit-normed
W9 §1 generalizes this to 5 methods, selectable via `--extraction`. See
`rgb_reports/representation_vector_methods.md` for the canonical method
catalog (formulas, token aggregation, theoretical motivation).

| name                | formula                                              |
|---------------------|------------------------------------------------------|
| `meandiff-pcs`      | unit(project_out_pcs(mean(fwd) - mean(rev), pcs))    |  (W8 default)
| `single-zero`       | unit(mean(fwd))                                      |  (no baseline)
| `single-neutral`    | unit(mean(fwd) - mean(neutral))                      |  (neutral-text baseline)
| `single-pcs`        | unit(project_out_pcs(mean(fwd), pcs))                |  (neutral-PC baseline)
| `single-ipip-mean`  | unit(mean(fwd) - mean(all_ipip_items))               |  (IPIP-format centroid baseline)

Pilot finding (W9 §1, Qwen7): single-zero/single-neutral/single-pcs are
degenerate (cosine ≈ +1.0 between all directions) because of residual-stream
anisotropy — the neutral-text baseline is too far from the chat-wrapped IPIP
manifold to capture the right shared component. `single-ipip-mean` works as
the matched-format baseline.

Item activations are cached to `results/phase_b_cache_ipip/` keyed by
HF repo so that the 4 methods can be run cheaply on the same activations.
Cache stores all layers; common_layer (~2/3 depth) is selected at
direction-build time.

Per-model analysis (parallel to facet_cluster.py for HEXACO):
- within-trait vs across-trait mean cosine
- nearest-neighbor within-trait purity
- 5-cluster hierarchical purity (Big Five)

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/ipip_facet_cluster.py
    PYTHONPATH=scripts .venv/bin/python scripts/ipip_facet_cluster.py \\
        --models Qwen7 --extraction single-zero single-neutral single-pcs

Defaults: --extraction meandiff-pcs (W8 §9 behavior, output unchanged).
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
IPIP_CACHE_DIR = Path("results/phase_b_cache_ipip")

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
EXTRACTION_METHODS = ["meandiff-pcs", "single-zero", "single-neutral", "single-pcs", "single-ipip-mean"]


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
                "fwd": [(i, fixes.get(i, items[i])) for i in fwd_iids],
                "rev": [(i, fixes.get(i, items[i])) for i in rev_iids],
            }
    return pool


def extract_or_load_activations(model_name):
    """Cache-or-extract per-item activations at all layers.

    Returns: (acts: ndarray (n_items, n_layers+1, hidden),
              meta: list of dicts with trait/facet/pole/item_id/item_text,
              n_layers: int)
    """
    repo = ALL_MODELS[model_name]
    tag = safe(repo)
    IPIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = IPIP_CACHE_DIR / f"{tag}_ipip_chat.pt"

    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        acts = blob["acts"] if isinstance(blob["acts"], np.ndarray) else blob["acts"].numpy()
        print(f"  Loaded cached activations: {acts.shape} from {cache_path}")
        return acts, blob["meta"], acts.shape[1] - 1

    pool = build_facet_pool()
    n_total = sum(len(v["fwd"]) + len(v["rev"]) for v in pool.values())
    print(f"  Extracting {n_total} IPIP item activations across {len(pool)} facets...")

    model, tok, device = load_model(model_name)

    all_acts = []
    meta = []
    for (trait, facet), texts in pool.items():
        for pole in ("fwd", "rev"):
            for item_id, item_text in texts[pole]:
                a = mdx.hidden_states_for_text(
                    model, tok, item_text, device,
                    split_prefix=None, chat_template=True,
                )
                all_acts.append(a.float().numpy())
                meta.append({
                    "trait": trait, "facet": facet, "pole": pole,
                    "item_id": item_id, "item_text": item_text,
                })

    acts = np.stack(all_acts)  # (n_items, n_layers+1, hidden)
    n_layers = acts.shape[1] - 1
    print(f"  Activations shape: {acts.shape}")

    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    torch.save({
        "acts": acts, "meta": meta, "model": repo,
        "format": "chat", "token_aggregation": "mean-all-skip0",
    }, cache_path)
    print(f"  Cached to {cache_path}")
    return acts, meta, n_layers


def load_neutral(model_name):
    """Load the cached neutral activations (used by both PC and neutral baselines)."""
    repo = ALL_MODELS[model_name]
    tag = safe(repo)
    neutral_path = CACHE_DIR / f"{tag}_neutral_chat.pt"
    if not neutral_path.exists():
        raise FileNotFoundError(f"missing neutral cache: {neutral_path}")
    neutral = torch.load(neutral_path, weights_only=False)
    if isinstance(neutral, torch.Tensor):
        return neutral.numpy()
    return neutral


def build_directions(acts, meta, common_layer, extraction, neutral_np):
    """Build per-facet directions from cached activations.

    Returns: (facet_names, D: ndarray (n_facets, hidden))
    """
    # Group activations by (trait, facet, pole) at common_layer
    by_facet = defaultdict(lambda: {"fwd": [], "rev": []})
    for i, m in enumerate(meta):
        by_facet[(m["trait"], m["facet"])][m["pole"]].append(acts[i, common_layer])

    needs_neutral_mean = extraction == "single-neutral"
    needs_pcs = extraction in ("meandiff-pcs", "single-pcs")
    needs_ipip_mean = extraction == "single-ipip-mean"

    neutral_mean = None
    pcs = None
    ipip_mean = None
    if needs_neutral_mean:
        neutral_mean = neutral_np[:, common_layer, :].mean(axis=0)
    if needs_pcs:
        neutral_layer_t = torch.from_numpy(neutral_np[:, common_layer, :])
        pcs, _, _ = mdx.compute_pc_projection(neutral_layer_t, 0.5)
    if needs_ipip_mean:
        ipip_mean = acts[:, common_layer, :].mean(axis=0)

    facet_names = []
    dir_rows = []
    for t in TRAIT_ORDER:
        for fname in FACETS[t]:
            polled = by_facet[(t, fname)]
            if not polled["fwd"]:
                print(f"  WARNING: facet {t}.{fname} missing forward items, skipping")
                continue

            fwd_mean = np.mean(polled["fwd"], axis=0)

            if extraction == "meandiff-pcs":
                if not polled["rev"]:
                    print(f"  WARNING: facet {t}.{fname} missing reverse items for contrast, skipping")
                    continue
                rev_mean = np.mean(polled["rev"], axis=0)
                d = unit(mdx.project_out_pcs(fwd_mean - rev_mean, pcs))
            elif extraction == "single-zero":
                d = unit(fwd_mean)
            elif extraction == "single-neutral":
                d = unit(fwd_mean - neutral_mean)
            elif extraction == "single-pcs":
                d = unit(mdx.project_out_pcs(fwd_mean, pcs))
            elif extraction == "single-ipip-mean":
                d = unit(fwd_mean - ipip_mean)
            else:
                raise ValueError(f"unknown extraction: {extraction}")

            facet_names.append(f"{t}:{fname}")
            dir_rows.append(d)

    return facet_names, np.vstack(dir_rows)


def analyze(model_name, extraction, facet_names, D, common_layer):
    n = len(facet_names)
    cos = D @ D.T
    parent = [f.split(":")[0] for f in facet_names]

    within, across = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if parent[i] == parent[j] else across).append(cos[i, j])
    print(f"\n--- {model_name} / {extraction} (layer {common_layer}, {n} facets) ---")
    print(f"  within-trait cos:  mean={np.mean(within):+.3f}  median={np.median(within):+.3f}  "
          f"n={len(within)}")
    print(f"  across-trait cos:  mean={np.mean(across):+.3f}  median={np.median(across):+.3f}  "
          f"n={len(across)}")
    ratio_denom = max(abs(np.mean(across)), 1e-3)
    print(f"  ratio (within/across): {np.mean(within) / ratio_denom:+.2f}x")

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
        "extraction": extraction,
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


def output_path(extraction):
    """Default 'meandiff-pcs' lands at the W8 §9 path; others get a tag."""
    if extraction == "meandiff-pcs":
        return Path("results/facets/ipip_facet_cluster.json")
    return Path(f"results/facets/ipip_facet_cluster_{extraction}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen7"],
                        help="Model names from hf_logprobs.MODELS. Default: Qwen7.")
    parser.add_argument("--extraction", nargs="+", default=["meandiff-pcs"],
                        choices=EXTRACTION_METHODS + ["all"],
                        help="One or more extraction methods. 'all' runs all four. "
                             "Default: meandiff-pcs (W8 §9 behavior).")
    args = parser.parse_args()

    methods = EXTRACTION_METHODS if "all" in args.extraction else args.extraction

    # Group results per-extraction; each method writes its own JSON.
    per_method_results = {m: {} for m in methods}

    for model in args.models:
        if model not in ALL_MODELS:
            print(f"Skipping unknown model: {model}")
            continue
        print(f"\n========== Model: {model} ==========")
        try:
            acts, meta, n_layers = extract_or_load_activations(model)
        except Exception as e:
            print(f"  ERROR extracting activations for {model}: {e}")
            continue
        common_layer = int(round(n_layers * 2 / 3))
        print(f"  Common layer: {common_layer}/{n_layers}")

        try:
            neutral_np = load_neutral(model)
        except Exception as e:
            print(f"  ERROR loading neutral cache for {model}: {e}")
            continue

        for method in methods:
            facet_names, D = build_directions(acts, meta, common_layer, method, neutral_np)
            res = analyze(model, method, facet_names, D, common_layer)
            per_method_results[method][model] = res

    Path("results/facets").mkdir(parents=True, exist_ok=True)
    for method, results in per_method_results.items():
        out_path = output_path(method)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {out_path}")

    # Cross-method summary if multiple methods run
    if len(methods) > 1 and per_method_results[methods[0]]:
        print(f"\n========== Cross-method summary ==========")
        print(f"  {'Model':<10} " + " ".join(f"{m:<22}" for m in methods))
        for model in args.models:
            if model not in per_method_results[methods[0]]:
                continue
            cells = []
            for m in methods:
                r = per_method_results[m].get(model, {})
                cells.append(f"w={r.get('within_mean', 0):+.3f} NN={r.get('nn_within_trait', '-')}/30 p={r.get('purity_5', 0):.2f}".ljust(22))
            print(f"  {model:<10} " + " ".join(cells))


if __name__ == "__main__":
    main()
