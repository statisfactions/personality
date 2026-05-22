#!/usr/bin/env python3
"""Export item-level matrices for reclustering / EFA analysis.

Produces in results/item_level/:
  item_meta.csv          — 288 rows: item_id, trait, facet, pole, item_text
  human_item_corr.csv    — 288x288 Pearson correlation matrix from IPIP300.por
                          (N=145,388 complete cases; columns/rows are item_ids)
  <model>_item_cos.csv   — 288x288 cosine similarity at common_layer
                          (10 files, one per cohort model)

The 288 items are the canonical IPIP-NEO-300 with W9 §3 deny-listed items
removed (12 items). Item order matches phase_b_cache_ipip; all matrices use
the same ordering so they can be loaded as drop-in replacements of each other.

Item-level data is the cleanest input for clustering / factor-analysis that
isn't pre-constrained by Johnson's facet aggregation (see W9 facet-design
caveat: facets are by-construction selected to load on traits, so clustering
on facet means recovers Big Five tautologically).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import torch

from hf_logprobs import MODELS as ALL_MODELS


def safe_name(s: str) -> str:
    """Mirror scripts/phase_b_sweep.py convention: HF repo -> filesystem-safe."""
    return s.replace("/", "_")


OUT_DIR = Path("results/item_level")
CACHE_DIR = Path("results/phase_b_cache_ipip")
POR_PATH = "data/kajonius_johnson_2019/data/kajonius_johnson_2019/IPIP300.por"

COHORT = ["Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7",
          "Gemma27", "Qwen32", "Gemma4"]


def load_canonical_item_order_from_cache():
    """Read one model's phase_b_cache_ipip to get the canonical 288-item order
    and metadata. This is what we'll align human + all-model matrices to."""
    sample = list(CACHE_DIR.glob("*_ipip_chat.pt"))[0]
    blob = torch.load(sample, weights_only=False)
    return blob["meta"]  # list of dicts in fixed order, length 288


def build_human_correlation_matrix(meta):
    """Per W9 §7.5.1, IPIP300.por stores reverse-keyed items already-reversed,
    so simple Pearson on raw item columns gives the correct correlation
    structure. Use N=145,388 complete-case subset (drop zero-coded missing)."""
    print(f"Loading {POR_PATH} ...")
    df, _ = pyreadstat.read_por(POR_PATH)
    print(f"  raw cases: {len(df)}")
    items_arr = df.loc[:, [f"I{n}" for n in range(1, 301)]].to_numpy(dtype="float64")
    mask = ~((items_arr == 0).any(axis=1) | np.isnan(items_arr).any(axis=1))
    items_arr = items_arr[mask]
    print(f"  complete cases: {items_arr.shape[0]}")

    # Subset to the 288 items in the same order as the model cache meta.
    item_ids = [m["item_id"] for m in meta]
    col_idx = [int(iid[len("ipip"):]) - 1 for iid in item_ids]  # ipip4 -> col 3
    items_arr = items_arr[:, col_idx]
    print(f"  subset to {items_arr.shape[1]} items matching cache order")

    print("Computing 288x288 Pearson correlation ...")
    corr = np.corrcoef(items_arr, rowvar=False)
    return corr, items_arr.shape[0]


def build_model_cosine_matrix(model_name, meta):
    """Load <model>_ipip_chat.pt, extract activations at common_layer
    (round(n_layers * 2/3) — W8 §9 convention), L2-normalize per item, return
    the 288x288 cosine matrix."""
    repo = ALL_MODELS[model_name]
    cache_path = CACHE_DIR / f"{safe_name(repo)}_ipip_chat.pt"
    blob = torch.load(cache_path, weights_only=False)
    cache_meta = blob["meta"]
    assert [m["item_id"] for m in cache_meta] == [m["item_id"] for m in meta], (
        f"{model_name} cache item order disagrees with canonical meta"
    )
    acts = blob["acts"]
    if not isinstance(acts, np.ndarray):
        acts = acts.numpy()
    # acts: (n_items, n_layers, hidden). n_layers includes embedding row,
    # so the residual stream layers are 0..n_layers-1.
    n_layers = acts.shape[1] - 1  # match ipip_facet_cluster.py convention
    common_layer = int(round(n_layers * 2 / 3))
    print(f"  {model_name}: acts {acts.shape}, common_layer {common_layer}/{n_layers}")
    X = acts[:, common_layer, :]                   # (288, hidden)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = X @ X.T                                  # (288, 288)
    return cos


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_canonical_item_order_from_cache()
    print(f"Canonical item order: {len(meta)} items")

    # 1. item_meta.csv
    meta_df = pd.DataFrame(meta)
    meta_df = meta_df[["item_id", "trait", "facet", "pole", "item_text"]]
    meta_df.to_csv(OUT_DIR / "item_meta.csv", index=False)
    print(f"Wrote {OUT_DIR}/item_meta.csv ({len(meta_df)} rows)")

    # 2. human_item_corr.csv
    human_corr, n_human = build_human_correlation_matrix(meta)
    item_ids = meta_df["item_id"].tolist()
    pd.DataFrame(human_corr, index=item_ids, columns=item_ids).to_csv(
        OUT_DIR / "human_item_corr.csv"
    )
    print(f"Wrote {OUT_DIR}/human_item_corr.csv (N={n_human:,}, "
          f"within range [{human_corr.min():+.3f}, {human_corr.max():+.3f}])")

    # 3. <model>_item_cos.csv for each cohort model
    for m in COHORT:
        try:
            cos = build_model_cosine_matrix(m, meta)
        except Exception as e:
            print(f"  SKIP {m}: {e}")
            continue
        pd.DataFrame(cos, index=item_ids, columns=item_ids).to_csv(
            OUT_DIR / f"{m}_item_cos.csv"
        )
        print(f"Wrote {OUT_DIR}/{m}_item_cos.csv "
              f"(range [{cos.min():+.3f}, {cos.max():+.3f}])")

    print(f"\nDone. All matrices in {OUT_DIR} are 288x288 with rows/cols "
          f"indexed by item_id in the same order. item_meta.csv has the "
          f"per-item trait/facet/pole/text annotations.")


if __name__ == "__main__":
    main()
