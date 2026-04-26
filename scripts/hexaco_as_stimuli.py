#!/usr/bin/env python3
"""HEXACO-100 items as RepE stimuli — fresh-stimulus check for report_week7 §4.1.

Per facet, computes:
    direction = mean(activations on forward-keyed items)
              − mean(activations on reverse-keyed items)
at ~2/3-depth layer, then neutral-PC-projected (matches facet_cluster.py's
md_projected pipeline). Each HEXACO item is wrapped as a user turn in the
model's chat template; activations averaged over content tokens (matches
hidden_states_for_text(split_prefix=None, chat_template=True)).

Output:
    results/hexaco_as_stimuli_directions.json — 24×24 cosine matrices per
        model + pairwise correlation between this matrix and the contrast-pair
        cosine matrix from facet_cluster.json.
    results/hexaco_as_stimuli_heatmap.html — per-model heatmaps.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/hexaco_as_stimuli.py
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS as ALL_MODELS, load_model


HEXACO_FILE = Path("instruments/hexaco100.json")
HOLDOUT_FILE = Path("instruments/contrast_pairs_holdout.json")
CACHE_DIR = Path("results/phase_b_cache")
TRAITS = ["H", "E", "X", "A", "C", "O"]
COHORT = ["Llama", "Gemma", "Phi4", "Qwen", "Llama8", "Gemma12", "Qwen7"]


def safe(s): return s.replace("/", "_")


def load_hexaco():
    """Return list of (item_id, text, scale, facet, reverse_keyed) per item."""
    with open(HEXACO_FILE) as f:
        data = json.load(f)
    out = []
    for iid, item in data["items"].items():
        out.append({
            "id": iid,
            "text": item["text"],
            "scale": item["scale"],          # H/E/X/A/C/O/ALT
            "facet": item.get("facet", ""),  # facet name within scale
            "reverse_keyed": bool(item.get("reverse_keyed", False)),
        })
    return out


def extract_item_activations(model, tokenizer, items, device):
    """Return (n_items, n_layers+1, hidden_dim) item activations.

    Each item wrapped as user turn (chat_template=True); activations averaged
    over content tokens (split_prefix=None). Mirrors extract_neutral_activations.
    """
    acts = []
    for i, item in enumerate(items):
        a = mdx.hidden_states_for_text(
            model, tokenizer, item["text"], device,
            split_prefix=None, chat_template=True,
        )
        acts.append(a)
        if (i + 1) % 25 == 0:
            print(f"    item {i+1}/{len(items)} done")
    return torch.stack(acts)


def facet_directions_from_items(items, item_acts, common_layer, neutral_layer):
    """Per-facet direction = mean(forward_keyed) − mean(reverse_keyed) at common_layer,
    projected out of top neutral PCs.

    Returns:
      facet_names: list of "TRAIT:facet" strings
      D: (n_facets, hidden_dim) of unit-norm directions
    """
    by_facet = defaultdict(lambda: {"fwd": [], "rev": []})
    for i, item in enumerate(items):
        # Skip altruism (ALT) — only 4 items, not in our 24-facet HEXACO matrix
        if item["scale"] not in TRAITS:
            continue
        if not item["facet"]:
            continue
        key = f"{item['scale']}:{item['facet']}"
        slot = "rev" if item["reverse_keyed"] else "fwd"
        by_facet[key][slot].append(i)

    facet_names, dir_rows = [], []
    pcs, _, _ = mdx.compute_pc_projection(neutral_layer, 0.5)

    for facet_name, slots in sorted(by_facet.items()):
        fwd_idxs = slots["fwd"]
        rev_idxs = slots["rev"]
        if len(fwd_idxs) == 0 or len(rev_idxs) == 0:
            print(f"    SKIP {facet_name}: fwd={len(fwd_idxs)} rev={len(rev_idxs)}")
            continue
        fwd_mean = item_acts[fwd_idxs, common_layer, :].float().mean(dim=0).numpy()
        rev_mean = item_acts[rev_idxs, common_layer, :].float().mean(dim=0).numpy()
        diff = fwd_mean - rev_mean
        direction = mdx.project_out_pcs(diff, pcs)
        n = np.linalg.norm(direction)
        if n > 1e-12:
            direction = direction / n
        facet_names.append(facet_name)
        dir_rows.append(direction)

    return facet_names, np.vstack(dir_rows)


def contrast_pair_facet_directions(tag, common_layer, neutral_layer, pair_source):
    """Reproduce facet_cluster.py / facet_viz.py per-facet directions from the
    24-pair holdout cache for one model. Returns (facet_names, D) with names
    formatted "TRAIT:facet" to match HEXACO-stimuli convention.
    """
    facet_names, dir_rows = [], []
    pcs, _, _ = mdx.compute_pc_projection(neutral_layer, 0.5)
    for trait in TRAITS:
        blob = torch.load(CACHE_DIR / f"{tag}_{trait}_chat_pairs.pt", weights_only=False)
        ph_h, pl_h = blob["ph_h"], blob["pl_h"]
        hold_pairs = pair_source["traits"][trait]["pairs"]
        by_facet = defaultdict(list)
        for i, p in enumerate(hold_pairs):
            by_facet[p["facet"]].append(i)
        for facet, idxs in by_facet.items():
            diff = (ph_h[idxs, common_layer, :] - pl_h[idxs, common_layer, :]).float().mean(dim=0).numpy()
            direction = mdx.project_out_pcs(diff, pcs)
            n = np.linalg.norm(direction)
            if n > 1e-12:
                direction = direction / n
            facet_names.append(f"{trait}:{facet}")
            dir_rows.append(direction)
    return facet_names, np.vstack(dir_rows)


def upper_tri_correlation(A, B):
    """Pearson r between flattened upper triangles (k=1) of two square matrices."""
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT)
    args = parser.parse_args()

    items = load_hexaco()
    print(f"Loaded {len(items)} HEXACO-100 items "
          f"({sum(1 for it in items if it['scale'] in TRAITS)} trait, "
          f"{sum(1 for it in items if it['scale'] not in TRAITS)} altruism)")

    with open(HOLDOUT_FILE) as f:
        holdout = json.load(f)

    output = {}
    for short in args.models:
        if short not in ALL_MODELS:
            print(f"unknown model: {short}")
            continue
        repo = ALL_MODELS[short]
        tag = safe(repo)

        print(f"\n=== {short} ({repo}) ===")
        # Load neutrals first (cheap) so we can derive n_layers from them —
        # matches facet_cluster.py's convention (acts include embedding layer
        # at index 0, so n_layers_dim = 1 + actual transformer layers).
        neutral_path = CACHE_DIR / f"{tag}_neutral_chat.pt"
        if not neutral_path.exists():
            print(f"  MISSING: {neutral_path}. Run phase_b_sweep first. Skipping.")
            continue
        neutral = torch.load(neutral_path, weights_only=False)
        if isinstance(neutral, torch.Tensor):
            neutral_np = neutral.numpy()
        else:
            neutral_np = neutral
        n_layers_dim = neutral_np.shape[1]
        common_layer = int(round(n_layers_dim * 2 / 3))
        print(f"  common_layer = {common_layer}/{n_layers_dim} (incl. embedding)")
        neutral_layer = torch.from_numpy(neutral_np[:, common_layer, :])

        model, tok, device = load_model(short)

        # Extract item activations
        print(f"  extracting {len(items)} items...")
        item_acts = extract_item_activations(model, tok, items, device)

        # Per-facet directions
        facet_names, D = facet_directions_from_items(
            items, item_acts, common_layer, neutral_layer,
        )
        cos = (D @ D.T)

        # Build contrast-pair cosine matrix inline using the same neutral / layer
        cp_facets, cp_D = contrast_pair_facet_directions(
            tag, common_layer, neutral_layer, holdout,
        )
        cp_cos = cp_D @ cp_D.T

        # Align facets between the two matrices (HEXACO and contrast-pair facet
        # name strings should match the HEXACO facet definitions exactly)
        if cp_facets == facet_names:
            r = upper_tri_correlation(cos, cp_cos)
            print(f"  cosine-matrix r(HEXACO-stim, contrast-pair) = {r:+.3f}")
        else:
            cp_index = {f: i for i, f in enumerate(cp_facets)}
            perm = [cp_index[f] for f in facet_names if f in cp_index]
            if len(perm) == len(facet_names):
                cp_cos_aligned = cp_cos[np.ix_(perm, perm)]
                r = upper_tri_correlation(cos, cp_cos_aligned)
                print(f"  cosine-matrix r(HEXACO-stim, contrast-pair) = {r:+.3f} (reordered)")
            else:
                print(f"  facet name mismatch: stim={facet_names}, cp={cp_facets}")
                r = None

        output[short] = {
            "common_layer": common_layer,
            "n_layers": n_layers_dim,
            "facets": facet_names,
            "cos": cos.tolist(),
            "r_vs_contrast_pair": r,
        }

        del model, tok, item_acts
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    out_path = Path("results/hexaco_as_stimuli_directions.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Summary: cosine-matrix r(HEXACO-stimuli, contrast-pair) per model ===")
    print(f"{'model':>10s}  {'r':>7s}  {'common_layer':>14s}")
    for short, entry in output.items():
        r = entry["r_vs_contrast_pair"]
        rstr = f"{r:+.3f}" if r is not None else "N/A"
        print(f"{short:>10s}  {rstr:>7s}  {entry['common_layer']}/{entry['n_layers']}")

    # Heatmap viz
    n = len(output)
    if n > 0:
        import math
        n_cols = 2
        n_rows = math.ceil(n / n_cols)
        fig = make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=list(output.keys()),
            horizontal_spacing=0.12, vertical_spacing=0.10,
        )
        for i, (short, entry) in enumerate(output.items()):
            row, col = i // n_cols + 1, i % n_cols + 1
            cos_arr = np.array(entry["cos"])
            cos_disp = cos_arr.copy()
            np.fill_diagonal(cos_disp, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=cos_disp, x=entry["facets"], y=entry["facets"],
                    colorscale="RdBu_r", zmin=-0.4, zmax=0.4,
                    showscale=(i == 0),
                    hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>cos=%{z:+.3f}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=row, col=col)
            fig.update_yaxes(tickfont=dict(size=8), autorange="reversed", row=row, col=col)
        fig.update_layout(
            title=dict(text="HEXACO-100 items as RepE stimuli — facet cosine similarity", x=0.5),
            height=600 * n_rows, width=1400,
        )
        hm_path = Path("results/hexaco_as_stimuli_heatmap.html")
        fig.write_html(str(hm_path))
        print(f"Wrote {hm_path}")


if __name__ == "__main__":
    main()
