#!/usr/bin/env python3
"""Goldberg's 52 Big Five markers as RepE stimuli — Sofroniew-style trait-level
direction extraction with single-adjective adjective stimuli.

Stimuli come from generate_trait_personas.MARKERS — 9-12 high-pole + 9-12
low-pole adjective phrases per Big Five trait (E, A, C, N, O).

Per-trait direction = mean(high-pole activations) − mean(low-pole activations)
at ~2/3-depth layer, neutral-PC-projected. Each marker wrapped as a user turn
in the model's chat template; activations averaged over content tokens.

Output:
    results/markers_as_stimuli.json — 5×5 cosine matrices per model + cross-
        model upper-tri correlations.
    results/markers_as_stimuli_heatmap.html — per-model 5×5 heatmaps.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/markers_as_stimuli.py
    PYTHONPATH=scripts .venv/bin/python scripts/markers_as_stimuli.py \
        --models Gemma12 Llama8 Qwen7
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import extract_meandiff_vectors as mdx
from hf_logprobs import MODELS as ALL_MODELS, load_model
from generate_trait_personas import MARKERS


CACHE_DIR = Path("results/phase_b_cache")
TRAITS = ["E", "A", "C", "N", "O"]  # Goldberg/Big Five order
COHORT = ["Gemma12", "Llama8", "Qwen7"]


def safe(s): return s.replace("/", "_")


def collect_marker_stimuli():
    """Return list of (text, trait, pole) for all 52×2 = 104 marker phrases."""
    stim = []
    for trait, poles in MARKERS.items():
        for pole, items in poles.items():
            for adj in items:
                stim.append({"text": adj, "trait": trait, "pole": pole})
    return stim


def extract_marker_activations(model, tok, stim, device):
    """Average activation over content tokens for each marker (chat-template wrapped)."""
    acts = []
    for i, s in enumerate(stim):
        a = mdx.hidden_states_for_text(
            model, tok, s["text"], device,
            split_prefix=None, chat_template=True,
        )
        acts.append(a)
        if (i + 1) % 25 == 0:
            print(f"    marker {i+1}/{len(stim)} done")
    return torch.stack(acts)


def trait_directions_from_markers(stim, marker_acts, common_layer, neutral_layer):
    """Per Big Five trait, direction = mean(high-pole) − mean(low-pole),
    neutral-PC-projected. Returns (trait_names, D unit-norm 5×hidden)."""
    pcs, _, _ = mdx.compute_pc_projection(neutral_layer, 0.5)
    trait_names, dir_rows = [], []
    for trait in TRAITS:
        high_idxs = [i for i, s in enumerate(stim) if s["trait"] == trait and s["pole"] == "high"]
        low_idxs  = [i for i, s in enumerate(stim) if s["trait"] == trait and s["pole"] == "low"]
        if not high_idxs or not low_idxs:
            print(f"    SKIP {trait}: high={len(high_idxs)} low={len(low_idxs)}")
            continue
        high_mean = marker_acts[high_idxs, common_layer, :].float().mean(dim=0).numpy()
        low_mean = marker_acts[low_idxs, common_layer, :].float().mean(dim=0).numpy()
        diff = high_mean - low_mean
        direction = mdx.project_out_pcs(diff, pcs)
        n = np.linalg.norm(direction)
        if n > 1e-12:
            direction = direction / n
        trait_names.append(trait)
        dir_rows.append(direction)
    return trait_names, np.vstack(dir_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT)
    args = parser.parse_args()

    stim = collect_marker_stimuli()
    n_high = sum(1 for s in stim if s["pole"] == "high")
    n_low = sum(1 for s in stim if s["pole"] == "low")
    print(f"Loaded {len(stim)} marker stimuli (high={n_high}, low={n_low})")
    for trait in TRAITS:
        h = sum(1 for s in stim if s["trait"] == trait and s["pole"] == "high")
        l = sum(1 for s in stim if s["trait"] == trait and s["pole"] == "low")
        print(f"  {trait}: high={h}, low={l}")

    output = {}
    for short in args.models:
        if short not in ALL_MODELS:
            print(f"unknown model: {short}")
            continue
        repo = ALL_MODELS[short]
        tag = safe(repo)

        print(f"\n=== {short} ({repo}) ===")
        neutral_path = CACHE_DIR / f"{tag}_neutral_chat.pt"
        if not neutral_path.exists():
            print(f"  MISSING: {neutral_path}. Skip.")
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

        print(f"  extracting {len(stim)} marker activations...")
        marker_acts = extract_marker_activations(model, tok, stim, device)

        trait_names, D = trait_directions_from_markers(
            stim, marker_acts, common_layer, neutral_layer,
        )
        cos = (D @ D.T)

        # Print 5×5 cosine matrix
        print(f"\n  5×5 trait-level cosine matrix (Goldberg markers):")
        print("       " + "  ".join(f"{t:>6s}" for t in trait_names))
        for i, t in enumerate(trait_names):
            print(f"  {t:>4s} " + "  ".join(f"{cos[i,j]:+6.3f}" for j in range(len(trait_names))))

        output[short] = {
            "common_layer": common_layer,
            "n_layers": n_layers_dim,
            "traits": trait_names,
            "cos": cos.tolist(),
        }

        del model, tok, marker_acts
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    out_path = Path("results/markers_as_stimuli.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # Cross-model agreement on the 5×5 matrices
    if len(output) >= 2:
        names = list(output.keys())
        print("\n=== Cross-model upper-tri correlation between 5×5 cosine matrices ===")
        print(f"{'':>10s} " + " ".join(f"{n:>8s}" for n in names))
        cos_per = {n: np.array(output[n]["cos"]) for n in names}
        iu = np.triu_indices(5, k=1)
        for a in names:
            row = []
            for b in names:
                if a == b:
                    row.append("    1.000")
                else:
                    r = float(np.corrcoef(cos_per[a][iu], cos_per[b][iu])[0, 1])
                    row.append(f"{r:+8.3f}")
            print(f"{a:>10s} " + " ".join(row))

    # Heatmap
    n = len(output)
    if n > 0:
        import math
        n_cols = min(n, 3)
        n_rows = math.ceil(n / n_cols)
        fig = make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=list(output.keys()),
            horizontal_spacing=0.10, vertical_spacing=0.15,
        )
        for i, (short, entry) in enumerate(output.items()):
            row, col = i // n_cols + 1, i % n_cols + 1
            cos_arr = np.array(entry["cos"])
            cos_disp = cos_arr.copy()
            np.fill_diagonal(cos_disp, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=cos_disp, x=entry["traits"], y=entry["traits"],
                    colorscale="RdBu_r", zmin=-0.4, zmax=0.4,
                    showscale=(i == 0),
                    text=[[f"{v:+.2f}" for v in row] for row in cos_arr],
                    texttemplate="%{text}", textfont=dict(size=12),
                    hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>cos=%{z:+.3f}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig.update_yaxes(autorange="reversed", row=row, col=col)
        fig.update_layout(
            title=dict(text="Goldberg 52 markers — Big Five trait cosine similarity", x=0.5),
            height=400 * n_rows, width=450 * n_cols,
        )
        hm_path = Path("results/markers_as_stimuli_heatmap.html")
        fig.write_html(str(hm_path))
        print(f"Wrote {hm_path}")


if __name__ == "__main__":
    main()
