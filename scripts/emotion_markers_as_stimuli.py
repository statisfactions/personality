#!/usr/bin/env python3
"""8 bipolar emotion axes as RepE stimuli — cross-domain test of high-bandwidth
preservation (W7 §11.5.4 #3 / to_try §16).

Mirrors markers_as_stimuli.py exactly: one short phrase per stimulus, chat-
template wrapped, mean(high-pole) − mean(low-pole) at ~2/3-depth, neutral-PC-
projected. Each emotion axis becomes one direction; cosine matrix between
the 8 axes per model; cross-model upper-tri correlation between cosine
matrices.

Comparison baseline: Goldberg 52 markers (Big Five) on the same 3-model
larger-cohort gave cross-model r = 0.98–0.99. If emotions show similar
fidelity, transformer cross-architecture preservation is not personality-
specific. If much weaker, personality stimuli may be special.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/emotion_markers_as_stimuli.py
    PYTHONPATH=scripts .venv/bin/python scripts/emotion_markers_as_stimuli.py \
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


CACHE_DIR = Path("results/phase_b_cache")
EMOTION_FILE = Path("instruments/emotion_markers.json")
COHORT = ["Gemma12", "Llama8", "Qwen7"]


def safe(s): return s.replace("/", "_")


def load_axes():
    with open(EMOTION_FILE) as f:
        d = json.load(f)
    return d["axes"]


def collect_stimuli(axes):
    """Return list of {text, axis, pole}."""
    stim = []
    for axis, poles in axes.items():
        for pole, items in poles.items():
            for adj in items:
                stim.append({"text": adj, "axis": axis, "pole": pole})
    return stim


def extract_activations(model, tok, stim, device):
    acts = []
    for i, s in enumerate(stim):
        a = mdx.hidden_states_for_text(
            model, tok, s["text"], device,
            split_prefix=None, chat_template=True,
        )
        acts.append(a)
        if (i + 1) % 25 == 0:
            print(f"    stim {i+1}/{len(stim)} done")
    return torch.stack(acts)


def axis_directions(stim, acts, common_layer, neutral_layer, axis_order):
    pcs, _, _ = mdx.compute_pc_projection(neutral_layer, 0.5)
    names, dirs = [], []
    for axis in axis_order:
        high_idxs = [i for i, s in enumerate(stim) if s["axis"] == axis and s["pole"] == "high"]
        low_idxs  = [i for i, s in enumerate(stim) if s["axis"] == axis and s["pole"] == "low"]
        if not high_idxs or not low_idxs:
            print(f"    SKIP {axis}: high={len(high_idxs)} low={len(low_idxs)}")
            continue
        high_mean = acts[high_idxs, common_layer, :].float().mean(dim=0).numpy()
        low_mean = acts[low_idxs, common_layer, :].float().mean(dim=0).numpy()
        diff = high_mean - low_mean
        d = mdx.project_out_pcs(diff, pcs)
        n = np.linalg.norm(d)
        if n > 1e-12:
            d = d / n
        names.append(axis)
        dirs.append(d)
    return names, np.vstack(dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=COHORT)
    args = parser.parse_args()

    axes = load_axes()
    axis_order = list(axes.keys())
    stim = collect_stimuli(axes)
    n_high = sum(1 for s in stim if s["pole"] == "high")
    n_low = sum(1 for s in stim if s["pole"] == "low")
    print(f"Loaded {len(stim)} emotion stimuli (high={n_high}, low={n_low}) over {len(axis_order)} axes")
    for axis in axis_order:
        h = sum(1 for s in stim if s["axis"] == axis and s["pole"] == "high")
        l = sum(1 for s in stim if s["axis"] == axis and s["pole"] == "low")
        print(f"  {axis}: high={h}, low={l}")

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

        print(f"  extracting {len(stim)} emotion-stimulus activations...")
        acts = extract_activations(model, tok, stim, device)

        names, D = axis_directions(stim, acts, common_layer, neutral_layer, axis_order)
        cos = D @ D.T

        print(f"\n  {len(names)}×{len(names)} axis cosine matrix:")
        print("       " + "  ".join(f"{n[:5]:>6s}" for n in names))
        for i, n in enumerate(names):
            print(f"  {n[:5]:>5s} " + "  ".join(f"{cos[i, j]:+6.3f}" for j in range(len(names))))

        output[short] = {
            "common_layer": common_layer,
            "n_layers": n_layers_dim,
            "axes": names,
            "cos": cos.tolist(),
        }

        del model, tok, acts
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    out_path = Path("results/emotion_markers_as_stimuli.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # Cross-model upper-tri correlation
    if len(output) >= 2:
        names_models = list(output.keys())
        n_axes = len(output[names_models[0]]["axes"])
        iu = np.triu_indices(n_axes, k=1)
        cos_per = {n: np.array(output[n]["cos"]) for n in names_models}
        print(f"\n=== Cross-model upper-tri correlation between {n_axes}×{n_axes} cosine matrices ===")
        print(f"{'':>10s} " + " ".join(f"{n:>10s}" for n in names_models))
        rs = []
        for a in names_models:
            row = []
            for b in names_models:
                if a == b:
                    row.append(f"{1.0:>+10.3f}")
                else:
                    r = float(np.corrcoef(cos_per[a][iu], cos_per[b][iu])[0, 1])
                    row.append(f"{r:>+10.3f}")
                    if a < b:
                        rs.append(r)
            print(f"{a:>10s} " + " ".join(row))
        if rs:
            print(f"\nMean cross-model upper-tri r: {np.mean(rs):+.3f} "
                  f"(range {min(rs):+.3f} to {max(rs):+.3f})")
            print("Personality markers baseline (W7 §8.5 same 3 models): "
                  "+0.986 (range +0.980 to +0.991)")

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
                    z=cos_disp, x=entry["axes"], y=entry["axes"],
                    colorscale="RdBu_r", zmin=-0.5, zmax=0.5,
                    showscale=(i == 0),
                    text=[[f"{v:+.2f}" for v in r] for r in cos_arr],
                    texttemplate="%{text}", textfont=dict(size=11),
                    hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>cos=%{z:+.3f}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig.update_yaxes(autorange="reversed", row=row, col=col)
        fig.update_layout(
            title=dict(text="8 emotion axes — cosine similarity (W7 §11.5.4 #3)", x=0.5),
            height=500 * n_rows, width=550 * n_cols,
        )
        hm_path = Path("results/emotion_markers_as_stimuli_heatmap.html")
        fig.write_html(str(hm_path))
        print(f"Wrote {hm_path}")


if __name__ == "__main__":
    main()
