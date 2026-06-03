#!/usr/bin/env python3
"""Representation vs introspection on adjective similarity (W15 §1; to_try.md #21).

The W14 arc showed the model's *resting* adjective geometry is encoder-like and
evaluation-dominated — positive and negative evaluation are near-orthogonal, so
Wonderful≈Awful (+0.41 cosine) rather than antonyms. But does the model *act* on
that geometry? Here we elicit the model's *judged* pairwise similarity (a
behavioral self-report) on a pole-spanning adjective subset, with a deliberately
**valence-neutral** anchor (1 = completely different … 7 = nearly the same — never
"opposite"), and compare three corners on the same words:
  behavioral (what it judges) · representational (its cosine) · human (525-PDA).

Sharp test: does judged similarity put pos-eval × neg-eval FAR APART (human-like
valence) while the representation puts them together (+0.41)? If so, behavior
overrides the associative geometry using symbolic valence knowledge — the read/write
gap, and the symbolic-vs-associative split, made behavioral.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_introspection.py \
           --models Qwen7,Gemma12 [--force]
"""
import argparse
import json
import os

import numpy as np

from hf_logprobs import MODELS, load_model, likert_distribution
from adjective_geom import model_matrix

GROUPS = {
    "pos_eval":   ["Wonderful", "Amazing", "Excellent", "Great"],
    "neg_eval":   ["Awful", "Disgusting", "Terrible", "Bad"],
    "warm":       ["Caring", "Kind", "Friendly", "Thoughtful"],
    "antagonism": ["Cruel", "Mean", "Rude", "Hostile"],
    "distress":   ["Anxious", "Insecure", "Moody", "Impulsive"],
    "intellect":  ["Intelligent", "Creative", "Curious", "Knowledgeable"],
    "neutral":    ["Ordinary", "Tall"],
}
ADJ = [w for ws in GROUPS.values() for w in ws]
GROUP_OF = {w: g for g, ws in GROUPS.items() for w in ws}
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
CACHE = {"semantic": "results/adjectives/introspect",
         "tom": "results/adjectives/introspect_tom"}
# training-stage checkpoints: probe with bare text so prompt FORMAT is held constant
# across base/SFT/DPO/RLVR (otherwise "behavior appears at SFT" is confounded with
# the chat template being introduced at SFT). Base also simply has no chat template.
BASE_MODELS = {"Qwen7Base", "Olmo2Base", "Olmo2SFT", "Olmo2DPO", "Olmo2Inst",
               "ZephyrSFT", "ZephyrDPO"}

# semantic = a dictionary task (how alike in meaning); tom = the HUMAN's task
# (persona = adjective a, rate how well adjective b describes that person —
# dispositional, matching the 525-PDA self-rating construct).
PROMPT = {
    "semantic": (
        "How similar in meaning are these two words, as descriptions of a person?\n"
        "Word 1: {a}\nWord 2: {b}\n"
        "Answer with one number from 1 to 7, where 1 = completely different in "
        "meaning and 7 = nearly the same in meaning.\nNumber:"),
    "tom": (
        "Consider a person who is very {a}.\n"
        "How accurately does the word \"{b}\" describe this person?\n"
        "Answer with one number from 1 to 7, where 1 = very inaccurate and "
        "7 = very accurate.\nNumber:"),
}


def ev(dist):
    return sum(int(k) * v for k, v in dist.items())


def elicit(short, mode="semantic", force=False, bare=False):
    bare = bare or short in BASE_MODELS          # base/staged ckpts: always bare text
    suffix = "_bare" if bare else ""
    os.makedirs(CACHE[mode], exist_ok=True)
    path = f"{CACHE[mode]}/{short}{suffix}.json"
    if os.path.exists(path) and not force:
        return json.load(open(path))
    model, tok, device = load_model(MODELS[short])
    n = len(ADJ)
    B = np.full((n, n), np.nan)
    Hent = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d, _, h = likert_distribution(
                model, tok, PROMPT[mode].format(a=ADJ[i].lower(), b=ADJ[j].lower()),
                device, digits=DIGITS, use_chat_template=not bare)
            B[i, j] = ev(d); Hent[i, j] = h
        print(f"  {short}/{mode}{suffix}: row {i+1}/{n} ({ADJ[i]})", flush=True)
    B_raw = B.copy()                                  # keep directional (asymmetry)
    B = np.nanmean(np.stack([B, B.T]), 0)            # symmetrize over order
    out = {"model": short, "mode": mode, "bare": bare, "adjectives": ADJ,
           "behavioral": B.tolist(), "directional": B_raw.tolist(),
           "entropy": Hent.tolist(), "mean_entropy": float(np.nanmean(Hent))}
    json.dump(out, open(path, "w"), indent=0)
    import gc
    import torch
    del model, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def offdiag(A):
    iu = np.triu_indices_from(A, 1); return A[iu]


def pc1frac(A):                      # single-axis (halo) dominance
    A = np.nan_to_num(A, nan=np.nanmean(A)); ev = np.sort(np.linalg.eigvalsh(A))[::-1]
    return float(ev[0] / np.sum(np.abs(ev)))


def block_mean(A, ga, gb):
    ia = [ADJ.index(w) for w in GROUPS[ga]]; ib = [ADJ.index(w) for w in GROUPS[gb]]
    sub = A[np.ix_(ia, ib)]
    return float(np.nanmean(sub))


SIZE = {"Qwen": "3B", "Gemma": "4B", "Llama": "3B", "Phi4": "4B", "Qwen7": "7B",
        "Llama8": "8B", "Gemma12": "12B", "Gemma27": "27B", "Qwen32": "32B",
        "Gemma4": "31B", "Aya": "8B", "FalconMamba": "7B"}
PARAMS = {"3B": 3, "4B": 4, "7B": 7, "8B": 8, "12B": 12, "27B": 27, "31B": 31, "32B": 32}


def compare(behavs):
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    full = np.array(human["correlation_matrix"], float); flab = list(human["labels"])
    idx = [flab.index(w) for w in ADJ]
    H = full[np.ix_(idx, idx)]

    def z(A):
        o = offdiag(A); return (o - np.nanmean(o)) / (np.nanstd(o) + 1e-9)

    def pn(A):                       # pos_eval×neg_eval block, z-scored within corner
        o = offdiag(A); m, s = np.nanmean(o), np.nanstd(o) + 1e-9
        return (block_mean(A, "pos_eval", "neg_eval") - m) / s
    hz = z(H); h_pn = pn(H)

    rows = []
    print(f"\n{'model':<10} corr(behav,human)  corr(repr,human)  corr(behav,repr) | "
          "posXneg z: human / repr / behav")
    for short, B in behavs.items():
        _, C, _, _ = model_matrix(
            f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers.pt",
            np.array(flab))
        R = C[np.ix_(idx, idx)]; B = np.array(B)
        bz, rz = z(B), z(R)
        r = {"model": short, "size": SIZE.get(short, "?"),
             "behav_human": float(np.corrcoef(bz, hz)[0, 1]),
             "repr_human": float(np.corrcoef(rz, hz)[0, 1]),
             "behav_repr": float(np.corrcoef(bz, rz)[0, 1]),
             "pn_repr": pn(R), "pn_behav": pn(B)}
        rows.append(r)
        print(f"{short:<10} {r['behav_human']:+.3f}            {r['repr_human']:+.3f}"
              f"           {r['behav_repr']:+.3f}        | "
              f"{h_pn:+.2f} / {r['pn_repr']:+.2f} / {r['pn_behav']:+.2f}")
    figure(rows, h_pn)


def figure(rows, h_pn):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = sorted(rows, key=lambda r: PARAMS.get(r["size"], 99))
    x = [f"{r['model']}<br>{r['size']}" for r in rows]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.13,
                        subplot_titles=(
        "pos-eval × neg-eval: representation MERGES, judgment FLIPS",
        "overall match to human"))
    for i, r in enumerate(rows):                       # dumbbells repr -> behav
        fig.add_trace(go.Scatter(x=[x[i], x[i]], y=[r["pn_repr"], r["pn_behav"]],
            mode="lines", line=dict(color="#bbb", width=2), showlegend=False,
            hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["pn_repr"] for r in rows], mode="markers",
        name="representation (cosine)", marker=dict(size=13, color="#d62728")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["pn_behav"] for r in rows], mode="markers",
        name="judgment (behavioral)", marker=dict(size=13, color="#1f77b4")), row=1, col=1)
    fig.add_hline(y=h_pn, line_dash="dash", line_color="#2ca02c", row=1, col=1,
                  annotation_text=f"human {h_pn:+.2f}", annotation_font_color="#2ca02c")
    fig.add_hline(y=0, line_color="#ddd", row=1, col=1)
    fig.update_yaxes(title_text="pos×neg similarity (z within corner)", row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=[r["repr_human"] for r in rows], name="repr↔human",
        marker_color="#f0a0a0", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=[r["behav_human"] for r in rows], name="behav↔human",
        marker_color="#9ec3e8", showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="matrix corr to human", range=[0, 0.9], row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Representation vs introspection: the model judges "
            "Wonderful≠Awful even though it represents them as neighbors</b><br><sub>"
            "Left: each model's representational cosine puts pos-eval×neg-eval ABOVE its "
            "own mean (the merge, red); its judged similarity flips BELOW, onto the human "
            "antonym value (blue, green line) — a sign reversal at every size, 3B→32B. "
            "Right: overall both corners are ~human; the divergence is localized to the "
            "evaluative merge. Symbolic valence overrides associative geometry.</sub>",
            x=0.01),
        width=1180, height=560, barmode="group", plot_bgcolor="white",
        font=dict(family="Helvetica, Arial"), legend=dict(x=0.0, y=-0.13, orientation="h"))
    out = "results/adjectives/introspection_vs_representation.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1180, height=560, scale=2)
    print(f"wrote {out} and .png")


def compare_tom(models):
    """4 corners on pos-eval×neg-eval: representation, semantic judgment, ToM
    (persona-conditioned dispositional) judgment, human. Tests whether the model's
    *dispositional* reasoning (the human's actual construct) splits or merges."""
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    full = np.array(human["correlation_matrix"], float); flab = list(human["labels"])
    idx = [flab.index(w) for w in ADJ]; H = full[np.ix_(idx, idx)]

    def pn(A):
        o = offdiag(A); m, s = np.nanmean(o), np.nanstd(o) + 1e-9
        return (block_mean(A, "pos_eval", "neg_eval") - m) / s

    def z(A):
        o = offdiag(A); return (o - np.nanmean(o)) / (np.nanstd(o) + 1e-9)

    def pc1frac(A):                  # single-axis (halo) dominance
        A = np.nan_to_num(A, nan=np.nanmean(A)); ev = np.sort(np.linalg.eigvalsh(A))[::-1]
        return float(ev[0] / np.sum(np.abs(ev)))
    hz = z(H)
    rows = []
    print(f"\n{'model':<8} pos×neg z: repr / semantic / ToM / human   | corr(ToM,human) "
          "corr(ToM,sem) corr(ToM,repr) | pc1 ToM/human")
    for short in models:
        Bs = np.array(json.load(open(f"{CACHE['semantic']}/{short}.json"))["behavioral"])
        Bt = np.array(json.load(open(f"{CACHE['tom']}/{short}.json"))["behavioral"])
        _, C, _, _ = model_matrix(
            f"results/adjectives/acts/{MODELS[short].replace('/', '_')}__pers.pt",
            np.array(flab))
        R = C[np.ix_(idx, idx)]
        rows.append({"model": short, "size": SIZE.get(short, "?"),
                     "repr": pn(R), "sem": pn(Bs), "tom": pn(Bt), "human": pn(H),
                     "tom_human": float(np.corrcoef(z(Bt), hz)[0, 1]),
                     "tom_repr": float(np.corrcoef(z(Bt), z(R))[0, 1]),
                     "pc1_tom": pc1frac(Bt), "pc1_human": pc1frac(H)})
        r = rows[-1]
        print(f"{short:<8} {r['repr']:+.2f} / {r['sem']:+.2f} / {r['tom']:+.2f} / "
              f"{r['human']:+.2f}      | {r['tom_human']:+.3f}        "
              f"{np.corrcoef(z(Bt),z(Bs))[0,1]:+.3f}       {r['tom_repr']:+.3f}     | "
              f"{r['pc1_tom']:.2f}/{r['pc1_human']:.2f}")
    figure_tom(rows, pn(H))


def figure_tom(rows, h_pn):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = sorted(rows, key=lambda r: PARAMS.get(r["size"], 99))
    x = [f"{r['model']}<br>{r['size']}" for r in rows]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.13,
                        subplot_titles=("pos-eval × neg-eval across 4 corners",
                                        "single-axis (halo) collapse"))
    for i, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=[x[i]] * 3, y=[r["repr"], r["sem"], r["tom"]],
            mode="lines", line=dict(color="#ccc", width=1.5), showlegend=False,
            hoverinfo="skip"), row=1, col=1)
    for key, name, color in (("repr", "representation", "#d62728"),
                             ("sem", "semantic judgment", "#1f77b4"),
                             ("tom", "ToM judgment (persona)", "#9467bd")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="markers",
            name=name, marker=dict(size=13, color=color)), row=1, col=1)
    fig.add_hline(y=h_pn, line_dash="dash", line_color="#2ca02c", row=1, col=1,
                  annotation_text=f"human {h_pn:+.2f}", annotation_font_color="#2ca02c")
    fig.add_hline(y=0, line_color="#eee", row=1, col=1)
    fig.update_yaxes(title_text="pos×neg similarity (z within corner)", row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=[r["pc1_tom"] for r in rows], marker_color="#9467bd",
        showlegend=False), row=1, col=2)
    fig.add_hline(y=rows[0]["pc1_human"], line_dash="dash", line_color="#2ca02c",
                  row=1, col=2, annotation_text="human", annotation_font_color="#2ca02c")
    fig.update_yaxes(title_text="ToM PC1 variance fraction", range=[0, 0.5], row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Persona/ToM judgment: the model's person-reasoning splits "
            "good/bad even harder than humans</b><br><sub>Left: representation MERGES "
            "pos×neg (red); both the semantic judgment (blue) and the ToM judgment "
            "(persona=A, rate B; purple) split it — ToM OVERSHOOTS the human (green). "
            "So the representation is the odd-one-out in both framings. Right: ToM "
            "collapses onto a single evaluative axis more than humans do (assistant "
            "halo).</sub>", x=0.01),
        width=1180, height=560, plot_bgcolor="white", font=dict(family="Helvetica, Arial"),
        legend=dict(x=0.0, y=-0.13, orientation="h"))
    out = "results/adjectives/introspection_tom.html"
    fig.write_html(out, include_plotlyjs="cdn")
    fig.write_image(out.replace(".html", ".png"), width=1180, height=560, scale=2)
    print(f"wrote {out} and .png")


def compare_template(models, mode):
    """Chat vs bare on the SAME instruct weights — decomposes the W15 effect into
    weights vs context (the chat template = assistant-persona activation). Loads the
    chat-template cache (<short>.json) and the bare cache (<short>_bare.json)."""
    human = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    H = np.array(human["correlation_matrix"], float); flab = list(human["labels"])
    idx = [flab.index(w) for w in ADJ]; Hs = H[np.ix_(idx, idx)]

    def pn(A):
        o = offdiag(A); m, s = np.nanmean(o), np.nanstd(o) + 1e-9
        return (block_mean(A, "pos_eval", "neg_eval") - m) / s
    print(f"\n[{mode}] chat vs bare (same weights)   human pos×neg z = {pn(Hs):+.2f}")
    print(f"{'model':<8} pos×neg z chat/bare | PC1 chat/bare | mean-entropy chat/bare")
    for short in models:
        ch = json.load(open(f"{CACHE[mode]}/{short}.json"))
        ba = json.load(open(f"{CACHE[mode]}/{short}_bare.json"))
        Bc, Bb = np.array(ch["behavioral"]), np.array(ba["behavioral"])
        ec = ch.get("mean_entropy", float("nan")); eb = ba.get("mean_entropy", float("nan"))
        print(f"{short:<8} {pn(Bc):+.2f} / {pn(Bb):+.2f}      | "
              f"{pc1frac(Bc):.2f} / {pc1frac(Bb):.2f}   | {ec:.2f} / {eb:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen7")
    ap.add_argument("--mode", default="semantic", choices=["semantic", "tom"])
    ap.add_argument("--bare", action="store_true",
                    help="force bare text on instruct models (raw-vs-chat study)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    models = args.models.split(",")
    behavs = {}
    for short in models:
        print(f"eliciting {short} ({args.mode}{'/bare' if args.bare else ''}) ...", flush=True)
        behavs[short] = np.array(elicit(short, args.mode, args.force, args.bare)["behavioral"])
    try:                                  # base/staged models lack the compare inputs
        if args.bare:
            compare_template(models, args.mode)
        elif args.mode == "tom":
            compare_tom(models)
        else:
            compare(behavs)
    except FileNotFoundError as e:
        print(f"(elicited + cached; skipping built-in compare — {e})")


if __name__ == "__main__":
    main()
