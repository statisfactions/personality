"""Reproduce arXiv:2509.10078's eta^2 / WMV construct-structure table and
item-heatmaps with the rescued VP scoring (W18 §7 follow-on).

Their RQ2: item scores grouped by construct — questionnaire items cluster
(eta^2 ~0.5), generation items scatter (~0.04, n.s.) -> "generation lacks
construct structure". We recompute item-level eta^2 three ways:
  QUESTIONNAIRE  our IPIP-300 item EVs (reverse-keyed), grouped by Big5 scale
  GEN-THEIRS     raw total log P per tagged response (v4 quantity, pos r>=.3)
  GEN-OURS       within-scenario z per-token preference (the reliable scorer)
eta^2 = SS_between/SS_total over (item, construct) pairs; WMV = mean
within-construct variance / total variance; p from 1000 label permutations.
Heatmap: models x pairs (sorted by construct), per-model z, three panels.

Usage: PYTHONPATH=scripts python scripts/vp_eta2.py
(needs results/vp_rescore/<m>.json from vp_rescore.py)
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = "results/vp_rescore"
MODELS = ["gemma3", "Qwen7", "llama3.2", "qwen2.5", "phi4"]
SURVEY = {"gemma3": "Gemma", "Qwen7": "Qwen7", "llama3.2": "Llama",
          "qwen2.5": "Qwen", "phi4": "Phi4"}
PVQ = ["Benevolence", "Universalism", "Self_Direction", "Stimulation",
       "Hedonism", "Achievement", "Power", "Security", "Conformity",
       "Tradition"]
BFI = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
       "Openness"]
CONSTRUCTS = PVQ + BFI
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
RNG = np.random.default_rng(11)


def eta2_wmv(scores, groups, nperm=1000):
    scores = np.asarray(scores, float)
    g = np.asarray(groups)
    gm = scores.mean()
    sst = ((scores - gm) ** 2).sum()
    labs = np.unique(g)

    def e2(gg):
        ssb = sum(((scores[gg == l].mean() - gm) ** 2) * (gg == l).sum()
                  for l in labs)
        return ssb / sst

    obs = e2(g)
    wmv = np.mean([scores[g == l].var() for l in labs]) / scores.var()
    null = [e2(RNG.permutation(g)) for _ in range(nperm)]
    p = (np.sum(np.array(null) >= obs) + 1) / (nperm + 1)
    return obs, wmv, float(np.mean(null)), p


def main():
    L = np.load(f"{OUT}/labels.npy")
    Q = np.repeat(np.arange(104), 5)

    # tagged (response, construct) pairs, pos r>=0.3 (their v4 rule)
    pairs = {"PVQ": [], "BFI": []}
    for c, name in enumerate(CONSTRUCTS):
        for i in np.where(L[:, c] >= 0.3)[0]:
            pairs["PVQ" if name in PVQ else "BFI"].append((i, name))
    print(f"tagged pairs: PVQ {len(pairs['PVQ'])}  BFI {len(pairs['BFI'])} "
          "(their v4: 286/228)")

    inst = json.load(open("instruments/ipip300.json"))
    rows_q, rows_t, rows_o = [], [], []
    print(f"\n{'model':9s} {'Q-BFI eta2 (p)':>18s} {'TH-PVQ':>14s} "
          f"{'TH-BFI':>14s} {'OURS-PVQ':>14s} {'OURS-BFI':>14s}")
    table = {}
    for m in MODELS:
        d = json.load(open(f"{OUT}/{m}.json"))
        lp = np.array(d["logprob"], float)
        ppt = lp / np.array(d["ntok"], float)
        pz = np.zeros(len(lp))
        for q in range(104):
            s = Q == q
            pz[s] = (ppt[s] - ppt[s].mean()) / (ppt[s].std() + 1e-9)

        res = {}
        for tag, quant in (("TH", lp), ("OURS", pz)):
            for fam in ("PVQ", "BFI"):
                sc = [quant[i] for i, _ in pairs[fam]]
                gg = [c for _, c in pairs[fam]]
                res[f"{tag}-{fam}"] = eta2_wmv(sc, gg)

        # questionnaire: IPIP-300 keyed item EVs grouped by scale
        sv = json.load(open(f"results/surveys/{SURVEY[m]}_ipip300.json"))
        ir = sv["item_results"]
        qs, qg = [], []
        for sk, se in inst["scales"].items():
            rev = set(se["reverse_keyed_item_ids"])
            for iid in se["item_ids"]:
                if iid not in ir:
                    continue
                ev = ir[iid]["expected_value"]
                qs.append(6 - ev if iid in rev else ev)
                qg.append(sk)
        res["Q-BFI"] = eta2_wmv(qs, qg)
        table[m] = res

        def f(k):
            e, w, b, p = res[k]
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            return f"{e:.3f}{star:3s}"
        print(f"{m:9s} {f('Q-BFI'):>18s} {f('TH-PVQ'):>14s} "
              f"{f('TH-BFI'):>14s} {f('OURS-PVQ'):>14s} {f('OURS-BFI'):>14s}")

        rows_q.append((np.array(qs) - np.mean(qs)) / np.std(qs))
        for store, quant in ((rows_t, lp), (rows_o, pz)):
            v = np.array([quant[i] for i, _ in pairs["PVQ"]] +
                         [quant[i] for i, _ in pairs["BFI"]])
            store.append((v - v.mean()) / v.std())

    print("\nmeans over models:")
    for k in ("Q-BFI", "TH-PVQ", "TH-BFI", "OURS-PVQ", "OURS-BFI"):
        e = np.mean([table[m][k][0] for m in MODELS])
        w = np.mean([table[m][k][1] for m in MODELS])
        b = np.mean([table[m][k][2] for m in MODELS])
        print(f"  {k:9s} eta2 {e:.3f}  WMV {w:.3f}  perm-baseline {b:.3f}")
    print("(their v4: questionnaire eta2 PVQ .526 / BFI .492; "
          "generation .040 p=.60 / .038 p=.73)")

    # ---- heatmaps ----
    # questionnaire columns already grouped by scale construction order
    gen_groups = [c for _, c in pairs["PVQ"]] + [c for _, c in pairs["BFI"]]
    order = np.argsort([CONSTRUCTS.index(c) for c in gen_groups],
                       kind="stable")
    gl = [gen_groups[i] for i in order]
    bounds = [i for i in range(1, len(gl)) if gl[i] != gl[i - 1]]

    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.10,
        subplot_titles=["QUESTIONNAIRE — IPIP-300 item EVs (keyed), grouped "
                        "by scale",
                        "GENERATION, their v4 scoring — raw total log P, "
                        "tagged pairs by construct",
                        "GENERATION, rescued scoring — within-scenario "
                        "per-token z, same pairs"])
    fig.add_trace(go.Heatmap(z=np.array(rows_q), colorscale="RdBu_r",
                             zmin=-2.5, zmax=2.5, showscale=False),
                  row=1, col=1)
    for ri, rows in ((2, rows_t), (3, rows_o)):
        Z = np.array(rows)[:, order]
        fig.add_trace(go.Heatmap(z=Z, colorscale="RdBu_r", zmin=-2.5,
                                 zmax=2.5, showscale=False), row=ri, col=1)
        for b in bounds:
            fig.add_vline(x=b - 0.5, line=dict(color=INK2, width=0.5),
                          row=ri, col=1)
    for i in range(1, 5):
        fig.add_vline(x=i * 60 - 0.5, line=dict(color=INK2, width=0.5),
                      row=1, col=1)
    fig.update_yaxes(tickvals=list(range(len(MODELS))), ticktext=MODELS,
                     tickfont=dict(size=8))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        title=dict(text="Construct structure, item level: questionnaire vs "
                        "generation under both scorings (blocks = structure; "
                        "cells are per-model z)", font=dict(size=13, color=INK)),
        width=1400, height=640, margin=dict(l=70, r=30, t=90, b=30),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    out = f"{OUT}/figs/vp_eta2_heatmap"
    import os
    os.makedirs(f"{OUT}/figs", exist_ok=True)
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"\nsaved {out}.png/.html")


if __name__ == "__main__":
    main()
