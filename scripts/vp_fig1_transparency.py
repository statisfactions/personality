"""Reproduce arXiv:2509.10078 Figure 1 (item-transparency heatmaps) with our
materials: IPIP-300 as the established instrument, VP responses as the
ecological one. (W18 §7 follow-on — this is the part of their paper we
endorse; it feeds the eta2-measures-transparency resolution.)

Four panels, all cosine similarities from all-mpnet-base-v2 embeddings:
  (a) IPIP-300 items x Big5 construct definitions   (expect diagonal bands)
  (b) VP responses x 15 construct definitions        (expect none)
  (c) IPIP-300 item x item, sorted by scale          (expect blocks)
  (d) VP response x response, sorted by primary tag  (expect none)
Prints their two summary stats per instrument: item-definition
discrimination (own-construct sim minus mean other-construct sim) and
within-vs-between construct item-similarity gap.

Usage: PYTHONPATH=scripts python scripts/vp_fig1_transparency.py
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer

OUT = "results/vp_rescore"
SP = "dependencies"  # scripts/fetch_dependencies.sh clones ValuePortrait here
PVQ = ["Benevolence", "Universalism", "Self_Direction", "Stimulation",
       "Hedonism", "Achievement", "Power", "Security", "Conformity",
       "Tradition"]
BFI = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
       "Openness"]
CONSTRUCTS = PVQ + BFI
DEFS = {
    "Benevolence": "preserving and enhancing the welfare of people one is "
                   "in frequent personal contact with; being helpful, "
                   "honest, forgiving, and loyal",
    "Universalism": "understanding, appreciation, tolerance, and protection "
                    "for the welfare of all people and for nature",
    "Self_Direction": "independent thought and action; choosing, creating, "
                      "exploring things on one's own",
    "Stimulation": "excitement, novelty, and challenge in life; a varied "
                   "and daring life",
    "Hedonism": "pleasure and sensuous gratification for oneself; enjoying "
                "life",
    "Achievement": "personal success through demonstrating competence "
                   "according to social standards; being ambitious",
    "Power": "social status and prestige, control or dominance over people "
             "and resources; authority and wealth",
    "Security": "safety, harmony, and stability of society, relationships, "
                "and of self",
    "Conformity": "restraint of actions and impulses likely to upset others "
                  "or violate social norms; obedience and politeness",
    "Tradition": "respect, commitment, and acceptance of the customs and "
                 "ideas of one's culture or religion; being humble and "
                 "devout",
    "Extraversion": "being outgoing, sociable, talkative, energetic, and "
                    "assertive around other people",
    "Agreeableness": "being kind, cooperative, warm, considerate, trusting, "
                     "and sympathetic toward others",
    "Conscientiousness": "being organized, responsible, hardworking, "
                         "careful, and self-disciplined",
    "Neuroticism": "tending to experience anxiety, worry, sadness, "
                   "moodiness, and emotional instability",
    "Openness": "being curious, imaginative, creative, and open to new "
                "ideas, art, and experiences",
}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    inst = json.load(open("instruments/ipip300.json"))
    ipip_items, ipip_scale = [], []
    for sk, se in inst["scales"].items():
        for iid in se["item_ids"]:
            ipip_items.append(inst["items"][iid])
            ipip_scale.append(sk)
    scale_names = list(dict.fromkeys(ipip_scale))

    vp = json.load(open(f"{SP}/ValuePortrait/data/query-response-tagged/"
                        "query-response-tagged.json"))
    L = np.load(f"{OUT}/labels.npy")
    vp_texts = [o["content"] for e in vp for o in e["outputs"]]
    # primary construct = strongest positive label (their tagging spirit)
    prim = [CONSTRUCTS[int(np.argmax(L[i]))] for i in range(len(vp_texts))]
    vp_order = np.argsort([CONSTRUCTS.index(p) for p in prim], kind="stable")

    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    E_def = enc.encode([DEFS[c] for c in CONSTRUCTS], normalize_embeddings=True)
    E_ipip = enc.encode(ipip_items, normalize_embeddings=True,
                        show_progress_bar=False)
    E_vp = enc.encode(vp_texts, normalize_embeddings=True,
                      show_progress_bar=False)

    S_ipip_def = E_ipip @ E_def[10:].T          # 300 x 5 (Big5 defs)
    S_vp_def = E_vp @ E_def.T                   # 520 x 15
    S_ipip = E_ipip @ E_ipip.T
    S_vp = E_vp @ E_vp.T

    # their two stats
    KEY = {"IPIP300-EXT": "Extraversion", "IPIP300-AGR": "Agreeableness",
           "IPIP300-CON": "Conscientiousness", "IPIP300-NEU": "Neuroticism",
           "IPIP300-OPE": "Openness"}
    own = np.array([S_ipip_def[i, BFI.index(KEY[ipip_scale[i]])]
                    for i in range(len(ipip_items))])
    oth = np.array([np.mean([S_ipip_def[i, j] for j in range(5)
                             if j != BFI.index(KEY[ipip_scale[i]])])
                    for i in range(len(ipip_items))])
    print(f"IPIP-300 item-definition discrimination: "
          f"{np.mean(own - oth):+.3f} (theirs, established: 0.13-0.22)")
    ownv = np.array([S_vp_def[i, CONSTRUCTS.index(prim[i])]
                     for i in range(len(vp_texts))])
    othv = np.array([np.mean(np.delete(S_vp_def[i], CONSTRUCTS.index(prim[i])))
                     for i in range(len(vp_texts))])
    print(f"VP item-definition discrimination:       "
          f"{np.mean(ownv - othv):+.3f} (theirs, VP: ~0)")

    def wb_gap(S, groups):
        g = np.asarray(groups)
        n = len(g)
        m = ~np.eye(n, dtype=bool)
        same = (g[:, None] == g[None, :]) & m
        return S[same].mean() - S[~same & m].mean()

    print(f"IPIP-300 within-vs-between item-sim gap:  "
          f"{wb_gap(S_ipip, ipip_scale):+.3f} (theirs: 0.07-0.15)")
    print(f"VP within-vs-between item-sim gap:        "
          f"{wb_gap(S_vp[np.ix_(vp_order, vp_order)], [prim[i] for i in vp_order]):+.3f} "
          "(theirs: ~0)")

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.12,
        subplot_titles=[
            "(a) IPIP-300 items × Big5 definitions",
            "(b) VP responses × 15 definitions",
            "(c) IPIP-300 item × item (by scale)",
            "(d) VP response × response (by primary label)"])
    fig.add_trace(go.Heatmap(z=S_ipip_def.T, colorscale="Viridis",
                             zmin=0, zmax=0.55, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=S_vp_def[vp_order].T, colorscale="Viridis",
                             zmin=0, zmax=0.55, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=S_ipip, colorscale="Viridis", zmin=0,
                             zmax=0.7, showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(z=S_vp[np.ix_(vp_order, vp_order)],
                             colorscale="Viridis", zmin=0, zmax=0.7,
                             showscale=False), row=2, col=2)
    fig.update_yaxes(tickvals=list(range(5)), ticktext=BFI,
                     tickfont=dict(size=7), row=1, col=1)
    fig.update_yaxes(tickvals=list(range(15)), ticktext=CONSTRUCTS,
                     tickfont=dict(size=6), row=1, col=2)
    for rc in ((2, 1), (2, 2)):
        fig.update_xaxes(showticklabels=False, row=rc[0], col=rc[1])
        fig.update_yaxes(showticklabels=False, row=rc[0], col=rc[1])
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        title=dict(text="Their Figure 1, our materials — item transparency: "
                        "established items announce their construct, "
                        "ecological items don't (all-mpnet-base-v2 cosines)",
                   font=dict(size=13, color=INK)),
        width=1250, height=900, margin=dict(l=90, r=30, t=90, b=30),
        paper_bgcolor=SURF, plot_bgcolor=SURF)
    out = f"{OUT}/figs/vp_fig1_transparency"
    fig.write_html(f"{out}.html")
    fig.write_image(f"{out}.png", scale=2)
    print(f"saved {out}.png/.html")


if __name__ == "__main__":
    main()
