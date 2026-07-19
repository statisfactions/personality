"""J-lens probe: how much of ENACT/REPRESENT is the adjective's J-lens vector,
and what do the vectors read out over the full vocabulary? (W18 §5)

Uses the pre-fitted Neuronpedia Jacobian lens (Anthropic 'Verbalizable
Workspace' paper; github.com/anthropics/jacobian-lens). The lens file holds
one J_l (d_model x d_model) per layer; the J-lens vector for token t at
layer l is v_t = J_l^T (g * u_t) with g the final RMSNorm weight and u_t the
lm_head row. No model forward passes needed.

Part 1 (smoke): matched-pair cosine between our recorded ENACT directions and
v_adj across all lens layers; top-1 retrieval among the single-token
adjectives; same for REPRESENT mid-layer acts (echo-confounded: the read
activation sits at the adjective's own token and the lens includes copy
pathways).

Part 2 (qualitative): per-adjective diagonal table with covariates (leak,
boot, enactability) and nearest-neighbour lens vectors; z-scored full-vocab
readouts (z per token across the adjective set kills the anisotropy junk).
Finding (2026-07-14, Qwen7): REPRESENT reads out as a dictionary entry
(word+synonyms+morphology, always); ENACT reads out as the persona's
speech-world when the persona works (religious -> preacher/prayed/God), junk
when it doesn't; failed virtue-personas collapse onto the assistant blob
(helpful/supportive/consistent), failed casual/negative ones onto a generic
wacky-character blob (funny/crazy/weird).

Usage: PYTHONPATH=scripts python scripts/jlens_enact_probe.py [--model Qwen7]
(only Qwen7 wired: lens repo path + enact npz; extend LENS_FILES for others —
pre-fitted lenses exist for gemma-3-4b/12b/27b-it, gemma-4-31b, llama3.1-8b-it.)
"""
import argparse
import json
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

import hf_logprobs as hf

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {
    "Qwen7": "qwen2.5-7b-it/jlens/Salesforce-wikitext/"
             "Qwen2.5-7B-Instruct_jacobian_lens.pt",
}


def load_unembed(repo):
    snap = snapshot_download(repo, allow_patterns=["*.safetensors*", "config.json"])
    wmap = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]

    def get(name):
        with safe_open(f"{snap}/{wmap[name]}", framework="pt") as f:
            return f.get_tensor(name)

    head = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    return get(head).float() * get("model.norm.weight").float()


def ccos(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)
    return A @ B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen7")
    args = ap.parse_args()
    m = args.model
    repo = hf.resolve(m)

    ze = np.load(f"results/persona_vectors/enact_mid/{m}.npz", allow_pickle=True)
    adjs = [str(a) for a in ze["adjectives"]]
    E = torch.tensor(ze["dir"].astype(np.float32))
    meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
    mid = meta["mid_layer"]
    rep = json.load(open(f"results/persona_vectors/{m}_pda_report.json"))["adjectives"]
    ena = json.load(open(f"results/adjectives/enactability/{m}_enactability.json"))["scores"]

    tok = AutoTokenizer.from_pretrained(repo)
    ids, keep = [], []
    for i, a in enumerate(adjs):
        t = tok.encode(" " + a, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
            keep.append(i)
    kad = [adjs[i] for i in keep]
    print(f"single-token adjectives: {len(keep)}/{len(adjs)}")

    Wg = load_unembed(repo)
    J = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[m]),
                   map_location="cpu", weights_only=False)["J"]
    Ek = E[keep]
    Utok = Wg[torch.tensor(ids)]

    fname = f"results/adjectives/acts/{repo.replace('/', '_')}__pers.pt"
    dr = torch.load(fname, map_location="cpu", weights_only=False)
    adjR = [str(a).lower() for a in dr["adjectives"]]
    X = torch.tensor(np.asarray(dr["acts"])[:, mid, :].astype(np.float32))
    X = X[[adjR.index(a.lower()) for a in kad]]

    print("\nlayer  diag_cos  z(diag)  top1  med_rank      (ENACT)")
    best = (None, -1e9)
    for l in sorted(J):
        C = ccos(Ek, Utok @ J[l].float())
        d = C.diagonal()
        off = C - torch.diag_embed(d)
        zsc = (d.mean().item() - off.mean().item()) / off.std().item()
        rank = (C >= d.unsqueeze(1)).sum(1)
        print(f"L{l:3d}  {d.mean():+.3f}   {zsc:5.2f}   "
              f"{(rank == 1).float().mean():.2f}  {rank.float().median():4.0f}")
        if zsc > best[1]:
            best = (l, zsc)
    L = best[0]
    assert L is not None
    Jl = J[L].float()
    V = Utok @ Jl

    CE = ccos(Ek, V)
    dE = CE.diagonal().numpy()
    CR = ccos(X, V)
    dR = CR.diagonal().numpy()
    print(f"\nbest lens layer L{L} (ENACT extraction layer = {mid})")
    rank = (CR >= CR.diagonal().unsqueeze(1)).sum(1)
    print(f"REPRESENT@L{mid}: diag {dR.mean():+.3f}, "
          f"top-1 {(rank == 1).float().mean():.2f} (echo-confounded)")

    def cov(a, key, src):
        return src.get(a, src.get(a.lower(), {})).get(key, np.nan)

    order = np.argsort(dE)
    print("\n--- ENACT far from own lens vector (bottom 15) / near (top 15) ---")
    for i in list(order[:15]) + list(order[-15:][::-1]):
        a = kad[i]
        nn = [kad[j] for j in CE[i].topk(4).indices.tolist() if j != i][:3]
        print(f"  {a:16s} cosE {dE[i]:+.3f}  cosR {dR[i]:+.3f}  "
              f"leak {cov(a, 'leak_rate', rep):.2f}  "
              f"enact {cov(a, 'enactability', ena):+.1f}  nn: {', '.join(nn)}")

    for key, src in (("leak_rate", rep), ("boot_cos_mean", rep)):
        x = np.array([cov(a, key, src) for a in kad])
        mask = ~np.isnan(x)
        print(f"corr(cosE, {key}): "
              f"{np.corrcoef(dE[mask], x[mask])[0, 1]:+.2f}")
    x = np.array([cov(a, "enactability", ena) for a in kad])
    mask = ~np.isnan(x)
    print(f"corr(cosE, enactability): {np.corrcoef(dE[mask], x[mask])[0, 1]:+.2f}")

    def topwords(zrow, k=10):
        out = []
        for t in zrow.topk(500).indices.tolist():
            s = tok.decode([t])
            if re.fullmatch(r" ?[A-Za-z][A-Za-z'-]+", s):
                out.append(s.strip())
            if len(out) == k:
                break
        return out

    SAMPLE = ([kad[i] for i in order[:6]] + [kad[i] for i in order[-6:]] +
              [a for a in ("funny", "honest", "quiet", "cruel", "curious", "lazy")
               if a in kad])
    for name, M in (("ENACT", Ek), ("REPRESENT", X)):
        Mc = M - M.mean(0)
        Lg = (Mc @ Jl.T) @ Wg.T
        Z = (Lg - Lg.mean(0)) / (Lg.std(0) + 1e-6)
        print(f"\n--- {name}: z-scored J-lens readout (top word tokens) ---")
        for a in dict.fromkeys(SAMPLE):
            i = kad.index(a)
            print(f"  {a:14s} [{dE[i]:+.2f}] " + ", ".join(topwords(Z[i])))

    json.dump({"adjectives": kad, "cosE": dE.tolist(), "cosR": dR.tolist(),
               "lens_layer": int(L)},
              open(f"results/steer_map/jlens_{m.lower()}_diag.json", "w"))
    print(f"\nsaved results/steer_map/jlens_{m.lower()}_diag.json")


if __name__ == "__main__":
    main()
