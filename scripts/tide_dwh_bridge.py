"""ΔW·h bridge (W19 §4): do Persona Cartography's OCEAN LoRAs displace
activations along OUR ENACT directions?

For each of their 10 souped adapters (5 traits x amp/sup, Llama-3.1-8B-it,
ocean_const_paired_dpo/-persona variant): apply ΔW = (alpha/r)·B@A in place,
teacher-force the 60 default-assistant rollouts (Llama8_pda_texts
__default__), capture layer-16 residual means over response tokens, and
take Δh = h_patched − h_base per text. Compare Δh against:
  - our ENACT span (523 winsorized dirs; top-k right singular vectors),
    in-span fraction vs random-vector null
  - trait-matched ENACT adjective directions (signed: amp +, sup −)
  - the assistant axis; amp/sup antipodality
Weight-space (their door) vs activation-space (ours): if fine-tuning's
displacement lives in the ENACT span, prompting, steering, and training
are three handles on one low-dim conduct subspace.

Usage: PYTHONPATH=scripts python scripts/tide_dwh_bridge.py
Output: results/tide_enact/Llama8_dwh.npz + printed analysis
"""
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import hf_logprobs as hf

OUT = "results/tide_enact"
REPO = "persona-cartography/monorepo"
BASE = "fine_tuning/llama-3.1-8b-it/ocean"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism"]
POLES = {"amplifier": "amplifying", "suppressor": "suppressing"}
MID = 16
MASSIVE = [290, 761, 2742, 4055]
ADJ_MATCH = {
    "openness": ["curious", "imaginative", "creative", "artistic",
                 "philosophical"],
    "conscientiousness": ["organized", "thorough", "responsible", "careful",
                          "practical"],
    "extraversion": ["extraverted", "outgoing", "talkative", "sociable",
                     "energetic"],
    "agreeableness": ["kind", "warm", "considerate", "cooperative",
                      "sympathetic"],
    "neuroticism": ["anxious", "nervous", "moody", "irritable", "insecure"],
}


def adapter_path(trait, pole):
    lora_dir = (f"{BASE}/{trait}/{pole}/ocean_const_paired_dpo/lora/"
                f"{trait}_{POLES[pole]}_full-persona")
    return lora_dir


def fetch_adapter(trait, pole):
    d = adapter_path(trait, pole)
    cfg = hf_hub_download(REPO, f"{d}/adapter_config.json",
                          repo_type="dataset", local_dir="dependencies/pc_monorepo")
    st = hf_hub_download(REPO, f"{d}/adapter_model.safetensors",
                         repo_type="dataset", local_dir="dependencies/pc_monorepo")
    return json.load(open(cfg)), st


def build_deltas(cfg, st_path):
    """Return {module_path: ΔW tensor} on CPU."""
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(st_path)
    deltas = {}
    for k in sd:
        if ".lora_A." not in k:
            continue
        kb = k.replace(".lora_A.", ".lora_B.")
        mod = k.split(".lora_A.")[0]
        mod = mod.replace("base_model.model.", "")
        A, B = sd[k].float(), sd[kb].float()
        deltas[mod] = (B @ A) * scale
    return deltas


def get_module(model, path):
    m = model
    for p in path.split("."):
        m = getattr(m, p)
    return m


@torch.no_grad()
def apply_deltas(model, deltas, sign):
    for path, dW in deltas.items():
        w = get_module(model, path).weight
        w += sign * dW.to(w.device, w.dtype)


@torch.no_grad()
def capture(model, tok, device, texts):
    """Mean layer-MID residual over assistant response tokens, per text."""
    outs = []
    for r in texts:
        msgs = [{"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["text"]}]
        pre_s = tok.apply_chat_template(msgs[:1], tokenize=False,
                                        add_generation_prompt=True)
        full_s = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False)
        pre = tok(pre_s, add_special_tokens=False).input_ids
        full = tok(full_s, add_special_tokens=False).input_ids
        ids = torch.tensor([full], device=device)
        hs = model(ids, output_hidden_states=True).hidden_states[MID][0]
        outs.append(hs[len(pre):].float().mean(0).cpu().numpy())
    return np.array(outs)


def main():
    os.makedirs(OUT, exist_ok=True)
    texts = [r for r in json.load(
        open("results/persona_vectors/Llama8_pda_texts.json"))["__default__"]]
    print(f"{len(texts)} baseline texts")

    model, tok, device = hf.load_model("Llama8", dtype=torch.bfloat16)
    H0 = capture(model, tok, device, texts)
    print("base captured", H0.shape)

    D = {}
    for trait in TRAITS:
        for pole in ("amplifier", "suppressor"):
            name = f"{trait[:4]}_{pole[:3]}"
            try:
                cfg, st = fetch_adapter(trait, pole)
            except Exception as e:
                print(f"[skip] {name}: {type(e).__name__} {str(e)[:120]}")
                continue
            deltas = build_deltas(cfg, st)
            apply_deltas(model, deltas, +1)
            H1 = capture(model, tok, device, texts)
            apply_deltas(model, deltas, -1)
            D[name] = (H1 - H0).mean(0)
            per = H1 - H0
            cons = np.mean([np.dot(per[i], D[name]) /
                            (np.linalg.norm(per[i]) * np.linalg.norm(D[name])
                             + 1e-9) for i in range(len(per))])
            print(f"  {name}: |Δh| {np.linalg.norm(D[name]):.1f}  "
                  f"per-text consistency {cons:.2f}", flush=True)
            del deltas

    np.savez_compressed(f"{OUT}/Llama8_dwh.npz",
                        **{k: v for k, v in D.items()},
                        base_mean=H0.mean(0))
    print(f"saved {OUT}/Llama8_dwh.npz")

    # ---------- analysis ----------
    ze = np.load("results/persona_vectors/enact_mid/Llama8.npz",
                 allow_pickle=True)
    E = ze["dir"].astype(np.float64)
    adjs = [str(a).lower() for a in ze["adjectives"]]
    axis = ze["axis"].astype(np.float64)

    def winso(X):
        X = np.atleast_2d(np.array(X, np.float64)).copy()
        keep = np.setdiff1d(np.arange(X.shape[1]), MASSIVE)
        std = E.std(0)
        cap = std[keep].max()
        for m_ in MASSIVE:
            if std[m_] > cap:
                X[:, m_] *= cap / std[m_]
        return X

    Ew = winso(E)
    U, S, Vt = np.linalg.svd(Ew - Ew.mean(0), full_matrices=False)
    rng = np.random.default_rng(5)

    print("\n== ΔW·h vs ENACT span ==")
    for k in (10, 45):
        P = Vt[:k]
        null = np.mean([np.linalg.norm(P @ v) ** 2
                        for v in rng.standard_normal((200, E.shape[1]))
                        / np.sqrt(E.shape[1])]) / 1.0
        print(f" k={k} (random-null in-span {null:.2f}):")
        for name, dh in D.items():
            v = winso(dh)[0]
            v = v / np.linalg.norm(v)
            print(f"   {name:9s} in-span {np.linalg.norm(P @ v)**2:.2f}")

    print("\n== trait-matched ENACT cosines (amp should be +, sup −) ==")
    for trait in TRAITS:
        match = [a for a in ADJ_MATCH[trait] if a in adjs]
        tdir = winso(Ew[[adjs.index(a) for a in match]].mean(0))[0]
        tdir /= np.linalg.norm(tdir)
        row = f"  {trait:17s} ({','.join(match[:3])}...)"
        for pole in ("amp", "sup"):
            name = f"{trait[:4]}_{pole}"
            if name in D:
                v = winso(D[name])[0]; v /= np.linalg.norm(v)
                row += f"  {pole} {np.dot(v, tdir):+.2f}"
        print(row)

    print("\n== amp/sup antipodality + assistant axis ==")
    ax = winso(axis)[0]; ax /= np.linalg.norm(ax)
    for trait in TRAITS:
        a, s = f"{trait[:4]}_amp", f"{trait[:4]}_sup"
        if a in D and s in D:
            va = winso(D[a])[0]; va /= np.linalg.norm(va)
            vs = winso(D[s])[0]; vs /= np.linalg.norm(vs)
            print(f"  {trait:17s} cos(amp,sup) {np.dot(va, vs):+.2f}   "
                  f"cos(amp,axis) {np.dot(va, ax):+.2f}  "
                  f"cos(sup,axis) {np.dot(vs, ax):+.2f}")


if __name__ == "__main__":
    main()
