"""Interpret REPRESENT's top PCs by their ENACT images (W17 §12).

REPRESENT's spectrum beyond the valence core was hard to read from input-side
adjective loadings and appeared model-idiosyncratic. This flips the lens: for
each top R-PC v_k, take its image W v_k under the fitted read->write map and
fingerprint it in ENACT space, where we have behaviorally-grounded coordinates:
the eval-antonym valence axis, ENACT's own PCs, the assistant axis, and
matched-filter axes for verbosity and asterisk/stage-direction formatting built
from the rollout texts.

Output: per model, per PC — transmitted gain, cosine profile against the
interpretable axes, top input-side adjectives (loadings) and top output-side
adjectives (ENACT directions aligned with the image). Saves
results/steer_map/rpc_fingerprints.json.

Usage: PYTHONPATH=scripts python scripts/rpc_fingerprints.py
"""
import json

import numpy as np
import torch
from sklearn.linear_model import Ridge

PAIRS = [("wonderful", "awful"), ("good", "bad"), ("kind", "cruel"),
         ("honest", "dishonest"), ("friendly", "unfriendly"),
         ("smart", "stupid")]
CASES = [("llama3.2", "meta-llama_Llama-3.2-3B-Instruct__pers.pt", "ablate"),
         ("qwen2.5", "Qwen_Qwen2.5-3B-Instruct__pers.pt", "ablate"),
         ("gemma3", "google_gemma-3-4b-it__pers.pt", "zscore")]
K = 200
N_PCS = 12
OUT = "results/steer_map/rpc_fingerprints.json"


def fingerprint(model, acts_file, scheme):
    d = torch.load(f"results/persona_vectors/{model}_pda.pt",
                   map_location="cpu", weights_only=False)
    mid, massive = d["report"]["mid_layer"], d["report"]["massive_dims"]
    adjs = sorted(d["report"]["adjectives"].keys())
    E = np.stack([d["directions"][a][mid] for a in adjs]).astype(np.float64)
    dr = torch.load(f"results/adjectives/acts/{acts_file}",
                    map_location="cpu", weights_only=False)
    adjR = [str(a).lower() for a in dr["adjectives"]]
    R = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
    R = R[[adjR.index(a) for a in adjs]]
    texts = json.load(open(f"results/persona_vectors/{model}_pda_texts.json"))
    if scheme == "ablate":
        E[:, massive] = 0
        R[:, massive] = 0
        sR = np.ones(R.shape[1])
        sE = np.ones(E.shape[1])
    else:
        sR = R.std(0) + 1e-8
        sE = E.std(0) + 1e-8
    Rz, Ez = (R - R.mean(0)) / sR, (E - E.mean(0)) / sE
    U, s, Vt = np.linalg.svd(Rz, full_matrices=False)
    W = Ridge(alpha=10.0).fit(Rz @ Vt[:K].T, Ez)
    Wm = W.coef_.T
    En = Ez / np.linalg.norm(Ez, axis=1, keepdims=True)
    ii = lambda a: adjs.index(a)

    vE = np.mean([(Ez[ii(a)] - Ez[ii(b)]) /
                  np.linalg.norm(Ez[ii(a)] - Ez[ii(b)]) for a, b in PAIRS],
                 axis=0)
    _, _, Ve = np.linalg.svd(Ez, full_matrices=False)
    aa = d["assistant_axis"][mid].astype(np.float64).copy()
    if scheme == "ablate":
        aa[massive] = 0
    else:
        aa = aa / sE
    lens = np.array([np.mean([len(e["text"]) for e in texts[a]])
                     for a in adjs])
    asts = np.array([np.mean([e["text"].count("*") / max(len(e["text"]), 1)
                              for e in texts[a]]) for a in adjs])
    mf = lambda y: ((y - y.mean())[:, None] * Ez).sum(0)
    axes = {"valence": vE, "E-PC1": Ve[0], "E-PC2": Ve[1], "E-PC3": Ve[2],
            "assist": aa, "verbose": mf(lens), "format*": mf(asts)}
    axes = {k: v / np.linalg.norm(v) for k, v in axes.items()}

    rows = []
    for k in range(N_PCS):
        m = Wm[k]
        mn = m / np.linalg.norm(m)
        lo = np.argsort(U[:, k])
        out = np.argsort(En @ mn)
        rows.append({
            "pc": k, "gain": float(np.linalg.norm(m)),
            "profile": {name: float(mn @ v) for name, v in axes.items()},
            "in_neg": [adjs[i] for i in lo[:5]],
            "in_pos": [adjs[i] for i in lo[-5:]],
            "out_neg": [adjs[i] for i in out[:5]],
            "out_pos": [adjs[i] for i in out[-5:]],
        })
    return rows


def main():
    result = {}
    for model, acts_file, scheme in CASES:
        rows = fingerprint(model, acts_file, scheme)
        result[model] = {"scheme": scheme, "pcs": rows}
        print(f"\n===== {model} ({scheme}) =====")
        for r in rows:
            prof = "  ".join(f"{n} {v:+.2f}" for n, v in r["profile"].items())
            print(f"PC{r['pc']:>2} gain {r['gain']:.2f} | {prof}")
            print(f"      in : [{','.join(r['in_neg'][:3])} | "
                  f"{','.join(r['in_pos'][-3:])}]")
            print(f"      out: [{','.join(r['out_neg'][:3])} | "
                  f"{','.join(r['out_pos'][-3:])}]")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
