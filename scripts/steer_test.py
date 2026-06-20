"""First steering test (W17 capstone, step A): does the OBSERVATIONAL
cluster-mean ENACT direction steer? llama3.2, judge-free Likert readout.

Scale sweep (find a working magnitude) + 8x8 specificity matrix (steer cluster
i, measure self-Likert on cluster j's probe adjective). Diagonal pop = steers;
off-diagonal flat = specific. Direction = mean of the cluster adjectives' ENACT
directions at mid layer, added at the residual (hook layers[mid-1]) scaled to a
fraction of the residual norm.
"""
import json
import numpy as np
import torch
import hf_logprobs as hf

SEL = json.load(open("instruments/trait_clusters.json"))["selected"]
# clean probe adjective per cluster tag (for the Likert readout)
PROBE = {"valuable": "wonderful", "imaginative": "creative",
         "familiar": "outgoing", "depressed": "lonely", "irritating": "rude",
         "rational": "rational", "procrastinating": "disorganized",
         "private": "quiet"}
TAGS = [c["tag"] for c in SEL]


def prompt(adj):
    return (f"Please rate how accurately this describes you, from 1 (very "
            f"inaccurate) to 5 (very accurate). Statement: I am {adj}. "
            f"Reply with a single number.")


def main():
    model, tok, device = hf.load_model("llama3.2")
    dtype = next(model.parameters()).dtype
    d = torch.load("results/persona_vectors/llama3.2_pda.pt", weights_only=False)
    mid = d["report"]["mid_layer"]
    dirs = d["directions"]
    cvec = {}
    for c in SEL:
        v = np.mean([dirs[a][mid].astype(np.float64) for a in c["members"]
                     if a in dirs], axis=0)
        cvec[c["tag"]] = v / np.linalg.norm(v)
    layer_mod = model.model.layers[mid - 1]   # output -> hidden_states[mid]

    # residual norm at mid (mean last-token over probe prompts)
    norms = []
    for adj in PROBE.values():
        enc = tok.apply_chat_template([{"role": "user", "content": prompt(adj)}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        norms.append(out.hidden_states[mid][0, -1, :].norm().item())
    rnorm = float(np.mean(norms))
    print(f"mid layer {mid}, hook layers[{mid-1}], resid_norm≈{rnorm:.0f}\n")

    def ev(adj, vec):
        h = None
        if vec is not None:
            delta = torch.tensor(vec, dtype=dtype, device=device)

            def hook(m, i, o):
                return (o[0] + delta,) + tuple(o[1:]) if isinstance(o, tuple) \
                    else o + delta
            h = layer_mod.register_forward_hook(hook)
        try:
            dist, _, _ = hf.likert_distribution(model, tok, prompt(adj), device,
                                                digits=("1", "2", "3", "4", "5"))
        finally:
            if h:
                h.remove()
        return sum(int(k) * v for k, v in dist.items())

    base = {t: ev(PROBE[t], None) for t in TAGS}

    # scale sweep on the eval cluster (diagonal cell)
    print("scale sweep — eval cluster steering → 'wonderful' self-Likert:")
    for a in (0.0, 0.04, 0.08, 0.16, 0.32):
        print(f"  α={a:.2f}: EV={ev('wonderful', a*rnorm*cvec['valuable']):.2f}"
              f"  (base {base['valuable']:.2f})")

    # 8x8 specificity matrix at a working alpha
    ALPHA = 0.16
    print(f"\n8×8 steering Δ(self-Likert) at α={ALPHA} "
          f"(rows=steered cluster, cols=probed trait); diagonal should pop:")
    print(f"{'steer\\probe':>14s}" + "".join(f"{t[:5]:>7s}" for t in TAGS))
    diag = []
    for ti in TAGS:
        vec = ALPHA * rnorm * cvec[ti]
        row = []
        for tj in TAGS:
            delta = ev(PROBE[tj], vec) - base[tj]
            row.append(delta)
            if ti == tj:
                diag.append(delta)
        print(f"{ti[:14]:>14s}" + "".join(f"{x:+7.2f}" for x in row))
    print(f"\nmean diagonal Δ = {np.mean(diag):+.2f}  "
          f"(steering toward a trait raises its own self-rating)")

    # --- valence-residualized: project the eval axis OUT of each direction ---
    veval = cvec["valuable"]
    rvec = {}
    for t in TAGS:
        r = cvec[t] - (cvec[t] @ veval) * veval
        rvec[t] = r / np.linalg.norm(r)
    print(f"\n8×8 with VALENCE (eval axis) projected OUT, α={ALPHA} "
          "(does the diagonal pop now?):")
    print(f"{'steer\\probe':>14s}" + "".join(f"{t[:5]:>7s}" for t in TAGS))
    diag2 = []
    for ti in TAGS:
        if ti == "valuable":
            continue
        vec = ALPHA * rnorm * rvec[ti]
        row = []
        for tj in TAGS:
            if tj == "valuable":
                continue
            delta = ev(PROBE[tj], vec) - base[tj]
            row.append((tj, delta))
            if ti == tj:
                diag2.append(delta)
        cells = "".join(f"{x:+7.2f}" for _, x in row)
        best = max(row, key=lambda kv: kv[1])[0]
        print(f"{ti[:14]:>14s}{cells}   max={best[:5]}")
    print(f"\nmean diagonal Δ (residualized) = {np.mean(diag2):+.2f}")


if __name__ == "__main__":
    main()
