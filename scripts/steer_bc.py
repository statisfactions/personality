"""Valence-controlled binary-choice steering test (W17 capstone).

Likert showed ENACT steering is ~all valence. BC fixes that: a forced choice
between the target trait and a SAME-VALENCE distractor cancels the valence
component by construction (a pure valence steer can't shift the choice; only
trait-specific steering can). So we use the RAW cluster direction.

Per cluster: pref(target vs same-valence distractor) measured in BOTH A/B
orders (cancels position bias), baseline vs steered toward the target cluster.
Δpref>0 = trait-specific, valence-controlled steering works. Within-pair
control: steering the DISTRACTOR's cluster should move pref the other way.
"""
import json
import numpy as np
import torch
import hf_logprobs as hf

# tag -> (target probe, same-valence distractor). pos triad / neg triad.
PAIRS = {
    "imaginative": ("creative", "rational"),       # creativity (pos)
    "familiar": ("outgoing", "rational"),          # extraversion (pos)
    "rational": ("rational", "creative"),          # rationality (pos)
    "depressed": ("lonely", "rude"),               # distress (neg) — repaired
    "irritating": ("rude", "lonely"),              # antagonism (neg) — repaired
    "procrastinating": ("disorganized", "rude"),   # disorganization (neg) — repaired
}
SEL = json.load(open("instruments/trait_clusters.json"))["selected"]


def bc_prompt(a, b):
    return (f"Which is more characteristic of you?\n(A) I am {a}.\n"
            f"(B) I am {b}.\nAnswer with a single letter, A or B.")


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
    layer_mod = model.model.layers[mid - 1]

    # residual norm at mid
    enc = tok.apply_chat_template([{"role": "user",
                                    "content": bc_prompt("kind", "calm")}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        rnorm = model(**enc, output_hidden_states=True
                      ).hidden_states[mid][0, -1, :].norm().item()

    def hooked(vec, fn):
        h = None
        if vec is not None:
            delta = torch.tensor(vec, dtype=dtype, device=device)
            h = layer_mod.register_forward_hook(
                lambda m, i, o: (o[0] + delta,) + tuple(o[1:])
                if isinstance(o, tuple) else o + delta)
        try:
            return fn()
        finally:
            if h:
                h.remove()

    def pref(target, distr, vec):   # preference for target, position-debiased
        lo_ab = hooked(vec, lambda: hf.bc_logodds(
            model, tok, bc_prompt(target, distr), device)[0])   # target=A
        lo_ba = hooked(vec, lambda: hf.bc_logodds(
            model, tok, bc_prompt(distr, target), device)[0])   # target=B
        return (lo_ab - lo_ba) / 2.0

    print(f"mid {mid}, resid_norm≈{rnorm:.0f}\n")
    # scale sweep on creativity
    tgt, dis = PAIRS["imaginative"]
    base = pref(tgt, dis, None)
    print(f"sweep — steer creativity → pref({tgt} vs {dis}), base={base:+.2f}:")
    for a in (0.0, 0.08, 0.16, 0.32):
        print(f"  α={a:.2f}: Δpref={pref(tgt, dis, a*rnorm*cvec['imaginative'])-base:+.2f}")

    ALPHA = 0.16
    print(f"\nvalence-controlled BC steering, α={ALPHA}:")
    print(f"{'cluster':>16s} {'target vs distractor':>26s} {'Δpref(self)':>12s} "
          f"{'Δpref(distractor-steer)':>24s}")
    diag = []
    for tag, (tgt, dis) in PAIRS.items():
        b = pref(tgt, dis, None)
        d_self = pref(tgt, dis, ALPHA*rnorm*cvec[tag]) - b
        # within-pair control: steer the distractor's cluster
        dtag = next(k for k, (t, _) in PAIRS.items() if t == dis)
        d_ctrl = pref(tgt, dis, ALPHA*rnorm*cvec[dtag]) - b
        diag.append(d_self)
        print(f"{tag[:16]:>16s} {tgt+' vs '+dis:>26s} {d_self:>+12.2f} "
              f"{d_ctrl:>+24.2f}")
    print(f"\nmean Δpref(self) = {np.mean(diag):+.2f}  "
          "(valence-controlled trait-specific steering; >0 = works)")


if __name__ == "__main__":
    main()
