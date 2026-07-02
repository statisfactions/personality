"""Steer with mapped-vs-recorded ENACT vectors on held-out adjectives (W17 §5).

Offline analysis (report_week17 §1-§4) showed ENACT directions are ~a low-rank
linear image of REPRESENT (held-out ridge R2 ~0.72), with a reliable but
human-orthogonal residual. This script asks the causal question: does the
mapped vector e_hat = W r steer persona behavior as well as the recorded ENACT
direction e? If yes, the functional content of a persona vector is exactly its
REPRESENT-predictable part. Controls: raw centered REPRESENT vector r and a
random unit vector (perturbation-size control).

Steering: add alpha * v (unit vectors, alpha = --frac of the mean residual
norm) to hidden_states[MID] — the output of decoder layer MID-1 — at every
position throughout generation. Evaluation questions are held out from
extraction (QUESTIONS[6:10]). The judge should be a different family from
--model; it rates each generation with a distributional-Likert EV over digits
1-7 for target-trait expression and for fluency.

Usage (PYTHONPATH=scripts), per model:
  python scripts/steer_mapped_vs_recorded.py --phase fit       --model llama3.2
  python scripts/steer_mapped_vs_recorded.py --phase calibrate --model llama3.2
  python scripts/steer_mapped_vs_recorded.py --phase generate  --model llama3.2 --frac 0.20
  python scripts/steer_mapped_vs_recorded.py --phase judge     --model llama3.2 --judge Qwen7 --frac 0.20
  python scripts/steer_mapped_vs_recorded.py --phase analyze   --model llama3.2 --frac 0.20
"""
import argparse
import json
import os
import zlib

import numpy as np
import torch

import hf_logprobs as hf
from extract_persona_vectors import QUESTIONS, build_inputs

OUT_DIR = "results/steer_map"
PDA_DIR = "results/persona_vectors"
ACTS_DIR = "results/adjectives/acts"
EVAL_QUESTIONS = QUESTIONS[6:10]  # held out from extraction (which used [:6])
# recorded_abl (massive dims zeroed) only exists for zscore-scheme bundles;
# generate/analyze take conditions from the vector bundle, not this constant.
CONDS = ("recorded", "recorded_abl", "mapped", "repr", "random")
DIGITS = ("1", "2", "3", "4", "5", "6", "7")
N_STRATUM = 12   # test adjectives per stratum (lowest / highest boot_cos)
RIDGE_K = 200    # REPRESENT PCs fed to the ridge map
RIDGE_ALPHA = 10.0


def _seed(*key):
    """Deterministic across processes (hash() is salted per interpreter)."""
    return zlib.crc32(repr(key).encode()) % (2**31)


def _vec_path(model):
    return f"{OUT_DIR}/{model}_vectors.npz"


def _gens_path(model, frac):
    return f"{OUT_DIR}/{model}_gens_f{int(round(frac * 100)):02d}.json"


def _acts_path(model):
    return f"{ACTS_DIR}/{hf.resolve(model).replace('/', '_')}__pers.pt"


def _decoder_layers(model):
    for attr in ("model", "language_model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "layers"):
            return m.layers
        if m is not None:
            for attr2 in ("model", "language_model"):
                m2 = getattr(m, attr2, None)
                if m2 is not None and hasattr(m2, "layers"):
                    return m2.layers
    raise AttributeError("cannot locate decoder layers")


def phase_fit(args):
    """Hold out the N lowest/highest-boot_cos adjectives, fit ridge
    W: REPRESENT -> ENACT on the rest, save test-set vector bundles.

    --denoise ablate: zero massive dims in both channels before fitting
      (fine for Llama/Qwen where they carry ~nothing; deletes 55-76% of
      centered variance on Gemma — do not use there).
    --denoise zscore: per-dim standardize both channels (train stats), fit in
      z-space, de-standardize predictions back to raw space. Keeps the
      massive-channel content (which on Gemma is most of the signal) without
      letting a 1000x-median dim dominate the least squares. Saved vectors
      are raw-space either way; steering happens in raw activation space."""
    from sklearn.linear_model import Ridge

    d = torch.load(f"{PDA_DIR}/{args.model}_pda.pt", map_location="cpu",
                   weights_only=False)
    mid = d["report"]["mid_layer"]
    massive = d["report"]["massive_dims"]
    rep = d["report"]["adjectives"]
    adjs = sorted(rep.keys())
    boot = np.array([rep[a]["boot_cos_mean"] for a in adjs])

    E = np.stack([d["directions"][a][mid] for a in adjs]).astype(np.float64)
    dr = torch.load(_acts_path(args.model), map_location="cpu",
                    weights_only=False)
    adjR = [str(a).lower() for a in dr["adjectives"]]
    R = np.asarray(dr["acts"])[:, mid, :].astype(np.float64)
    R = R[[adjR.index(a) for a in adjs]]
    if args.denoise == "ablate":
        E[:, massive] = 0.0
        R[:, massive] = 0.0

    o = np.argsort(boot)
    test_names = [adjs[i] for i in o[:N_STRATUM]] + \
                 [adjs[i] for i in o[-N_STRATUM:]]
    te = np.array([adjs.index(a) for a in test_names])
    tr = np.array([i for i in range(len(adjs)) if i not in set(te)])

    Rmu, Emu = R[tr].mean(0), E[tr].mean(0)
    if args.denoise == "zscore":
        sR = R[tr].std(0) + 1e-8
        sE = E[tr].std(0) + 1e-8
    else:
        sR = np.ones(R.shape[1])
        sE = np.ones(E.shape[1])
    _, _, Vt = np.linalg.svd((R[tr] - Rmu) / sR, full_matrices=False)
    feats = lambda X: ((X - Rmu) / sR) @ Vt[:RIDGE_K].T
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(feats(R[tr]), (E[tr] - Emu) / sE)
    Ehat = ridge.predict(feats(R[te])) * sE + Emu

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    for j, a in enumerate(test_names):
        print(f"  {a:>16} boot {boot[adjs.index(a)]:.2f} "
              f"cos(ehat,e) {cos(Ehat[j], E[adjs.index(a)]):+.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(_vec_path(args.model),
             adjs=np.array(test_names), Ehat=Ehat,
             E=np.stack([E[adjs.index(a)] for a in test_names]),
             Rc=np.stack([R[adjs.index(a)] - Rmu for a in test_names]),
             boot=np.array([boot[adjs.index(a)] for a in test_names]),
             massive=np.array(massive), mid=mid,
             scheme=np.array(args.denoise))
    print(f"wrote {_vec_path(args.model)} (denoise={args.denoise})")


def load_vectors(model_name):
    z = np.load(_vec_path(model_name), allow_pickle=True)
    adjs = [str(a) for a in z["adjs"]]
    massive = list(z["massive"]) if "massive" in z.files else []
    scheme = str(z["scheme"]) if "scheme" in z.files else "ablate"
    rng = np.random.default_rng(0)
    vecs = {}
    for j, a in enumerate(adjs):
        e_abl = z["E"][j].copy()
        e_abl[massive] = 0.0  # no-op for legacy ablate-scheme bundles
        raw = {"recorded": z["E"][j], "recorded_abl": e_abl,
               "mapped": z["Ehat"][j], "repr": z["Rc"][j],
               "random": rng.standard_normal(z["E"].shape[1])}
        vecs[a] = {k: (v / np.linalg.norm(v)).astype(np.float32)
                   for k, v in raw.items()}
    if scheme == "ablate":
        # recorded is already massive-ablated; the _abl condition is identical
        for a in vecs:
            vecs[a].pop("recorded_abl")
    return adjs, vecs, int(z["mid"]), list(z["boot"])


class Steerer:
    """Adds alpha*v to the output of one decoder layer while active."""

    def __init__(self, model, hs_index, device, dtype):
        # hidden_states[i] = output of decoder layer i-1
        self.layer = _decoder_layers(model)[hs_index - 1]
        self.device, self.dtype = device, dtype
        self.v = None
        self.alpha = 0.0
        self.handle = self.layer.register_forward_hook(self._hook)

    def set(self, v, alpha):
        self.v = (None if v is None else
                  torch.tensor(v, device=self.device, dtype=self.dtype))
        self.alpha = alpha

    def _hook(self, module, inp, out):
        if self.v is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.alpha * self.v,) + out[1:]
        return out + self.alpha * self.v

    def remove(self):
        self.handle.remove()


def residual_norm(model, tok, device, hs_index):
    ids = build_inputs(tok, "", EVAL_QUESTIONS[0], device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return float(out.hidden_states[hs_index][0].float().norm(dim=-1).mean())


def generate_one(model, tok, device, question, seed, max_new_tokens=100):
    torch.manual_seed(seed)
    ids = build_inputs(tok, "", question, device)
    with torch.no_grad():
        out = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.95,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def phase_calibrate(args):
    adjs, vecs, mid, _ = load_vectors(args.model)
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    rn = residual_norm(model, tok, device, mid)
    print(f"mean residual norm at hidden_states[{mid}]: {rn:.1f}")
    steer = Steerer(model, mid, device, torch.bfloat16)
    for adj in (adjs[-7], adjs[5]):  # one clean, one clumsy
        for frac in (0.15, 0.3, 0.6):
            steer.set(vecs[adj]["recorded"], frac * rn)
            for qi in (0, 1):
                text = generate_one(model, tok, device, EVAL_QUESTIONS[qi],
                                    seed=1000 + qi)
                print(f"\n=== {adj} frac={frac} q{qi} ===\n{text}")
    steer.set(None, 0.0)
    for qi in (0, 1):
        print(f"\n=== baseline q{qi} ===\n"
              f"{generate_one(model, tok, device, EVAL_QUESTIONS[qi], seed=1000 + qi)}")
    steer.remove()


def phase_generate(args):
    adjs, vecs, mid, _ = load_vectors(args.model)
    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    rn = residual_norm(model, tok, device, mid)
    alpha = args.frac * rn
    print(f"alpha = {args.frac} * {rn:.1f} = {alpha:.2f}")
    steer = Steerer(model, mid, device, torch.bfloat16)
    gens = []
    conds = [c for c in CONDS if c in vecs[adjs[0]]]
    n_total = len(adjs) * len(conds) * len(EVAL_QUESTIONS) * args.rollouts
    done = 0
    for adj in adjs:
        for cond in conds:
            steer.set(vecs[adj][cond], alpha)
            for qi, q in enumerate(EVAL_QUESTIONS):
                for r in range(args.rollouts):
                    text = generate_one(model, tok, device, q,
                                        _seed(adj, cond, qi, r))
                    gens.append({"adj": adj, "cond": cond, "q": qi,
                                 "rollout": r, "text": text})
                    done += 1
            print(f"  {adj}/{cond}  ({done}/{n_total})", flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    steer.set(None, 0.0)
    for qi, q in enumerate(EVAL_QUESTIONS):
        for r in range(args.rollouts):
            gens.append({"adj": None, "cond": "baseline", "q": qi,
                         "rollout": r,
                         "text": generate_one(model, tok, device, q,
                                              _seed("__baseline__", qi, r))})
    steer.remove()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_gens_path(args.model, args.frac), "w") as f:
        json.dump({"model": args.model, "frac": args.frac, "alpha": alpha,
                   "mid": mid, "questions": EVAL_QUESTIONS, "gens": gens},
                  f, indent=1)
    print(f"wrote {_gens_path(args.model, args.frac)} ({len(gens)} generations)")


def _judge_ev(model, tok, device, prompt):
    """Expected value over the digit distribution (likert_distribution's
    third return is entropy, not EV)."""
    dist, _, _ = hf.likert_distribution(model, tok, prompt, device,
                                        digits=DIGITS)
    tot = sum(dist.values())
    return sum(int(k) * p for k, p in dist.items()) / tot


def phase_judge(args):
    data = json.load(open(_gens_path(args.model, args.frac)))
    gens = data["gens"]
    adjs = sorted({g["adj"] for g in gens if g["adj"]})
    model, tok, device = hf.load_model(args.judge, dtype=torch.bfloat16)

    def target_prompt(q, text, adj):
        return (f'Someone was asked: "{q}"\n\nThey replied:\n"""{text}"""\n\n'
                f"On a scale from 1 to 7, how {adj} does this person seem? "
                f"Answer with a single digit from 1 to 7.")

    def fluency_prompt(q, text):
        return (f'Someone was asked: "{q}"\n\nThey replied:\n"""{text}"""\n\n'
                f"On a scale from 1 (garbled nonsense) to 7 (perfectly clear "
                f"and coherent), how coherent is this reply? "
                f"Answer with a single digit from 1 to 7.")

    for i, g in enumerate(gens):
        q = data["questions"][g["q"]]
        g["fluency"] = _judge_ev(model, tok, device,
                                 fluency_prompt(q, g["text"]))
        if g["adj"]:
            g["target_ev"] = _judge_ev(model, tok, device,
                                       target_prompt(q, g["text"], g["adj"]))
        else:  # baseline: rate on every test adjective
            g["target_ev"] = {a: _judge_ev(model, tok, device,
                                           target_prompt(q, g["text"], a))
                              for a in adjs}
        if (i + 1) % 50 == 0:
            print(f"  judged {i + 1}/{len(gens)}", flush=True)
    data["judge"] = args.judge
    out_path = _gens_path(args.model, args.frac).replace("_gens_", "_judged_")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {out_path}")


def phase_analyze(args):
    data = json.load(open(
        _gens_path(args.model, args.frac).replace("_gens_", "_judged_")))
    gens = data["gens"]
    adjs, vecs, _, boot = load_vectors(args.model)
    conds = [c for c in CONDS if c in vecs[adjs[0]]]
    base = [g for g in gens if g["cond"] == "baseline"]
    base_ev = {a: np.mean([g["target_ev"][a] for g in base]) for a in adjs}
    base_flu = np.mean([g["fluency"] for g in base])
    print(f"baseline fluency {base_flu:.2f}")
    rows = {}
    for a in adjs:
        rows[a] = {}
        for cond in conds:
            sel = [g for g in gens if g["adj"] == a and g["cond"] == cond]
            rows[a][cond] = {
                "delta": np.mean([g["target_ev"] for g in sel]) - base_ev[a],
                "fluency": np.mean([g["fluency"] for g in sel]),
            }
    clumsy, clean = adjs[:N_STRATUM], adjs[N_STRATUM:]
    hdr = "".join(f"{c:>18}" for c in conds)
    print(f"\n{'adj':>16} {'boot':>5}" + hdr + "   (delta target EV / fluency)")
    for grp, name in ((clumsy, "CLUMSY"), (clean, "CLEAN")):
        print(f"--- {name} ---")
        for a in grp:
            b = boot[adjs.index(a)]
            cells = "".join(
                f"  {rows[a][c]['delta']:+5.2f}/{rows[a][c]['fluency']:4.1f}   "
                for c in conds)
            print(f"{a:>16} {b:5.2f}" + cells)
        for c in conds:
            dd = np.mean([rows[a][c]["delta"] for a in grp])
            fl = np.mean([rows[a][c]["fluency"] for a in grp])
            print(f"{name + ' mean ' + c:>40}: delta {dd:+.2f}  fluency {fl:.2f}")
    out = {"model": args.model, "frac": args.frac, "baseline_ev": base_ev,
           "baseline_fluency": base_flu, "rows": rows}
    path = _gens_path(args.model, args.frac).replace("_gens_", "_analysis_")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["fit", "calibrate", "generate", "judge",
                             "analyze"])
    ap.add_argument("--model", default="llama3.2",
                    help="steered model (hf_logprobs short name)")
    ap.add_argument("--judge", default="Qwen7",
                    help="judge model — use a different family than --model")
    ap.add_argument("--denoise", default="zscore",
                    choices=["ablate", "zscore"],
                    help="fit-phase massive-dim handling (zscore keeps their "
                         "content; mandatory for Gemma)")
    ap.add_argument("--frac", type=float, default=0.20,
                    help="steering norm as fraction of mean residual norm")
    ap.add_argument("--rollouts", type=int, default=2,
                    help="rollouts per (adjective, condition, question)")
    args = ap.parse_args()
    globals()[f"phase_{args.phase}"](args)


if __name__ == "__main__":
    main()
