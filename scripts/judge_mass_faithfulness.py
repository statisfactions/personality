"""Is the position-0 digit readout faithful for JUDGE models that preamble?
(2026-11-01: 25 of 53 JUDGE-wide models put < 50% of next-token mass on the
digits; e.g. Llama-2 emits '▁I would rate...' with digit mass ~1e-6.)

For each model, a fixed sample of ordered pairs: greedy-generate the answer
(48 tokens), parse the first standalone 1-7, and compare with the stored
renormalized EV and argmax at position 0. Reports parse rate, r(EV, generated),
and exact argmax agreement.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/judge_mass_faithfulness.py \
           --models m1 m2 ... [--n 150]
Output: results/adjectives/introspect_full/mass_faithfulness.json (merged per model)
"""
import argparse, json, os, re
import numpy as np, torch
from hf_logprobs import load_model
from adjective_introspection import PROMPT

ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+")
ap.add_argument("--n", type=int, default=150); args = ap.parse_args()
D = "results/adjectives/introspect_full/"; OUT = D + "mass_faithfulness.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for m in args.models:
    z = np.load(D + m.replace("/", "_") + "_tom_likely_dir.npz", allow_pickle=True)
    adj = [str(a) for a in z["adjectives"]]; B, P, Mass = z["B"], z["P"].astype(float), z["Mass"].astype(float)
    rng = np.random.default_rng(0); n = len(adj); pairs = []
    while len(pairs) < args.n:
        i, j = rng.integers(n, size=2)
        if i != j: pairs.append((int(i), int(j)))
    model, tok, dev = load_model(m)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    gen, raw = [], []
    for i, j in pairs:
        msgs = [{"role": "user", "content": PROMPT["tom_likely"].format(a=adj[i].lower(), b=adj[j].lower())}]
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if t.rstrip().endswith("<think>"): t = t.rstrip() + "\n\n</think>\n\n"
        enc = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        s = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True); raw.append(s)
        mt = re.search(r"(?<![\d.])([1-7])(?![\d.])", s)
        gen.append(int(mt.group(1)) if mt else np.nan)
    gen = np.array(gen, float); ok = ~np.isnan(gen)
    ev = np.array([B[i, j] for i, j in pairs], float); am = np.array([P[i, j].argmax() + 1 for i, j in pairs])
    r = float(np.corrcoef(ev[ok], gen[ok])[0, 1]) if ok.sum() > 5 and gen[ok].std() > 0 else float("nan")
    res[m] = {"n": len(pairs), "parse_rate": float(ok.mean()), "r_ev_gen": r,
              "argmax_agree": float((am[ok] == gen[ok]).mean()) if ok.any() else float("nan"),
              "median_mass": float(np.median([Mass[i, j] for i, j in pairs])),
              "examples": raw[:5]}
    print(f"{m:46s} mass {res[m]['median_mass']:.3f} parse {ok.mean():.2f} r {r:.2f} "
          f"argmax {res[m]['argmax_agree']:.2f} | {raw[0][:70]!r}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    del model; torch.mps.empty_cache()
