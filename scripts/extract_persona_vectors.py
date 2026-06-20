"""Generation-based persona-vector extraction (Lu et al. 2026 recipe, scaled down).

For each adjective, induce a persona via system prompt ("You are someone who is
{adj}..."), sample rollouts to generic everyday questions, and average hidden
states over the *response* tokens. The per-adjective direction is

    d_adj = mean_adj - mean(role means)          (vs-mean; Lu et al.)

and the assistant axis is

    a = mean_default - mean(role means)          (Lu et al., explicit formula)

Crucially, vs-mean extraction does NOT assume antonym poles are antipodal —
the angle between d_wonderful and d_awful is a measurement, not a construction.
(Chen et al.'s pos-vs-neg differencing builds antipodality in; we keep it as a
derived comparison: d_pos - d_neg.)

Precision: hidden states are cast to fp32 on device BEFORE token averaging
(bf16 keeps ~3 significant digits; the trait component is ~0.15% of activation
norm, so bf16-accumulated means lose it — worst on Gemma-3 at ~1e5 norms).
Cross-rollout accumulation happens in fp64 on CPU (MPS has no fp64).

Diagnostics:
  - bootstrap coherence: resample rollouts B times, cosine(boot dir, full dir)
  - lexical leakage: rollouts whose text mentions the assigned adjective
  - W2 norm check: corr(per-rollout projection on d, per-rollout norm)
  - per-dim dominance: top-10 dims' share of ||d||^2

Usage (smoke):
  PYTHONPATH=scripts python scripts/extract_persona_vectors.py \
      --model llama3.2 --smoke --out results/persona_vectors
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch

import hf_logprobs as hf

# Generic everyday questions: occasions to *act* in character without inviting
# self-description (no "tell me about yourself" — that rewards narrating the
# assigned attribute, which would let unenactable adjectives fake a direction).
QUESTIONS = [
    "I have a free Saturday and no plans. What should I do with it?",
    "My coworker keeps taking credit for my ideas in meetings. How should I handle it?",
    "What's the best way to plan a dinner party for six people?",
    "I found a wallet on the sidewalk with $200 in it. What should I do?",
    "My friend wants to borrow my car for the weekend. What should I consider?",
    "How should I prepare for a job interview next week?",
    "My neighbor's dog barks all night. What would you do?",
    "I just got an unexpected bonus at work. Any thoughts on what to do with it?",
    "What advice would you give someone moving to a new city alone?",
    "A stranger in line at the grocery store starts chatting with me. How should I respond?",
    "I'm picking a gift for a colleague I don't know well. Any ideas?",
    "My team's project deadline just got moved up two weeks. What now?",
]

# Persona system prompts (Lu et al. used 5 variants per role).
SYS_TEMPLATES = [
    "You are someone who is {adj}. Respond to everything as this person would.",
    "Roleplay as a person who is deeply {adj}. Stay in character no matter what.",
    "Adopt the persona of someone {adj}. Answer all questions as that person.",
    "You are role-playing a character whose defining trait is being {adj}.",
    "Imagine you are a person who is {adj} to your core. Reply as that person.",
]

# Subtle register: naturalistic, downscaled ("a bit"), explicit not-a-
# performance. Still an instruction (all induction is roleplay; the default
# assistant is itself a persona — there is no zero), but the quietest one.
SUBTLE_TEMPLATES = [
    "You are a normal person who happens to be a bit {adj}. "
    "Just respond naturally, as yourself.",
    "You're an ordinary person — nothing unusual — though somewhat {adj}. "
    "Answer however you normally would.",
    "Respond as a regular person. If anything, you lean a little {adj}.",
    "You are an everyday person who is slightly {adj}. "
    "Reply naturally, not as a performance.",
    "Just be yourself: an ordinary person who happens to be a bit {adj}.",
]

# Plain register: bare attribution, no intensity word, no performance cue.
PLAIN_TEMPLATES = [
    "You are {adj}.",
    "You are a {adj} person.",
    "You are someone {adj}.",
    "Consider yourself {adj}.",
    "You tend to be {adj}.",
]

# Assistant-anchored register: keep the deployed HHH assistant as the base and
# add the trait as a perturbation. The drift this induces ((HHH+trait) −
# default) is the literal "how far does the assistant move toward X" — the
# assistant-drift measurement target, in the assistant's own frame.
ASSISTANT_TEMPLATES = [
    "You are a helpful, harmless, and honest AI assistant who also happens to "
    "be a bit {adj}. Respond as you normally would.",
    "You are an honest, helpful, harmless assistant. You also happen to be a "
    "little {adj}.",
    "You're a helpful, harmless, honest AI assistant — and you happen to be "
    "somewhat {adj}. Answer as usual.",
    "As a helpful, harmless, and honest assistant who is also a bit {adj}, "
    "respond to the user.",
    "You are a helpful, honest, harmless AI assistant that also happens to be "
    "slightly {adj}.",
]

# Candidate replacements for SUBTLE_TEMPLATES on the full run (the "normal
# person" wording collides with the 525-PDA traits normal/ordinary/natural/
# unusual). subtleA keeps an explicit HUMAN anchor with non-trait words
# (real/individual/person/actual/human; NOT "genuine" — that's a trait) —
# hypothesised to help express very-not-assistant traits. subtleB drops the
# anchor entirely — cleanest re collisions, but may let the model stay in
# assistant mode for anti-HHH traits.
SUBTLE_A_TEMPLATES = [
    "You are a real individual — just a person who happens to be a bit {adj}. "
    "Respond naturally, as yourself, not as a performance.",
    "You're an actual human being, not an assistant — you happen to be "
    "somewhat {adj}. Answer as yourself.",
    "You are a real person who happens to be a little {adj}. "
    "Reply naturally, not as a performance.",
]
SUBTLE_B_TEMPLATES = [
    "You happen to be a bit {adj}. Respond naturally, as yourself — "
    "this isn't a performance.",
    "You're somewhat {adj}. Just answer as you naturally would, "
    "not as a performance.",
    "You happen to be a little {adj}. Reply naturally, as yourself.",
]

INDUCTION = {"performance": SYS_TEMPLATES, "subtle": SUBTLE_TEMPLATES,
             "plain": PLAIN_TEMPLATES, "assistant": ASSISTANT_TEMPLATES,
             "subtleA": SUBTLE_A_TEMPLATES, "subtleB": SUBTLE_B_TEMPLATES}

SMOKE_ADJECTIVES = [
    # eval-antonym pair (the merge test)
    "wonderful", "awful",
    # a classic E pair + two singles
    "talkative", "quiet", "kind", "dishonest",
    # physical placebos (enactability negative controls)
    "slim", "handsome",
]

PLACEBOS = ["slim", "handsome", "tall", "muscular"]

# 30-adjective pole-spanning set: W15's 7-group design (pos/neg eval, warm,
# antagonism, distress, intellect) widened with E/C poles and the placebo arm.
SCALED30_ADJECTIVES = [
    "wonderful", "excellent", "great",            # pos_eval
    "awful", "terrible", "bad",                   # neg_eval
    "kind", "warm", "cruel", "rude",              # warm / antagonism
    "honest", "dishonest",                        # H pole pair
    "talkative", "quiet", "shy", "outgoing",      # E poles
    "anxious", "moody", "calm",                   # N / distress
    "organized", "careless", "thorough",          # C poles
    "creative", "curious", "intelligent",         # O / intellect
    "lazy", "energetic",                          # activity pair
    *PLACEBOS[:3],                                # slim, handsome, tall
]

DEFAULT_KEY = "__default__"


def build_inputs(tokenizer, system, question, device):
    """Chat-templated prompt; fall back to folding system into the user turn
    for templates that reject a system role (older Gemma)."""
    if system:
        try:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": question}]
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt")
            return _ids_tensor(ids).to(device)
        except Exception:
            question = system + "\n\n" + question
    messages = [{"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")
    return _ids_tensor(ids).to(device)


def _ids_tensor(ids):
    """apply_chat_template returns a BatchEncoding in newer transformers."""
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def rollout_mean_states(model, tokenizer, input_ids, device, max_new_tokens,
                        temperature, top_p, seed, skip_first=0, avg_window=0):
    """Generate one sampled rollout, then re-forward the full sequence and
    return (mean hidden states over response tokens, response text, n_resp).

    Returns hidden states of shape (n_layers+1, hidden_dim), fp32 on CPU.
    The fp32 cast happens BEFORE the token mean (see module docstring).
    """
    torch.manual_seed(seed)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq = out_ids[0]
    # Strip trailing eos/pad from the response span.
    end = seq.shape[0]
    while end > prompt_len and seq[end - 1].item() == tokenizer.eos_token_id:
        end -= 1
    start = prompt_len + skip_first
    if end - start < 1:
        return None, "", 0
    text = tokenizer.decode(seq[prompt_len:end], skip_special_tokens=True)

    # avg_window caps the averaged span at the first N response tokens, so
    # personas with very different response lengths (quiet: 10 tokens, slim:
    # 100) contribute comparable position ranges (length/position confound).
    end_w = min(end, start + avg_window) if avg_window > 0 else end
    with torch.no_grad():
        outputs = model(seq[:end].unsqueeze(0), output_hidden_states=True)
    states = torch.stack([
        hs[0, start:end_w, :].float().mean(dim=0)
        for hs in outputs.hidden_states
    ]).cpu()  # (n_layers+1, hidden_dim) fp32
    return states, text, end - start


def leaks(text, adj):
    """Crude lexical-leakage check: response mentions the assigned adjective
    (whole word, allowing simple suffixes: kind/kindness, quiet/quietly)."""
    return re.search(r"\b" + re.escape(adj.lower()) + r"\w{0,4}\b",
                     text.lower()) is not None


def bootstrap_coherence(X, g, f, B, rng):
    """Resample rollouts (rows of X, one layer) B times with the grand mean
    g held fixed; return cosines between each bootstrap direction and the
    full-data direction f. All fp64."""
    f = f / np.linalg.norm(f)
    n = X.shape[0]
    cos = np.empty(B)
    for b in range(B):
        d = X[rng.integers(0, n, n)].mean(axis=0) - g
        cos[b] = d @ f / np.linalg.norm(d)
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adjectives", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help=f"use the {len(SMOKE_ADJECTIVES)}-adjective smoke set")
    ap.add_argument("--scaled30", action="store_true",
                    help="use the 30-adjective pole-spanning set")
    ap.add_argument("--pda", action="store_true",
                    help="use the full 525-PDA adjective list (lowercased)")
    ap.add_argument("--placebos", nargs="+", default=PLACEBOS,
                    help="adjectives tagged as physical placebos in the report")
    ap.add_argument("--avg-window", type=int, default=0,
                    help="average only the first N response tokens (0 = all)")
    ap.add_argument("--tag", default=None, help="output filename tag")
    ap.add_argument("--induction", default="performance",
                    choices=list(INDUCTION),
                    help="persona system-prompt register (gradient of "
                         "induction strength)")
    ap.add_argument("--no-save-acts", action="store_true",
                    help="don't save per-rollout activations (full-525 runs: "
                         "~11GB/model; directions+diagnostics still saved)")
    ap.add_argument("--acts-stride", type=int, default=1,
                    help="keep per-rollout acts only at every Nth layer "
                         "(+ mid). Bounds RAM/disk on big models; condition "
                         "MEANS (and so directions) stay full-layer.")
    ap.add_argument("--n-questions", type=int, default=6)
    ap.add_argument("--n-sys", type=int, default=2,
                    help="persona system-prompt variants per adjective")
    ap.add_argument("--rollouts", type=int, default=2,
                    help="sampled rollouts per (sysprompt, question)")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--skip-first", type=int, default=0,
                    help="response tokens to drop from the front of the span")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/persona_vectors")
    args = ap.parse_args()

    if args.pda:
        from extract_adjectives import load_adjectives
        adjectives = sorted({a.lower() for a in load_adjectives()})
    else:
        adjectives = args.adjectives or (
            SMOKE_ADJECTIVES if args.smoke else
            SCALED30_ADJECTIVES if args.scaled30 else None)
    if not adjectives:
        ap.error("provide --adjectives, --smoke, --scaled30, or --pda")

    dtype = getattr(torch, args.dtype)
    model, tokenizer, device = hf.load_model(args.model, dtype=dtype)
    # get_text_config(): Gemma-3 is multimodal, layers live in text_config.
    n_layers = model.config.get_text_config().num_hidden_layers
    mid = (n_layers + 1) // 2  # middle of hidden_states (incl. embedding row)

    questions = QUESTIONS[:args.n_questions]
    sys_templates = INDUCTION[args.induction][:args.n_sys]
    conditions = [DEFAULT_KEY] + list(adjectives)

    # Per-rollout acts are kept (in RAM and on disk) only at `keep` layers;
    # condition means accumulate at ALL layers in fp64 running sums, so the
    # directions stay full-depth regardless of stride.
    keep = (sorted(set(range(0, n_layers + 1, args.acts_stride)) | {mid})
            if args.acts_stride > 1 else list(range(n_layers + 1)))
    mid_idx = keep.index(mid)

    tag = args.tag or ("smoke" if args.smoke else
                       "scaled30" if args.scaled30 else
                       "pda" if args.pda else f"{len(adjectives)}adj")
    if args.induction != "performance":  # keep performance files unsuffixed
        tag = f"{tag}_{args.induction}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-condition checkpoints: each completed condition is saved
    # immediately (atomic rename) and skipped on resume, so a crash loses at
    # most one adjective's rollouts. Seeds are derived per condition, not
    # from a global counter, so a resumed run generates the same rollouts an
    # uninterrupted one would have.
    ckpt_dir = out_dir / f"{args.model}_{tag}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    acts = {}    # condition -> list of (len(keep), hidden) fp32 arrays
    texts = {}   # condition -> list of {sys, question, text, n_resp, leak}
    sums = {}    # condition -> fp64 (n_layers+1, hidden) running sum
    for ci, cond in enumerate(conditions):
        ck = ckpt_dir / f"{cond}.pt"
        if ck.exists():
            blob = torch.load(ck, weights_only=False)
            if blob["keep"] == keep:
                acts[cond] = list(blob["acts"])
                texts[cond] = blob["texts"]
                sums[cond] = blob["sum"]
                print(f"  {cond}: resumed {len(acts[cond])} rollouts "
                      f"from checkpoint")
                continue
            print(f"  {cond}: checkpoint layer-set mismatch, re-running")
        seed = args.seed + ci * 100_000
        sys_prompts = [""] if cond == DEFAULT_KEY else \
            [t.format(adj=cond) for t in sys_templates]
        acts[cond], texts[cond] = [], []
        # Default condition gets the same rollout budget as one persona.
        per_sys = args.rollouts * (len(sys_templates) if cond == DEFAULT_KEY
                                   else 1)
        for sp in sys_prompts:
            for q in questions:
                input_ids = build_inputs(tokenizer, sp, q, device)
                for _ in range(per_sys):
                    seed += 1
                    states, text, n_resp = rollout_mean_states(
                        model, tokenizer, input_ids, device,
                        args.max_new_tokens, args.temperature, args.top_p,
                        seed, args.skip_first, args.avg_window)
                    if states is None:
                        continue
                    arr = states.numpy()
                    sums[cond] = sums.get(cond, 0.0) + arr.astype(np.float64)
                    acts[cond].append(arr[keep])
                    texts[cond].append({
                        "sys": sp, "question": q, "text": text,
                        "n_resp": n_resp,
                        "leak": cond != DEFAULT_KEY and leaks(text, cond),
                    })
        tmp = ck.with_suffix(".tmp")
        torch.save({"acts": np.stack(acts[cond]).astype(np.float32),
                    "texts": texts[cond], "sum": sums[cond],
                    "keep": keep}, tmp)
        os.replace(tmp, ck)
        n_leak = sum(t["leak"] for t in texts[cond])
        print(f"  {cond}: {len(acts[cond])} rollouts, {n_leak} leaky")
        if device == "mps":
            torch.mps.empty_cache()

    # --- Directions (fp64 on CPU, all layers) ------------------------------
    cond_means = {c: sums[c] / len(acts[c]) for c in conditions}
    role_means = [cond_means[a] for a in adjectives]
    grand_mean = np.mean(role_means, axis=0)        # mean of role means (Lu)
    directions = {a: cond_means[a] - grand_mean for a in adjectives}
    assistant_axis = cond_means[DEFAULT_KEY] - grand_mean

    # --- Massive-activation dims (Gemma-3: single dims at 1000x median) ----
    # These channels are condition-dependent on Gemma (they do NOT cancel in
    # the vs-mean contrast), so cosines that include them are artifact. We
    # flag dims with mean |act| >= 20x median at the mid layer and report
    # diagnostics both raw and with those dims zeroed.
    all_mid = np.concatenate([np.stack(acts[c])[:, mid_idx, :]
                              for c in conditions], axis=0).astype(np.float64)
    mean_abs = np.abs(all_mid).mean(axis=0)
    massive = np.where(mean_abs >= 20 * np.median(mean_abs))[0]
    mask = np.ones(all_mid.shape[1])
    mask[massive] = 0.0
    if len(massive):
        print(f"\n  !! {len(massive)} massive dims (>=20x median mean|act|) "
              f"at layer {mid}: {[int(i) for i in massive[:10]]}"
              f"{'...' if len(massive) > 10 else ''}")

    # --- Diagnostics at the middle layer -----------------------------------
    rng = np.random.default_rng(args.seed)
    report = {"model": args.model, "mid_layer": mid, "n_layers": n_layers,
              "massive_dims": [int(i) for i in massive],
              "adjectives": {}, "pairwise_cos_mid": {}, "assistant_axis": {}}

    def unit(v):
        return v / np.linalg.norm(v)

    print(f"\n=== diagnostics at layer {mid} (of {n_layers + 1} rows) ===")
    for a in adjectives:
        d = directions[a][mid]
        X = np.stack([r[mid_idx] for r in acts[a]]).astype(np.float64)
        cos = bootstrap_coherence(X, grand_mean[mid], d, args.bootstrap, rng)
        proj = X @ unit(d)
        norms = np.linalg.norm(X, axis=1)
        lens = np.array([t["n_resp"] for t in texts[a]], dtype=np.float64)
        norm_r = float(np.corrcoef(proj, norms)[0, 1])
        # zero length variance (all rollouts at the token cap) -> undefined
        len_r = (float(np.corrcoef(proj, lens)[0, 1])
                 if lens.std() > 0 else 0.0)
        Xa = X * mask
        proj_a = Xa @ unit(d * mask)
        norm_r_abl = float(np.corrcoef(proj_a, np.linalg.norm(Xa, axis=1))[0, 1])
        dom = float(np.sort(d ** 2)[-10:].sum() / (d ** 2).sum())
        leak_rate = float(np.mean([t["leak"] for t in texts[a]]))
        report["adjectives"][a] = {
            "n_rollouts": len(acts[a]),
            "placebo": a in args.placebos,
            "dir_norm": float(np.linalg.norm(d)),
            "boot_cos_mean": float(cos.mean()),
            "boot_cos_p05": float(np.percentile(cos, 5)),
            "norm_corr": norm_r,
            "norm_corr_ablated": norm_r_abl,
            "len_corr": len_r,
            "top10_dim_share": dom,
            "leak_rate": leak_rate,
        }
        print(f"  {a:>12s}: ||d||={np.linalg.norm(d):8.2f}  "
              f"boot_cos={cos.mean():.3f} (p05={np.percentile(cos, 5):.3f})  "
              f"norm_r={norm_r:+.2f}  len_r={len_r:+.2f}  "
              f"top10dim={dom:.2f}  leak={leak_rate:.2f}"
              + ("  [placebo]" if a in args.placebos else ""))

    report["pairwise_cos_mid_ablated"] = {}
    print("\n  pairwise cosines (mid layer, raw / massive-dims-zeroed):")
    for i, a in enumerate(adjectives):
        for b in adjectives[i + 1:]:
            c = float(unit(directions[a][mid]) @ unit(directions[b][mid]))
            ca = float(unit(directions[a][mid] * mask)
                       @ unit(directions[b][mid] * mask))
            report["pairwise_cos_mid"][f"{a}|{b}"] = c
            report["pairwise_cos_mid_ablated"][f"{a}|{b}"] = ca
    for k, v in sorted(report["pairwise_cos_mid"].items(),
                       key=lambda kv: -abs(kv[1])):
        print(f"    {k:>24s}: {v:+.3f} / "
              f"{report['pairwise_cos_mid_ablated'][k]:+.3f}")

    aa = assistant_axis[mid]
    report["assistant_axis"] = {
        "norm": float(np.linalg.norm(aa)),
        "cos_to_adjectives": {a: float(unit(aa) @ unit(directions[a][mid]))
                              for a in adjectives},
        "cos_to_adjectives_ablated": {
            a: float(unit(aa * mask) @ unit(directions[a][mid] * mask))
            for a in adjectives},
    }
    print(f"\n  assistant axis: ||a||={np.linalg.norm(aa):.2f} (raw / ablated)")
    for a in adjectives:
        print(f"    cos(a, {a}): "
              f"{report['assistant_axis']['cos_to_adjectives'][a]:+.3f} / "
              f"{report['assistant_axis']['cos_to_adjectives_ablated'][a]:+.3f}")

    # --- Save ---------------------------------------------------------------
    base = out_dir / f"{args.model}_{tag}"
    torch.save({
        "acts": (None if args.no_save_acts else
                 {c: np.stack(acts[c]).astype(np.float32)
                  for c in conditions}),
        "acts_layers": keep,
        "directions": {a: directions[a].astype(np.float32)
                       for a in adjectives},
        "assistant_axis": assistant_axis.astype(np.float32),
        "grand_mean": grand_mean.astype(np.float32),
        "report": report,
        "config": vars(args),
        "questions": questions,
        "sys_templates": sys_templates,
    }, f"{base}.pt")
    with open(f"{base}_texts.json", "w") as f:
        json.dump(texts, f, indent=1)
    with open(f"{base}_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsaved {base}.pt / _texts.json / _report.json")
    shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
